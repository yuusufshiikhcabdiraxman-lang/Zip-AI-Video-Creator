import os
import json
import uuid
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Zip-AI Video Creator", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).resolve().parent
JOBS = BASE / "video_jobs"
HISTORY = BASE / "history.json"
JOBS.mkdir(exist_ok=True)

CLIP_SECONDS = 10
MAX_DURATION = 120


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    aspect_ratio: str = "9:16"
    duration: int = Field(default=10, ge=10, le=120)
    quality: str = "high"
    language: str = "so"
    image_url: Optional[str] = None


def env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(500, f"{name} is not configured")
    return value


def history():
    if not HISTORY.exists():
        return []
    try:
        data = json.loads(HISTORY.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(data):
    HISTORY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def error_text(data):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        x = data.get("detail") or data.get("error") or data.get("message")
        if isinstance(x, dict):
            return error_text(x)
        if x:
            return str(x)
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)


def create_prediction(prompt, aspect, quality, language, image_url=None):
    token = env("REPLICATE_API_TOKEN")
    version = env("REPLICATE_MODEL_VERSION")
    payload = {
        "version": version,
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect,
            "duration": CLIP_SECONDS,
            "quality": quality,
            "language": language,
        },
    }
    if image_url:
        payload["input"]["image"] = image_url

    try:
        r = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json=payload, timeout=60
        )
        data = r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"Replicate connection error: {e}")
    except ValueError:
        raise HTTPException(502, "Replicate returned invalid JSON.")

    if not r.ok:
        raise HTTPException(r.status_code, error_text(data))
    if not data.get("id"):
        raise HTTPException(502, "Replicate did not return a prediction ID.")
    return data


def get_prediction(pid):
    token = env("REPLICATE_API_TOKEN")
    try:
        r = requests.get(
            f"https://api.replicate.com/v1/predictions/{pid}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        data = r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Replicate connection error: {e}")
    except ValueError:
        raise RuntimeError("Replicate returned invalid JSON.")
    if not r.ok:
        raise RuntimeError(error_text(data))
    return data


def output_url(output):
    if isinstance(output, str) and output.startswith(("http://", "https://")):
        return output
    if isinstance(output, list):
        for x in output:
            if isinstance(x, str) and x.startswith(("http://", "https://")):
                return x
    if isinstance(output, dict):
        for x in output.values():
            if isinstance(x, str) and x.startswith(("http://", "https://")):
                return x
    return None


def download(url, path):
    try:
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not download generated clip: {e}")


def merge_clips(clips, final):
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is not installed. Add FFmpeg to Render before using 2-minute videos."
        )
    concat = final.parent / "concat.txt"
    lines = [f"file '{str(p).replace(chr(92), '/').replace(chr(39), chr(39)+chr(92)+chr(39))}'"
             for p in clips]
    concat.write_text("\n".join(lines), encoding="utf-8")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", str(final)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=900
        )
        if r.returncode:
            raise RuntimeError("FFmpeg merge failed: " + r.stderr[-2000:])
    finally:
        concat.unlink(missing_ok=True)


def run_job(job_id, req):
    jobdir = JOBS / job_id
    jobdir.mkdir(parents=True, exist_ok=True)
    count = (req.duration + 9) // 10

    record = {
        "id": job_id, "prompt": req.prompt, "duration": req.duration,
        "total_clips": count, "status": "processing", "progress": 0,
        "aspect_ratio": req.aspect_ratio, "quality": req.quality,
        "language": req.language, "output": None, "error": None,
    }
    h = history(); h.insert(0, record); save_history(h[:50])
    clips = []

    try:
        for i in range(count):
            segment_prompt = (
                f"{req.prompt}\n\n"
                f"This is segment {i+1} of {count} of one continuous video. "
                "Keep the same characters, location, clothing, style, lighting, "
                "camera language and visual identity. Continue naturally."
            )
            p = create_prediction(
                segment_prompt, req.aspect_ratio, req.quality,
                req.language, req.image_url if i == 0 else None
            )
            pid = p["id"]
            while True:
                data = get_prediction(pid)
                status = data.get("status")
                if status == "succeeded":
                    url = output_url(data.get("output"))
                    if not url:
                        raise RuntimeError(f"Clip {i+1} finished without a video URL.")
                    path = jobdir / f"clip_{i+1:03d}.mp4"
                    download(url, path)
                    clips.append(path)
                    break
                if status in ("failed", "canceled"):
                    raise RuntimeError(f"Clip {i+1} failed: {error_text(data.get('error'))}")
                time.sleep(5)

            record["progress"] = int(((i + 1) / count) * 90)
            h = history()
            for x in h:
                if x.get("id") == job_id:
                    x.update(record)
            save_history(h[:50])

        final = jobdir / "final.mp4"
        merge_clips(clips, final)

        # Exact trimming for durations such as 15, 25, 35 seconds.
        if req.duration % 10:
            exact = jobdir / "final_exact.mp4"
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(final), "-t", str(req.duration),
                 "-c", "copy", "-movflags", "+faststart", str(exact)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300
            )
            if r.returncode == 0:
                final.unlink(missing_ok=True)
                exact.rename(final)

        record["status"] = "succeeded"
        record["progress"] = 100
        record["output"] = f"/api/jobs/{job_id}/video"

    except Exception as e:
        record["status"] = "failed"
        record["error"] = str(e)

    h = history()
    for x in h:
        if x.get("id") == job_id:
            x.update(record)
    save_history(h[:50])
    return record


@app.post("/api/generate")
def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    # Save the job immediately so the browser can poll its progress.
    count = (req.duration + 9) // 10
    record = {
        "id": job_id,
        "prompt": req.prompt,
        "duration": req.duration,
        "total_clips": count,
        "status": "queued",
        "progress": 0,
        "aspect_ratio": req.aspect_ratio,
        "quality": req.quality,
        "language": req.language,
        "output": None,
        "error": None,
    }
    h = history()
    h.insert(0, record)
    save_history(h[:50])

    background_tasks.add_task(run_job, job_id, req)

    return {
        "success": True,
        "job_id": job_id,
        "duration": req.duration,
        "total_clips": count,
        "status": "queued",
    }


@app.get("/api/jobs/{job_id}")
def job(job_id):
    for x in history():
        if x.get("id") == job_id:
            return x
    raise HTTPException(404, "Job not found.")


@app.get("/api/jobs/{job_id}/video")
def video(job_id):
    path = JOBS / job_id / "final.mp4"
    if not path.exists():
        raise HTTPException(404, "Final video is not ready.")
    return FileResponse(path, media_type="video/mp4", filename=f"zip_ai_{job_id}.mp4")


@app.get("/api/history")
def api_history():
    return history()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "The image is empty.")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image is too large. Maximum size is 15MB.")

    import base64
    data = base64.b64encode(content).decode("ascii")
    return {"success": True, "url": f"data:{file.content_type};base64,{data}"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "clip_seconds": CLIP_SECONDS,
        "max_duration": MAX_DURATION,
        "ffmpeg": bool(shutil.which("ffmpeg")),
    }


HTML = r"""
<!doctype html><html lang="so"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zip-AI Video Creator</title>
<style>
body{margin:0;background:linear-gradient(180deg,#18082a,#08080b);color:white;font-family:Arial}
.wrap{max-width:900px;margin:auto;padding:40px 16px}.logo{width:65px;height:65px;margin:auto;border-radius:18px;
display:flex;align-items:center;justify-content:center;font-size:36px;font-weight:bold;background:#8b36ee}
h1{text-align:center;font-size:40px}.sub{text-align:center;color:#aaa}.card{margin-top:20px;padding:22px;
background:#15131a;border:1px solid #312d37;border-radius:18px}label{display:block;margin:13px 0 7px;color:#ccc}
textarea,input,select{width:100%;padding:13px;border-radius:10px;border:1px solid #36313c;background:#0d0c10;color:white;font-size:16px;box-sizing:border-box}
textarea{min-height:150px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.tabs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{border:0;border-radius:11px;padding:15px;color:white;font-weight:bold;cursor:pointer}.tab{background:#1b1820;border:1px solid #393440}
.active{background:#35165d}.primary{width:100%;margin-top:20px;background:#8d36ef;font-size:17px}
.msg{margin-top:15px;padding:14px;border:1px solid #39323f;border-radius:10px;white-space:pre-wrap;word-break:break-word}
.hidden{display:none}.small{color:#928c99;font-size:13px;margin-top:7px}.video{margin-top:10px;padding:13px;background:#100f13;border:1px solid #2d2932;border-radius:10px}
a{color:#c27aff}@media(max-width:650px){.grid{grid-template-columns:1fr}h1{font-size:31px}}
</style></head><body><div class="wrap">
<div class="logo">Z</div><h1>Zip-AI Video Creator</h1>
<div class="sub">Hal prompt → 10-second clips → HAL video ilaa 2 daqiiqo.</div>
<div class="card">
<div class="tabs"><button id="t" class="tab active" onclick="mode='text';toggle()">Qoraal → Video</button>
<button id="i" class="tab" onclick="mode='image';toggle()">Sawir → Video</button></div>
<label>Prompt</label><textarea id="prompt" placeholder="Qor hal prompt oo qeexaya muuqaalka..."></textarea>
<div id="ibox" class="hidden"><label>Sawir</label><input id="image" type="file" accept="image/*">
<div id="up" class="small"></div></div>
<div class="grid"><div><label>Luuqad</label><select id="lang"><option value="so">Somali</option><option value="en">English</option><option value="ar">Arabic</option></select></div>
<div><label>Aspect Ratio</label><select id="aspect"><option>9:16</option><option>16:9</option><option>1:1</option></select></div>
<div><label>Duration</label><select id="duration">
<option value="10">10 seconds</option><option value="20">20 seconds</option><option value="30">30 seconds</option>
<option value="40">40 seconds</option><option value="50">50 seconds</option><option value="60">60 seconds — 1 minute</option>
<option value="70">70 seconds</option><option value="80">80 seconds</option><option value="90">90 seconds</option>
<option value="100">100 seconds</option><option value="110">110 seconds</option><option value="120" selected>120 seconds — 2 minutes</option>
</select><div class="small">Kling: 10s clip kasta, kadib FFmpeg ayaa isku dara.</div></div>
<div><label>Quality</label><select id="quality"><option value="high">High</option><option value="fast">Fast</option></select></div></div>
<button id="go" class="primary" onclick="generate()">✨ Samee Video</button><div id="msg" class="msg hidden"></div>
</div><div class="card"><h2>History</h2><div id="hist"></div></div>
</div>
<script>
let mode="text",imageUrl=null;
function toggle(){document.getElementById("t").classList.toggle("active",mode==="text");document.getElementById("i").classList.toggle("active",mode==="image");document.getElementById("ibox").classList.toggle("hidden",mode!=="image")}
function show(x,err=false){let m=document.getElementById("msg");m.classList.remove("hidden");m.textContent=x;m.style.borderColor=err?"#71303a":"#39323f"}
function et(d){let x=d&&(d.detail??d.error??d.message??d);if(typeof x==="object"){try{return JSON.stringify(x,null,2)}catch(e){}}return String(x||"Unknown error")}
document.getElementById("image").addEventListener("change",async e=>{
let f=e.target.files[0];if(!f)return;let fd=new FormData();fd.append("file",f);
try{let r=await fetch("/api/upload",{method:"POST",body:fd}),d=await r.json();if(!r.ok)throw Error(et(d));imageUrl=d.url;document.getElementById("up").textContent="✅ Sawirka waa diyaar."}catch(x){imageUrl=null;document.getElementById("up").textContent="❌ "+x.message}});
async function generate(){
let b=document.getElementById("go"),p=document.getElementById("prompt").value.trim(),duration=Number(document.getElementById("duration").value);
if(!p)return show("Fadlan geli prompt.",true);
if(mode==="image"&&!imageUrl)return show("Fadlan geli sawirka.",true);

b.disabled=true;
b.textContent="⏳ Video-ga waa la diyaarinayaa...";
show("⏳ Job-ka waa la bilaabay.\n\nWaxaa la samaynayaa "+Math.ceil(duration/10)+" clips oo 10 seconds ah.\nKadib FFmpeg ayaa isku daraya HAL video.\n\nFadlan ha xirin bogga.");

try{
let r=await fetch("/api/generate",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({
prompt:p,
duration,
aspect_ratio:document.getElementById("aspect").value,
quality:document.getElementById("quality").value,
language:document.getElementById("lang").value,
image_url:mode==="image"?imageUrl:null
})
});
let d=await r.json();
if(!r.ok)throw Error(et(d));

await pollJob(d.job_id);
}catch(e){
show("❌ "+(e.message||e),true);
b.disabled=false;
b.textContent="✨ Samee Video";
}
}

async function pollJob(jobId){
let b=document.getElementById("go");

while(true){
let r=await fetch("/api/jobs/"+jobId);
let d=await r.json();

if(!r.ok)throw Error(et(d));

if(d.status==="queued"){
show("⏳ Job-ka saf ayuu ku jiraa.\n\nClips: "+d.total_clips+"\nProgress: 0%");
}
else if(d.status==="processing"){
let pct=Number(d.progress||0);
let done=Math.min(d.total_clips||0,Math.round((pct/90)*(d.total_clips||0)));
show("⏳ Video-ga waa la samaynayaa...\n\nClips: "+d.total_clips+"\nLa dhammeeyey: "+done+"\nProgress: "+pct+"%");
}
else if(d.status==="succeeded"){
show("✅ HAL video ayaa diyaar ah.\n\nDuration: "+d.duration+" seconds\nClips: "+d.total_clips+"\nProgress: 100%\n\n🎬 Fur HAL video-ga:\n"+location.origin+d.output);
loadHistory();
b.disabled=false;
b.textContent="✨ Samee Video";
return;
}
else if(d.status==="failed"){
throw Error(d.error||"Video generation failed.");
}

await new Promise(resolve=>setTimeout(resolve,5000));
}
}

async function loadHistory(){let h=document.getElementById("hist");try{let r=await fetch("/api/history"),a=await r.json();h.innerHTML="";a.forEach(x=>{let d=document.createElement("div");d.className="video";d.textContent=(x.prompt||"")+" | "+(x.duration||"?")+"s | "+(x.status||"");if(x.output){let z=document.createElement("a");z.href=x.output;z.target="_blank";z.textContent=" 🎬 Fur HAL Video";d.appendChild(z)}h.appendChild(d)})}catch(e){}}
loadHistory();
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


