import os
import json
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Zip-AI Video Creator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history.json"

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    aspect_ratio: str = "9:16"
    duration: int = Field(default=5, ge=1, le=30)
    quality: str = "fast"
    language: str = "so"

def history():
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def save_history(items):
    try:
        HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"Could not save history: {exc}")

def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(500, f"{name} is not configured")
    return value

def create_replicate_prediction(prompt: str, aspect_ratio: str, duration: int, quality: str, language: str = "so", image_url: Optional[str] = None):
    version = required_env("REPLICATE_MODEL_VERSION")
    model_input = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "quality": quality,
        "language": language,
    }
    if image_url:
        model_input["image"] = image_url
    try:
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Bearer {required_env('REPLICATE_API_TOKEN')}", "Content-Type": "application/json"},
            json={"version": version, "input": model_input},
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(502, f"Replicate connection error: {exc}")
    if not response.ok:
        raise HTTPException(response.status_code, f"Replicate error: {response.text[:1000]}")
    try:
        return response.json()
    except ValueError:
        raise HTTPException(502, "Replicate returned invalid JSON")

def upload_to_cloudinary(image: UploadFile, image_bytes: bytes) -> str:
    cloud_name = required_env("CLOUDINARY_CLOUD_NAME")
    upload_preset = required_env("CLOUDINARY_UPLOAD_PRESET")
    try:
        response = requests.post(
            f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload",
            data={"upload_preset": upload_preset},
            files={"file": (image.filename or "image.jpg", image_bytes, image.content_type or "image/jpeg")},
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(502, f"Cloudinary connection error: {exc}")
    if not response.ok:
        raise HTTPException(response.status_code, f"Cloudinary error: {response.text[:1000]}")
    try:
        data = response.json()
    except ValueError:
        raise HTTPException(502, "Cloudinary returned invalid JSON")
    image_url = data.get("secure_url")
    if not image_url:
        raise HTTPException(502, "Cloudinary did not return an image URL")
    return image_url

@app.get("/", response_class=HTMLResponse)
def root():
    return """<!doctype html>
<html lang="so">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zip-AI Video Creator</title>
<style>
:root{color-scheme:dark;--bg:#09090b;--card:#121216;--line:#27272f;--text:#f5f5f5;--muted:#a1a1aa;--accent:#7c3aed;--accent2:#a855f7}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -10%,#24103d 0,#09090b 45%);font-family:Arial,sans-serif;color:var(--text);min-height:100vh}
.wrap{max-width:900px;margin:auto;padding:24px 16px 50px}.hero{text-align:center;padding:30px 10px 20px}.logo{width:58px;height:58px;border-radius:16px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;margin:0 auto 14px;font-size:27px;font-weight:800}.hero h1{margin:0;font-size:32px}.hero p{color:var(--muted);margin:9px 0 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:rgba(18,18,22,.94);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 12px 35px rgba(0,0,0,.25)}.full{grid-column:1/-1}label{display:block;font-size:13px;color:#d4d4d8;margin:0 0 7px}input,select,textarea{width:100%;background:#0d0d10;border:1px solid #30303a;color:var(--text);border-radius:11px;padding:12px;font-size:15px;outline:none}textarea{min-height:130px;resize:vertical}input:focus,select:focus,textarea:focus{border-color:var(--accent2)}.row{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.btn{width:100%;border:0;border-radius:12px;padding:13px 16px;font-weight:700;font-size:15px;cursor:pointer;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}.btn:disabled{opacity:.55;cursor:not-allowed}.secondary{background:#202027;border:1px solid #34343d}.tabs{display:flex;gap:8px;margin-bottom:15px}.tab{flex:1;padding:11px;border-radius:10px;border:1px solid #34343d;background:#17171c;color:#ddd;cursor:pointer}.tab.active{background:#2b1745;border-color:#7c3aed}.hidden{display:none}.status{margin-top:14px;padding:12px;border-radius:11px;background:#0d0d10;border:1px solid var(--line);color:#d4d4d8;white-space:pre-wrap}.result{margin-top:12px}.result a{color:#c084fc;word-break:break-all}.item{padding:13px;border:1px solid var(--line);border-radius:12px;margin-top:10px}.small{font-size:12px;color:var(--muted)}.spinner{display:inline-block;width:14px;height:14px;border:2px solid #777;border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:7px}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:700px){.grid{grid-template-columns:1fr}.full{grid-column:auto}.row{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero"><div class="logo">Z</div><h1>Zip-AI Video Creator</h1><p>Qoraal ama sawir u beddel muuqaal AI ah.</p></section>
  <div class="grid">
    <section class="card full">
      <div class="tabs"><button class="tab active" id="textTab" onclick="setMode('text')">Qoraal → Video</button><button class="tab" id="imageTab" onclick="setMode('image')">Sawir → Video</button></div>
      <div id="textMode">
        <label>Prompt / Qoraalka muuqaalka</label><textarea id="prompt" placeholder="Tusaale: Magaalada Muqdisho habeenkii, nalal qurux badan, cinematic..."></textarea>
      </div>
      <div id="imageMode" class="hidden">
        <label>Sawirka</label><input id="image" type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp">
        <div style="height:10px"></div><label>Prompt (ikhtiyaari)</label><textarea id="imagePrompt" placeholder="Sharax dhaqdhaqaaqa aad rabto..."></textarea>
      </div>
      <div style="height:14px"></div>
      <div class="row">
        <div><label>Language</label><select id="language"><option value="so">Somali</option><option value="en">English</option></select></div>
        <div><label>Aspect ratio</label><select id="ratio"><option value="9:16">9:16 — TikTok/Reels</option><option value="16:9">16:9 — YouTube</option><option value="1:1">1:1 — Square</option></select></div>
        <div><label>Duration (seconds)</label><input id="duration" type="number" min="1" max="30" value="5"></div>
        <div><label>Quality</label><select id="quality"><option value="fast">Fast</option><option value="high">High</option></select></div>
      </div>
      <div style="height:15px"></div><button class="btn" id="generateBtn" onclick="generate()">✨ Samee Video</button>
      <div id="status" class="status hidden"></div><div id="result" class="result"></div>
    </section>
    <section class="card full"><h2 style="margin:0 0 5px;font-size:20px">Videos-kii hore</h2><div class="small">Natiijooyinka waxaa lagu kaydiyaa history-ga server-ka.</div><button class="btn secondary" style="margin-top:12px" onclick="loadHistory()">↻ Cusboonaysii</button><div id="history"></div></section>
  </div>
</div>
<script>
let mode='text', pollTimer=null;
function setMode(m){mode=m;document.getElementById('textMode').classList.toggle('hidden',m!=='text');document.getElementById('imageMode').classList.toggle('hidden',m!=='image');document.getElementById('textTab').classList.toggle('active',m==='text');document.getElementById('imageTab').classList.toggle('active',m==='image');clearResult();}
function clearResult(){document.getElementById('status').classList.add('hidden');document.getElementById('result').innerHTML='';}
function showStatus(t){const e=document.getElementById('status');e.classList.remove('hidden');e.innerHTML=t;}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function generate(){
 const btn=document.getElementById('generateBtn');btn.disabled=true;document.getElementById('result').innerHTML='';
 const prompt=mode==='text'?document.getElementById('prompt').value.trim():document.getElementById('imagePrompt').value.trim();
 const language=document.getElementById('language').value, ratio=document.getElementById('ratio').value, duration=document.getElementById('duration').value, quality=document.getElementById('quality').value;
 try{
   let response;
   if(mode==='text'){
     if(!prompt) throw new Error('Fadlan geli prompt.');
     response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,language,aspect_ratio:ratio,duration:Number(duration),quality})});
   }else{
     const file=document.getElementById('image').files[0];if(!file) throw new Error('Fadlan dooro sawir.');
     const fd=new FormData();fd.append('prompt',prompt);fd.append('language',language);fd.append('aspect_ratio',ratio);fd.append('duration',duration);fd.append('quality',quality);fd.append('image',file);
     response=await fetch('/api/generate-image',{method:'POST',body:fd});
   }
   const data=await response.json();if(!response.ok) throw new Error(data.detail||'Wax baa khaldamay.');
   const id=data.id;showStatus('<span class="spinner"></span> Video-ga waa la bilaabay...<br><span class="small">ID: '+esc(id)+'</span>');
   if(id) poll(id); else finish(data);
 }catch(e){showStatus('❌ '+esc(e.message));btn.disabled=false;}
}
async function poll(id){
 clearTimeout(pollTimer);
 try{const r=await fetch('/api/predictions/'+encodeURIComponent(id));const d=await r.json();if(!r.ok)throw new Error(d.detail||'Status error');
   if(['succeeded','failed','canceled'].includes(d.status)){finish(d);document.getElementById('generateBtn').disabled=false;loadHistory();return;}
   showStatus('<span class="spinner"></span> Status: '+esc(d.status||'processing')+'<br><span class="small">Waxaan sugaynaa natiijada...</span>');pollTimer=setTimeout(()=>poll(id),5000);
 }catch(e){showStatus('⚠️ '+esc(e.message)+'<br><span class="small">Waxaad isku dayi kartaa Cusboonaysii.</span>');document.getElementById('generateBtn').disabled=false;}
}
function finish(d){let out=d.output;let html='<div class="item"><b>Status:</b> '+esc(d.status||'done');if(Array.isArray(out))out=out[0];if(typeof out==='string'&&out.startsWith('http'))html+='<br><br><a href="'+esc(out)+'" target="_blank" rel="noopener">▶ Fur/Daawo Video</a>';else if(out)html+='<br><pre style="white-space:pre-wrap;overflow:auto">'+esc(JSON.stringify(out,null,2))+'</pre>';document.getElementById('result').innerHTML=html;showStatus('✅ Codsiga waa dhammaaday.');}
async function loadHistory(){const box=document.getElementById('history');box.innerHTML='<div class="small" style="margin-top:12px">Loading...</div>';try{const r=await fetch('/api/videos');const a=await r.json();if(!a.length){box.innerHTML='<div class="small" style="margin-top:12px">Weli video lama sameyn.</div>';return;}box.innerHTML=a.slice(0,10).map(x=>'<div class="item"><b>'+esc(x.status||'unknown')+'</b><div class="small">'+esc(x.prompt)+'</div><div class="small">'+esc(x.aspect_ratio)+' · '+esc(x.duration)+'s · '+esc(x.quality)+'</div></div>').join('');}catch(e){box.innerHTML='<div class="small">History lama soo qaadi karin.</div>';}}
loadHistory();
</script>
</body></html>"""

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/videos")
def videos():
    return list(reversed(history()))

@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is required")
    pred = create_replicate_prediction(req.prompt.strip(), req.aspect_ratio, req.duration, req.quality, req.language)
    item = {
        "id": pred.get("id", str(uuid.uuid4())),
        "prompt": req.prompt.strip(),
        "language": req.language,
        "aspect_ratio": req.aspect_ratio,
        "duration": req.duration,
        "quality": req.quality,
        "status": pred.get("status", "starting"),
        "output": pred.get("output"),
    }
    items = history()
    items.append(item)
    save_history(items)
    return pred

@app.post("/api/generate-image")
async def generate_image(
    prompt: str = "",
    language: str = "so",
    aspect_ratio: str = "9:16",
    duration: int = 5,
    quality: str = "fast",
    image: UploadFile = File(...),
):
    extension = Path(image.filename or "").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Use JPG, PNG, or WEBP")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Image file is empty")
    image_url = upload_to_cloudinary(image, image_bytes)
    clean_prompt = prompt.strip() or "Create a video from this image."
    pred = create_replicate_prediction(clean_prompt, aspect_ratio, duration, quality, language, image_url)
    item = {
        "id": pred.get("id", str(uuid.uuid4())),
        "prompt": clean_prompt,
        "language": language,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "quality": quality,
        "status": pred.get("status", "starting"),
        "output": pred.get("output"),
        "image_url": image_url,
    }
    items = history()
    items.append(item)
    save_history(items)
    return pred

@app.get("/api/predictions/{prediction_id}")
def prediction(prediction_id: str):
    token = required_env("REPLICATE_API_TOKEN")
    try:
        response = requests.get(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise HTTPException(502, f"Replicate connection error: {exc}")
    if not response.ok:
        raise HTTPException(response.status_code, f"Replicate error: {response.text[:1000]}")
    try:
        data = response.json()
    except ValueError:
        raise HTTPException(502, "Replicate returned invalid JSON")
    items = history()
    for item in items:
        if item.get("id") == prediction_id:
            item["status"] = data.get("status")
            item["output"] = data.get("output")
            break
    save_history(items)
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
