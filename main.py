import json, os, time, uuid
from pathlib import Path
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

BASE = Path(__file__).resolve().parent

UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
DATA = BASE / "backend" / "videos.json"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
DATA.parent.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    DATA.write_text("[]", encoding="utf-8")
DATA.parent.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    DATA.write_text("[]", encoding="utf-8")
app = FastAPI(title="Zip Video AI API")

origin = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origin == "*" else [origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    language: str = "so"
    aspect_ratio: str = "9:16"
    duration: int = 5
    quality: str = "1080p"
    speed: str = "fast"
    quality: str = "fast"

def history():
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_history(items):
    DATA.write_text(json.dumps(items[-100:], ensure_ascii=False, indent=2), encoding="utf-8")

def create_replicate_prediction(prompt: str, aspect_ratio: str, duration: int, quality: str = "fast", image_url=None):
    token = os.getenv("REPLICATE_API_TOKEN")
    model = os.getenv("REPLICATE_MODEL", "kwaivgi/kling-v2.1")
    if not token:
        raise HTTPException(500, "REPLICATE_API_TOKEN is not configured")

    # This starter sends a common text/image-to-video shape.
    # Model-specific fields may need adjustment to match the selected Replicate model.
    inp = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        # Quality/speed are normalized here. Exact support depends on the selected Replicate model.
        "quality": quality,
        "speed": speed,
    }
    # Quality is normalized here so the UI can offer Fast/HD/4K.
    # Exact 4K parameters depend on the Replicate model selected.
    if quality not in {"fast", "hd", "4k"}:
        raise HTTPException(400, "Quality must be fast, hd, or 4k")
    inp["quality"] = quality
    if image_url:
        inp["image"] = image_url

    r = requests.post(
        "https://api.replicate.com/v1/models/" + model + "/predictions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"input": inp},
        timeout=30,
    )
    if not r.ok:
        raise HTTPException(r.status_code, f"Replicate error: {r.text[:500]}")
    return r.json()

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
    pred = create_replicate_prediction(req.prompt, req.aspect_ratio, req.duration, req.quality)
    item = {
        "id": pred.get("id", str(uuid.uuid4())),
        "prompt": req.prompt,
        "language": req.language,
        "aspect_ratio": req.aspect_ratio,
        "duration": req.duration,
        "quality": req.quality,
        "status": pred.get("status", "starting"),
        "output": pred.get("output"),
        "created_at": int(time.time()),
    }
    items = history()
    items.append(item)
    save_history(items)
    return item

@app.post("/api/generate-image")
async def generate_image(
    prompt: str = Form(...),
    language: str = Form("so"),
    aspect_ratio: str = Form("9:16"),
    duration: int = Form(5),
    quality: str = Form("fast"),
    image: UploadFile = File(...),
):
    ext = Path(image.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Use JPG, PNG, or WEBP")
    name = f"{uuid.uuid4()}{ext}"
    path = UPLOADS / name
    path.write_bytes(await image.read())

    # Replicate generally needs a publicly reachable image URL for hosted models.
    # For a production Render deployment, upload this file to object storage first.
    raise HTTPException(
        501,
        "Image-to-video needs public object storage (S3/R2/Cloudinary) for the uploaded image. "
        "The upload is saved locally; connect storage and pass its public URL to Replicate."
    )

@app.get("/api/predictions/{prediction_id}")
def prediction(prediction_id: str):
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(500, "REPLICATE_API_TOKEN is not configured")
    r = requests.get(
        f"https://api.replicate.com/v1/predictions/{prediction_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if not r.ok:
        raise HTTPException(r.status_code, f"Replicate error: {r.text[:500]}")
    data = r.json()
    items = history()
    for item in items:
        if item["id"] == prediction_id:
            item["status"] = data.get("status")
            item["output"] = data.get("output")
            break
    save_history(items)
    return data
