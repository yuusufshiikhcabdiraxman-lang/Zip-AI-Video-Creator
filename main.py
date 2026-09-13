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

UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

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

def history():
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_history(items):
    DATA.write_text(json.dumps(items[-100:], ensure_ascii=False, indent=2), encoding="utf-8")

def create_replicate_prediction(
    prompt: str,
    aspect_ratio: str,
    duration: int,
    quality: str = "fast",
    image_url: str = None
):
    token = os.getenv("REPLICATE_API_TOKEN")
    model = os.getenv("REPLICATE_MODEL", "kwaivgi/kling-v2.1")

    if not token:
        raise HTTPException(
            500,
            "REPLICATE_API_TOKEN is not configured"
        )

    inp = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "quality": quality
    }

    r = requests.post(
        f"https://api.replicate.com/v1/models/{model}/predictions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={"input": inp},
        timeout=60
    )

    if not r.ok:
        raise HTTPException(
            r.status_code,
            f"Replicate error: {r.text[:500]}"
        )

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
    pred = create_replicate_prediction(req.prompt, req.aspect_ratio, req.duration, req.quality, req.speed)
    item = {
        "id": pred.get("id", str(uuid.uuid4())),
        "prompt": req.prompt,
        "language": req.language,
        "aspect_ratio": req.aspect_ratio,
        "duration": req.duration,
        "quality": req.quality,
        "status": pred.get("status", "starting"),
        "output": pred.get("output"),
        @app.post("/api/generate-image")
async def generate_image(
    prompt: str = Form(...),
    language: str = Form("so"),
    aspect_ratio: str = Form("9:16"),
    duration: int = Form(5),
    quality: str = Form("fast"),
    image: UploadFile = File(...)
):
    ext = Path(image.filename or "").suffix.lower()

    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Use JPG, PNG, or WEBP")

    image_bytes = await image.read()

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET")

    if not cloud_name or not upload_preset:
        raise HTTPException(
            500,
            "Cloudinary is not configured"
        )
            "Cloudinary is not configured"
        )

    upload_url = (
        f"https://api.cloudinary.com/v1_1/"
        f"{cloud_name}/image/upload"
    )

    upload_response = requests.post(
        upload_url,
        data={
            "upload_preset": upload_preset
        },
        files={
            "file": (
                image.filename or "image.jpg",
                image_bytes,
                image.content_type or "image/jpeg"
            )
        },
        timeout=60
    )

    if not upload_response.ok:
        raise HTTPException(
            upload_response.status_code,
            f"Cloudinary error: {upload_response.text[:500]}"
        )

    image_url = upload_response.json().get("secure_url")

    if not image_url:
        raise HTTPException(
            500,
            "Cloudinary did not return an image URL"
        )

    pred = create_replicate_prediction(
        prompt,
        aspect_ratio,
        duration,
        quality,
        image_url=image_url
    )

    return pred
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
