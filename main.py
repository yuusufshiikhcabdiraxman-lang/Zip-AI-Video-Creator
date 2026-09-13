import os
import json
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
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

@app.get("/")
def root():
    return {"name": "Zip-AI Video Creator", "status": "running"}

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
