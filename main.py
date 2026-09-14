import os
import json
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="Zip-AI Video Creator",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history.json"


# =========================================================
# SETTINGS
# =========================================================

MAX_DURATION = 120


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    aspect_ratio: str = "9:16"
    duration: int = Field(default=120, ge=1, le=120)
    quality: str = "high"
    language: str = "so"
    image_url: Optional[str] = None


# =========================================================
# HISTORY
# =========================================================

def get_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_history(items):
    try:
        HISTORY_FILE.write_text(
            json.dumps(
                items,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save history: {exc}"
        )


# =========================================================
# ENVIRONMENT
# =========================================================

def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise HTTPException(
            status_code=500,
            detail=f"{name} is not configured"
        )

    return value


# =========================================================
# REPLICATE
# =========================================================

def create_replicate_prediction(
    prompt: str,
    aspect_ratio: str,
    duration: int,
    quality: str,
    language: str = "so",
    image_url: Optional[str] = None
):
    token = required_env("REPLICATE_API_TOKEN")
    version = required_env("REPLICATE_MODEL_VERSION")

    # 2 Daqiiqo = 120 seconds
    duration = max(1, min(int(duration), MAX_DURATION))

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
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Prefer": "wait"
            },
            json={
                "version": version,
                "input": model_input
            },
            timeout=60
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Replicate connection error: {exc}"
        )

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Replicate returned an invalid response."
        )

    if not response.ok:
        error = data.get("detail") or data.get("error")

        if isinstance(error, dict):
            error = (
                error.get("message")
                or error.get("detail")
                or json.dumps(error, ensure_ascii=False)
            )

        if not error:
            error = "Replicate rejected the video request."

        raise HTTPException(
            status_code=response.status_code,
            detail=str(error)
        )

    return data


# =========================================================
# CLOUDINARY
# =========================================================

def upload_to_cloudinary(file_bytes: bytes, filename: str):
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
    api_key = os.getenv("CLOUDINARY_API_KEY", "").strip()
    api_secret = os.getenv("CLOUDINARY_API_SECRET", "").strip()

    if not all([cloud_name, api_key, api_secret]):
        raise HTTPException(
            status_code=500,
            detail="Cloudinary environment variables are not configured."
        )

    import hashlib
    import time

    timestamp = int(time.time())

    signature_string = (
        f"timestamp={timestamp}{api_secret}"
    )

    signature = hashlib.sha1(
        signature_string.encode("utf-8")
    ).hexdigest()

    url = (
        f"https://api.cloudinary.com/v1_1/"
        f"{cloud_name}/image/upload"
    )

    try:
        response = requests.post(
            url,
            data={
                "api_key": api_key,
                "timestamp": timestamp,
                "signature": signature
            },
            files={
                "file": (
                    filename,
                    file_bytes
                )
            },
            timeout=60
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cloudinary connection error: {exc}"
        )

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Cloudinary returned an invalid response."
        )

    if not response.ok:
        error = data.get("error", "Image upload failed.")

        if isinstance(error, dict):
            error = error.get("message", "Image upload failed.")

        raise HTTPException(
            status_code=502,
            detail=str(error)
        )

    return data.get("secure_url")


# =========================================================
# API ROUTES
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Zip-AI Video Creator",
        "max_duration": MAX_DURATION
    }


@app.get("/api/history")
def api_history():
    return get_history()


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type is missing."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The image is empty."
        )

    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image is too large. Maximum size is 15MB."
        )

    filename = file.filename or f"{uuid.uuid4()}.jpg"

    image_url = upload_to_cloudinary(
        content,
        filename
    )

    return {
        "success": True,
        "url": image_url
    }


@app.post("/api/generate")
def generate_video(request: GenerateRequest):

    duration = int(request.duration)

    # Hubi in aanu ka badnaan 120 seconds
    if duration < 1:
        duration = 1

    if duration > 120:
        duration = 120

    result = create_replicate_prediction(
        prompt=request.prompt,
        aspect_ratio=request.aspect_ratio,
        duration=duration,
        quality=request.quality,
        language=request.language,
        image_url=request.image_url
    )

    item = {
        "id": str(uuid.uuid4()),
        "prompt": request.prompt,
        "aspect_ratio": request.aspect_ratio,
        "duration": duration,
        "quality": request.quality,
        "language": request.language,
        "image_url": request.image_url,
        "prediction_id": result.get("id"),
        "status": result.get("status"),
        "output": result.get("output"),
        "error": result.get("error")
    }

    history = get_history()
    history.insert(0, item)

    save_history(history[:50])

    return {
        "success": True,
        "prediction": result,
        "history_item": item
    }


# =========================================================
# WEB APP
# =========================================================

@app.get("/", response_class=HTMLResponse)
def root():

    html = r"""
<!DOCTYPE html>
<html lang="so">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Zip-AI Video Creator</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
    background:
        linear-gradient(
            180deg,
            #170827 0%,
            #0b0910 55%,
            #07070a 100%
        );
    color: #fff;
}

.container {
    width: min(950px, 94%);
    margin: 0 auto;
    padding: 55px 0 80px;
}

.logo {
    width: 70px;
    height: 70px;
    margin: 0 auto 18px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
    font-weight: 900;
    background:
        linear-gradient(
            135deg,
            #8c35ff,
            #b347ff
        );
    box-shadow:
        0 15px 50px rgba(133, 50, 255, .30);
}

h1 {
    text-align: center;
    margin: 0;
    font-size: clamp(30px, 6vw, 50px);
}

.subtitle {
    text-align: center;
    color: #a9a4b0;
    font-size: 18px;
    margin: 10px 0 35px;
}

.card {
    background: rgba(20, 18, 24, .90);
    border: 1px solid #2b2831;
    border-radius: 20px;
    padding: 24px;
    box-shadow:
        0 20px 70px rgba(0,0,0,.30);
}

.tabs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 22px;
}

.tab {
    border: 1px solid #34303b;
    background: #151319;
    color: #ddd8e4;
    border-radius: 12px;
    padding: 15px;
    font-size: 16px;
    cursor: pointer;
}

.tab.active {
    background: #2d1650;
    border-color: #873cff;
    color: white;
}

label {
    display: block;
    margin: 14px 0 8px;
    color: #c9c4cf;
    font-size: 15px;
}

textarea,
input,
select {
    width: 100%;
    border: 1px solid #302d35;
    border-radius: 12px;
    background: #0d0c10;
    color: white;
    padding: 14px;
    font-size: 16px;
    outline: none;
}

textarea {
    min-height: 170px;
    resize: vertical;
}

textarea:focus,
input:focus,
select:focus {
    border-color: #8b3dff;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
}

.button {
    width: 100%;
    margin-top: 20px;
    border: 0;
    border-radius: 13px;
    padding: 17px;
    background:
        linear-gradient(
            90deg,
            #8132ed,
            #ad48f7
        );
    color: white;
    font-size: 17px;
    font-weight: 700;
    cursor: pointer;
}

.button:disabled {
    opacity: .55;
    cursor: wait;
}

.upload {
    display: none;
}

.upload-label {
    display: block;
    text-align: center;
    padding: 25px;
    border: 1px dashed #65418a;
    border-radius: 13px;
    background: #111016;
    cursor: pointer;
    color: #bbb5c4;
}

.message {
    margin-top: 15px;
    padding: 14px;
    border-radius: 12px;
    background: #0d0c10;
    border: 1px solid #2c2931;
    white-space: pre-wrap;
    word-break: break-word;
}

.videos {
    margin-top: 25px;
}

.video-item {
    padding: 15px;
    margin-top: 10px;
    background: #111016;
    border: 1px solid #29262f;
    border-radius: 12px;
}

.video-item a {
    color: #bd79ff;
    word-break: break-all;
}

.hidden {
    display: none;
}

.small {
    color: #8f8997;
    font-size: 13px;
    margin-top: 8px;
}

@media(max-width:650px) {
    .grid {
        grid-template-columns: 1fr;
    }

    .card {
        padding: 16px;
    }

    .container {
        padding-top: 35px;
    }
}

</style>
</head>

<body>

<div class="container">

    <div class="logo">Z</div>

    <h1>Zip-AI Video Creator</h1>

    <div class="subtitle">
        Qoraal ama sawir u beddel muuqaal AI ah.
    </div>

    <div class="card">

        <div class="tabs">
            <button
                id="textTab"
                class="tab active"
                onclick="setMode('text')">
                Qoraal → Video
            </button>

            <button
                id="imageTab"
                class="tab"
                onclick="setMode('image')">
                Sawir → Video
            </button>
        </div>

        <label>
            Prompt / Qoraalka muuqaalka
        </label>

        <textarea
            id="prompt"
            placeholder="Qor waxa aad rabto in muuqaalka uu sameeyo..."></textarea>

        <div
            id="imageBox"
            class="hidden">

            <label>
                Dooro sawir
            </label>

            <label
                class="upload-label"
                for="imageInput">

                📷 Taabo si aad sawir u geliso

            </label>

            <input
                id="imageInput"
                class="upload"
                type="file"
                accept="image/*">

            <div
                id="uploadStatus"
                class="small"></div>

        </div>

        <div class="grid">

            <div>
                <label>Luuqad</label>

                <select id="language">

                    <option value="so">
                        Somali
                    </option>

                    <option value="en">
                        English
                    </option>

                    <option value="ar">
                        Arabic
                    </option>

                </select>
            </div>

            <div>
                <label>Aspect ratio</label>

                <select id="aspect">

                    <option value="9:16">
                        9:16 — TikTok/Reels
                    </option>

                    <option value="16:9">
                        16:9 — YouTube
                    </option>

                    <option value="1:1">
                        1:1 — Square
                    </option>

                </select>
            </div>

            <div>
                <label>
                    Duration — seconds
                </label>

                <input
                    id="duration"
                    type="number"
                    min="1"
                    max="120"
                    value="120">

                <div class="small">
                    Ugu badnaan 120 seconds (2 daqiiqo)
                </div>
            </div>

            <div>
                <label>Quality</label>

                <select id="quality">

                    <option value="high">
                        High
                    </option>

                    <option value="fast">
                        Fast
                    </option>

                </select>
            </div>

        </div>

        <button
            id="generateButton"
            class="button"
            onclick="generateVideo()">

            ✨ Samee Video

        </button>

        <div
            id="message"
            class="message hidden"></div>

    </div>

    <div class="card videos">

        <h2>Videos-kii hore</h2>

        <div class="small">
            Natiijooyinka waxaa lagu kaydiyaa history-ga server-ka.
        </div>

        <button
            class="button"
            onclick="loadHistory()"
            style="margin-top:15px;background:#232229">

            ↻ Cusboonaysii

        </button>

        <div id="history"></div>

    </div>

</div>


<script>

let mode = "text";
let imageUrl = null;


function setMode(newMode) {

    mode = newMode;

    document
        .getElementById("textTab")
        .classList.toggle(
            "active",
            mode === "text"
        );

    document
        .getElementById("imageTab")
        .classList.toggle(
            "active",
            mode === "image"
        );

    document
        .getElementById("imageBox")
        .classList.toggle(
            "hidden",
            mode !== "image"
        );
}


function showMessage(text, error=false) {

    const box =
        document.getElementById("message");

    box.classList.remove("hidden");

    box.textContent = text;

    if (error) {
        box.style.borderColor = "#6e2730";
    } else {
        box.style.borderColor = "#3b3150";
    }
}


function getErrorMessage(data) {

    if (!data) {
        return "Cilad aan la aqoon ayaa dhacday.";
    }

    let detail =
        data.detail ??
        data.error ??
        data.message;

    if (detail === undefined) {
        return JSON.stringify(
            data,
            null,
            2
        );
    }

    // Tani waxay xallinaysaa
    // "[object Object]"
    if (typeof detail === "object") {

        if (detail.message) {
            return String(detail.message);
        }

        if (detail.detail) {
            return String(detail.detail);
        }

        try {
            return JSON.stringify(
                detail,
                null,
                2
            );
        } catch(e) {
            return "Cilad ayaa dhacday.";
        }
    }

    return String(detail);
}


async function uploadImage() {

    const input =
        document.getElementById("imageInput");

    const status =
        document.getElementById("uploadStatus");

    if (!input.files.length) {
        return;
    }

    const file = input.files[0];

    status.textContent =
        "⏳ Sawirka waa la upload-gareynayaa...";

    const formData = new FormData();

    formData.append(
        "file",
        file
    );

    try {

        const response =
            await fetch(
                "/api/upload",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                getErrorMessage(data)
            );
        }

        imageUrl = data.url;

        status.textContent =
            "✅ Sawirka waa la geliyey.";

    } catch(error) {

        imageUrl = null;

        status.textContent =
            "❌ " + error.message;
    }
}


document
    .getElementById("imageInput")
    .addEventListener(
        "change",
        uploadImage
    );


async function generateVideo() {

    const button =
        document.getElementById(
            "generateButton"
        );

    const prompt =
        document
            .getElementById("prompt")
            .value
            .trim();

    const aspect =
        document
            .getElementById("aspect")
            .value;

    let duration =
        Number(
            document
                .getElementById("duration")
                .value
        );

    const quality =
        document
            .getElementById("quality")
            .value;

    const language =
        document
            .getElementById("language")
            .value;

    if (!prompt) {

        showMessage(
            "Fadlan marka hore qor prompt-ka.",
            true
        );

        return;
    }

    if (!Number.isFinite(duration)) {
        duration = 120;
    }

    // 2 daqiiqo = 120 seconds
    duration = Math.max(
        1,
        Math.min(
            120,
            Math.floor(duration)
        )
    );

    if (
        mode === "image" &&
        !imageUrl
    ) {

        showMessage(
            "Fadlan marka hore geli sawirka.",
            true
        );

        return;
    }

    button.disabled = true;

    button.textContent =
        "⏳ Video-ga waa la samaynayaa...";

    showMessage(
        "⏳ Fadlan sug. Video-ga ayaa la diyaarinayaa..."
    );

    try {

        const response =
            await fetch(
                "/api/generate",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({

                        prompt: prompt,

                        aspect_ratio:
                            aspect,

                        duration:
                            duration,

                        quality:
                            quality,

                        language:
                            language,

                        image_url:
                            mode === "image"
                                ? imageUrl
                                : null

                    })
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                getErrorMessage(data)
            );
        }

        const prediction =
            data.prediction || {};

        const output =
            prediction.output;

        let resultText =
            "✅ Codsiga waa la aqbalay.\n\n";

        if (output) {

            if (Array.isArray(output)) {

                resultText +=
                    output
                        .map(
                            item =>
                                String(item)
                        )
                        .join("\n");

            } else {

                resultText +=
                    String(output);
            }

        } else {

            resultText +=
                "Status: " +
                String(
                    prediction.status ||
                    "processing"
                );

            if (prediction.id) {

                resultText +=
                    "\nPrediction ID: " +
                    prediction.id;
            }
        }

        showMessage(resultText);

        loadHistory();

    } catch(error) {

        showMessage(
            "❌ " + (
                error.message ||
                "Cilad ayaa dhacday."
            ),
            true
        );

    } finally {

        button.disabled = false;

        button.textContent =
            "✨ Samee Video";
    }
}


async function loadHistory() {

    const box =
        document.getElementById(
            "history"
        );

    try {

        const response =
            await fetch(
                "/api/history"
            );

        const items =
            await response.json();

        if (!items.length) {

            box.innerHTML =
                '<div class="small" style="margin-top:15px">' +
                'Weli video lama samayn.' +
                '</div>';

            return;
        }

        box.innerHTML = "";

        items.forEach(item => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "video-item";

            const prompt =
                document.createElement(
                    "div"
                );

            prompt.textContent =
                item.prompt || "";

            const info =
                document.createElement(
                    "div"
                );

            info.className =
                "small";

            info.textContent =
                "Duration: " +
                (item.duration || "?") +
                "s | Status: " +
                (item.status || "unknown");

            div.appendChild(prompt);
            div.appendChild(info);

            if (item.output) {

                const outputs =
                    Array.isArray(
                        item.output
                    )
                        ? item.output
                        : [item.output];

                outputs.forEach(url => {

                    if (
                        typeof url ===
                        "string" &&
                        (
                            url.startsWith(
                                "http://"
                            ) ||
                            url.startsWith(
                                "https://"
                            )
                        )
                    ) {

                        const a =
                            document.createElement(
                                "a"
                            );

                        a.href = url;
                        a.target = "_blank";
                        a.rel = "noopener";
                        a.textContent =
                            "🎬 Fur Video";

                        div.appendChild(a);
                    }
                });
            }

            if (item.error) {

                const error =
                    document.createElement(
                        "div"
                    );

                error.style.color =
                    "#ff7b86";

                error.style.marginTop =
                    "8px";

                error.textContent =
                    "❌ " +
                    (
                        typeof item.error ===
                        "object"
                            ? JSON.stringify(
                                item.error
                            )
                            : item.error
                    );

                div.appendChild(error);
            }

            box.appendChild(div);
        });

    } catch(error) {

        box.innerHTML =
            '<div class="small">History lama soo dejin karin.</div>';
    }
}


loadHistory();

</script>

</body>
</html>
"""

    return HTMLResponse(content=html)


# =========================================================
# STARTUP
# =========================================================

if __name__ == "__main__":
    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
