# Zip Video AI

A starter full-stack AI video generator for Somali / English / Arabic.

## Features
- Text → Video
- Image → Video
- Talking-animal / animal-story presets
- 9:16, 16:9, 1:1
- 720p / 1080p / 4K quality selector
- ⚡ Fast / 🎬 Quality generation modes
- Loading state + elapsed timer
- Preview + download
- My Videos history
- Replicate API integration
- Render deployment configuration

## Run locally

### 1. Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### 2. Frontend
Open `frontend/index.html` directly, or serve it:
```bash
cd frontend
python -m http.server 5173
```
Then open http://localhost:5173.

Set `API_BASE` in `frontend/app.js` if your backend is not on localhost:8000.

## Replicate

Add your Replicate token to `.env`:
```env
REPLICATE_API_TOKEN=r8_...
REPLICATE_MODEL=kwaivgi/kling-v2.1
```

The backend uses Replicate's HTTP API. Because video model names and input schemas can change, verify the model/input fields you want in your Replicate account before production use.

## Render

`render.yaml` contains a web-service definition for the FastAPI backend. Deploy the backend first, then point `API_BASE` in the frontend to the deployed backend URL.

## Important production notes
- Add authentication before exposing My Videos publicly.
- Replace local JSON history with Postgres/S3 or another persistent store on Render.
- Add a background job queue for long video generation.
- Restrict upload size and file types.
- Keep the Replicate token server-side; never put it in frontend JavaScript.


## Quality modes

The UI now provides:
- **⚡ Fast** — speed-first mode.
- **HD** — balanced quality/speed.
- **🔥 4K** — requests the highest-quality mode.

Actual 4K output is model-dependent. The configured Replicate model must support a 4K/ultra-high-resolution output and the model-specific input field may need to be mapped in `backend/main.py`. The app intentionally does not fake 4K by merely upscaling a lower-resolution video.


## 4K and Fast Mode

The UI and API now accept `quality=4K|1080p|720p` and `speed=fast|quality`.
**Actual 4K output and speed depend on the Replicate model you select.** Not every video model accepts these exact fields. Before production, map the fields in `create_replicate_prediction()` to the current model's documented input schema. If the model does not support 4K, the backend cannot magically upscale it to true native 4K.
