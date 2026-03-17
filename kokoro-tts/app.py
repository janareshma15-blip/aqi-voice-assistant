import base64
import io
import os
import traceback
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro

app = FastAPI()

RUNTIME_DIR = Path("/app/runtime/kokoro-runtime")
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

KOKORO_HF_REPO = os.getenv("KOKORO_HF_REPO", "onnx-community/Kokoro-82M-v1.0-ONNX")

kokoro = None
kokoro_error = None
model_path = None
voice_path = None


class TTSRequest(BaseModel):
    text: str


@app.on_event("startup")
def startup_load():
    global kokoro, kokoro_error, model_path, voice_path
    try:
        print(f"Downloading/loading Kokoro from repo: {KOKORO_HF_REPO}", flush=True)

        model_path = hf_hub_download(
            repo_id=KOKORO_HF_REPO,
            filename="onnx/model.onnx",
            local_dir=str(RUNTIME_DIR),
        )
        print(f"Model file: {model_path}", flush=True)

        voice_path = hf_hub_download(
            repo_id=KOKORO_HF_REPO,
            filename="voices/af.bin",
            local_dir=str(RUNTIME_DIR),
        )
        print(f"Voice file: {voice_path}", flush=True)

        kokoro = Kokoro(str(model_path), str(voice_path))
        kokoro_error = None
        print("Kokoro initialized successfully.", flush=True)
    except Exception as e:
        kokoro = None
        kokoro_error = f"{type(e).__name__}: {e}"
        print("Kokoro startup failed:", kokoro_error, flush=True)
        traceback.print_exc()


@app.get("/health")
def health():
    return {
        "ok": kokoro is not None,
        "service": "kokoro-tts",
        "loaded": kokoro is not None,
        "model_path": str(model_path) if model_path else None,
        "voice_path": str(voice_path) if voice_path else None,
        "error": kokoro_error,
    }


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    if kokoro is None:
        raise HTTPException(status_code=503, detail=f"Kokoro not loaded: {kokoro_error}")

    try:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        audio, sr = kokoro.create(
            text=text,
            voice="af_heart",
            speed=1.0,
            lang="en-us"
        )

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)

        return {
            "mime_type": "audio/wav",
            "audio_base64": base64.b64encode(buf.read()).decode("utf-8")
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
