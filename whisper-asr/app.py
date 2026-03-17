import base64
import io
import os

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI()

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)


class AudioRequest(BaseModel):
    audio_base64: str
    language: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "service": "whisper-asr"}


@app.post("/transcribe")
def transcribe(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        segments, info = model.transcribe(
            audio,
            language=req.language,
            beam_size=1,
            best_of=1,
            vad_filter=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return {"text": text, "language": getattr(info, "language", None)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
