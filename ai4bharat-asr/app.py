import base64
import io
import os

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel

app = FastAPI()

AI4BHARAT_MODEL_NAME = os.getenv(
    "AI4BHARAT_MODEL_NAME",
    "ai4bharat/indic-conformer-600m-multilingual"
)

model = AutoModel.from_pretrained(
    AI4BHARAT_MODEL_NAME,
    trust_remote_code=True,
)
model.eval()


class AudioRequest(BaseModel):
    audio_base64: str


@app.get("/health")
def health():
    return {"ok": True, "service": "ai4bharat-asr"}


@app.post("/transcribe")
def transcribe(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        wav = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            result = model(wav, "ta", "ctc")

        text = result[0] if isinstance(result, (list, tuple)) else str(result)
        return {"text": text.strip(), "language": "ta"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
