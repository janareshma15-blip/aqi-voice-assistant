import base64
import io
import os
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from transformers import VitsModel, AutoTokenizer

app = FastAPI()

MMS_TAMIL_HF_REPO = os.getenv("MMS_TAMIL_HF_REPO", "facebook/mms-tts-tam")
MODEL_DIR = Path("/app/runtime/hf-cache/mms-tts-tam")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not any(MODEL_DIR.iterdir()):
    snapshot_download(
        repo_id=MMS_TAMIL_HF_REPO,
        local_dir=str(MODEL_DIR),
    )

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = VitsModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
model.eval()


class TTSRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "service": "mms-tamil-tts"}


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    try:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform

        audio = output.squeeze().cpu().numpy()

        buf = io.BytesIO()
        sf.write(buf, audio, model.config.sampling_rate, format="WAV")
        buf.seek(0)

        return {
            "mime_type": "audio/wav",
            "audio_base64": base64.b64encode(buf.read()).decode("utf-8")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
