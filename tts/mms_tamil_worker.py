from pathlib import Path
import os
import sys
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

MODEL_PATH = Path(os.environ.get("MMS_TAMIL_MODEL_DIR", "")).resolve()


def main():
    if len(sys.argv) < 3:
        print("Usage: python mms_tamil_worker.py <text> <output_wav>")
        sys.exit(1)

    text = sys.argv[1].strip()
    output_wav = sys.argv[2]

    if not text:
        print("Empty text")
        sys.exit(1)

    if not str(MODEL_PATH) or not MODEL_PATH.exists():
        print(f"Model path not found: {MODEL_PATH}")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True
        )

        model = VitsModel.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True
        )

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform

        audio = output.squeeze().cpu().numpy()
        sf.write(output_wav, audio, model.config.sampling_rate)
        print(f"Saved: {output_wav}")

    except Exception as e:
        print(f"MMS Tamil TTS failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
