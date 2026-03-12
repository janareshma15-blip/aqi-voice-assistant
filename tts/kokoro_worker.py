from pathlib import Path
import sys
import soundfile as sf
from kokoro_onnx import Kokoro

BASE_DIR = Path("/home/URK22AI1081/Projects/AQI")
RUNTIME_DIR = BASE_DIR / "runtime" / "kokoro-runtime"

MODEL_FILE = RUNTIME_DIR / "kokoro-v1.0.onnx"
VOICES_FILE = RUNTIME_DIR / "voices-v1.0.bin"


def main():
    if len(sys.argv) < 3:
        print("Usage: python kokoro_worker.py <text> <output_wav>")
        sys.exit(1)

    text = sys.argv[1].strip()
    output_wav = sys.argv[2]

    if not text:
        print("Empty text")
        sys.exit(1)

    if not MODEL_FILE.exists():
        print(f"Kokoro model file not found: {MODEL_FILE}")
        sys.exit(1)

    if not VOICES_FILE.exists():
        print(f"Kokoro voices file not found: {VOICES_FILE}")
        sys.exit(1)

    try:
        kokoro = Kokoro(str(MODEL_FILE), str(VOICES_FILE))
        audio, sr = kokoro.create(
            text=text,
            voice="af_heart",
            speed=1.0,
            lang="en-us"
        )

        sf.write(output_wav, audio, sr)
        print(f"Saved: {output_wav}")

    except Exception as e:
        print(f"Kokoro TTS failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
