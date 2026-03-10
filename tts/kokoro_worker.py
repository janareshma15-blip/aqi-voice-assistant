import sys
import soundfile as sf
from kokoro_onnx import Kokoro

MODEL_FILE = "/home/URK22AI1081/kokoro-runtime/kokoro-v1.0.onnx"
VOICES_FILE = "/home/URK22AI1081/kokoro-runtime/voices-v1.0.bin"


def main():
    if len(sys.argv) < 3:
        print("Usage: python kokoro_worker.py '<text>' <output_wav>")
        sys.exit(1)

    text = sys.argv[1].strip()
    output_wav = sys.argv[2].strip()

    kokoro = Kokoro(MODEL_FILE, VOICES_FILE)

    samples, sample_rate = kokoro.create(
        text,
        voice="af_sarah",
        speed=1.0,
        lang="en-us",
    )

    sf.write(output_wav, samples, sample_rate)
    print(output_wav)


if __name__ == "__main__":
    main()
