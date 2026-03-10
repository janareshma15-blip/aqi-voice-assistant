import sys
import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer

MODEL_PATH = "/home/URK22AI1081/aqi-tts-models/mms-tts-tam"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = VitsModel.from_pretrained(MODEL_PATH)


def main():
    if len(sys.argv) < 3:
        print("Usage: python mms_tamil_worker.py '<text>' <output_wav>")
        sys.exit(1)

    text = sys.argv[1].strip()
    output_wav = sys.argv[2].strip()

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    audio = output.squeeze().cpu().numpy()
    wavfile.write(output_wav, rate=model.config.sampling_rate, data=audio)

    print(output_wav)


if __name__ == "__main__":
    main()
