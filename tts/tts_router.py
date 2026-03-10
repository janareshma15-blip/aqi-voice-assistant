import os
import uuid
import subprocess

AUDIO_DIR = "/home/URK22AI1081/aqi-assistant/static/audio"


def detect_lang(text: str) -> str:
    tamil_chars = 0
    latin_chars = 0

    for ch in text:
        if "\u0B80" <= ch <= "\u0BFF":
            tamil_chars += 1
        elif ("a" <= ch.lower() <= "z"):
            latin_chars += 1

    if tamil_chars > latin_chars:
        return "ta"
    return "en"


def generate_tts(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    os.makedirs(AUDIO_DIR, exist_ok=True)

    file_id = uuid.uuid4().hex
    output_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")

    lang = detect_lang(text)

    try:
        if lang == "ta":
            cmd = [
                "python",
                "/home/URK22AI1081/aqi-assistant/tts/mms_tamil_worker.py",
                text,
                output_path
            ]
        else:
            cmd = [
                "conda", "run", "-n", "kokoro312",
                "python",
                "/home/URK22AI1081/aqi-assistant/tts/kokoro_worker.py",
                text,
                output_path
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print("TTS error:", result.stderr)
            return None

        return f"/static/audio/{file_id}.wav"

    except Exception as e:
        print("TTS exception:", str(e))
        return None
