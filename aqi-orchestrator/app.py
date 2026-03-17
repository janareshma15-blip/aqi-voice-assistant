import asyncio
import base64
import io
import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Optional, List, Dict, Any, Tuple

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

APP_PORT = int(os.getenv("APP_PORT", "9002"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://gemma-vllm:8000/v1").rstrip("/")
WHISPER_ASR_URL = os.getenv("WHISPER_ASR_URL", "http://whisper-asr:9010").rstrip("/")
AI4BHARAT_ASR_URL = os.getenv("AI4BHARAT_ASR_URL", "http://ai4bharat-asr:9011").rstrip("/")
KOKORO_TTS_URL = os.getenv("KOKORO_TTS_URL", "http://kokoro-tts:9020").rstrip("/")
MMS_TAMIL_TTS_URL = os.getenv("MMS_TAMIL_TTS_URL", "http://mms-tamil-tts:9021").rstrip("/")

MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "900"))
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

RMS_THRESHOLD = 0.0065
END_OF_SPEECH_SILENCE_MS = 320
PARTIAL_TRANSCRIBE_EVERY_MS = 350
PARTIAL_WINDOW_MS = 1400
MIN_VOICED_AUDIO_MS = 260
FINALIZE_COOLDOWN_MS = 650
MIN_TRANSCRIPT_CHARS = 2

sessions = defaultdict(lambda: deque(maxlen=12))
session_locks = defaultdict(Lock)

profiles = defaultdict(lambda: {
    "reply_lang": "auto",
})

TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")

SYSTEM_PROMPT_BASE = (
    "You are an AI engineering assistant for AQI and AI systems work. "
    "Stay focused on AQI, air-quality systems, AI/ML, FastAPI, dashboards, APIs, deployment, debugging, architecture, "
    "documentation, sensors, inference, and academic engineering topics. "
    "If asked who you are, say exactly: 'I am an AI engineering assistant for AQI and AI systems work.' "
    "If the user wants English, reply in English. If the user wants Tamil, reply in Tamil. "
    "For voice, keep answers natural and clear. "
    "When giving code, return complete runnable code in fenced code blocks."
)

http_client = httpx.AsyncClient(timeout=None)


class VoiceSession:
    def __init__(self):
        self.audio_buffer = bytearray()
        self.partial_text = ""
        self.last_partial_run_ms = 0.0
        self.last_finalize_ms = -999999.0
        self.started_at = time.time()
        self.speech_started = False
        self.voiced_audio_ms = 0.0
        self.silence_after_speech_ms = 0.0
        self.voice_mode = "auto"
        self.reply_lang = "auto"


voice_sessions = {}


def contains_tamil_script(text: str) -> bool:
    return bool(TAMIL_CHAR_RE.search(text or ""))


def infer_reply_lang(user_text: str, session_id: str) -> str:
    forced = profiles[session_id]["reply_lang"]
    if forced in {"en", "ta"}:
        return forced
    return "ta" if contains_tamil_script(user_text) else "en"


def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)
    audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0


def rms_from_pcm16(audio_bytes: bytes) -> float:
    audio = pcm16_bytes_to_float32(audio_bytes)
    return 0.0 if audio.size == 0 else float(np.sqrt(np.mean(np.square(audio))))


def audio_ms(audio_bytes: bytes) -> float:
    num_samples = len(audio_bytes) / SAMPLE_WIDTH
    return (num_samples / SAMPLE_RATE) * 1000.0


def get_last_window_bytes(audio_bytes: bytes, window_ms: float) -> bytes:
    total_samples_needed = int((window_ms / 1000.0) * SAMPLE_RATE)
    total_bytes_needed = total_samples_needed * SAMPLE_WIDTH
    return audio_bytes if len(audio_bytes) <= total_bytes_needed else audio_bytes[-total_bytes_needed:]


def wav_base64_from_pcm16(audio_bytes: bytes) -> str:
    audio = pcm16_bytes_to_float32(audio_bytes)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


async def safe_send_text(websocket: WebSocket, payload: Dict[str, Any]) -> bool:
    try:
        if websocket.client_state != WebSocketState.CONNECTED:
            return False
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:
        return False


def websocket_is_connected(websocket: WebSocket) -> bool:
    try:
        return websocket.client_state == WebSocketState.CONNECTED
    except Exception:
        return False


async def call_whisper_asr(audio_bytes: bytes, language: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "audio_base64": wav_base64_from_pcm16(audio_bytes),
        "language": language
    }
    r = await http_client.post(f"{WHISPER_ASR_URL}/transcribe", json=payload)
    r.raise_for_status()
    return r.json()


async def call_ai4bharat_asr(audio_bytes: bytes) -> Dict[str, Any]:
    payload = {
        "audio_base64": wav_base64_from_pcm16(audio_bytes)
    }
    r = await http_client.post(f"{AI4BHARAT_ASR_URL}/transcribe", json=payload)
    r.raise_for_status()
    return r.json()


async def call_tts(text: str, lang: str) -> Dict[str, Any]:
    url = f"{MMS_TAMIL_TTS_URL}/synthesize" if lang == "ta" else f"{KOKORO_TTS_URL}/synthesize"
    r = await http_client.post(url, json={"text": text})
    r.raise_for_status()
    return r.json()


async def resolve_model_name() -> str:
    r = await http_client.get(f"{VLLM_BASE_URL}/models", timeout=20)
    r.raise_for_status()
    data = r.json()
    ids = [m["id"] for m in data.get("data", []) if "id" in m]
    if not ids:
        return "gemma-3-12b-it-text"
    return ids[0]


def clean_reply_for_voice(text: str) -> str:
    text = re.sub(r"[*_`#>-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_complete_sentences(buffer: str, final: bool = False) -> Tuple[List[str], str]:
    pattern = re.compile(r"(.+?[.!?।！？]+)(?=\s|$)", re.S)
    sentences = []
    last_end = 0
    for m in pattern.finditer(buffer):
        sentence = m.group(1).strip()
        if sentence:
            sentences.append(sentence)
        last_end = m.end()
    remainder = buffer[last_end:].lstrip() if last_end > 0 else buffer
    if final and remainder.strip():
        sentences.append(remainder.strip())
        remainder = ""
    return sentences, remainder


async def send_tts_chunk(websocket: WebSocket, sentence: str, lang: str) -> bool:
    sentence = clean_reply_for_voice(sentence)
    if not sentence:
        return False
    audio_obj = await call_tts(sentence, lang)
    return await safe_send_text(websocket, {
        "type": "tts_sentence_ready",
        "text": sentence,
        "mime_type": audio_obj["mime_type"],
        "audio_base64": audio_obj["audio_base64"]
    })


def prepare_llm_request_messages(session_id: str, user_input: str):
    with session_locks[session_id]:
        history = list(sessions[session_id])
        system_prompt = SYSTEM_PROMPT_BASE
        reply_lang = infer_reply_lang(user_input, session_id)
        system_prompt += " Reply in Tamil." if reply_lang == "ta" else " Reply in English."
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]
    return messages


def save_turn(session_id: str, user_input: str, assistant_reply: str):
    with session_locks[session_id]:
        sessions[session_id].append({"role": "user", "content": user_input})
        sessions[session_id].append({"role": "assistant", "content": assistant_reply[:2500]})


async def stream_vllm_chat(websocket: WebSocket, session_id: str, user_input: str) -> bool:
    model = await resolve_model_name()
    messages = prepare_llm_request_messages(session_id, user_input)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "stream": True,
    }

    full_reply = ""
    sentence_buffer = ""
    reply_lang = infer_reply_lang(user_input, session_id)

    async with http_client.stream("POST", f"{VLLM_BASE_URL}/chat/completions", json=payload) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not websocket_is_connected(websocket):
                return False

            if not line or not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except Exception:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {}).get("content", "")
            if not delta:
                continue

            full_reply += delta
            sentence_buffer += delta

            ok = await safe_send_text(websocket, {
                "type": "llm_delta",
                "text": delta
            })
            if not ok:
                return False

            ready_sentences, remainder = split_complete_sentences(sentence_buffer, final=False)
            if ready_sentences:
                for sent in ready_sentences:
                    await send_tts_chunk(websocket, sent, reply_lang)
                sentence_buffer = remainder

    final_reply = clean_reply_for_voice(full_reply)

    await safe_send_text(websocket, {
        "type": "llm_done",
        "text": final_reply
    })

    remaining_sentences, _ = split_complete_sentences(sentence_buffer, final=True)
    for sent in remaining_sentences:
        await send_tts_chunk(websocket, sent, reply_lang)

    save_turn(session_id, user_input, final_reply)
    return True


async def handle_user_text(websocket: WebSocket, session_id: str, user_text: str, typed: bool = False) -> bool:
    user_text = user_text.strip()
    if not user_text:
        return False

    if typed:
        ok = await safe_send_text(websocket, {"type": "typed_user", "text": user_text})
        if not ok:
            return False

    return await stream_vllm_chat(websocket, session_id, user_text)


def should_run_partial(session: VoiceSession) -> bool:
    if not session.speech_started:
        return False
    total_ms = audio_ms(session.audio_buffer)
    return (total_ms - session.last_partial_run_ms) >= PARTIAL_TRANSCRIBE_EVERY_MS


def can_finalize_again(session: VoiceSession, now_ms: float) -> bool:
    return (now_ms - session.last_finalize_ms) >= FINALIZE_COOLDOWN_MS


def should_finalize(session: VoiceSession, now_ms: float) -> bool:
    return (
        session.speech_started
        and session.voiced_audio_ms >= MIN_VOICED_AUDIO_MS
        and session.silence_after_speech_ms >= END_OF_SPEECH_SILENCE_MS
        and can_finalize_again(session, now_ms)
    )


def reset_utterance(session: VoiceSession, now_ms: float):
    session.audio_buffer = bytearray()
    session.partial_text = ""
    session.last_partial_run_ms = 0.0
    session.speech_started = False
    session.voiced_audio_ms = 0.0
    session.silence_after_speech_ms = 0.0
    session.last_finalize_ms = now_ms


async def choose_asr_text(audio_bytes: bytes, voice_mode: str) -> str:
    if voice_mode == "en":
        result = await call_whisper_asr(audio_bytes, "en")
        return result.get("text", "").strip()

    if voice_mode == "ta":
        result = await call_ai4bharat_asr(audio_bytes)
        text = result.get("text", "").strip()
        if text:
            return text
        result = await call_whisper_asr(audio_bytes, None)
        return result.get("text", "").strip()

    result = await call_whisper_asr(audio_bytes, None)
    text = result.get("text", "").strip()
    lang = result.get("language")

    if lang == "ta" or contains_tamil_script(text):
        ta_result = await call_ai4bharat_asr(audio_bytes)
        ta_text = ta_result.get("text", "").strip()
        if ta_text:
            return ta_text

    return text


async def finalize_voice_utterance(websocket: WebSocket, session_id: str, session: VoiceSession, now_ms: float) -> None:
    final_bytes = bytes(session.audio_buffer)

    if not final_bytes or not session.speech_started or session.voiced_audio_ms < MIN_VOICED_AUDIO_MS:
        reset_utterance(session, now_ms)
        return

    final_text = await choose_asr_text(final_bytes, session.voice_mode)
    final_text = final_text.strip()

    if final_text and len(final_text) >= MIN_TRANSCRIPT_CHARS:
        sent = await safe_send_text(websocket, {
            "type": "final_transcript",
            "text": final_text
        })
        if sent:
            await handle_user_text(websocket, session_id, final_text, typed=False)
    else:
        await safe_send_text(websocket, {"type": "discarded_voice", "text": "discarded short or unclear speech"})

    reset_utterance(session, now_ms)


@app.get("/")
def root():
    index_path = BASE_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.on_event("shutdown")
async def on_shutdown():
    await http_client.aclose()


@app.websocket("/ws/voice/{session_id}")
async def ws_voice(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = VoiceSession()
    session.reply_lang = profiles[session_id]["reply_lang"]
    voice_sessions[session_id] = session

    await safe_send_text(websocket, {"type": "info", "text": "connected"})
    await safe_send_text(websocket, {
        "type": "config",
        "voice_mode": session.voice_mode,
        "reply_lang": profiles[session_id]["reply_lang"]
    })

    try:
        while True:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                break

            if data.get("type") == "websocket.disconnect":
                break

            if "text" in data and data["text"] is not None:
                raw_text = data["text"]

                try:
                    msg = json.loads(raw_text)
                except Exception:
                    msg = {}

                msg_type = msg.get("type")

                if msg_type == "text_chat":
                    ok = await handle_user_text(websocket, session_id, (msg.get("text") or "").strip(), typed=True)
                    if not ok and not websocket_is_connected(websocket):
                        break
                    continue

                if msg_type == "set_voice_mode":
                    mode = (msg.get("mode") or "auto").lower()
                    if mode not in {"auto", "en", "ta"}:
                        mode = "auto"
                    session.voice_mode = mode
                    continue

                if msg_type == "set_reply_lang":
                    mode = (msg.get("mode") or "auto").lower()
                    if mode not in {"auto", "en", "ta"}:
                        mode = "auto"
                    profiles[session_id]["reply_lang"] = mode
                    session.reply_lang = mode
                    continue

                if msg_type == "clear_session":
                    with session_locks[session_id]:
                        sessions[session_id] = deque(maxlen=12)
                    reset_utterance(session, (time.time() - session.started_at) * 1000.0)
                    await safe_send_text(websocket, {"type": "cleared"})
                    continue

                if msg_type == "stop_voice":
                    now_ms = (time.time() - session.started_at) * 1000.0
                    await finalize_voice_utterance(websocket, session_id, session, now_ms)
                    continue

            elif "bytes" in data and data["bytes"] is not None:
                chunk = data["bytes"]
                chunk_ms = audio_ms(chunk)
                now_ms = (time.time() - session.started_at) * 1000.0
                chunk_rms = rms_from_pcm16(chunk)

                if chunk_rms >= RMS_THRESHOLD:
                    if not session.speech_started:
                        session.speech_started = True
                        session.audio_buffer = bytearray()
                        session.partial_text = ""
                        session.last_partial_run_ms = 0.0
                        session.voiced_audio_ms = 0.0
                        session.silence_after_speech_ms = 0.0

                    session.audio_buffer.extend(chunk)
                    session.voiced_audio_ms += chunk_ms
                    session.silence_after_speech_ms = 0.0
                else:
                    if session.speech_started:
                        session.audio_buffer.extend(chunk)
                        session.silence_after_speech_ms += chunk_ms

                if should_run_partial(session):
                    partial_bytes = get_last_window_bytes(bytes(session.audio_buffer), PARTIAL_WINDOW_MS)
                    partial_text = await choose_asr_text(partial_bytes, session.voice_mode)
                    session.last_partial_run_ms = audio_ms(session.audio_buffer)

                    if partial_text and partial_text != session.partial_text and len(partial_text) >= MIN_TRANSCRIPT_CHARS:
                        session.partial_text = partial_text
                        await safe_send_text(websocket, {
                            "type": "partial_transcript",
                            "text": partial_text
                        })

                if should_finalize(session, now_ms):
                    await finalize_voice_utterance(websocket, session_id, session, now_ms)

    finally:
        voice_sessions.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, reload=False)
