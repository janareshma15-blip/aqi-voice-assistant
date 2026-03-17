import os

# ============================================================
# FORCE CPU BEFORE TORCH IMPORT
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict, deque
from threading import Lock
from faster_whisper import WhisperModel
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import asyncio
import base64
import json
import numpy as np
import re
import subprocess
import sys
import time
import uuid
import torch

# Hard-disable CUDA usage in this process
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

from transformers import AutoModel
from huggingface_hub import snapshot_download, hf_hub_download
from better_profanity import profanity

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except Exception:
    AnalyzerEngine = None
    PRESIDIO_AVAILABLE = False


app = FastAPI()

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECTS_AQI_DIR = BASE_DIR.parent
RUNTIME_DIR = PROJECTS_AQI_DIR / "runtime"
HF_CACHE_DIR = RUNTIME_DIR / "hf-cache"
TMP_AUDIO_DIR = RUNTIME_DIR / "tmp_audio"
KOKORO_RUNTIME_DIR = RUNTIME_DIR / "kokoro-runtime"

TTS_DIR = BASE_DIR / "tts"
KOKORO_WORKER = TTS_DIR / "kokoro_worker.py"
MMS_TAMIL_WORKER = TTS_DIR / "mms_tamil_worker.py"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
KOKORO_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# APP CONFIG
# ============================================================
APP_PORT = int(os.getenv("APP_PORT", "9002"))

MAX_INPUT_CHARS = 8000
MODEL_SAFE_INPUT_CHARS = 1800
HISTORY_MAX_CHARS = 1400

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

AI4BHARAT_MODEL_NAME = os.getenv(
    "AI4BHARAT_MODEL_NAME",
    "ai4bharat/indic-conformer-600m-multilingual"
)

KOKORO_HF_REPO = os.getenv("KOKORO_HF_REPO", "onnx-community/Kokoro-82M-v1.0-ONNX")
MMS_TAMIL_HF_REPO = os.getenv("MMS_TAMIL_HF_REPO", "facebook/mms-tts-tam")

KOKORO_MODEL_FILE = KOKORO_RUNTIME_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = KOKORO_RUNTIME_DIR / "voices-v1.0.bin"
KOKORO_ENV_NAME = os.getenv("KOKORO_ENV_NAME", "kokoro312")

MMS_TAMIL_MODEL_DIR = HF_CACHE_DIR / "mms-tts-tam"

RMS_THRESHOLD = float(os.getenv("RMS_THRESHOLD", "0.007"))
END_OF_SPEECH_SILENCE_MS = int(os.getenv("END_OF_SPEECH_SILENCE_MS", "300"))
PARTIAL_TRANSCRIBE_EVERY_MS = int(os.getenv("PARTIAL_TRANSCRIBE_EVERY_MS", "320"))
PARTIAL_WINDOW_MS = int(os.getenv("PARTIAL_WINDOW_MS", "700"))
LANG_HINT_MIN_AUDIO_MS = int(os.getenv("LANG_HINT_MIN_AUDIO_MS", "450"))
LANG_HINT_CHECK_EVERY_MS = int(os.getenv("LANG_HINT_CHECK_EVERY_MS", "1400"))
MIN_VOICED_AUDIO_MS = int(os.getenv("MIN_VOICED_AUDIO_MS", "180"))
FINALIZE_COOLDOWN_MS = int(os.getenv("FINALIZE_COOLDOWN_MS", "900"))
MIN_TRANSCRIPT_CHARS = 1

ENABLE_PARTIAL_ASR = os.getenv("ENABLE_PARTIAL_ASR", "0") == "1"
MAX_SPOKEN_REPLY_CHARS = 1200

# ============================================================
# LLM
# ============================================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:9005/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma-3-12b-it-text")

client = OpenAI(base_url=LLM_BASE_URL, api_key="dummy")
async_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="dummy")

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = (
    "You are a professional AI engineering assistant for an AQI monitoring and AI systems project. "
    "Your role is limited to AQI monitoring, AI ML, FastAPI, dashboards, APIs, deployment, debugging, architecture, "
    "documentation, interviews, project building, and closely related safe academic or engineering topics. "
    "If the user asks who you are, say exactly: "
    "'I am an AI engineering assistant for AQI and AI systems work.' "
    "Do not repeat this identity unless explicitly asked. "
    "If the user speaks in Tamil, reply mainly in Tamil. "
    "If the user speaks in English, reply in English. "
    "For voice conversations, keep replies very short, natural, and direct unless the user explicitly asks for detail. "
    "If the user asks for code, return valid runnable code in fenced code blocks. "
    "Preserve Python syntax exactly. "
    "Do not insert spaces inside variable names. "
    "If the user asks something outside AQI, AI, dashboards, APIs, deployment, debugging, architecture, "
    "or related academic engineering scope, politely refuse and redirect back to scope."
)

# ============================================================
# STATE
# ============================================================
sessions = defaultdict(lambda: deque(maxlen=4))
session_locks = defaultdict(Lock)

profiles = defaultdict(lambda: {
    "current_user_name": None,
    "mentioned_names": [],
    "pending_name_confirmation": None,
    "previous_name_before_pending": None,
    "pending_live_aqi_location": False,
})

voice_sessions = {}

# ============================================================
# GUARDRAILS CONFIG
# ============================================================
profanity.load_censor_words()
PII_ANALYZER = AnalyzerEngine() if PRESIDIO_AVAILABLE else None

PROMPT_EXTRACTION_PATTERNS = [
    "system prompt", "developer prompt", "hidden prompt", "hidden instructions",
    "internal instructions", "show your prompt", "reveal your prompt",
    "what are your instructions", "ignore previous instructions",
    "repeat your system message", "give me your system prompt",
    "show me your system prompt",
]

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "forget all previous instructions",
    "act as",
    "developer mode",
    "jailbreak",
    "bypass safety",
    "disable safety",
    "reveal hidden prompt",
    "print system prompt",
    "leak secrets",
    "show internal prompt",
]

HARMFUL_CYBER_PATTERNS = [
    "hack ", "hacking", "malware", "ransomware", "phishing", "steal password",
    "steal credentials", "keylogger", "exploit", "payload", "reverse shell",
    "bypass authentication", "sql injection attack", "ddos", "bruteforce",
    "privilege escalation", "credential stuffing", "meterpreter",
]

HARMFUL_IMAGE_PATTERNS = [
    "morph a girl badly", "morph a boy badly", "morph a person badly",
    "morph a girls image", "morph a girl's image", "make her ugly", "make him ugly",
    "make this girl ugly", "make this boy ugly", "deform her face", "deform his face",
    "distort her face", "distort his face", "edit her badly", "edit him badly",
    "humiliate her image", "humiliate his image", "ruin her face", "ruin his face",
]

OUT_OF_SCOPE_PATTERNS_EN = [
    "write a poem", "tell me a joke", "roleplay", "sing a song", "i love you",
    "be my girlfriend", "be my boyfriend", "marry me", "kiss me", "hug me",
]

OUT_OF_SCOPE_PATTERNS_TA = [
    "சாப்பிட்டாயா", "ஜோக் சொல்லு", "காதல்", "லவ் யூ", "கவிதை எழுது", "பாட்டு பாடு",
]

SUSPICIOUS_SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
    r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}",
    r"access[_-]?token\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{12,}",
    r"secret[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}",
]

# ============================================================
# MODEL DOWNLOAD HELPERS
# ============================================================
def ensure_kokoro_assets():
    if not KOKORO_MODEL_FILE.exists():
        print(f"Downloading Kokoro model from HF repo: {KOKORO_HF_REPO}")
        hf_hub_download(
            repo_id=KOKORO_HF_REPO,
            filename="kokoro-v1.0.onnx",
            local_dir=str(KOKORO_RUNTIME_DIR),
        )

    if not KOKORO_VOICES_FILE.exists():
        print(f"Downloading Kokoro voices from HF repo: {KOKORO_HF_REPO}")
        downloaded_voices = hf_hub_download(
            repo_id=KOKORO_HF_REPO,
            filename="voices/voices-v1.0.bin",
            local_dir=str(KOKORO_RUNTIME_DIR),
        )
        downloaded_voices_path = Path(downloaded_voices)
        if downloaded_voices_path.exists() and downloaded_voices_path != KOKORO_VOICES_FILE:
            KOKORO_VOICES_FILE.write_bytes(downloaded_voices_path.read_bytes())

def ensure_mms_tamil_model():
    if MMS_TAMIL_MODEL_DIR.exists() and any(MMS_TAMIL_MODEL_DIR.iterdir()):
        return
    print(f"Downloading MMS Tamil model from HF repo: {MMS_TAMIL_HF_REPO}")
    snapshot_download(
        repo_id=MMS_TAMIL_HF_REPO,
        local_dir=str(MMS_TAMIL_MODEL_DIR),
    )

# ============================================================
# MODEL LOADING
# ============================================================
print("Ensuring Hugging Face assets...")
ensure_kokoro_assets()
ensure_mms_tamil_model()

print("Loading faster-whisper English ASR...")
print(
    f"ASR config -> size={WHISPER_MODEL_SIZE}, "
    f"device={WHISPER_DEVICE}, compute_type={WHISPER_COMPUTE_TYPE}"
)
whisper_model_en = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)
print("faster-whisper English ASR loaded.")

print("Loading AI4Bharat Tamil ASR...")
print(f"AI4Bharat model -> {AI4BHARAT_MODEL_NAME}")
try:
    ai4bharat_asr = AutoModel.from_pretrained(
        AI4BHARAT_MODEL_NAME,
        trust_remote_code=True,
        device="cpu",
    )
    ai4bharat_asr.eval()
    print("AI4Bharat Tamil ASR loaded.")
except Exception as e:
    print("Failed to load AI4Bharat Tamil ASR:", str(e))
    raise

# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class EvalDecision:
    kind: str
    reply: Optional[str] = None
    user_input: Optional[str] = None
    error: Optional[str] = None
    flags: Optional[List[str]] = None

class VoiceSession:
    def __init__(self):
        self.audio_buffer = bytearray()
        self.partial_text = ""
        self.last_partial_run_ms = 0.0
        self.last_lang_check_ms = 0.0
        self.last_finalize_ms = -999999.0
        self.started_at = time.time()
        self.detected_lang: Optional[str] = None
        self.lang_locked = False
        self.speech_started = False
        self.voiced_audio_ms = 0.0
        self.silence_after_speech_ms = 0.0

# ============================================================
# BASIC HELPERS
# ============================================================
TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")

def clean_text_basic(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def contains_tamil_script(text: str) -> bool:
    return bool(TAMIL_CHAR_RE.search(text or ""))

def is_tamil_text(text: str) -> bool:
    return sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF") > 0

def contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower().strip()
    return any(k in t for k in keywords)

def normalize_domain_terms(text: str) -> str:
    if not text:
        return text
    replacements = {
        r"\bafi\b": "AQI",
        r"\baqi\b": "AQI",
        r"\ba q i\b": "AQI",
        r"\bair quality index\b": "AQI",
        r"\bpm 2\.?5\b": "PM2.5",
        r"\bpm 10\b": "PM10",
    }
    out = text
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

# ============================================================
# PROFILE HELPERS
# ============================================================
def normalize_name(name: str) -> str:
    return name.strip().capitalize()

def add_mentioned_name(session_id: str, name: str):
    name = normalize_name(name)
    if name and name not in profiles[session_id]["mentioned_names"]:
        profiles[session_id]["mentioned_names"].append(name)

def extract_name_candidate(text: str):
    lowered = text.lower().strip()
    patterns = [
        r"^my name is\s+([a-zA-Z]{2,30})$",
        r"^i am\s+([a-zA-Z]{2,30})$",
        r"^i'm\s+([a-zA-Z]{2,30})$",
    ]
    for pattern in patterns:
        m = re.match(pattern, lowered)
        if m:
            return normalize_name(m.group(1))
    return None

def update_profile_memory(session_id: str, text: str):
    candidate = extract_name_candidate(text)
    if not candidate:
        return None
    current = profiles[session_id]["current_user_name"]
    add_mentioned_name(session_id, candidate)
    if current is None or current == candidate:
        profiles[session_id]["current_user_name"] = candidate
        profiles[session_id]["pending_name_confirmation"] = None
        profiles[session_id]["previous_name_before_pending"] = None
        return ("accepted", candidate)
    profiles[session_id]["pending_name_confirmation"] = candidate
    profiles[session_id]["previous_name_before_pending"] = current
    return ("confirm_change", candidate)

def maybe_handle_name_confirmation(session_id: str, text: str):
    pending = profiles[session_id]["pending_name_confirmation"]
    previous = profiles[session_id]["previous_name_before_pending"]
    if not pending:
        return None
    t = text.lower().strip()
    if t in {"yes", "yeah", "yep", "correct", "confirm", "okay", "ok"}:
        profiles[session_id]["current_user_name"] = pending
        profiles[session_id]["pending_name_confirmation"] = None
        profiles[session_id]["previous_name_before_pending"] = None
        return f"Okay, I will remember that your current name is {pending}."
    if t in {"no", "nope", "cancel", "wrong"}:
        profiles[session_id]["pending_name_confirmation"] = None
        profiles[session_id]["previous_name_before_pending"] = None
        return f"Okay, I will continue to remember your current name as {previous}."
    return f"Your current stored name is still {previous}. If you want me to update it to {pending}, reply yes. Otherwise reply no."

def direct_memory_answer(session_id: str, user_input: str):
    t = user_input.lower().strip()
    current_name = profiles[session_id]["current_user_name"]
    if t in {"who am i", "what is my name", "whats my name", "what's my name"}:
        return f"You are {current_name}." if current_name else "You have not told me your name in this session yet."
    if t in {"who are you", "what are you", "what do you do"}:
        return "I am an AI engineering assistant for AQI and AI systems work."
    if user_input.strip() in {"நீங்கள் யார்", "நீ யார்", "நீ என்ன செய்கிறாய்"}:
        return "நான் AQI மற்றும் AI அமைப்புகளுக்கான AI பொறியியல் உதவியாளர்."
    return None

# ============================================================
# PII / SECURITY HELPERS
# ============================================================
def detect_pii_entities(text: str) -> List[Dict[str, Any]]:
    if not text or not PII_ANALYZER:
        return []
    try:
        results = PII_ANALYZER.analyze(text=text, language="en")
        return [{"entity_type": r.entity_type, "score": float(r.score)} for r in results]
    except Exception:
        return []

def contains_secret_like_data(text: str) -> bool:
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SUSPICIOUS_SECRET_PATTERNS)

def contains_bad_words(text: str) -> bool:
    try:
        return profanity.contains_profanity(text or "")
    except Exception:
        return False

# ============================================================
# AQI HELPERS
# ============================================================
def get_aqi_depth(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["elaborate", "detailed", "full explanation", "விரிவாக"]):
        return "elaborate"
    if any(k in t for k in ["summary", "brief", "சுருக்கம்"]):
        return "summary"
    return "details"

def is_live_aqi_request(text: str) -> bool:
    t = text.lower().strip()
    live_keywords = ["current", "today", "live", "now", "near me", "location", "இப்போது", "தற்போதைய"]
    aqi_keywords = ["aqi", "air quality", "air pollution", "காற்று", "மாசு", "காற்று தரம்"]
    return contains_any(t, live_keywords) and contains_any(t, aqi_keywords)

def is_aqi_concept_request(text: str) -> bool:
    t = text.lower().strip()
    concept_keywords = ["what is", "details", "summary", "explanation", "meaning", "explain", "விளக்கம்", "என்றால் என்ன"]
    aqi_keywords = ["aqi", "air quality", "air pollution", "காற்று", "மாசு", "காற்று தரம்"]
    architecture_keywords = ["architecture", "pipeline", "workflow", "அமைப்பு", "வடிவமைப்பு"]
    return contains_any(t, concept_keywords) and contains_any(t, aqi_keywords) and not contains_any(t, architecture_keywords) and not is_live_aqi_request(text)

def is_aqi_architecture_request(text: str) -> bool:
    t = text.lower().strip()
    architecture_keywords = ["architecture", "pipeline", "workflow", "modules", "components", "அமைப்பு", "வடிவமைப்பு"]
    aqi_keywords = ["aqi", "air quality", "air pollution", "காற்று", "மாசு", "காற்று தரம்"]
    return contains_any(t, architecture_keywords) and contains_any(t, aqi_keywords)

def is_aqi_code_request(text: str) -> bool:
    t = text.lower()
    code_keywords = ["code", "python", "fastapi", "api", "program", "script", "function", "implement", "கோடு", "நிரல்"]
    aqi_keywords = ["aqi", "air quality", "காற்று", "மாசு"]
    return contains_any(t, code_keywords) and contains_any(t, aqi_keywords)

def extract_location_from_text(text: str) -> Optional[str]:
    lower_t = text.lower().strip()
    patterns = [
        r"current aqi in (.+)$",
        r"today aqi in (.+)$",
        r"live aqi in (.+)$",
        r"aqi in (.+)$",
        r"air quality in (.+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, lower_t)
        if m:
            loc = m.group(1).strip(" .,")
            if loc:
                return loc.title()
    return None

def direct_greeting_reply(user_input: str):
    t = user_input.lower().strip()
    if t in {"hi", "hey", "hello"}:
        return "Hi! How can I help you with your AQI project today?"
    if user_input.strip() in {"வணக்கம்", "ஹாய்", "ஹலோ"}:
        return "வணக்கம்! உங்கள் AQI திட்டத்தில் என்ன உதவி வேண்டும்?"
    return None

def direct_aqi_answer(user_input: str):
    tamil = is_tamil_text(user_input)
    if is_aqi_concept_request(user_input):
        depth = get_aqi_depth(user_input)
        if depth == "elaborate":
            return None
        if tamil:
            if depth == "summary":
                return "AQI என்பது காற்றின் தரத்தை அளவிடும் ஒரு குறியீடு. எண் குறைவாக இருந்தால் காற்று நல்லது."
            return "AQI என்பது காற்றின் தரத்தை அளவிடும் ஒரு குறியீடு. இது PM2.5, PM10, NO2, SO2, CO மற்றும் O3 போன்ற மாசுபடுத்திகளின் அளவை அடிப்படையாகக் கொண்டு கணக்கிடப்படுகிறது."
        if depth == "summary":
            return "AQI is a measure of air quality. A lower AQI means cleaner air."
        return "AQI is a measure of air quality. It is calculated using pollutants such as PM2.5, PM10, NO2, SO2, CO, and O3."
    if is_aqi_architecture_request(user_input):
        if tamil:
            return "AQI அமைப்பு பொதுவாக தரவு சேகரிப்பு, தரவு செயலாக்கம், AI அல்லது ML மாதிரி, மற்றும் dashboard காட்சிப்படுத்தல் ஆகிய பகுதிகளை கொண்டுள்ளது."
        return "An AQI system usually includes data collection, data processing, AI or ML modeling, and dashboard visualization."
    return None

def handle_live_aqi_flow(session_id: str, user_input: str):
    location = extract_location_from_text(user_input)
    if profiles[session_id]["pending_live_aqi_location"] and location:
        profiles[session_id]["pending_live_aqi_location"] = False
        return f"You asked for the current AQI in {location}. Current live AQI lookup is not connected yet in this app, but I can help you wire an API or data source for it."
    if is_live_aqi_request(user_input) and location:
        return f"You asked for the current AQI in {location}. Current live AQI lookup is not connected yet in this app, but I can help you wire an API or data source for it."
    if is_live_aqi_request(user_input):
        profiles[session_id]["pending_live_aqi_location"] = True
        return "தற்போதைய AQI மதிப்பை சொல்ல நான் உங்கள் நகரம் அல்லது இடத்தின் பெயரைத் தெரிந்து கொள்ள வேண்டும்." if is_tamil_text(user_input) else "To provide the current AQI, I need your city or location name."
    return None

# ============================================================
# AUDIO HELPERS
# ============================================================
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

# ============================================================
# ASR
# ============================================================
def detect_language_hint_whisper(audio_bytes: bytes) -> Optional[str]:
    try:
        audio = pcm16_bytes_to_float32(audio_bytes)
        if audio.size == 0:
            return None
        segments, info = whisper_model_en.transcribe(audio, language=None, beam_size=1, vad_filter=False)
        lang = getattr(info, "language", None)
        text = "".join(seg.text for seg in segments).strip()
        if lang == "en":
            return "en"
        if contains_tamil_script(text):
            return "ta"
        return None
    except Exception:
        return None

def transcribe_with_whisper_en(audio_bytes: bytes, language: Optional[str] = None) -> str:
    audio = pcm16_bytes_to_float32(audio_bytes)
    if audio.size == 0:
        return ""
    try:
        segments, _ = whisper_model_en.transcribe(audio, language=language, beam_size=1, vad_filter=False)
        return clean_text("".join(seg.text for seg in segments).strip())
    except Exception as e:
        print("Whisper EN transcription error:", str(e))
        return ""

def transcribe_with_ai4bharat_tamil(audio_bytes: bytes) -> str:
    audio = pcm16_bytes_to_float32(audio_bytes)
    if audio.size == 0:
        return ""
    try:
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        result = ai4bharat_asr(wav, "ta", "ctc")
        if isinstance(result, (list, tuple)):
            text = result[0]
        else:
            text = str(result)
        return clean_text(text)
    except Exception as e:
        print("AI4Bharat Tamil transcription error:", str(e))
        return ""

def transcribe_audio_routed(audio_bytes: bytes, language_hint: Optional[str]) -> Tuple[str, str]:
    if not audio_bytes:
        return "", "none"
    if language_hint == "ta":
        text = normalize_domain_terms(transcribe_with_ai4bharat_tamil(audio_bytes))
        if text:
            return text, "ai4bharat_tamil"
    if language_hint == "en":
        text = normalize_domain_terms(transcribe_with_whisper_en(audio_bytes, language="en"))
        if text:
            return text, "whisper_en_cpu"
    text_en = normalize_domain_terms(transcribe_with_whisper_en(audio_bytes, language="en"))
    if text_en and not contains_tamil_script(text_en):
        return text_en, "whisper_en_fallback"
    text_ta = normalize_domain_terms(transcribe_with_ai4bharat_tamil(audio_bytes))
    if text_ta:
        return text_ta, "ai4bharat_tamil_fallback"
    text_auto = normalize_domain_terms(transcribe_with_whisper_en(audio_bytes, language=None))
    if text_auto:
        return text_auto, "whisper_auto_tamil_cpu" if contains_tamil_script(text_auto) else "whisper_auto_cpu"
    return "", "none"

# ============================================================
# TTS
# ============================================================
def detect_lang(text: str) -> str:
    tamil_chars = 0
    latin_chars = 0
    for ch in text:
        if "\u0B80" <= ch <= "\u0BFF":
            tamil_chars += 1
        elif "a" <= ch.lower() <= "z":
            latin_chars += 1
    return "ta" if tamil_chars > latin_chars else "en"

def generate_english_tts_kokoro(text: str, output_path: Path) -> bool:
    if not KOKORO_WORKER.exists() or not KOKORO_MODEL_FILE.exists() or not KOKORO_VOICES_FILE.exists():
        print("Kokoro paths missing.")
        return False
    cmd = ["conda", "run", "-n", KOKORO_ENV_NAME, "python", str(KOKORO_WORKER), text, str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print("Kokoro TTS stderr:", result.stderr)
        return False
    return output_path.exists()

def generate_tamil_tts_mms(text: str, output_path: Path) -> bool:
    if not MMS_TAMIL_WORKER.exists():
        print("Tamil worker missing.")
        return False
    env = os.environ.copy()
    env["MMS_TAMIL_MODEL_DIR"] = str(MMS_TAMIL_MODEL_DIR)
    cmd = [sys.executable, str(MMS_TAMIL_WORKER), text, str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(BASE_DIR), env=env)
    if result.returncode != 0:
        print("Tamil TTS stderr:", result.stderr)
        return False
    return output_path.exists()

def wav_file_to_base64(path: Path) -> Optional[str]:
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None

def strip_code_blocks_for_tts(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", "I have provided the code on screen.", text)

def clean_reply_for_voice(text: str) -> str:
    text = re.sub(r"[*_`#>-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def make_full_spoken_text(reply_text: str) -> str:
    spoken = strip_code_blocks_for_tts(clean_reply_for_voice(reply_text))
    if len(spoken) > MAX_SPOKEN_REPLY_CHARS:
        spoken = spoken[:MAX_SPOKEN_REPLY_CHARS].rsplit(" ", 1)[0].strip()
        if spoken and spoken[-1] not in ".!?":
            spoken += "."
    return spoken

def generate_tts_base64(text: str) -> Optional[Dict[str, str]]:
    text = text.strip()
    if not text:
        return None
    file_id = uuid.uuid4().hex
    output_path = TMP_AUDIO_DIR / f"{file_id}.wav"
    lang = detect_lang(text)
    try:
        ok = generate_tamil_tts_mms(text, output_path) if lang == "ta" else generate_english_tts_kokoro(text, output_path)
        if not ok or not output_path.exists():
            return None
        b64 = wav_file_to_base64(output_path)
        output_path.unlink(missing_ok=True)
        if not b64:
            return None
        return {"mime_type": "audio/wav", "audio_base64": b64}
    except Exception as e:
        print("TTS exception:", str(e))
        output_path.unlink(missing_ok=True)
        return None

# ============================================================
# PROMPT / HISTORY HELPERS
# ============================================================
def choose_max_tokens(user_input: str, voice_mode: bool = False) -> int:
    text = user_input.lower().strip()
    if voice_mode:
        return 140 if is_aqi_code_request(text) else 100
    return 220 if is_aqi_code_request(text) else 160

def trim_history_for_small_context(history, max_chars=HISTORY_MAX_CHARS):
    trimmed = []
    total_chars = 0
    for msg in reversed(history):
        content = msg.get("content", "")
        msg_len = len(content)
        if total_chars + msg_len > max_chars:
            remaining = max_chars - total_chars
            if remaining > 160:
                trimmed.append({"role": msg["role"], "content": content[-remaining:]})
            break
        trimmed.append(msg)
        total_chars += msg_len
    return list(reversed(trimmed))

def build_system_prompt_for_request(user_input: str) -> str:
    prompt = SYSTEM_PROMPT
    if is_aqi_code_request(user_input):
        prompt += " Return only valid runnable code in fenced code blocks. Preserve Python syntax exactly."
    return prompt

def compress_long_input(text: str, limit: int = MODEL_SAFE_INPUT_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit]

def prepare_llm_request_messages(session_id: str, user_input: str):
    llm_input = compress_long_input(user_input)
    with session_locks[session_id]:
        history = list(sessions[session_id])
        cleaned = []
        expected = "user"
        for m in history:
            if m.get("role") == expected:
                cleaned.append(m)
                expected = "assistant" if expected == "user" else "user"
        history = cleaned
        if history and history[-1]["role"] == "user":
            history.pop()
        history.append({"role": "user", "content": llm_input})
        history = trim_history_for_small_context(history, max_chars=HISTORY_MAX_CHARS)
        effective_system_prompt = build_system_prompt_for_request(user_input)
        request_messages = [{"role": "system", "content": effective_system_prompt}] + history
    return llm_input, request_messages

def save_conversation_turn(session_id: str, user_input: str, assistant_reply: str):
    llm_input = compress_long_input(user_input)
    with session_locks[session_id]:
        history = list(sessions[session_id])
        if history and history[-1]["role"] == "user":
            history.pop()
        history.append({"role": "user", "content": llm_input})
        history.append({"role": "assistant", "content": assistant_reply})
        sessions[session_id] = deque(history[-4:], maxlen=4)

def direct_followup_answer(session_id: str, user_input: str):
    t = user_input.lower().strip()
    if t in {"yes", "yeah", "yep", "ok", "okay"}:
        history = list(sessions[session_id])
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                if "```" in last_assistant or "def " in last_assistant:
                    return "Please ask what you want next, for example: explain the code, simplify the code, or give FastAPI version."
                break
    return None

# ============================================================
# GUARDRAILS
# ============================================================
def run_security_guardrails(user_input: str) -> Optional[EvalDecision]:
    t = user_input.lower().strip()
    flags = []
    if any(p in t for p in PROMPT_EXTRACTION_PATTERNS):
        flags.append("prompt_extraction")
        return EvalDecision("direct", "I can’t reveal my system prompt, hidden instructions, or internal configuration.", flags=flags)
    if any(p in t for p in PROMPT_INJECTION_PATTERNS):
        flags.append("prompt_injection")
        return EvalDecision("direct", "I can’t follow prompt-injection or instruction-bypass requests.", flags=flags)
    if any(p in t for p in HARMFUL_CYBER_PATTERNS):
        flags.append("harmful_cyber")
        return EvalDecision("direct", "I can’t help with hacking, malware, unauthorized access, or harmful cyber activity.", flags=flags)
    if any(p in t for p in HARMFUL_IMAGE_PATTERNS):
        flags.append("harmful_image")
        return EvalDecision("direct", "I can’t help create degrading, humiliating, or harmful image edits of a person.", flags=flags)
    if contains_bad_words(user_input):
        flags.append("profanity")
        return EvalDecision("direct", "I won’t engage with abusive or vulgar language. Please keep the conversation respectful and project-focused.", flags=flags)
    if contains_secret_like_data(user_input):
        flags.append("secret_like_data")
        return EvalDecision("direct", "Your message appears to contain secret-like credentials or sensitive tokens. Please remove or mask them before sending.", flags=flags)
    pii_entities = detect_pii_entities(user_input)
    high_conf_pii = [e for e in pii_entities if e["score"] >= 0.65]
    if high_conf_pii:
        flags.append("pii_detected")
        entity_names = sorted(set(e["entity_type"] for e in high_conf_pii))
        return EvalDecision("direct", f"Your message appears to contain sensitive personal information ({', '.join(entity_names)}). Please remove or mask it before continuing.", flags=flags)
    if any(p in t for p in OUT_OF_SCOPE_PATTERNS_EN) or any(p in t for p in OUT_OF_SCOPE_PATTERNS_TA):
        flags.append("out_of_scope")
        return EvalDecision("direct", "I’m strictly limited to AQI project and approved assistant tasks.", flags=flags)
    return None

# ============================================================
# EVALUATION
# ============================================================
def evaluate_user_input(session_id: str, raw_user_input: str) -> dict:
    user_input = normalize_domain_terms(clean_text_basic(raw_user_input))
    if not user_input:
        return {"kind": "error", "error": "Empty message."}
    if len(user_input) > MAX_INPUT_CHARS:
        return {"kind": "error", "error": f"Your message is too long. Max allowed is about {MAX_INPUT_CHARS} characters."}
    if user_input.lower().strip() in {"how are you", "how r you"}:
        return {"kind": "direct", "reply": "I’m ready to help you with your AQI and AI project work."}
    confirmation_reply = maybe_handle_name_confirmation(session_id, user_input)
    if confirmation_reply:
        return {"kind": "direct", "reply": confirmation_reply}
    guardrail_decision = run_security_guardrails(user_input)
    if guardrail_decision:
        return {"kind": guardrail_decision.kind, "reply": guardrail_decision.reply, "flags": guardrail_decision.flags or []}
    name_update = update_profile_memory(session_id, user_input)
    if name_update:
        kind, candidate = name_update
        if kind == "accepted":
            return {"kind": "direct", "reply": f"Glad to know your name, {candidate}. How can I help you with your project today?"}
        if kind == "confirm_change":
            current = profiles[session_id]["previous_name_before_pending"]
            return {"kind": "direct", "reply": f"Earlier you said your name is {current}. Now you said {candidate}. Should I update your current name to {candidate}? Reply with yes or no."}
    memory_reply = direct_memory_answer(session_id, user_input)
    if memory_reply:
        return {"kind": "direct", "reply": memory_reply}
    followup_reply = direct_followup_answer(session_id, user_input)
    if followup_reply:
        return {"kind": "direct", "reply": followup_reply}
    live_aqi_reply = handle_live_aqi_flow(session_id, user_input)
    if live_aqi_reply:
        return {"kind": "direct", "reply": live_aqi_reply}
    greet_reply = direct_greeting_reply(user_input)
    if greet_reply:
        return {"kind": "direct", "reply": greet_reply}
    aqi_reply = direct_aqi_answer(user_input)
    if aqi_reply:
        return {"kind": "direct", "reply": aqi_reply}
    return {"kind": "llm", "user_input": user_input}

# ============================================================
# VOICE STATE HELPERS
# ============================================================
def should_run_partial(session: VoiceSession) -> bool:
    if not ENABLE_PARTIAL_ASR or not session.speech_started:
        return False
    total_ms = audio_ms(session.audio_buffer)
    return (total_ms - session.last_partial_run_ms) >= PARTIAL_TRANSCRIBE_EVERY_MS

def should_check_lang_hint(session: VoiceSession) -> bool:
    total_ms = audio_ms(session.audio_buffer)
    enough_audio = total_ms >= LANG_HINT_MIN_AUDIO_MS
    due = (total_ms - session.last_lang_check_ms) >= LANG_HINT_CHECK_EVERY_MS
    return enough_audio and due and not session.lang_locked and session.speech_started

def can_finalize_again(session: VoiceSession, now_ms: float) -> bool:
    return (now_ms - session.last_finalize_ms) >= FINALIZE_COOLDOWN_MS

def should_finalize(session: VoiceSession, now_ms: float) -> bool:
    return session.speech_started and session.voiced_audio_ms >= MIN_VOICED_AUDIO_MS and session.silence_after_speech_ms >= END_OF_SPEECH_SILENCE_MS and can_finalize_again(session, now_ms)

def reset_utterance(session: VoiceSession, now_ms: float):
    session.audio_buffer = bytearray()
    session.partial_text = ""
    session.last_partial_run_ms = 0.0
    session.last_lang_check_ms = 0.0
    session.detected_lang = None
    session.lang_locked = False
    session.speech_started = False
    session.voiced_audio_ms = 0.0
    session.silence_after_speech_ms = 0.0
    session.last_finalize_ms = now_ms

# ============================================================
# REPLY / TTS HELPERS
# ============================================================
async def send_full_reply_audio(websocket: WebSocket, reply_text: str):
    spoken_text = make_full_spoken_text(reply_text)
    audio_obj = await asyncio.to_thread(generate_tts_base64, spoken_text)
    if audio_obj:
        await websocket.send_text(json.dumps({
            "type": "tts_sentence_ready",
            "text": spoken_text,
            "mime_type": audio_obj["mime_type"],
            "audio_base64": audio_obj["audio_base64"]
        }))

async def send_direct_reply(websocket: WebSocket, reply_text: str):
    clean_reply = clean_reply_for_voice(reply_text)
    await websocket.send_text(json.dumps({
        "type": "llm_done",
        "text": clean_reply
    }))
    await send_full_reply_audio(websocket, clean_reply)

async def stream_llm_and_tts(websocket: WebSocket, session_id: str, user_input: str):
    _, request_messages = prepare_llm_request_messages(session_id, user_input)

    await websocket.send_text(json.dumps({
        "type": "info",
        "text": "generating reply"
    }))

    full_reply = ""

    try:
        stream = await async_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=request_messages,
            temperature=0.2,
            max_tokens=choose_max_tokens(user_input, voice_mode=True),
            stream=True,
        )

        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""

            if not delta:
                continue

            full_reply += delta

            await websocket.send_text(json.dumps({
                "type": "llm_delta",
                "text": delta
            }))

        save_conversation_turn(session_id, user_input, full_reply)

        clean_full_reply = clean_reply_for_voice(full_reply)

        await websocket.send_text(json.dumps({
            "type": "llm_done",
            "text": clean_full_reply
        }))

        await send_full_reply_audio(websocket, clean_full_reply)

    except Exception as e:
        print("LLM stream error:", str(e))
        await websocket.send_text(json.dumps({
            "type": "error",
            "text": "I could not process that input properly. Please try again with a clear AQI-related question."
        }))

async def handle_user_text(websocket: WebSocket, session_id: str, user_text: str, typed: bool = False):
    user_text = normalize_domain_terms((user_text or "").strip())
    print("handle_user_text received:", repr(user_text))
    if not user_text:
        return
    if typed:
        await websocket.send_text(json.dumps({"type": "typed_user", "text": user_text}))
    result = evaluate_user_input(session_id, user_text)
    if result["kind"] == "error":
        await websocket.send_text(json.dumps({"type": "error", "text": result["error"]}))
        return
    if result["kind"] == "direct":
        if result.get("flags"):
            print("Guardrail flags:", result["flags"])
        await send_direct_reply(websocket, result["reply"])
        return
    await stream_llm_and_tts(websocket, session_id, result["user_input"])

# ============================================================
# WEBSOCKET ONLY ENDPOINT
# ============================================================
@app.websocket("/ws/voice/{session_id}")
async def ws_voice(websocket: WebSocket, session_id: str):
    print("WebSocket connection request received for:", session_id)
    await websocket.accept()
    print("WebSocket accepted.")

    session = VoiceSession()
    voice_sessions[session_id] = session

    await websocket.send_text(json.dumps({"type": "info", "text": "connected"}))

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
                    await handle_user_text(
                        websocket=websocket,
                        session_id=session_id,
                        user_text=(msg.get("text") or "").strip(),
                        typed=True
                    )
                    continue

                if msg_type == "clear_session":
                    with session_locks[session_id]:
                        sessions[session_id] = deque(maxlen=4)
                        profiles[session_id] = {
                            "current_user_name": None,
                            "mentioned_names": [],
                            "pending_name_confirmation": None,
                            "previous_name_before_pending": None,
                            "pending_live_aqi_location": False,
                        }

                    reset_utterance(session, (time.time() - session.started_at) * 1000.0)
                    await websocket.send_text(json.dumps({"type": "cleared"}))
                    continue

                if msg_type == "stop_voice":
                    now_ms = (time.time() - session.started_at) * 1000.0
                    final_bytes = bytes(session.audio_buffer)

                    if final_bytes and session.speech_started and session.voiced_audio_ms >= MIN_VOICED_AUDIO_MS:
                        print("Finalize debug:", {
                            "voiced_audio_ms": session.voiced_audio_ms,
                            "silence_after_speech_ms": session.silence_after_speech_ms,
                            "detected_lang": session.detected_lang,
                            "buffer_ms": audio_ms(session.audio_buffer),
                        })

                        final_text, route_used = await asyncio.to_thread(
                            transcribe_audio_routed,
                            final_bytes,
                            session.detected_lang
                        )
                        final_text = normalize_domain_terms(final_text.strip())
                        print("Final route used:", route_used)
                        print("Final text:", repr(final_text))

                        if final_text and len(final_text) >= MIN_TRANSCRIPT_CHARS:
                            await websocket.send_text(json.dumps({
                                "type": "final_transcript",
                                "text": final_text
                            }))
                            await handle_user_text(
                                websocket=websocket,
                                session_id=session_id,
                                user_text=final_text,
                                typed=False
                            )
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "discarded_voice",
                                "text": "empty or too short after ASR"
                            }))

                    reset_utterance(session, now_ms)
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
                        session.last_lang_check_ms = 0.0
                        session.detected_lang = None
                        session.lang_locked = False
                        session.voiced_audio_ms = 0.0
                        session.silence_after_speech_ms = 0.0

                    session.audio_buffer.extend(chunk)
                    session.voiced_audio_ms += chunk_ms
                    session.silence_after_speech_ms = 0.0
                else:
                    if session.speech_started:
                        session.audio_buffer.extend(chunk)
                        session.silence_after_speech_ms += chunk_ms

                if should_check_lang_hint(session):
                    lang_bytes = get_last_window_bytes(bytes(session.audio_buffer), PARTIAL_WINDOW_MS)
                    lang_hint = await asyncio.to_thread(detect_language_hint_whisper, lang_bytes)
                    session.last_lang_check_ms = audio_ms(session.audio_buffer)

                    if lang_hint:
                        session.detected_lang = lang_hint
                        if lang_hint in {"ta", "en"}:
                            session.lang_locked = True

                        await websocket.send_text(json.dumps({
                            "type": "info",
                            "text": f"listening ({lang_hint})"
                        }))

                if should_run_partial(session):
                    partial_bytes = get_last_window_bytes(bytes(session.audio_buffer), PARTIAL_WINDOW_MS)

                    if session.detected_lang == "ta":
                        partial_text = await asyncio.to_thread(transcribe_with_ai4bharat_tamil, partial_bytes)
                    elif session.detected_lang == "en":
                        partial_text = await asyncio.to_thread(transcribe_with_whisper_en, partial_bytes, "en")
                    else:
                        partial_text = await asyncio.to_thread(transcribe_with_whisper_en, partial_bytes, None)

                    partial_text = normalize_domain_terms(partial_text.strip())
                    session.last_partial_run_ms = audio_ms(session.audio_buffer)

                    if partial_text and partial_text != session.partial_text and len(partial_text) >= MIN_TRANSCRIPT_CHARS:
                        session.partial_text = partial_text
                        await websocket.send_text(json.dumps({
                            "type": "partial_transcript",
                            "text": partial_text
                        }))

                if should_finalize(session, now_ms):
                    print("Finalize debug:", {
                        "voiced_audio_ms": session.voiced_audio_ms,
                        "silence_after_speech_ms": session.silence_after_speech_ms,
                        "detected_lang": session.detected_lang,
                        "buffer_ms": audio_ms(session.audio_buffer),
                    })

                    final_bytes = bytes(session.audio_buffer)
                    final_text, route_used = await asyncio.to_thread(
                        transcribe_audio_routed,
                        final_bytes,
                        session.detected_lang
                    )
                    final_text = normalize_domain_terms(final_text.strip())
                    print("Final route used:", route_used)
                    print("Final text:", repr(final_text))

                    if final_text and len(final_text) >= MIN_TRANSCRIPT_CHARS:
                        await websocket.send_text(json.dumps({
                            "type": "final_transcript",
                            "text": final_text
                        }))
                        await handle_user_text(
                            websocket=websocket,
                            session_id=session_id,
                            user_text=final_text,
                            typed=False
                        )
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "discarded_voice",
                            "text": "empty or too short after ASR"
                        }))

                    reset_utterance(session, now_ms)

    except Exception as e:
        print("Voice pipeline error:", str(e))
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": f"Voice pipeline error: {str(e)}"
            }))
        except Exception:
            pass
    finally:
        voice_sessions.pop(session_id, None)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, reload=False)
