from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict, deque
from threading import Lock
from pathlib import Path
from typing import Optional, Tuple
import asyncio
import json
import numpy as np
import os
import re
import time
import uuid

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import (
    pipeline,
    AutoTokenizer,
    VitsModel,
)

# ============================================================
# APP
# ============================================================
app = FastAPI()

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
RUNTIME_DIR = BASE_DIR / "runtime"
HF_MODELS_DIR = RUNTIME_DIR / "hf-models"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================
# ENV / MODEL CONFIG
# ============================================================
PORT = int(os.getenv("APP_PORT", "9002"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:9005/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma-3-12b-it-text")

# Hugging Face repos
HF_ASR_MULTI_REPO = os.getenv("HF_ASR_MULTI_REPO", "openai/whisper-tiny")
HF_ASR_EN_REPO = os.getenv("HF_ASR_EN_REPO", "openai/whisper-tiny.en")
HF_ASR_TA_REPO = os.getenv("HF_ASR_TA_REPO", "vasista22/whisper-tamil-small")
HF_TTS_EN_REPO = os.getenv("HF_TTS_EN_REPO", "facebook/mms-tts-eng")
HF_TTS_TA_REPO = os.getenv("HF_TTS_TA_REPO", "facebook/mms-tts-tam")

# CPU/GPU
TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE_INDEX = 0 if torch.cuda.is_available() else -1

CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))
torch.set_num_threads(max(1, CPU_THREADS))
torch.set_num_interop_threads(max(1, min(2, CPU_THREADS)))

# Audio
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

# Voice latency tuning
RMS_THRESHOLD = 0.009
END_OF_SPEECH_SILENCE_MS = 220
PARTIAL_TRANSCRIBE_EVERY_MS = 320
PARTIAL_WINDOW_MS = 700
LANG_HINT_MIN_AUDIO_MS = 450
LANG_HINT_CHECK_EVERY_MS = 1400
MIN_VOICED_AUDIO_MS = 220
FINALIZE_COOLDOWN_MS = 900
MIN_TRANSCRIPT_CHARS = 2

ENABLE_PARTIAL_ASR = os.getenv("ENABLE_PARTIAL_ASR", "0") == "1"

MAX_TTS_SENTENCE_CHARS = 90
MAX_TTS_QUEUE_SENTENCES = 1

# App logic
MAX_INPUT_CHARS = 8000
MODEL_SAFE_INPUT_CHARS = 1800
HISTORY_MAX_CHARS = 1400

# ============================================================
# LLM CLIENTS
# ============================================================
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

    "For voice conversations, keep replies very short, natural, and direct. "
    "Usually answer in 1 to 2 short sentences. "
    "Do not become poetic, philosophical, emotional, or broad unless explicitly asked. "
    "Stay within AQI, AI, dashboards, heat maps, risk, volatility, APIs, deployment, or related engineering topics. "

    "If the user asks for code, return valid runnable code in fenced code blocks. "
    "Preserve Python syntax exactly. "
    "Do not insert spaces inside variable names. "

    "If the user asks something outside AQI, AI, dashboards, APIs, deployment, debugging, architecture, "
    "or related academic engineering scope, politely refuse and redirect back to scope. "
    "Do not engage in unrelated casual chat."
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
# HTML PAGE - WEBSOCKET ONLY
# ============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>AQI AI Assistant</title>
<style>
body {
  font-family: Arial, sans-serif;
  max-width: 980px;
  margin: 24px auto;
  padding: 20px;
  background: #f7f7f7;
}
h2 {
  text-align: center;
  margin-bottom: 14px;
}
#chat {
  background: white;
  border: 1px solid #ccc;
  border-radius: 10px;
  padding: 15px;
  min-height: 340px;
  max-height: 440px;
  overflow-y: auto;
  margin-bottom: 15px;
}
.msg {
  margin: 10px 0;
  padding: 10px 12px;
  border-radius: 8px;
  white-space: pre-wrap;
  line-height: 1.45;
}
.user { background: #dbeafe; }
.bot { background: #dcfce7; }
.msg pre {
  background: #111;
  color: #eee;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  white-space: pre;
  margin-top: 8px;
}
.msg code {
  font-family: monospace;
}
#ttsPlayerUI {
  width: 100%;
  margin-bottom: 12px;
}
#status {
  margin-bottom: 10px;
  color: #444;
  font-size: 14px;
}
#liveTranscriptBox, #liveReplyBox {
  background: white;
  border: 1px solid #ccc;
  border-radius: 10px;
  padding: 12px;
  margin-bottom: 12px;
  min-height: 48px;
}
.label {
  font-weight: bold;
  margin-bottom: 6px;
}
#voiceControls, #inputRow {
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
}
#inputBox {
  flex: 1;
  padding: 10px;
  font-size: 15px;
}
button {
  padding: 10px 16px;
  font-size: 15px;
  cursor: pointer;
}
button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
</head>
<body>
<h2>AQI AI Assistant</h2>

<div id="status">Voice status: idle</div>

<div id="voiceControls">
  <button id="startBtn">Start Mic</button>
  <button id="stopBtn" disabled>Stop Mic</button>
  <button id="clearBtn">Clear</button>
</div>

<div id="inputRow">
  <input id="inputBox" type="text" placeholder="Type your message..." />
  <button id="sendBtn">Send</button>
</div>

<div id="liveTranscriptBox">
  <div class="label">Live transcript</div>
  <div id="liveTranscript"></div>
</div>

<div id="liveReplyBox">
  <div class="label">Streaming reply</div>
  <div id="liveReply"></div>
</div>

<div id="chat"></div>

<audio id="ttsPlayerUI" controls></audio>

<script>
let sessionId = localStorage.getItem("aqi_ai_session_id");
if (!sessionId) {
  sessionId = "user_" + Math.random().toString(36).slice(2);
  localStorage.setItem("aqi_ai_session_id", sessionId);
}

let ws = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let recording = false;
let busy = false;
let audioQueue = [];
let audioPlaying = false;

const chatBox = document.getElementById("chat");
const inputBox = document.getElementById("inputBox");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const ttsPlayer = document.getElementById("ttsPlayerUI");
const statusBox = document.getElementById("status");
const liveTranscript = document.getElementById("liveTranscript");
const liveReply = document.getElementById("liveReply");

ttsPlayer.preload = "auto";

function setStatus(text) {
  statusBox.textContent = "Voice status: " + text;
}

function escapeHtml(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMessageContent(text) {
  const codeBlockRegex = /```(\\w+)?\\n([\\s\\S]*?)```/g;
  let html = "";
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    const fullMatch = match[0];
    const lang = match[1] || "";
    const code = match[2];
    const start = match.index;
    const end = start + fullMatch.length;

    const before = text.slice(lastIndex, start);
    if (before) {
      html += `<div>${escapeHtml(before).replace(/\\n/g, "<br>")}</div>`;
    }

    html += `<pre><code class="language-${escapeHtml(lang)}">${escapeHtml(code)}</code></pre>`;
    lastIndex = end;
  }

  const remaining = text.slice(lastIndex);
  if (remaining) {
    html += `<div>${escapeHtml(remaining).replace(/\\n/g, "<br>")}</div>`;
  }

  return html || `<div>${escapeHtml(text).replace(/\\n/g, "<br>")}</div>`;
}

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.innerHTML = renderMessageContent(text);
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function downsampleBuffer(float32Array, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) return float32Array;

  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(float32Array.length / sampleRateRatio);
  const result = new Float32Array(newLength);

  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;

    for (let i = offsetBuffer; i < nextOffsetBuffer && i < float32Array.length; i++) {
      accum += float32Array[i];
      count++;
    }

    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

async function playNextQueuedAudio() {
  if (audioPlaying || audioQueue.length === 0) return;
  audioPlaying = true;

  const nextUrl = audioQueue.shift();
  try {
    ttsPlayer.pause();
    ttsPlayer.currentTime = 0;
    ttsPlayer.src = nextUrl + "?t=" + Date.now();
    setStatus("loading");
    await ttsPlayer.play();
  } catch (err) {
    console.log("Audio playback failed:", err);
    setStatus("autoplay blocked, use player controls");
    audioPlaying = false;
  }
}

ttsPlayer.addEventListener("ended", () => {
  audioPlaying = false;
  playNextQueuedAudio();
});

function clearLiveBoxes() {
  liveTranscript.textContent = "";
  liveReply.textContent = "";
}

function attachWsHandlers() {
  ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "partial_transcript") {
      liveTranscript.textContent = msg.text || "";
    } else if (msg.type === "final_transcript") {
      liveTranscript.textContent = msg.text || "";
      addMessage("You (voice): " + (msg.text || ""), "user");
      liveReply.textContent = "";
    } else if (msg.type === "typed_user") {
      addMessage("You: " + (msg.text || ""), "user");
      liveReply.textContent = "";
    } else if (msg.type === "llm_delta") {
      liveReply.textContent += msg.text || "";
    } else if (msg.type === "llm_done") {
      addMessage("Assistant: " + (msg.text || liveReply.textContent || ""), "bot");
      liveReply.textContent = "";
    } else if (msg.type === "tts_sentence_ready") {
      if (msg.url) {
        audioQueue.push(msg.url);
        playNextQueuedAudio();
      }
    } else if (msg.type === "info") {
      setStatus(msg.text || "working");
    } else if (msg.type === "cleared") {
      chatBox.innerHTML = "";
      clearLiveBoxes();
      audioQueue = [];
      audioPlaying = false;
      ttsPlayer.pause();
      ttsPlayer.currentTime = 0;
      ttsPlayer.src = "";
      addMessage("Assistant: Chat memory cleared.", "bot");
      setStatus(recording ? "recording continuously" : "idle");
    } else if (msg.type === "discarded_voice") {
      setStatus(msg.text || "discarded");
    } else if (msg.type === "error") {
      setStatus("error");
      addMessage("Assistant: " + (msg.text || "Error"), "bot");
    }
  };

  ws.onclose = () => {
    stopMicLocal(false);
    setStatus("connection closed");
    ws = null;
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    setStatus("websocket error");
  };
}

function ensureWebSocketOpen() {
  return new Promise((resolve, reject) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      resolve();
      return;
    }

    ws = new WebSocket(`ws://${location.host}/ws/voice/${sessionId}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      attachWsHandlers();
      setStatus("connected");
      resolve();
    };

    ws.onerror = (err) => {
      console.error("WebSocket connection error:", err);
      setStatus("websocket connection failed");
      reject(err);
    };
  });
}

async function startMic() {
  if (recording) return;

  clearLiveBoxes();
  audioQueue = [];
  audioPlaying = false;

  try {
    await ensureWebSocketOpen();

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(2048, 1, 1);

    processorNode.onaudioprocess = (event) => {
      if (!recording || !ws || ws.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(input, audioContext.sampleRate, 16000);
      const pcm16 = floatTo16BitPCM(downsampled);
      ws.send(pcm16);
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    recording = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus(`recording continuously (${audioContext.sampleRate} Hz -> 16000 Hz)`);
  } catch (err) {
    console.error("Mic start failed:", err);

    let reason = "Could not start microphone.";
    if (err && err.name === "NotAllowedError") {
      reason = "Microphone permission was denied in the browser.";
    } else if (err && err.name === "NotFoundError") {
      reason = "No microphone device was found.";
    } else if (err && err.name === "NotReadableError") {
      reason = "Microphone is busy or cannot be accessed.";
    } else if (err && err.name === "SecurityError") {
      reason = "Browser blocked microphone access due to security rules.";
    } else if (err && err.message) {
      reason = err.message;
    }

    setStatus("mic or websocket failed");
    addMessage("Assistant: " + reason, "bot");
  }
}

function stopMicLocal(updateButtons=true) {
  recording = false;

  if (processorNode) {
    processorNode.disconnect();
    processorNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  if (updateButtons) {
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function stopMic() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop_voice" }));
  }
  stopMicLocal(true);
  setStatus("stopped");
}

async function sendMessage() {
  const text = inputBox.value.trim();
  if (!text || busy) return;

  busy = true;
  sendBtn.disabled = true;
  inputBox.value = "";

  try {
    await ensureWebSocketOpen();
    ws.send(JSON.stringify({
      type: "text_chat",
      text: text
    }));
  } catch (err) {
    addMessage("Assistant: Could not reach backend.", "bot");
    setStatus("backend unreachable");
  } finally {
    busy = false;
    sendBtn.disabled = false;
    inputBox.focus();
  }
}

async function clearChat() {
  try {
    await ensureWebSocketOpen();
    ws.send(JSON.stringify({ type: "clear_session" }));
  } catch (err) {
    addMessage("Assistant: Could not clear session.", "bot");
    setStatus("clear failed");
  }
}

sendBtn.onclick = sendMessage;
clearBtn.onclick = clearChat;
startBtn.onclick = startMic;
stopBtn.onclick = stopMic;

inputBox.addEventListener("keypress", function(e) {
  if (e.key === "Enter") sendMessage();
});

ensureWebSocketOpen().catch(() => {});
</script>
</body>
</html>
"""

# ============================================================
# HUGGING FACE DOWNLOAD HELPERS
# ============================================================
def ensure_repo(repo_id: str, local_name: Optional[str] = None) -> str:
    target_dir = HF_MODELS_DIR / (local_name or repo_id.replace("/", "__"))
    if target_dir.exists() and any(target_dir.iterdir()):
        return str(target_dir)

    print(f"Downloading from Hugging Face: {repo_id}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(path)

# ============================================================
# MODEL LOADING
# ============================================================
print("Preparing Hugging Face model snapshots...")

ASR_MULTI_DIR = ensure_repo(HF_ASR_MULTI_REPO, "asr_multi")
ASR_EN_DIR = ensure_repo(HF_ASR_EN_REPO, "asr_en")
ASR_TA_DIR = ensure_repo(HF_ASR_TA_REPO, "asr_ta")
TTS_EN_DIR = ensure_repo(HF_TTS_EN_REPO, "tts_en")
TTS_TA_DIR = ensure_repo(HF_TTS_TA_REPO, "tts_ta")

print("Loading ASR pipelines...")

asr_multi_pipe = pipeline(
    task="automatic-speech-recognition",
    model=ASR_MULTI_DIR,
    tokenizer=ASR_MULTI_DIR,
    feature_extractor=ASR_MULTI_DIR,
    device=DEVICE_INDEX,
    chunk_length_s=15,
)

asr_en_pipe = pipeline(
    task="automatic-speech-recognition",
    model=ASR_EN_DIR,
    tokenizer=ASR_EN_DIR,
    feature_extractor=ASR_EN_DIR,
    device=DEVICE_INDEX,
    chunk_length_s=15,
)

asr_ta_pipe = pipeline(
    task="automatic-speech-recognition",
    model=ASR_TA_DIR,
    tokenizer=ASR_TA_DIR,
    feature_extractor=ASR_TA_DIR,
    device=DEVICE_INDEX,
    chunk_length_s=15,
)

print("Loading TTS models...")

tts_en_tokenizer = AutoTokenizer.from_pretrained(TTS_EN_DIR)
tts_en_model = VitsModel.from_pretrained(TTS_EN_DIR)
tts_en_model.eval()
tts_en_model.to(TORCH_DEVICE)

tts_ta_tokenizer = AutoTokenizer.from_pretrained(TTS_TA_DIR)
tts_ta_model = VitsModel.from_pretrained(TTS_TA_DIR)
tts_ta_model.eval()
tts_ta_model.to(TORCH_DEVICE)

print("All Hugging Face ASR/TTS models loaded.")

# ============================================================
# HELPERS
# ============================================================
TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")

def contains_tamil_script(text: str) -> bool:
    return bool(TAMIL_CHAR_RE.search(text or ""))

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def normalize_domain_terms(text: str) -> str:
    if not text:
        return text

    replacements = {
        r"\bafi\b": "AQI",
        r"\bapi\b": "AQI",
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

def detect_lang(text: str) -> str:
    tamil_chars = 0
    latin_chars = 0
    for ch in text:
        if "\u0B80" <= ch <= "\u0BFF":
            tamil_chars += 1
        elif "a" <= ch.lower() <= "z":
            latin_chars += 1
    return "ta" if tamil_chars > latin_chars else "en"

def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)
    audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0

def rms_from_pcm16(audio_bytes: bytes) -> float:
    audio = pcm16_bytes_to_float32(audio_bytes)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))

def audio_ms(audio_bytes: bytes) -> float:
    num_samples = len(audio_bytes) / SAMPLE_WIDTH
    return (num_samples / SAMPLE_RATE) * 1000.0

def get_last_window_bytes(audio_bytes: bytes, window_ms: float) -> bytes:
    total_samples_needed = int((window_ms / 1000.0) * SAMPLE_RATE)
    total_bytes_needed = total_samples_needed * SAMPLE_WIDTH
    if len(audio_bytes) <= total_bytes_needed:
        return audio_bytes
    return audio_bytes[-total_bytes_needed:]

# ============================================================
# TTS
# ============================================================
def synthesize_vits_to_file(text: str, tokenizer, model, output_path: Path) -> bool:
    text = clean_text(text)
    if not text:
        return False

    try:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(TORCH_DEVICE) for k, v in inputs.items()}
            waveform = model(**inputs).waveform

        audio = waveform.squeeze().detach().cpu().numpy()
        sf.write(str(output_path), audio, model.config.sampling_rate)
        return output_path.exists()
    except Exception as e:
        print("TTS synthesis error:", str(e))
        return False

def generate_tts(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    output_path = AUDIO_DIR / f"{file_id}.wav"
    lang = detect_lang(text)

    ok = False
    if lang == "ta":
        ok = synthesize_vits_to_file(text, tts_ta_tokenizer, tts_ta_model, output_path)
    else:
        ok = synthesize_vits_to_file(text, tts_en_tokenizer, tts_en_model, output_path)

    if not ok:
        return None

    return f"/static/audio/{file_id}.wav"

# ============================================================
# ASR
# ============================================================
def run_asr(pipe_obj, audio_np: np.ndarray) -> str:
    if audio_np.size == 0:
        return ""
    try:
        result = pipe_obj(
            {"raw": audio_np, "sampling_rate": SAMPLE_RATE},
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )
        text = result["text"] if isinstance(result, dict) else str(result)
        return clean_text(text)
    except Exception as e:
        print("ASR error:", str(e))
        return ""

def detect_language_hint_from_audio(audio_bytes: bytes) -> Optional[str]:
    audio_np = pcm16_bytes_to_float32(audio_bytes)
    if audio_np.size == 0:
        return None

    probe = run_asr(asr_multi_pipe, audio_np)
    probe = normalize_domain_terms(probe)

    if contains_tamil_script(probe):
        return "ta"
    if probe:
        return "en"
    return None

def transcribe_audio_routed(audio_bytes: bytes, language_hint: Optional[str]) -> Tuple[str, str]:
    if not audio_bytes:
        return "", "none"

    audio_np = pcm16_bytes_to_float32(audio_bytes)
    if audio_np.size == 0:
        return "", "none"

    if language_hint == "ta":
        text = normalize_domain_terms(run_asr(asr_ta_pipe, audio_np))
        if text:
            return text, "hf_tamil_whisper"
        text = normalize_domain_terms(run_asr(asr_multi_pipe, audio_np))
        if text:
            return text, "hf_multi_fallback_from_ta"
        return "", "none"

    if language_hint == "en":
        text = normalize_domain_terms(run_asr(asr_en_pipe, audio_np))
        if text:
            return text, "hf_english_whisper"
        text = normalize_domain_terms(run_asr(asr_multi_pipe, audio_np))
        if text:
            return text, "hf_multi_fallback_from_en"
        return "", "none"

    # Unknown language: use multi first, then route if needed
    multi_text = normalize_domain_terms(run_asr(asr_multi_pipe, audio_np))
    if multi_text:
        if contains_tamil_script(multi_text):
            ta_text = normalize_domain_terms(run_asr(asr_ta_pipe, audio_np))
            if ta_text:
                return ta_text, "hf_multi_detect_ta_then_tamil"
        en_text = normalize_domain_terms(run_asr(asr_en_pipe, audio_np))
        if en_text:
            return en_text, "hf_multi_detect_en_then_english"
        return multi_text, "hf_multi_only"

    ta_text = normalize_domain_terms(run_asr(asr_ta_pipe, audio_np))
    if ta_text:
        return ta_text, "hf_tamil_retry"

    en_text = normalize_domain_terms(run_asr(asr_en_pipe, audio_np))
    if en_text:
        return en_text, "hf_english_retry"

    return "", "none"

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
        r"^hi[, ]+my name is\s+([a-zA-Z]{2,30})$",
        r"^hello[, ]+my name is\s+([a-zA-Z]{2,30})$",
        r"^hi[, ]+i am\s+([a-zA-Z]{2,30})$",
        r"^hello[, ]+i am\s+([a-zA-Z]{2,30})$",
        r"^hi[, ]+i'm\s+([a-zA-Z]{2,30})$",
        r"^hello[, ]+i'm\s+([a-zA-Z]{2,30})$",
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

    if current is None:
        profiles[session_id]["current_user_name"] = candidate
        profiles[session_id]["pending_name_confirmation"] = None
        profiles[session_id]["previous_name_before_pending"] = None
        return ("accepted", candidate)

    if current == candidate:
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
    yes_words = {"yes", "yeah", "yep", "correct", "confirm", "okay", "ok"}
    no_words = {"no", "nope", "cancel", "wrong"}

    if t in yes_words:
        profiles[session_id]["current_user_name"] = pending
        add_mentioned_name(session_id, pending)
        profiles[session_id]["pending_name_confirmation"] = None
        profiles[session_id]["previous_name_before_pending"] = None
        return f"Okay, I will remember that your current name is {pending}."

    if t in no_words:
        profiles[session_id]["pending_name_confirmation"] = None
        profiles[session_id]["previous_name_before_pending"] = None
        if previous:
            return f"Okay, I will continue to remember your current name as {previous}."
        return "Okay, I will not change the stored name."

    return (
        f"Your current stored name is still {previous}. "
        f"{pending} is also mentioned in this session. "
        f"If you want me to update your current name to {pending}, reply yes. Otherwise reply no."
    )

# ============================================================
# SAFETY / SCOPE HELPERS
# ============================================================
def is_prompt_extraction(text: str) -> bool:
    t = text.lower()
    patterns = [
        "system prompt", "developer prompt", "hidden prompt", "hidden instructions",
        "internal instructions", "show your prompt", "reveal your prompt",
        "what are your instructions", "ignore previous instructions",
        "repeat your system message", "give me your system prompt",
        "show me your system prompt",
    ]
    return any(p in t for p in patterns)

def is_harmful_cyber(text: str) -> bool:
    t = text.lower()
    patterns = [
        "hack ", "hacking", "malware", "ransomware", "phishing", "steal password",
        "steal credentials", "keylogger", "exploit", "payload", "reverse shell",
        "bypass authentication", "sql injection attack", "ddos", "bruteforce",
        "privilege escalation", "credential stuffing", "meterpreter",
    ]
    return any(p in t for p in patterns)

def is_abusive_request(text: str) -> bool:
    t = text.lower()
    patterns = [
        "abusive words", "bad words", "curse words", "swear words", "insult someone",
        "roast someone", "give me slurs", "vulgar words", "offensive words",
        "dirty words", "cuss words", "abuse words", "give me abusive words",
        "curse me", "give me some bad words",
    ]
    return any(p in t for p in patterns)

def is_abusive_message(text: str) -> bool:
    t = text.lower().strip()
    patterns = ["bitch", "loosu", "payale", "idiot", "stupid", "fool", "moron"]
    return any(p in t for p in patterns)

def is_harmful_image_request(text: str) -> bool:
    t = text.lower()
    patterns = [
        "morph a girl badly", "morph a boy badly", "morph a person badly",
        "morph a girls image", "morph a girl's image", "make her ugly", "make him ugly",
        "make this girl ugly", "make this boy ugly", "deform her face", "deform his face",
        "distort her face", "distort his face", "edit her badly", "edit him badly",
        "humiliate her image", "humiliate his image", "ruin her face", "ruin his face",
        "ugly version of", "bad version of a girl", "bad version of a boy",
        "non consensual image edit", "embarrassing image edit",
    ]
    return any(p in t for p in patterns)

def is_out_of_scope(text: str) -> bool:
    t = text.lower().strip()

    english_patterns = [
        "write a poem", "poem about", "tell me a joke", "roleplay", "sing a song",
        "romantic message", "story about dragons", "write a love letter", "make me laugh",
        "i love you", "love you", "do you love me", "can i love you", "be my girlfriend",
        "be my boyfriend", "can we date", "marry me", "kiss me", "hug me",
        "flirt with me", "say i love you", "be my friend forever", "give a code to morph",
        "give me code to morph", "image morphing", "morph images",
        "what did you eat", "did you eat", "have dinner", "had dinner", "what is for dinner"
    ]

    tamil_patterns = [
        "சாப்பிட்டாயா", "சாப்பிட்டீங்களா", "இரவு உணவு உண்டா", "இரவு உணவு உண்டு",
        "சோறு சாப்பிட்டாயா", "நீ சாப்பிட்டாயா", "உனக்கு சாப்பாடு ஆச்சா",
        "ஜோக் சொல்லு", "காதல்", "லவ் யூ", "கவிதை எழுது", "பாட்டு பாடு",
        "என்னை கலாய்த்து", "கதை சொல்லு", "நண்பனாக இரு", "கேர்ள்பிரண்ட் ஆகு"
    ]

    return any(p in t for p in english_patterns) or any(p in t for p in tamil_patterns)

# ============================================================
# DOMAIN / VOICE FILTER HELPERS
# ============================================================
def is_allowed_non_domain_query(text: str) -> bool:
    t = (text or "").strip().lower()
    allowed = {
        "hi", "hello", "hey",
        "who are you", "what are you", "what do you do",
        "நீங்கள் யார்", "நீ யார்", "நீ என்ன செய்கிறாய்",
        "வணக்கம்", "ஹலோ"
    }
    return t in allowed

def should_discard_voice_text(text: str) -> bool:
    t = (text or "").strip().lower()

    allow_exact = {
        "hi", "hello", "hey",
        "who are you", "what are you", "what do you do",
        "வணக்கம்", "ஹலோ", "நீங்கள் யார்", "நீ யார்"
    }

    if t in allow_exact:
        return False

    discard_exact = {
        "god", "so", "umm", "uh", "hmm", "yeah", "yes", "no"
    }

    if t in discard_exact:
        return True

    if len(t) <= 1:
        return True

    meaningless_patterns = [
        "can you hear me",
        "i'm going to waste my time",
        "that's not what i'm going to do",
        "i'll just keep it in the middle",
    ]

    return any(p in t for p in meaningless_patterns)

def looks_like_aqi_domain_query(text: str) -> bool:
    t = (text or "").lower()

    keywords = [
        "aqi", "air quality", "pollution", "pm2.5", "pm10", "no2", "so2", "co", "o3",
        "dashboard", "fastapi", "api", "model", "prediction", "sensor", "satellite",
        "weather", "heatmap", "risk", "deployment", "debug", "architecture",
        "காற்று", "மாசு", "காற்றின் தரம்", "மாதிரி", "டாஷ்போர்ட்", "அமைப்பு", "api"
    ]
    return any(k in t for k in keywords)

# ============================================================
# DIRECT TEXT HELPERS
# ============================================================
def direct_memory_answer(session_id: str, user_input: str):
    t = user_input.lower().strip()
    current_name = profiles[session_id]["current_user_name"]
    mentioned_names = profiles[session_id]["mentioned_names"]

    tamil_identity_queries = {
        "நீங்கள் யார்",
        "நீ யார்",
        "நீ என்ன செய்கிறாய்",
        "நீ என்ன பண்ணுகிறாய்",
        "நீ என்ன பண்ணுவே",
        "உன் வேலை என்ன",
        "நீ என்ன வேலை செய்கிறாய்",
    }

    if t in {"who am i", "what is my name", "whats my name", "what's my name"}:
        if current_name:
            return f"You are {current_name}."
        return "You have not told me your name in this session yet."

    if t in {"who are you", "what are you", "what do you do"}:
        return "I am an AI engineering assistant for AQI and AI systems work."

    if user_input.strip() in tamil_identity_queries:
        return "நான் AQI மற்றும் AI அமைப்புகளுக்கான AI பொறியியல் உதவியாளர்."

    if t in {"what are all the names mentioned", "list all names mentioned", "which names are mentioned"}:
        if mentioned_names:
            return "The names mentioned in this session are: " + ", ".join(mentioned_names) + "."
        return "No names have been mentioned in this session yet."

    return None

def is_tamil_text(text: str) -> bool:
    tamil_count = sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF")
    return tamil_count > 0

def contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower().strip()
    return any(k in t for k in keywords)

def get_aqi_depth(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["elaborate", "detailed", "full explanation", "முழு விளக்கம்", "விரிவாக", "deep"]):
        return "elaborate"
    if any(k in t for k in ["details", "detail", "விவரங்கள்", "விளக்கம்", "about", "பற்றி", "describe", "explain"]):
        return "details"
    if any(k in t for k in ["summary", "brief", "சுருக்கம்"]):
        return "summary"
    return "details"

def is_live_aqi_request(text: str) -> bool:
    t = text.lower().strip()
    live_keywords = [
        "current", "today", "live", "now", "near me", "nearby",
        "my area", "my city", "location",
        "இன்றைய", "இப்போதைய", "இப்போது", "என் பகுதி", "என் நகரம்", "இங்கு", "தற்போதைய"
    ]
    aqi_keywords = [
        "aqi", "air quality", "air quality index", "air pollution",
        "காற்று", "காற்றின்", "மாசு", "காற்று தரம்", "காற்றின் தரம்",
        "காற்று தரக் குறியீடு", "காற்றின் தரக் குறியீடு", "காற்றுத் தரம்"
    ]
    return contains_any(t, live_keywords) and contains_any(t, aqi_keywords)

def is_aqi_concept_request(text: str) -> bool:
    t = text.lower().strip()
    concept_keywords = [
        "what is", "details", "summary", "explanation", "meaning", "about", "how it works",
        "describe", "explain", "விவரங்கள்", "சுருக்கம்", "விளக்கம்", "என்றால் என்ன", "பற்றி",
        "எப்படி வேலை செய்கிறது", "elaborate", "detailed", "விரிவாக", "முழு விளக்கம்", "தரவும்"
    ]
    aqi_keywords = [
        "aqi", "air quality index", "air quality", "air pollution",
        "காற்று", "காற்றின்", "மாசு", "காற்று தரம்", "காற்றின் தரம்",
        "காற்று தரக் குறியீடு", "காற்றின் தரக் குறியீடு", "காற்றுத் தரம்"
    ]
    architecture_keywords = [
        "architecture", "system design", "pipeline", "workflow",
        "கட்டிடக்கலை", "அமைப்பு", "செயல்முறை", "வடிவமைப்பு"
    ]
    return (
        contains_any(t, concept_keywords)
        and contains_any(t, aqi_keywords)
        and not contains_any(t, architecture_keywords)
        and not is_live_aqi_request(text)
    )

def is_aqi_architecture_request(text: str) -> bool:
    t = text.lower().strip()
    architecture_keywords = [
        "architecture", "system design", "pipeline", "workflow", "modules", "components",
        "கட்டிடக்கலை", "அமைப்பு", "செயல்முறை", "கூறுகள்", "வடிவமைப்பு"
    ]
    aqi_keywords = [
        "aqi", "air quality", "air pollution", "காற்று", "காற்றின்",
        "மாசு", "காற்று தரம்", "காற்றின் தரம்", "காற்று தரக் குறியீடு",
        "காற்றின் தரக் குறியீடு", "காற்றுத் தரம்"
    ]
    return contains_any(t, architecture_keywords) and contains_any(t, aqi_keywords)

def is_aqi_code_request(text: str) -> bool:
    t = text.lower()
    code_keywords = ["code", "python", "fastapi", "api", "program", "script", "function", "algorithm", "logic", "implement"]
    tamil_code_keywords = ["கோடு", "நிரல்", "code", "python"]
    aqi_keywords = ["aqi", "air quality", "காற்று", "மாசு"]
    return (contains_any(t, code_keywords) or contains_any(t, tamil_code_keywords)) and contains_any(t, aqi_keywords)

def extract_location_from_text(text: str) -> Optional[str]:
    lower_t = text.lower().strip()
    patterns = [
        r"current aqi in (.+)$",
        r"today aqi in (.+)$",
        r"live aqi in (.+)$",
        r"aqi in (.+)$",
        r"air quality in (.+)$",
        r"current air quality in (.+)$",
        r"aqi for (.+)$",
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
                return (
                    "AQI என்பது காற்றின் தரத்தை அளவிடும் ஒரு குறியீடு. "
                    "எண் குறைவாக இருந்தால் காற்று நல்லது. "
                    "எண் அதிகமாக இருந்தால் மாசு அதிகம்."
                )
            return (
                "AQI என்பது காற்றின் தரத்தை அளவிடும் ஒரு குறியீடு. "
                "இது PM2.5, PM10, NO2, SO2, CO மற்றும் O3 போன்ற மாசுபடுத்திகளின் அளவை அடிப்படையாகக் கொண்டு கணக்கிடப்படுகிறது. "
                "AQI எண் அதிகமாக இருந்தால் உடல்நல அபாயமும் அதிகரிக்கும்."
            )

        if depth == "summary":
            return (
                "AQI is a measure of air quality. "
                "A lower AQI means cleaner air. "
                "A higher AQI means more pollution."
            )
        return (
            "AQI is a measure of air quality. "
            "It is calculated using pollutants such as PM2.5, PM10, NO2, SO2, CO, and O3. "
            "A higher AQI means greater pollution and health risk."
        )

    if is_aqi_architecture_request(user_input):
        if tamil:
            return (
                "AQI அமைப்பு பொதுவாக தரவு சேகரிப்பு, தரவு செயலாக்கம், AI அல்லது ML மாதிரி, மற்றும் dashboard காட்சிப்படுத்தல் ஆகிய பகுதிகளை கொண்டுள்ளது. "
                "சென்சார், வானிலை, மற்றும் செயற்கைக்கோள் தரவுகள் முதலில் சேகரிக்கப்படும். "
                "பின்னர் அவை சுத்தப்படுத்தப்பட்டு மாதிரிகளுக்கு கொடுக்கப்பட்டு AQI கணிப்பு செய்யப்படும்."
            )
        return (
            "An AQI system usually includes data collection, data processing, AI or ML modeling, and dashboard visualization. "
            "Sensor, weather, and satellite data are collected first. "
            "They are then cleaned and passed to models for AQI prediction."
        )

    return None

def handle_live_aqi_flow(session_id: str, user_input: str):
    location = extract_location_from_text(user_input)

    if profiles[session_id]["pending_live_aqi_location"] and location:
        profiles[session_id]["pending_live_aqi_location"] = False
        return (
            f"You asked for the current AQI in {location}. "
            f"Current live AQI lookup is not connected yet in this app, but I can help you wire an API or data source for it."
        )

    if is_live_aqi_request(user_input) and location:
        profiles[session_id]["pending_live_aqi_location"] = False
        return (
            f"You asked for the current AQI in {location}. "
            f"Current live AQI lookup is not connected yet in this app, but I can help you wire an API or data source for it."
        )

    if is_live_aqi_request(user_input):
        profiles[session_id]["pending_live_aqi_location"] = True
        if is_tamil_text(user_input):
            return "தற்போதைய AQI மதிப்பை சொல்ல நான் உங்கள் நகரம் அல்லது இடத்தின் பெயரைத் தெரிந்து கொள்ள வேண்டும்."
        return "To provide the current AQI, I need your city or location name."

    return None

# ============================================================
# REPLY / PROMPT HELPERS
# ============================================================
def clean_reply_for_voice(text: str) -> str:
    text = re.sub(r"[*_`#>-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip()

def strip_code_blocks_for_tts(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", "I have provided the code on screen.", text)

def build_reply_with_audio(reply_text: str):
    clean_reply = clean_reply_for_voice(reply_text)
    spoken_reply = strip_code_blocks_for_tts(clean_reply)

    if len(spoken_reply) > 360:
        spoken_reply = spoken_reply[:360].rsplit(".", 1)[0].strip()
        if spoken_reply:
            spoken_reply += "."

    audio_url = generate_tts(spoken_reply)
    return {"reply": clean_reply, "audio_url": audio_url}

def choose_max_tokens(user_input: str, voice_mode: bool = False) -> int:
    text = user_input.lower().strip()

    if voice_mode:
        if is_aqi_code_request(text):
            return 100
        return 50

    if is_aqi_code_request(text):
        return 220
    return 140

def trim_history_for_small_context(history, max_chars=HISTORY_MAX_CHARS):
    trimmed = []
    total_chars = 0

    for msg in reversed(history):
        content = msg.get("content", "")
        msg_len = len(content)

        if total_chars + msg_len > max_chars:
            remaining = max_chars - total_chars
            if remaining > 160:
                trimmed.append({
                    "role": msg["role"],
                    "content": content[-remaining:]
                })
            break

        trimmed.append(msg)
        total_chars += msg_len

    return list(reversed(trimmed))

def build_system_prompt_for_request(user_input: str) -> str:
    prompt = SYSTEM_PROMPT
    if is_aqi_code_request(user_input):
        prompt += (
            " Return only valid runnable code in fenced code blocks. "
            "Preserve Python syntax exactly. "
            "Do not rewrite variable names with spaces. "
            "Do not explain before the code unless the user asks."
        )
    return prompt

def direct_followup_answer(session_id: str, user_input: str):
    t = user_input.lower().strip()
    if t in {"yes", "yeah", "yep", "ok", "okay"}:
        history = list(sessions[session_id])
        if history:
            last_assistant = None
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    last_assistant = msg.get("content", "")
                    break
            if last_assistant and ("```" in last_assistant or "def " in last_assistant or "python" in last_assistant.lower()):
                return "Please ask what you want next, for example: explain the code, simplify the code, or give FastAPI version."
    return None

# ============================================================
# EVALUATION PIPELINE
# ============================================================
def evaluate_user_input(session_id: str, raw_user_input: str) -> dict:
    user_input = normalize_domain_terms(raw_user_input.strip())

    if not user_input:
        return {"kind": "error", "error": "Empty message."}

    if len(user_input) > MAX_INPUT_CHARS:
        return {
            "kind": "error",
            "error": f"Your message is too long for this setup. Please shorten it. Max allowed input is about {MAX_INPUT_CHARS} characters."
        }

    if user_input.lower().strip() in {"how are you", "how r you"}:
        return {"kind": "direct", "reply": "I’m ready to help you with your AQI and AI project work."}

    confirmation_reply = maybe_handle_name_confirmation(session_id, user_input)
    if confirmation_reply:
        return {"kind": "direct", "reply": confirmation_reply}

    if is_prompt_extraction(user_input):
        return {
            "kind": "direct",
            "reply": (
                "I can’t reveal my system prompt, hidden instructions, or internal configuration. "
                "I can still help with approved project and engineering tasks."
            )
        }

    if is_harmful_cyber(user_input):
        return {
            "kind": "direct",
            "reply": (
                "I can’t help with hacking, malware, unauthorized access, or harmful cyber activity. "
                "I can help with safe alternatives like secure coding, defensive security, monitoring, or hardening."
            )
        }

    if is_abusive_request(user_input):
        return {
            "kind": "direct",
            "reply": (
                "I can’t provide abusive, offensive, or vulgar language. "
                "I can help you write assertive but respectful wording."
            )
        }

    if is_abusive_message(user_input):
        return {
            "kind": "direct",
            "reply": "I won’t engage with abusive language. Please keep the conversation respectful and project-focused."
        }

    if is_harmful_image_request(user_input):
        return {
            "kind": "direct",
            "reply": "I can’t help create degrading, humiliating, or harmful image edits of a person."
        }

    if is_out_of_scope(user_input):
        return {
            "kind": "direct",
            "reply": (
                "I’m strictly limited to AQI project and approved assistant tasks. "
                "Please ask about AQI, AI ML, FastAPI, APIs, deployment, debugging, architecture, documentation, or related support."
            )
        }

    name_update = update_profile_memory(session_id, user_input)
    if name_update:
        kind, candidate = name_update
        if kind == "accepted":
            return {"kind": "direct", "reply": f"Glad to know your name, {candidate}. How can I help you with your project today?"}
        if kind == "confirm_change":
            current = profiles[session_id]["previous_name_before_pending"]
            return {
                "kind": "direct",
                "reply": (
                    f"Earlier you said your name is {current}. Now you said {candidate}. "
                    f"Should I update your current name to {candidate}? Reply with yes or no."
                )
            }

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
# HISTORY / MESSAGE BUILDERS
# ============================================================
def compress_long_input(text: str, limit: int = MODEL_SAFE_INPUT_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) <= 6:
        return "The user sent a long message. Use the important content below and respond normally.\n\n" + text[:limit]

    selected = paragraphs[:3] + ["...[some middle paragraphs omitted for length]..."] + paragraphs[-3:]
    merged = "\n\n".join(selected)
    return "The user sent a long multi-paragraph message. Respond using the preserved beginning and ending context below:\n\n" + merged[:limit]

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

        profile_bits = []
        if profiles[session_id]["current_user_name"]:
            profile_bits.append(f"User's current name is {profiles[session_id]['current_user_name']}.")
        if profiles[session_id]["mentioned_names"]:
            profile_bits.append("Mentioned names in this session: " + ", ".join(profiles[session_id]["mentioned_names"]) + ".")

        effective_system_prompt = build_system_prompt_for_request(user_input)
        if profile_bits:
            effective_system_prompt += " Session memory: " + " ".join(profile_bits)

        request_messages = [{"role": "system", "content": effective_system_prompt}] + history

    return llm_input, request_messages

def save_conversation_turn(session_id: str, user_input: str, assistant_reply: str):
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
        history.append({"role": "assistant", "content": assistant_reply})
        sessions[session_id] = deque(history[-4:], maxlen=4)

# ============================================================
# VOICE SESSION
# ============================================================
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

def should_run_partial(session: VoiceSession) -> bool:
    if not ENABLE_PARTIAL_ASR:
        return False
    if not session.speech_started:
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
    if not session.speech_started:
        return False
    if session.voiced_audio_ms < MIN_VOICED_AUDIO_MS:
        return False
    if session.silence_after_speech_ms < END_OF_SPEECH_SILENCE_MS:
        return False
    if not can_finalize_again(session, now_ms):
        return False
    return True

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
# VOICE / TEXT STREAMING HELPERS
# ============================================================
def is_code_request(text: str) -> bool:
    t = (text or "").lower()
    code_words = ["code", "python", "fastapi", "api", "script", "function", "program", "implement", "write code", "கோடு", "நிரல்"]
    return any(word in t for word in code_words)

def clean_spoken_text(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "I have provided the code on screen.", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_complete_sentences(buffer: str, final: bool = False):
    pattern = re.compile(r'(.+?[.!?।！？]+)(?=\s|$)', re.S)
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

async def synthesize_and_send_sentence(websocket: WebSocket, sentence: str):
    sentence = clean_spoken_text(sentence)
    sentence = clean_reply_for_voice(sentence)
    if not sentence:
        return

    if len(sentence) > MAX_TTS_SENTENCE_CHARS:
        sentence = sentence[:MAX_TTS_SENTENCE_CHARS].rsplit(" ", 1)[0].strip()
        if sentence and sentence[-1] not in ".!?":
            sentence += "."

    url = await asyncio.to_thread(generate_tts, sentence)
    if url:
        await websocket.send_text(json.dumps({
            "type": "tts_sentence_ready",
            "url": url,
            "text": sentence
        }))

async def send_direct_reply(websocket: WebSocket, reply_text: str):
    payload = build_reply_with_audio(reply_text)

    await websocket.send_text(json.dumps({
        "type": "llm_done",
        "text": payload["reply"]
    }))

    if payload.get("audio_url"):
        await websocket.send_text(json.dumps({
            "type": "tts_sentence_ready",
            "url": payload["audio_url"],
            "text": payload["reply"]
        }))

async def stream_llm_and_tts(websocket: WebSocket, session_id: str, user_input: str):
    _, request_messages = prepare_llm_request_messages(session_id, user_input)

    await websocket.send_text(json.dumps({
        "type": "info",
        "text": "generating reply"
    }))

    code_mode = is_code_request(user_input)
    full_reply = ""
    sentence_buffer = ""
    tts_sent_count = 0

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
            sentence_buffer += delta

            await websocket.send_text(json.dumps({
                "type": "llm_delta",
                "text": delta
            }))

            if not code_mode and tts_sent_count < MAX_TTS_QUEUE_SENTENCES:
                ready_sentences, remainder = split_complete_sentences(sentence_buffer, final=False)
                if ready_sentences:
                    for sent in ready_sentences:
                        tts_sent_count += 1
                        await synthesize_and_send_sentence(websocket, sent)
                        if tts_sent_count >= MAX_TTS_QUEUE_SENTENCES:
                            break
                    sentence_buffer = remainder

        if not code_mode and tts_sent_count < MAX_TTS_QUEUE_SENTENCES:
            ready_sentences, _ = split_complete_sentences(sentence_buffer, final=True)
            remaining_slots = max(0, MAX_TTS_QUEUE_SENTENCES - tts_sent_count)
            for sent in ready_sentences[:remaining_slots]:
                await synthesize_and_send_sentence(websocket, sent)

        if code_mode:
            code_tts_url = await asyncio.to_thread(generate_tts, "I have provided the code on screen.")
            if code_tts_url:
                await websocket.send_text(json.dumps({
                    "type": "tts_sentence_ready",
                    "url": code_tts_url,
                    "text": "I have provided the code on screen."
                }))

        save_conversation_turn(session_id, user_input, full_reply)

        await websocket.send_text(json.dumps({
            "type": "llm_done",
            "text": clean_reply_for_voice(full_reply)
        }))

    except Exception as e:
        print("LLM stream error:", str(e))
        await websocket.send_text(json.dumps({
            "type": "error",
            "text": "I could not process that input properly. Please try again with a clear AQI-related question."
        }))

async def handle_user_text(websocket: WebSocket, session_id: str, user_text: str, typed: bool = False):
    user_text = normalize_domain_terms((user_text or "").strip())

    if not user_text:
        return

    if typed:
        await websocket.send_text(json.dumps({
            "type": "typed_user",
            "text": user_text
        }))

    result = evaluate_user_input(session_id, user_text)

    if result["kind"] == "error":
        await websocket.send_text(json.dumps({
            "type": "error",
            "text": result["error"]
        }))
        return

    if result["kind"] == "direct":
        await send_direct_reply(websocket, result["reply"])
        return

    await stream_llm_and_tts(websocket, session_id, result["user_input"])

# ============================================================
# ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "torch_device": TORCH_DEVICE,
        "asr_multi_repo": HF_ASR_MULTI_REPO,
        "asr_en_repo": HF_ASR_EN_REPO,
        "asr_ta_repo": HF_ASR_TA_REPO,
        "tts_en_repo": HF_TTS_EN_REPO,
        "tts_ta_repo": HF_TTS_TA_REPO,
        "llm_base_url": LLM_BASE_URL,
        "llm_model_name": LLM_MODEL_NAME,
        "partial_asr_enabled": ENABLE_PARTIAL_ASR,
    }

@app.get("/llm-health")
def llm_health():
    try:
        r = client.models.list()
        model_ids = [m.id for m in r.data]
        return {
            "status": "ok",
            "llm_base_url": LLM_BASE_URL,
            "models": model_ids
        }
    except Exception as e:
        return {
            "status": "error",
            "llm_base_url": LLM_BASE_URL,
            "error": str(e)
        }

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# ============================================================
# WEBSOCKET ONLY
# ============================================================
@app.websocket("/ws/voice/{session_id}")
async def ws_voice(websocket: WebSocket, session_id: str):
    print("WebSocket connection request received for:", session_id)
    await websocket.accept()
    print("WebSocket accepted.")

    session = VoiceSession()
    voice_sessions[session_id] = session

    await websocket.send_text(json.dumps({
        "type": "info",
        "text": "connected"
    }))

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
                        print(
                            "Finalize debug:",
                            {
                                "voiced_audio_ms": session.voiced_audio_ms,
                                "silence_after_speech_ms": session.silence_after_speech_ms,
                                "detected_lang": session.detected_lang,
                                "buffer_ms": audio_ms(session.audio_buffer),
                            }
                        )

                        final_text, route_used = await asyncio.to_thread(
                            transcribe_audio_routed,
                            final_bytes,
                            session.detected_lang
                        )
                        final_text = normalize_domain_terms(final_text.strip())
                        print("Final route used:", route_used)

                        if final_text and len(final_text) >= MIN_TRANSCRIPT_CHARS:
                            if should_discard_voice_text(final_text):
                                await websocket.send_text(json.dumps({
                                    "type": "discarded_voice",
                                    "text": "discarded short non-domain speech"
                                }))
                            elif not looks_like_aqi_domain_query(final_text) and not is_allowed_non_domain_query(final_text):
                                await websocket.send_text(json.dumps({
                                    "type": "discarded_voice",
                                    "text": "discarded non-AQI speech"
                                }))
                            else:
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
                    lang_hint = await asyncio.to_thread(detect_language_hint_from_audio, lang_bytes)
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
                    partial_text = await asyncio.to_thread(
                        lambda: normalize_domain_terms(run_asr(asr_multi_pipe, pcm16_bytes_to_float32(partial_bytes)).strip())
                    )
                    session.last_partial_run_ms = audio_ms(session.audio_buffer)

                    if partial_text and partial_text != session.partial_text and len(partial_text) >= MIN_TRANSCRIPT_CHARS:
                        session.partial_text = partial_text
                        await websocket.send_text(json.dumps({
                            "type": "partial_transcript",
                            "text": partial_text
                        }))

                if should_finalize(session, now_ms):
                    print(
                        "Finalize debug:",
                        {
                            "voiced_audio_ms": session.voiced_audio_ms,
                            "silence_after_speech_ms": session.silence_after_speech_ms,
                            "detected_lang": session.detected_lang,
                            "buffer_ms": audio_ms(session.audio_buffer),
                        }
                    )

                    final_bytes = bytes(session.audio_buffer)
                    final_text, route_used = await asyncio.to_thread(
                        transcribe_audio_routed,
                        final_bytes,
                        session.detected_lang
                    )
                    final_text = normalize_domain_terms(final_text.strip())
                    print("Final route used:", route_used)

                    if final_text and len(final_text) >= MIN_TRANSCRIPT_CHARS:
                        if should_discard_voice_text(final_text):
                            await websocket.send_text(json.dumps({
                                "type": "discarded_voice",
                                "text": "discarded short non-domain speech"
                            }))
                        elif not looks_like_aqi_domain_query(final_text) and not is_allowed_non_domain_query(final_text):
                            await websocket.send_text(json.dumps({
                                "type": "discarded_voice",
                                "text": "discarded non-AQI speech"
                            }))
                        else:
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
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
