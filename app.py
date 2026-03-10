from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from openai import AsyncOpenAI
from faster_whisper import WhisperModel
import asyncio
import json
import numpy as np
import os
import re
import time
from typing import Optional

from tts.tts_router import generate_tts

app = FastAPI()

# --------------------------------------------------
# vLLM / Gemma
# --------------------------------------------------
llm_client = AsyncOpenAI(
    base_url="http://127.0.0.1:9004/v1",
    api_key="dummy",
)

SYSTEM_PROMPT = (
    "You are a professional AI engineering assistant for an AQI monitoring and AI systems project. "
    "Reply naturally, clearly, and concisely. "
    "If the user speaks mostly in Tamil, reply mainly in Tamil. "
    "If the user speaks in English, reply in English. "
    "Keep normal spoken replies short, usually 2 to 4 sentences. "
    "Do not repeat your identity unless explicitly asked. "
    "If asked who you are, say exactly: I am an AI engineering assistant for AQI and AI systems work. "
    "If the user asks for code, return valid code in fenced code blocks."
)

# --------------------------------------------------
# ASR config
# For lowest friction, start on CPU.
# If you later have spare GPU VRAM, switch to:
# ASR_DEVICE="cuda", ASR_COMPUTE_TYPE="float16"
# --------------------------------------------------
ASR_MODEL_SIZE = "base"
ASR_DEVICE = "cpu"
ASR_COMPUTE_TYPE = "int8"

asr_model = WhisperModel(
    ASR_MODEL_SIZE,
    device=ASR_DEVICE,
    compute_type=ASR_COMPUTE_TYPE,
)

# --------------------------------------------------
# Audio / streaming config
# Browser sends mono PCM16 at 16kHz
# --------------------------------------------------
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
SILENCE_RMS_THRESHOLD = 0.010
END_OF_SPEECH_SILENCE_MS = 700
PARTIAL_TRANSCRIBE_EVERY_MS = 900
MIN_FINAL_AUDIO_MS = 600

# Sentence streaming
MAX_TTS_SENTENCE_CHARS = 240
MAX_TTS_QUEUE_SENTENCES = 6

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AQI Voice Assistant</title>
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
            min-height: 360px;
            max-height: 480px;
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
        .partial { background: #fff7cc; }

        #controls {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
        }
        button {
            padding: 10px 16px;
            font-size: 15px;
            cursor: pointer;
        }
        #status {
            margin-bottom: 10px;
            color: #444;
            font-size: 14px;
        }
        #transcriptBox, #replyBox {
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
        #textRow {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
        }
        #textInput {
            flex: 1;
            padding: 10px;
            font-size: 15px;
        }
    </style>
</head>
<body>
    <h2>AQI Voice Assistant</h2>

    <div id="status">Status: idle</div>

    <div id="controls">
        <button id="startBtn">Start Mic</button>
        <button id="stopBtn" disabled>Stop Mic</button>
        <button id="clearBtn">Clear</button>
    </div>

    <div id="textRow">
        <input id="textInput" type="text" placeholder="Type a text query too..." />
        <button id="sendTextBtn">Send Text</button>
    </div>

    <div id="transcriptBox">
        <div class="label">Live transcript</div>
        <div id="liveTranscript"></div>
    </div>

    <div id="replyBox">
        <div class="label">Streaming reply</div>
        <div id="liveReply"></div>
    </div>

    <div id="chat"></div>

<script>
let ws = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let recording = false;

let audioQueue = [];
let audioPlaying = false;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const sendTextBtn = document.getElementById("sendTextBtn");
const textInput = document.getElementById("textInput");

const statusBox = document.getElementById("status");
const liveTranscript = document.getElementById("liveTranscript");
const liveReply = document.getElementById("liveReply");
const chatBox = document.getElementById("chat");

function addMessage(text, cls) {
    const div = document.createElement("div");
    div.className = "msg " + cls;
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function setStatus(text) {
    statusBox.textContent = "Status: " + text;
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

async function playNextAudioInQueue() {
    if (audioPlaying || audioQueue.length === 0) return;
    audioPlaying = true;

    const nextUrl = audioQueue.shift();
    const audio = new Audio(nextUrl + "?t=" + Date.now());

    audio.onended = () => {
        audioPlaying = false;
        playNextAudioInQueue();
    };

    audio.onerror = () => {
        audioPlaying = false;
        playNextAudioInQueue();
    };

    try {
        await audio.play();
    } catch (err) {
        console.log("Audio play blocked:", err);
        audioPlaying = false;
    }
}

async function startMic() {
    if (recording) return;

    liveTranscript.textContent = "";
    liveReply.textContent = "";
    audioQueue = [];
    audioPlaying = false;

    ws = new WebSocket(`ws://${location.host}/ws/voice`);
    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);

            processorNode.onaudioprocess = (event) => {
                if (!recording || !ws || ws.readyState !== WebSocket.OPEN) return;
                const input = event.inputBuffer.getChannelData(0);
                const pcm16 = floatTo16BitPCM(input);
                ws.send(pcm16);
            };

            sourceNode.connect(processorNode);
            processorNode.connect(audioContext.destination);

            recording = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            setStatus("recording");
        } catch (err) {
            setStatus("mic error");
            console.error(err);
        }
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "partial_transcript") {
            liveTranscript.textContent = msg.text || "";
        } else if (msg.type === "final_transcript") {
            liveTranscript.textContent = msg.text || "";
            addMessage("You (voice): " + (msg.text || ""), "user");
            liveReply.textContent = "";
        } else if (msg.type === "llm_delta") {
            liveReply.textContent += msg.text || "";
        } else if (msg.type === "llm_done") {
            addMessage("Assistant: " + (msg.text || liveReply.textContent || ""), "bot");
        } else if (msg.type === "tts_sentence_ready") {
            if (msg.url) {
                audioQueue.push(msg.url);
                playNextAudioInQueue();
            }
        } else if (msg.type === "info") {
            setStatus(msg.text || "working");
        } else if (msg.type === "error") {
            setStatus("error");
            addMessage("Assistant: " + (msg.text || "Error"), "bot");
        }
    };

    ws.onclose = () => {
        stopMicLocal();
        setStatus("closed");
    };

    ws.onerror = () => {
        setStatus("socket error");
    };
}

function stopMicLocal() {
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

    startBtn.disabled = false;
    stopBtn.disabled = true;
}

function stopMic() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }));
        ws.close();
    }
    stopMicLocal();
    setStatus("stopped");
}

function sendTextQuery() {
    const text = textInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    liveReply.textContent = "";
    addMessage("You (text): " + text, "user");
    ws.send(JSON.stringify({ type: "text_query", text }));
    textInput.value = "";
}

startBtn.onclick = startMic;
stopBtn.onclick = stopMic;
sendTextBtn.onclick = sendTextQuery;

textInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendTextQuery();
});

clearBtn.onclick = () => {
    liveTranscript.textContent = "";
    liveReply.textContent = "";
    chatBox.innerHTML = "";
    audioQueue = [];
    audioPlaying = false;
    setStatus("idle");
};
</script>
</body>
</html>
"""

class VoiceSession:
    def __init__(self):
        self.audio_buffer = bytearray()
        self.last_voice_ms = 0.0
        self.partial_text = ""
        self.last_partial_run_ms = 0.0
        self.started_at = time.time()

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

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    audio = pcm16_bytes_to_float32(audio_bytes)
    if audio.size == 0:
        return ""

    segments, _ = asr_model.transcribe(
        audio,
        language=None,
        vad_filter=False,
        beam_size=1
    )
    text = "".join(seg.text for seg in segments).strip()
    return text

def should_run_partial(session: VoiceSession) -> bool:
    total_ms = audio_ms(session.audio_buffer)
    return (total_ms - session.last_partial_run_ms) >= PARTIAL_TRANSCRIBE_EVERY_MS

def should_finalize(session: VoiceSession, now_ms: float) -> bool:
    total_ms = audio_ms(session.audio_buffer)
    silence_ms = now_ms - session.last_voice_ms
    return total_ms >= MIN_FINAL_AUDIO_MS and silence_ms >= END_OF_SPEECH_SILENCE_MS

def is_code_request(text: str) -> bool:
    t = text.lower()
    code_words = ["code", "python", "fastapi", "api", "script", "function", "program", "implement", "write code"]
    return any(word in t for word in code_words)

def clean_spoken_text(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "I have provided the code on screen.", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_complete_sentences(buffer: str, final: bool = False):
    # sentence punctuation across English / Tamil-friendly punctuation
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

async def stream_llm_and_tts(websocket: WebSocket, user_text: str):
    await websocket.send_text(json.dumps({
        "type": "info",
        "text": "generating reply"
    }))

    code_mode = is_code_request(user_text)

    stream = await llm_client.chat.completions.create(
        model="gemma-3-4b-it",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        temperature=0.2,
        max_tokens=220 if not code_mode else 420,
        stream=True,
    )

    full_reply = ""
    sentence_buffer = ""
    tts_sent_count = 0

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
        ready_sentences, remainder = split_complete_sentences(sentence_buffer, final=True)
        for sent in ready_sentences[: max(0, MAX_TTS_QUEUE_SENTENCES - tts_sent_count)]:
            await synthesize_and_send_sentence(websocket, sent)

    if code_mode:
        await websocket.send_text(json.dumps({
            "type": "tts_sentence_ready",
            "url": await asyncio.to_thread(generate_tts, "I have provided the code on screen."),
            "text": "I have provided the code on screen."
        }))

    await websocket.send_text(json.dumps({
        "type": "llm_done",
        "text": full_reply
    }))

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    session = VoiceSession()

    await websocket.send_text(json.dumps({
        "type": "info",
        "text": "mic connected"
    }))

    try:
        while True:
            data = await websocket.receive()

            if "text" in data and data["text"] is not None:
                raw_text = data["text"]
                try:
                    msg = json.loads(raw_text)
                except Exception:
                    msg = {}

                if msg.get("type") == "stop":
                    final_text = await asyncio.to_thread(transcribe_audio_bytes, bytes(session.audio_buffer))
                    final_text = final_text.strip()
                    if final_text:
                        await websocket.send_text(json.dumps({
                            "type": "final_transcript",
                            "text": final_text
                        }))
                        await stream_llm_and_tts(websocket, final_text)
                    break

                if msg.get("type") == "text_query":
                    user_text = (msg.get("text") or "").strip()
                    if user_text:
                        await stream_llm_and_tts(websocket, user_text)
                    continue

            elif "bytes" in data and data["bytes"] is not None:
                chunk = data["bytes"]
                session.audio_buffer.extend(chunk)

                now_ms = (time.time() - session.started_at) * 1000.0
                chunk_rms = rms_from_pcm16(chunk)

                if chunk_rms >= SILENCE_RMS_THRESHOLD:
                    session.last_voice_ms = now_ms

                if should_run_partial(session):
                    partial_text = await asyncio.to_thread(transcribe_audio_bytes, bytes(session.audio_buffer))
                    partial_text = partial_text.strip()
                    session.last_partial_run_ms = audio_ms(session.audio_buffer)

                    if partial_text and partial_text != session.partial_text:
                        session.partial_text = partial_text
                        await websocket.send_text(json.dumps({
                            "type": "partial_transcript",
                            "text": partial_text
                        }))

                if should_finalize(session, now_ms):
                    final_text = await asyncio.to_thread(transcribe_audio_bytes, bytes(session.audio_buffer))
                    final_text = final_text.strip()

                    if final_text:
                        await websocket.send_text(json.dumps({
                            "type": "final_transcript",
                            "text": final_text
                        }))
                        await stream_llm_and_tts(websocket, final_text)

                    session.audio_buffer = bytearray()
                    session.partial_text = ""
                    session.last_partial_run_ms = 0.0
                    session.last_voice_ms = now_ms

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": f"Voice pipeline error: {str(e)}"
            }))
        except Exception:
            pass
