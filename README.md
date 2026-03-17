# AQI Voice Assistant

AQI Voice Assistant is a multilingual voice-and-text assistant built for AQI and AI systems work.  
It supports real-time WebSocket-based interaction and integrates ASR, LLM, and TTS components for both English and Tamil.

---

## Project Overview

This project provides an end-to-end assistant that can:

- accept typed user input
- accept spoken user input through microphone
- transcribe speech into text
- process the text through direct routing or LLM
- generate response text
- convert the response into speech
- return both text and audio to the frontend through WebSocket

The current milestone focuses on a working CPU-stable setup for local development and testing.

---

## Current Milestone

This version includes:

- WebSocket-based typed and voice interaction
- English ASR using `faster-whisper`
- Tamil ASR using `AI4Bharat Indic Conformer`
- English TTS using Kokoro ONNX
- Tamil TTS using MMS Tamil
- full reply audio playback
- frontend WebSocket UI with reconnect support
- Hugging Face model download and local runtime setup
- AQI-focused assistant routing with safety guardrails

---

## Current Scope

The assistant is currently limited to:

- AQI-related queries
- AI/ML project discussions
- FastAPI, APIs, architecture, deployment, and debugging
- AQI assistant project-support interactions

It is not intended for broad general-purpose casual chat beyond limited greeting and identity responses.

---

## Tech Stack

### Backend
- Python
- FastAPI
- WebSocket
- OpenAI-compatible client
- AsyncOpenAI / OpenAI SDK

### Speech Recognition
- English ASR: `faster-whisper`
- Tamil ASR: `ai4bharat/indic-conformer-600m-multilingual`

### Text to Speech
- English TTS: Kokoro ONNX
- Tamil TTS: MMS Tamil

### Model Hosting / Inference
- Hugging Face for model asset download
- vLLM-compatible LLM endpoint for assistant responses

### Frontend
- HTML
- CSS
- JavaScript
- WebSocket client
- browser microphone capture
- browser audio playback

---

## Project Structure

```text
aqi-assistant/
│
├── app.py
├── index.html
├── README.md
├── PROGRESS_UPDATE.md
│
└── tts/
    ├── kokoro_worker.py
    ├── mms_tamil_worker.py





    └── tts_router.pyFile Description

app.py
Main FastAPI WebSocket backend. Handles:

text input

voice input

ASR routing

direct response routing

guardrails

LLM streaming

full reply TTS generation

index.html
Frontend interface for:

typed input

microphone input

transcript display

streaming reply display

audio playback

reconnect support

tts/kokoro_worker.py
English TTS worker using Kokoro ONNX.

tts/mms_tamil_worker.py
Tamil TTS worker using MMS Tamil model.

Architecture Overview

The system contains three main layers.

1. Frontend Layer

The frontend runs in the browser and is responsible for:

generating and storing session ID

opening WebSocket connection

sending typed messages

capturing microphone audio

converting audio to PCM16 at 16 kHz

receiving transcript, text reply, and audio reply

playing returned audio

2. Backend Layer

The FastAPI backend is responsible for:

maintaining WebSocket sessions

buffering voice input

detecting speech end

routing ASR based on language

applying guardrails

generating direct or LLM-based responses

generating TTS output

sending text and audio results back through WebSocket

3. Model Layer

The model layer includes:

English ASR: faster-whisper

Tamil ASR: AI4Bharat Indic Conformer

English TTS: Kokoro

Tamil TTS: MMS Tamil

LLM: OpenAI-compatible vLLM endpoint

End-to-End Pipeline
Typed Input Pipeline

User types a message in the frontend.

Frontend sends the message through WebSocket as JSON.

Backend receives the text.

Backend checks whether:

it matches a direct response rule

it should be blocked by guardrails

it should go to the LLM

Backend generates the text reply.

Backend generates full reply TTS.

Backend sends:

final text reply

full audio reply

Frontend displays text and plays audio.

Voice Input Pipeline

User starts microphone in frontend.

Frontend captures live audio from browser microphone.

Audio is downsampled to 16 kHz PCM16.

Binary audio chunks are streamed to backend over WebSocket.

Backend buffers incoming audio.

Backend uses RMS threshold and silence duration to detect end of speech.

Backend uses Whisper-based language hinting.

Backend routes audio to:

English ASR via faster-whisper

Tamil ASR via AI4Bharat

Final transcript is generated.

Transcript is sent to frontend.

Transcript is processed through the same text pipeline as typed input.

Backend generates text response.

Backend generates full reply TTS.

Frontend receives and plays audio.

WebSocket Communication

The backend uses this WebSocket endpoint:

/ws/voice/{session_id}
Session ID

The frontend generates a session ID automatically and stores it in browser local storage.

This session ID is used by the backend to maintain:

recent conversation history

name memory

follow-up state

Message Types Sent by Frontend

text_chat

clear_session

stop_voice

binary PCM audio chunks

Message Types Sent by Backend

info

typed_user

partial_transcript

final_transcript

llm_delta

llm_done

tts_sentence_ready

cleared

discarded_voice

error

ASR Design
English ASR

English speech is transcribed using faster-whisper.

Why it was chosen:

lightweight

CPU-friendly

suitable for real-time English transcription

Tamil ASR

Tamil speech is transcribed using AI4Bharat Indic Conformer.

Why it was chosen:

stronger Tamil support

better fit for Indic speech than English-only ASR fallback

Routing Logic

ASR routing works using language hinting:

if language hint is English → Whisper English path

if language hint is Tamil → AI4Bharat Tamil path

if unclear → fallback routing is used

TTS Design
English TTS

English reply audio is generated using Kokoro ONNX through a worker script.

Tamil TTS

Tamil reply audio is generated using MMS Tamil through a worker script.

Full Reply Audio

The pipeline was updated so that:

direct replies produce full reply audio

LLM replies also produce full final reply audio

This ensures that the entire visible text reply is spoken back to the user.

Frontend Design

The frontend provides:

session-based WebSocket connection

connect and reconnect controls

typed input

microphone start and stop

transcript display

streaming reply display

audio playback

session reset support

Reconnect Logic

Reconnect logic was added so that if the WebSocket disconnects:

frontend can reconnect automatically

or reconnect manually through button control

Hugging Face Model Handling

At startup, the backend ensures required model assets exist locally.

Ensured Assets

Kokoro ONNX model

Kokoro voice file

MMS Tamil model files

This allows the assistant to run using locally cached assets after initial setup.

Stability and Fixes Completed

The following major fixes were completed during this milestone:

WebSocket-based assistant flow

The assistant communication was moved to a WebSocket-based message flow.

Full reply TTS

Fixed issue where only the first sentence was spoken.

AI4Bharat loading fix

Replaced incorrect pipeline-based loading with direct AutoModel.from_pretrained(..., trust_remote_code=True) loading.

CPU-safe Tamil ASR setup

Forced CPU-safe handling to avoid CUDA allocation issues during Tamil ASR initialization.

Voice transcript debug flow

Added logging for:

final route used

final text

user text received by backend

Model double-loading prevention

Used uvicorn.run(app, ...) instead of module string-based startup to avoid repeated top-level imports.

Frontend session handling

Frontend now auto-generates session ID and reuses it via local storage.

Run Backend
WHISPER_DEVICE=cpu \
WHISPER_COMPUTE_TYPE=int8 \
WHISPER_MODEL_SIZE=tiny \
APP_PORT=9002 \
AI4BHARAT_MODEL_NAME=ai4bharat/indic-conformer-600m-multilingual \
python app.py
Run Frontend

Serve the frontend page locally:

python -m http.server 5500

Then open:

http://127.0.0.1:5500/index.html
WebSocket Endpoint
ws://127.0.0.1:9002/ws/voice/<session_id>

session_id is generated automatically by the frontend and stored in browser local storage.

Current Limitations

Some limitations still remain:

Tamil ASR is currently CPU-based, so latency may be higher

TTS uses worker scripts, which add subprocess overhead

AI4Bharat emits internal frame-duration warnings

browser still needs static page loading for the UI

live AQI data source is not yet integrated

Current Status

The assistant now supports:

typed WebSocket chat

voice WebSocket chat

multilingual ASR

multilingual TTS

full reply audio playback

session-based memory flow

reconnect-capable frontend

This milestone establishes a working AQI assistant foundation for further development.

Next Planned Improvements

Possible next steps include:

integrate live AQI API or data source

improve Tamil ASR latency

reduce TTS subprocess overhead

add richer AQI response intelligence

add authentication and user-specific memory

improve frontend styling and UX

add deployment guide and architecture diagrams

Author

janareshma15-blip
