# AQI Voice Assistant — Work Progress

## Objective
Converted the AQI assistant into a multi-service Docker architecture.

## Services
- gemma-vllm — port 9005
- whisper-asr — port 9010
- ai4bharat-asr — port 9011
- kokoro-tts — port 9020
- mms-tamil-tts — port 9021
- aqi-orchestrator — port 9002

## Current working services
- gemma-vllm ✅
- whisper-asr ✅
- ai4bharat-asr ✅
- mms-tamil-tts ✅
- aqi-orchestrator ✅

## Current blocker
### kokoro-tts
- Container builds and starts successfully
- Hugging Face download path fixed
- Health endpoint responds
- Model file loaded from:
  - `/app/runtime/kokoro-runtime/onnx/model.onnx`
- Voice file loaded from:
  - `/app/runtime/kokoro-runtime/voices/af.bin`
- Current runtime error:
  - `ValueError: This file contains pickled (object) data`
- Reason:
  - `kokoro_onnx` expects a different voice file format than the Hugging Face `af.bin` file currently being downloaded

## Verified outputs
### AI4Bharat health
- `curl http://127.0.0.1:9011/health`
- Returned: `{"ok":true,"service":"ai4bharat-asr"}`

### Kokoro health
- `curl http://127.0.0.1:9020/health`
- Returned:
  - `ok: false`
  - `loaded: false`
  - error about pickled object data in voice file

## Files created
- `.env`
- `docker-compose.yml`
- `WORK_PROGRESS.md`
- `whisper-asr/`
- `ai4bharat-asr/`
- `kokoro-tts/`
- `mms-tamil-tts/`
- `aqi-orchestrator/`

## Next steps
1. Resolve Kokoro voice-file compatibility issue
2. Re-test full voice pipeline
3. Push final stable version after Kokoro fix
