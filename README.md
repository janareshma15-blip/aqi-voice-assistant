# AQI Multiservice

## Services
- gemma-vllm -> 9005
- whisper-asr -> 9010
- ai4bharat-asr -> 9011
- kokoro-tts -> 9020
- mms-tamil-tts -> 9021
- aqi-orchestrator -> 9002

## Current status
- Working: gemma-vllm, whisper-asr, ai4bharat-asr, mms-tamil-tts, aqi-orchestrator
- Blocked: kokoro-tts voice file compatibility issue

## Notes
- Environment secrets are stored in `.env`
- Runtime/cache files are under `shared/runtime/`
- Detailed work log is in `docs/WORK_PROGRESS.md`
