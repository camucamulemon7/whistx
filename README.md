# whistx

`whistx` is a browser-based transcription app built around an OpenAI-compatible ASR API.
It records microphone audio, shared screen audio, or both, sends chunked audio to a FastAPI backend, and streams finalized transcript segments back to the UI.

## What It Does

- Real-time chunk-based transcription over WebSocket
- OpenAI-compatible ASR backend support
  - OpenAI `whisper-1`
  - OpenAI-compatible Whisper deployments
  - Other compatible backends configured via `ASR_BASE_URL` / `ASR_MODEL`
- Audio source selection
  - Microphone
  - Screen/tab audio
  - Microphone + screen audio mix
- Context carry-over between chunks
- Client-side VAD-assisted chunk finalization
- Audio preprocessing before ASR
- Transcript export
  - `txt`
  - `jsonl`
  - `srt`
- Optional speaker diarization with `pyannote.audio`
- Optional transcript summarization with an LLM backend
- Optional transcript proofreading with an LLM backend

## Current Architecture

### Frontend

Files: [`web/`](./web)

- Uses `MediaRecorder` and WebSocket
- Captures audio from the selected source
- Uses lightweight RMS-based VAD in the browser
- Finalizes chunks with this policy:
  - keep recording until a minimum segment length is reached
  - cut earlier when silence is detected
  - force cut at the configured max chunk length
- Sends `start`, `chunk`, and `stop` messages to the backend

### Backend

Files: [`server/`](./server)

- FastAPI application
- `ws://.../ws/transcribe` receives audio chunks and returns finalized transcript segments
- Applies audio preprocessing with `ffmpeg`
- Maintains short context memory per session
  - recent transcript lines
  - extracted key terms
- Writes transcript artifacts to disk
- Can run post-session diarization
- Exposes REST endpoints for summary and proofreading

### ASR Flow

1. Browser records a chunk
2. Backend decodes and preprocesses audio
3. Backend appends a small overlap from the previous chunk
4. Backend sends the chunk to the configured ASR API
5. Finalized text is normalized and stored
6. UI updates immediately with final segments

## Requirements

- Python 3.10+
- `ffmpeg`
- Chrome or Edge recommended
- OpenAI-compatible ASR API key
- Optional for diarization:
  - Hugging Face token with access to `pyannote/speaker-diarization-3.1`
  - compatible `torch` / `torchaudio`

## Quick Start

### Local

```bash
cp .env.example .env
./run.sh
```

Then open:

- `http://localhost:8005`

### Docker

```bash
cp .env.example .env
./start.sh
```

### Podman (rootless)

```bash
cp .env.example .env
./podman-run.sh
```

## Minimal Configuration

At minimum, set these in `.env`:

```env
ASR_API_KEY=your_api_key
```

If you use an OpenAI-compatible local or self-hosted backend, also set:

```env
ASR_BASE_URL=http://localhost:8000/v1
ASR_MODEL=whisper-1
```

## Important Environment Variables

### ASR

- `ASR_API_KEY`
- `ASR_BASE_URL`
- `ASR_MODEL`
- `ASR_DEFAULT_LANGUAGE`
- `ASR_DEFAULT_PROMPT`
- `ASR_DEFAULT_TEMPERATURE`
- `ASR_PREPROCESS_ENABLED`
- `ASR_PREPROCESS_SAMPLE_RATE`
- `ASR_OVERLAP_MS`
- `ASR_CONTEXT_PROMPT_ENABLED`
- `ASR_CONTEXT_MAX_CHARS`
- `ASR_CONTEXT_RECENT_LINES`
- `ASR_CONTEXT_TERM_LIMIT`
- `ASR_MAX_QUEUE_SIZE`
- `ASR_MAX_CHUNK_BYTES`

### UI

- `APP_BRAND_TITLE`
- `APP_BRAND_TAGLINE`
- `APP_UI_BANNERS`
- `APP_PROMPT_TEMPLATES`

### Summary

- `SUMMARY_API_KEY`
- `SUMMARY_BASE_URL`
- `SUMMARY_MODEL`
- `SUMMARY_TEMPERATURE`
- `SUMMARY_INPUT_MAX_CHARS`
- `SUMMARY_SYSTEM_PROMPT`
- `SUMMARY_PROMPT_TEMPLATE`

### Proofread

- `PROOFREAD_API_KEY`
- `PROOFREAD_BASE_URL`
- `PROOFREAD_MODEL`
- `PROOFREAD_TEMPERATURE`
- `PROOFREAD_INPUT_MAX_CHARS`
- `PROOFREAD_SYSTEM_PROMPT`
- `PROOFREAD_PROMPT_TEMPLATE`

### Diarization

- `DIARIZATION_ENABLED`
- `DIARIZATION_HF_TOKEN`
- `DIARIZATION_MODEL`
- `DIARIZATION_DEVICE`
- `DIARIZATION_SAMPLE_RATE`
- `DIARIZATION_NUM_SPEAKERS`
- `DIARIZATION_MIN_SPEAKERS`
- `DIARIZATION_MAX_SPEAKERS`
- `DIARIZATION_WORK_DIR`
- `DIARIZATION_KEEP_CHUNKS`
- `DIARIZATION_FFMPEG_BIN`
- `HF_HUB_DISABLE_XET`

See [`.env.example`](./.env.example) for the full template.

## Prompt Templates

You can define prompt template buttons from `.env`.

Example:

```env
APP_PROMPT_TEMPLATES=[
  {"id":"soc","label":"SoC Design","content":"SoC, ASIC, AXI, STA, PnR"},
  {"id":"nand","label":"NAND","content":"NAND, ONFI, LDPC, BBT"}
]
```

Use `\n` for line breaks inside `.env` values.

## Diarization Setup

To enable speaker diarization:

1. Accept the gated model terms on Hugging Face for:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
2. Set:

```env
DIARIZATION_ENABLED=1
DIARIZATION_HF_TOKEN=hf_xxx
```

3. Restart the app

Notes:

- Diarization is applied after recording stops
- Speaker labels are patched into stored transcript records after batch processing
- When diarization is disabled, speaker-count controls are hidden in the UI

## API Endpoints

### Health

- `GET /api/health`

### WebSocket Transcription

- `ws://<host>:<port>/ws/transcribe`

Client messages:

```json
{"type":"start","sessionId":"sess-xxx","language":"ja","audioSource":"mic","prompt":"domain terms"}
```

```json
{
  "type": "chunk",
  "seq": 0,
  "offsetMs": 0,
  "durationMs": 20000,
  "mimeType": "audio/webm;codecs=opus",
  "audio": "<base64>"
}
```

```json
{"type":"stop"}
```

### Summary

- `POST /api/summarize`

### Proofread

- `POST /api/proofread`

### Transcript Downloads

- `/api/transcript/{session_id}.txt`
- `/api/transcript/{session_id}.jsonl`
- `/api/transcript/{session_id}.srt`

## Notes on Accuracy

Current accuracy-oriented measures include:

- source-aware preprocessing for `mic`, `display`, and `both`
- overlap between adjacent chunks
- VAD-assisted chunk finalization
- short context memory instead of unbounded transcript concatenation
- Japanese spacing normalization for transcript cleanup
- repetition suppression for obvious ASR failure cases

## Security Notes

- Do not commit `.env`
- Treat transcript data as sensitive if it contains internal conversations
- Rotate API keys immediately if they are exposed
- Review any banner text or prompt templates before publishing screenshots or demos

## License

MIT. See [LICENSE](./LICENSE).
