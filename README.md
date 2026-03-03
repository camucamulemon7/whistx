# whistx

`whistx` is a real-time transcription app built on OpenAI `whisper-1`.
It captures browser audio, sends chunked audio over WebSocket, transcribes on the server, and streams finalized segments back to the UI.

## Features

- Real-time chunk-based transcription using OpenAI `whisper-1`
- Context carry-over between chunks for better continuity
- Built-in silence handling
  - Client-side lightweight VAD (skip silent chunks)
  - Server-side hallucination suppression for common silence artifacts
- Audio source selection in UI
  - Microphone
  - Display audio (screen/tab share audio)
  - Microphone + display audio mix
- Transcript export
  - TXT
  - JSONL
  - SRT
- One-click LLM summary from transcript text (`/api/summarize`)

## Architecture

- **Frontend**: static files in `web/`
  - Uses `MediaRecorder` + WebSocket
  - Creates fixed-size chunks (default 20s, configurable 12-30s)
- **Backend**: FastAPI in `server/`
  - `/ws/transcribe`: receives chunks, calls OpenAI transcription API, emits final segments
  - `/api/summarize`: summarizes transcript text with a chat model
  - `/api/transcript/{session_id}.{txt|jsonl|srt}`: download outputs

## Requirements

- Python 3.10+
- Chrome or Edge (latest recommended)
- OpenAI API key

## Quick Start (Local)

```bash
cd /home/remon1129/ai/whistx
cp .env.example .env
# Set OPENAI_API_KEY in .env
./run.sh
```

Then open:

- `http://localhost:8005`

## Quick Start (Docker)

```bash
cd /home/remon1129/ai/whistx
cp .env.example .env
# Set OPENAI_API_KEY in .env
./start.sh
```

For Podman:

```bash
./podman-run.sh
```

### Rootless Podman Notes

`podman-run.sh` is tuned for rootless mode by default:

- `--userns=keep-id` is enabled (`PODMAN_USERNS=keep-id`)
- volume mount defaults to `:Z` label option (`PODMAN_VOLUME_OPTS=Z`)

If your environment does not need SELinux relabeling, set:

```bash
PODMAN_VOLUME_OPTS=
```

## Environment Variables

Required:

- `OPENAI_API_KEY`

Important optional settings:

- `OPENAI_BASE_URL` (default: OpenAI)
- `WHISPER_MODEL` (default: `whisper-1`)
- `DEFAULT_LANGUAGE` (default: `ja`)
- `DEFAULT_PROMPT`
- `DEFAULT_TEMPERATURE` (default: `0.0`)
- `CONTEXT_PROMPT_ENABLED` (default: `1`)
- `CONTEXT_MAX_CHARS` (default: `1000`)
- `MAX_CHUNK_BYTES` (default: `12582912`)

Summary settings:

- `SUMMARY_API_KEY` (falls back to `OPENAI_API_KEY`)
- `SUMMARY_BASE_URL` (falls back to `OPENAI_BASE_URL`)
- `SUMMARY_MODEL` (default: `gpt-4o-mini`)
- `SUMMARY_TEMPERATURE` (default: `0.2`)
- `SUMMARY_INPUT_MAX_CHARS` (default: `16000`)

Runtime settings:

- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8005`)
- `WS_PATH` (default: `/ws/transcribe`)
- `DATA_DIR` (default: `data/transcripts`)
- `PODMAN_USERNS` (default: `keep-id`, used by `podman-run.sh`)
- `PODMAN_VOLUME_OPTS` (default: `Z`, used by `podman-run.sh`)

## API

### Health

- `GET /api/health`

### WebSocket Transcription

- `ws://<host>:<port>/ws/transcribe` (`wss://` when HTTPS)

Client messages:

- `start`

```json
{"type":"start","sessionId":"sess-xxx","language":"ja","prompt":"domain terms"}
```

- `chunk`

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

- `stop`

```json
{"type":"stop"}
```

Server messages:

- `info`
- `final`
- `error`
- `conn`

### Summary

- `POST /api/summarize`

```json
{
  "text": "full transcript text",
  "language": "ja"
}
```

## Zoom / Webex Notes

- Choose **Display audio** or **Mic + Display audio** in the UI.
- In the browser share dialog, enable audio sharing (for tab share: "Share tab audio").
- If display sharing stops, recording stops automatically.

## Security Notes

- Do not commit `.env`.
- Use `.env.example` as a template.
- Rotate API keys if accidentally exposed.

## License

MIT License. See [LICENSE](./LICENSE).
