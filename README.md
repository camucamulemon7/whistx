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
- Optional speaker diarization using `pyannote.audio`
  - Segment-level speaker labels (`SPK_00`, `SPK_01`, ...)
  - Speaker labels are applied after recording stops
  - UI toggle to enable/disable diarization per session
- One-click LLM summary from transcript text (`/api/summarize`)
- One-click LLM proofreading/correction (`/api/proofread`)
  - Rendered in a separate panel next to Transcript (desktop layout)

## Architecture

- **Frontend**: static files in `web/`
  - Uses `MediaRecorder` + WebSocket
  - Creates fixed-size chunks (default 20s, configurable 12-30s)
- **Backend**: FastAPI in `server/`
  - `/ws/transcribe`: receives chunks, calls OpenAI transcription API, emits final segments
  - `/api/summarize`: summarizes transcript text with a chat model
  - `/api/proofread`: proofreads transcript text with a chat model
  - `/api/transcript/{session_id}.{txt|jsonl|srt}`: download outputs

## Requirements

- Python 3.10+
- Chrome or Edge (latest recommended)
- OpenAI API key
- (Optional for diarization) Hugging Face token with access to `pyannote/speaker-diarization-3.1`

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

Proofread settings:

- `PROOFREAD_API_KEY` (falls back to `SUMMARY_API_KEY`, then `OPENAI_API_KEY`)
- `PROOFREAD_BASE_URL` (falls back to `SUMMARY_BASE_URL`, then `OPENAI_BASE_URL`)
- `PROOFREAD_MODEL` (default: `gpt-4o-mini`)
- `PROOFREAD_TEMPERATURE` (default: `0.0`)
- `PROOFREAD_INPUT_MAX_CHARS` (default: `24000`)

UI banner settings:

- `UI_BANNERS` (JSON array, default: empty)
- `WEBUI_BANNERS` (alias of `UI_BANNERS`)

Example:

```env
UI_BANNERS=[{"id":"notice-1","type":"warning","title":"Notice","message":"Do not input confidential information.","dismissible":true}]
```

You can also set plain text (non-JSON). In that case it is shown as a single info banner:

```env
UI_BANNERS=Do not share confidential content in this workspace.
```

Diarization settings (`pyannote.audio`):

- `DIARIZATION_ENABLED` (default: `0`)
- `DIARIZATION_HF_TOKEN` (required when diarization is enabled)
- `DIARIZATION_MODEL` (default: `pyannote/speaker-diarization-3.1`)
- `DIARIZATION_DEVICE` (default: `auto`)
- `DIARIZATION_SAMPLE_RATE` (default: `16000`)
- `DIARIZATION_NUM_SPEAKERS` (default: `0`, auto)
- `DIARIZATION_MIN_SPEAKERS` (default: `0`)
- `DIARIZATION_MAX_SPEAKERS` (default: `0`)
- `DIARIZATION_WORK_DIR` (default: `data/diarization`)
- `DIARIZATION_KEEP_CHUNKS` (default: `0`)
- `HF_HUB_DISABLE_XET` (default: `1` inside app when diarization is used)
- `FFMPEG_BIN` (default: `ffmpeg`)

Runtime settings:

- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8005`)
- `WS_PATH` (default: `/ws/transcribe`)
- `DATA_DIR` (default: `data/transcripts`)
- `PODMAN_USERNS` (default: `keep-id`, used by `podman-run.sh`)
- `PODMAN_VOLUME_OPTS` (default: `Z`, used by `podman-run.sh`)

## Speaker Diarization Setup (pyannote.audio)

1. Accept model terms on Hugging Face for:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
2. Set `.env`:

```env
DIARIZATION_ENABLED=1
DIARIZATION_HF_TOKEN=hf_xxx
# Optional tuning
DIARIZATION_NUM_SPEAKERS=0
DIARIZATION_MIN_SPEAKERS=0
DIARIZATION_MAX_SPEAKERS=0
```

3. Restart server (`./run.sh` or container scripts).

You can control speaker count from UI per session:

- `Auto`: let pyannote estimate number of speakers
- `Fixed`: force a fixed count (e.g. 2)
- `Range`: constrain min/max speakers

When diarization is OFF, speaker-count controls are hidden in the UI.

When enabled, transcript lines are emitted in real time as before, then speaker labels are patched in after `stop`.
Exports (`.txt`, `.jsonl`, `.srt`) include speaker labels once diarization completes.

### Troubleshooting diarization

If you see an error related to `torchaudio.AudioMetaData`, your local `torch/torchaudio` is too new for the current pyannote stack.

```bash
cd /home/remon1129/ai/whistx
source .venv/bin/activate
pip install --upgrade "torch>=2.2,<2.9" "torchaudio>=2.2,<2.9"
```

Then restart the app.

If you still see `hf_hub_download() got an unexpected keyword argument 'use_auth_token'`
after updating this repository, reinstall dependencies:

```bash
cd /home/remon1129/ai/whistx
source .venv/bin/activate
pip install -r requirements.txt
```

If diarization fails with `Weights only load failed` / `_pickle.UnpicklingError`
on `torch>=2.6`, reinstall dependencies and restart:

```bash
cd /home/remon1129/ai/whistx
source .venv/bin/activate
pip install -r requirements.txt
```

If it still fails, remove cached pyannote checkpoints and retry:

```bash
rm -rf ~/.cache/torch/pyannote
rm -rf ~/.cache/huggingface/hub/models--pyannote--*
```

If diarization fails with errors around `NoneType` / `eval` or model download,
your Hugging Face token likely does not have access to one of the gated models.
Accept terms for both:

- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

Then restart the server.

## API

### Health

- `GET /api/health`

### WebSocket Transcription

- `ws://<host>:<port>/ws/transcribe` (`wss://` when HTTPS)

Client messages:

- `start`

```json
{"type":"start","sessionId":"sess-xxx","language":"ja","prompt":"domain terms","diarizationEnabled":true}
```

Optional diarization parameters in `start`:

```json
{
  "type": "start",
  "diarizationEnabled": true,
  "diarizationNumSpeakers": 0,
  "diarizationMinSpeakers": 0,
  "diarizationMaxSpeakers": 0
}
```

- `diarizationNumSpeakers > 0`: fixed speaker count
- otherwise `diarizationMinSpeakers` / `diarizationMaxSpeakers` are used as range

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

### Proofread

- `POST /api/proofread`

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
