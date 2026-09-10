# whistx

`whistx` is a browser-based transcription app built around an OpenAI-compatible Whisper ASR API.
It records microphone audio, shared screen audio, or both, and provides live transcripts, chapter summaries with captured screens, and a meeting assistant grounded in transcript citations.

## What It Does

- Live PCM transcription with partial text, LocalAgreement-2, durable audio ACKs, and reconnect/replay
- Chapter summaries with decisions, actions, citations, and actual captured screen images
- Streaming meeting questions grounded in the selected meeting
- Post-recording audio re-recognition with original text retained
- Legacy chunk-based transcription over WebSocket
- OpenAI-compatible Whisper ASR backend support
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
- Optional speaker diarization with `pyannote.audio`
- Optional transcript summarization with an LLM backend
- Optional transcript proofreading with an LLM backend

By default, diarization dependencies are not installed in local or container setups unless explicitly enabled.

## Current Architecture

### Frontend

Files: [`web/`](./web)

- Uses `AudioWorklet` (16 kHz mono PCM, one-second packets) and a resumable WebSocket by default
- Uses `MediaRecorder` and the existing chunk protocol when live capture is disabled or AudioWorklet is unavailable
- The following chunk/VAD policy describes that legacy path
- Captures audio from the selected source
- Uses lightweight RMS-based VAD in the browser
- Finalizes chunks with this policy:
  - keep recording until a minimum segment length is reached
  - cut earlier when silence is detected
  - force cut at the configured max chunk length
- Sends `start`, `chunk`, and `stop` messages to the backend

### Backend

Files: [`server/`](./server)


Current backend layout after the refactor:

- `server/app.py`: thin entrypoint
- `server/core/application.py`: app wiring and router registration
- `server/api/routes/*.py`: HTTP entrypoints
- `server/api/ws/transcribe.py`: WebSocket entrypoint
- `server/services/*.py`: auth/admin/history business logic
- `server/repositories/*.py`: SQLAlchemy access boundaries
- `server/core/config/*.py`: split configuration modules
- `server/runtime.py`: realtime resource composition and lifecycle
- `server/transcription/*.py`: live sessions, chunk protocol, and ASR worker
- `docs/architecture.md`: dependency direction and source-of-truth guide

- FastAPI application
- `ws://.../ws/transcribe/live` journals PCM and returns partial/final events; microphone and shared audio can remain separate
- `ws://.../ws/transcribe` retains the existing chunk protocol
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

Qwen3-ASR realtime is supported through `ASR_BACKEND=qwen3_vllm` (see below). Voxtral realtime remains unsupported. `ASR_BACKEND=whisper` retains the Whisper-compatible HTTP backend.

## Meeting workspace

1. Log in, choose the audio source, and start recording. **ライブ文字起こし** in the settings enables the new PCM path. Browser capture requires localhost or HTTPS; shared audio depends on the browser and selected share target.
2. Ask questions such as **決まったこと** or **直近5分の要点** in the meeting assistant. Answers include links to the supporting utterances and identify the transcript time used. Unstable recognition hypotheses are excluded from its evidence.
3. Stop recording and wait for finalization. Optionally use **音声から再認識** before saving to history. This runs the configured ASR again on saved utterance audio; it does not guarantee an accuracy improvement. Failures before the commit retain the current transcript. The first original transcript is retained, and each revised record includes `originalText`.
4. Select **会議の要約** and **要約する**. Chapter cards contain cited key points, decisions, actions, open questions, and time-matched captured screens. **共有資料** shows the captured images. **画像付きで保存** exports a self-contained HTML recap with embedded images.
5. Save the meeting to history. The chapter recap, assistant turns, audio, and screenshots travel with the history artifacts. Its ZIP includes `meeting.json` and, after re-recognition, `transcript.original.jsonl`. New transcript revisions mark an existing recap as stale; regenerate it to update the sources.

Both live adapters use the existing `.env` `ASR_BASE_URL`, `ASR_MODEL`, and API key. The Whisper adapter implements rolling HTTP inference and LocalAgreement-2; the Qwen adapter uses vLLM realtime WebSockets and concurrent long-interval re-recognition. Neither is a SimulStreaming decoder. `SUMMARY_*` configures both chapter summaries and meeting QA. If `PROOFREAD_MODEL` is unset, it falls back to `SUMMARY_MODEL`. No new model server or schema migration is required for meeting insights; the application's existing database migrations are still required.

For a Xinference instance on the same host, the tested base URL is `http://localhost:9997/v1` with `whisper-large-v3-turbo`. Inside a container, use an address reachable from that container instead of assuming container localhost reaches the host. Existing `.env` values must be loaded by the launch command (as `run.sh` already does).

The new endpoint is `<APP_WS_PATH>/live`. Configure the proxy for WebSocket upgrades on that path and disable response buffering for `/api/meeting/ask` and `/api/meeting/refine`. Authentication, origin checks, connection quotas, and expensive API limits apply. Recap, QA, and refinement require login. Optional final speaker labeling reuses the configured `pyannote.audio` diarizer; without it, input-track labels are available. Diarization dependencies are not installed automatically by this feature.

Live capture keeps at most 16 MiB of unacknowledged PCM in browser memory, uses a 120-second server backlog threshold, and limits a session to four hours. Short disconnects replay from the server ACK. A full page reload loses browser-only packets; the server retains acknowledged audio. Capture failures expose a download for pending audio. The energy gate suppresses silence but is not neural VAD. Whisper recognition uses windows up to 24 seconds; long speech at window boundaries and difficult Japanese audio still require accuracy evaluation.

Screenshots are actual captured frames, not generated images. The assistant currently grounds answers in transcript text; it does not OCR or interpret image contents. Long meetings use bounded lexical retrieval, so answers are not guaranteed to cover every relevant statement. Citation validation verifies IDs and source ranges, not semantic truth. Inspect the linked utterance/audio for consequential decisions.

### Measure the configured ASR

Use an uncompressed 16 kHz mono, signed 16-bit PCM WAV:

```bash
python scripts/bench_streaming_asr.py sample.wav --language ja --realtime --reference reference.txt --output benchmark.json
```

The harness invokes the same live adapter and `.env` ASR, records actual partial/final events, first-partial and finalization timing, request counts, and optional raw character edit rate (CER). Temporary artifacts are isolated; the harness bypasses application quotas and excludes browser/WebSocket latency. It sends the supplied audio to the configured ASR endpoint. Use representative Japanese recordings and references before drawing accuracy conclusions.

Validation performed during implementation: Python service/regression tests, JavaScript tests, browser layout and recording lifecycle tests, mocked live capture/refinement and chapter/QA interactions, a real configured LLM recap/QA call on a fictional Japanese meeting, and real Xinference inference on a short public English audio fixture. These checks do not establish Japanese meeting accuracy, sustained multi-hour performance, or real-device screen-audio compatibility.

### Qwen3-ASR: one vLLM process, two inference lanes

Run one audio-enabled vLLM server with the realtime architecture and priority scheduling. This configuration uses the same resident weights for both APIs; Whistx does not load an ASR model or a forced aligner:

```bash
# In the vLLM environment, using the installed version's audio extra:
pip install 'vllm[audio]==0.29.0'
vllm serve Qwen/Qwen3-ASR-1.7B \
  --served-model-name Qwen3-ASR-1.7B --port 8004 \
  --hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}' \
  --scheduling-policy priority --max-model-len 8192
```

Use `.env` values below, then restart Whistx with its usual launch command:

```dotenv
ASR_BACKEND=qwen3_vllm
ASR_BASE_URL=http://localhost:8004/v1
ASR_MODEL=Qwen3-ASR-1.7B
ASR_DEFAULT_LANGUAGE=auto
ASR_REALTIME_WINDOW_SECONDS=5
ASR_HIGH_ACCURACY_ENABLED=1
ASR_HIGH_ACCURACY_WINDOW_SECONDS=60
ASR_HIGH_ACCURACY_PRIORITY=10
ASR_HIGH_ACCURACY_MAX_RT_LAG_SECONDS=2.0
ASR_HIGH_ACCURACY_TIMEOUT_SECONDS=180.0
```

Configure `ASR_API_KEY` if the server requires authentication. Containers must use a host address reachable from the container. Keep `SUMMARY_*` / `PROOFREAD_*` pointing to the existing text model endpoints. To switch back, set `ASR_BACKEND=whisper` together with the Whisper server URL/model. An unfinished recording must be resumed with its original backend/model; its interval sizes remain fixed from creation.

- **Realtime:** the browser sends 250 ms packets of PCM16 / 16 kHz / mono. Saved PCM is forwarded as it arrives to `/v1/realtime`. A bounded realtime window (default 5 seconds, configurable 1–10) has its own upstream WebSocket; incoming deltas update the active text, and completion commits a sample-anchored record. This limits upstream context reuse and aligns windows with high-accuracy boundaries. Choose Automatic for mixed Japanese/English streaming. With Japanese or English selected, each short window uses the same model’s `/audio/transcriptions` endpoint with the selected language; updates arrive at window completion rather than as token deltas. High-accuracy and manual re-recognition preserve that language selection. The vLLM Realtime protocol itself has no language parameter.
- **High accuracy:** once realtime has committed a complete 30–120 second interval, a separate async worker reads that continuous range from the same PCM journal and sends WAV audio to `/v1/chat/completions` with the **same URL/model** and lower scheduling priority (`10`; realtime uses `0`). The glossary and ASR prompt provide contextual vocabulary. The persisted high-accuracy cursor is a disk-backed queue, with at most one in-flight high-accuracy request across app workers sharing the transcript directory. Realtime can continue during that request. Admission waits when realtime is behind; priority scheduling does not guarantee GPU preemption or a hard latency bound.
- **Mixed-language safeguard:** a coarse Japanese/Latin-script coverage check catches wholesale language loss or translation relative to realtime text. When it triggers, the same model re-recognizes consecutive source groups separately (at most eight groups). If a group still loses a language, its realtime text is retained; an entirely unusable revision is skipped. This adds requests and can shorten the fallback context. It is not a complete omission/accuracy detector. Manual re-recognition also rejects detected language loss.
- **Replacement:** only the exact listed realtime segment IDs, track, and complete sample interval can be replaced. Each revision is appended and fsynced before the checkpoint advances. JSONL readers and downloads materialize revisions; TXT, meeting QA, recap and history consume the revised transcript. Runtime ZIPs include a separate revision journal. The browser applies revisions during recording, ignores superseded late events and reconciles the transcript on reconnect. Original realtime text, sample ranges and audio remain in `realtimeSegments` / `originalText`; history ZIPs include `transcript.revisions.jsonl`.
- **Timestamps:** ranges come from the retained PCM sample clock, not text length or response arrival time. A high-accuracy row covers its full 30–120 second source interval. This backend does **not** claim word-level alignment. The optional existing diarizer can label these intervals; a long interval can contain several speakers, so retained realtime ranges remain useful for inspection. No additional ASR/alignment model is installed.
- **Recovery:** upstream failures retain acknowledged PCM and current realtime text. High-accuracy failures retry during recording; failed stop processing leaves the session unfinalized and exposes retry. Silence intervals advance coverage without inserting hallucinated text. On reconnect, sample positions and materialized records are recovered from disk. Browser-only, unacknowledged packets still require the existing pending-audio recovery behavior.

Validate the server separately before a meeting:

```bash
python scripts/probe_qwen_asr.py --url http://localhost:8004/v1
# Add --audio sample.wav to verify actual recognition; the default is silence.
python scripts/bench_streaming_asr.py sample.wav --realtime --output benchmark.json
```

The probe checks both APIs concurrently and returns failure when either lane fails. `Please install vllm[audio]` is a server dependency error. Realtime support needs the architecture override; nonzero batch priorities need priority scheduling. The health capability `asrReady` means the app adapter is configured, not a completed upstream audio probe.

Measured on the configured local vLLM 0.29.0 endpoint with an 80.26-second repeated [Qwen public English fixture](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py), 30-second high-accuracy windows: first partial 5.12 seconds; revisions at 31.70 and 61.61 seconds while capture continued; final tail completed about 1.08 seconds after capture. A separate 38.48-second fixture alternating the [JSUT Japanese sample distributed by ttslearn](https://r9y9.github.io/ttslearn/latest/_modules/ttslearn/util.html) and that English sample reproduced majority-language omission in a plain long request. The safeguard re-recognized four source groups, retained Japanese and English in the resulting transcript, and applied the first revision at 32.95 seconds, before capture ended. These are adapter timings and script-retention checks, not microphone-to-browser latency or accuracy scores. Default vLLM buffering produces multi-second updates; evaluate representative meetings before tuning the realtime window.

Protocol/configuration references: [vLLM realtime connection](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/speech_to_text/realtime/connection.py), [Qwen realtime architecture](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_asr_realtime.py), [vLLM scheduler](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/).

## Requirements

- Python 3.12+
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
# Set APP_SESSION_SECRET in .env to a unique value, e.g. generated with:
python3 -c 'import secrets; print(secrets.token_hex(32))'
./run.sh
```

`run.sh` uses `python3` by default and rejects Python versions older than 3.12.
Set `DEV_PYTHON_BIN=python3.12` when the Python 3.12 executable is not your default `python3`.

If you want local diarization support, set one of these before running:

```env
DIARIZATION_ENABLED=1
```

or

```env
INSTALL_DIARIZATION_DEPS=1
```

Then open:

- `http://localhost:8005`

### Docker

```bash
cp .env.example .env
# Set APP_SESSION_SECRET in .env to a unique value, e.g. generated with:
python3 -c 'import secrets; print(secrets.token_hex(32))'
./start.sh
```

The container image installs the exact versions and hashes recorded in `requirements.lock`. Dependency declarations remain in `requirements.txt`; diarization-only declarations live in `requirements-diarization.txt`.

### Dependency locking

- Runtime declarations: `requirements.txt`
- Diarization declarations: `requirements-diarization.txt`
- Development declarations (including Ruff): `requirements-dev.txt`
- Generated locks: the matching `*.lock` files; do not edit these by hand

Regenerate all lock files with the same `uv` version used by the container:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ghcr.io/astral-sh/uv:0.8.17-python3.12-bookworm-slim \
  uv pip compile requirements.txt --python-version 3.12 --generate-hashes -o requirements.lock
docker run --rm -v "$PWD:/workspace" -w /workspace ghcr.io/astral-sh/uv:0.8.17-python3.12-bookworm-slim \
  uv pip compile requirements-diarization.txt --python-version 3.12 --generate-hashes -o requirements-diarization.lock
docker run --rm -v "$PWD:/workspace" -w /workspace ghcr.io/astral-sh/uv:0.8.17-python3.12-bookworm-slim \
  uv pip compile requirements-dev.txt --python-version 3.12 --generate-hashes -o requirements-dev.lock
```

Local startup and container builds consume the runtime lock. CI consumes the development lock so all three environments resolve the same runtime versions.

### Configuration registry

[`server/core/config/registry.py`](./server/core/config/registry.py) is the canonical registry for application environment variables. It records each variable's type, default, accepted range, secret status, and aliases. `.env.advanced.example`, the Python loaders, and container forwarding are checked against it by the test suite. `.env.example` contains the smaller starter configuration.

Aliases listed in `deprecated_aliases` are compatibility-only and scheduled for removal in the next major release. New deployments should use the canonical `APP_*`, `ASR_*`, `SUMMARY_*`, `PROOFREAD_*`, and `DIARIZATION_*` names.

### Guest transcription

Guest WebSocket transcription is disabled by default. Authenticated, active users are not subject to guest quotas; missing, expired, or inactive sessions do not count as authenticated.

To expose guest transcription, set `ALLOW_GUEST_TRANSCRIPTION=1` and review all of these limits:

- `GUEST_WS_MAX_PER_IP` and `GUEST_WS_MAX_CONNECTIONS`
- `GUEST_WS_MAX_DURATION_SECONDS`
- `GUEST_WS_MAX_AUDIO_BYTES`
- `GUEST_WS_MAX_ASR_REQUESTS`

Unauthenticated connections are rejected with WebSocket code `4401` when guest use is disabled. Guest limit violations use `4429` for connection limits and `4408` for duration, audio, or ASR-request limits. The UI only offers guest mode when the server reports it as enabled.

### Reverse proxy security

For a public reverse-proxy deployment, set `APP_PUBLIC_URL` to the canonical HTTPS origin and list accepted HTTP Host values in `APP_ALLOWED_HOSTS`.

Forwarded headers are ignored by default. Enable them only with both:

```env
APP_TRUST_PROXY_HEADERS=1
APP_TRUSTED_PROXY_IPS=127.0.0.1,10.0.0.0/8
```

`APP_TRUSTED_PROXY_IPS` accepts IP addresses and CIDR networks for the direct proxy peer. When the peer is not trusted, `X-Forwarded-For`, `X-Forwarded-Host`, `X-Forwarded-Proto`, and `Forwarded` do not affect client-IP rate limits, OIDC callback URLs, or Secure cookie decisions. Prefer `APP_PUBLIC_URL` for OIDC so callback construction does not depend on request headers.

### Session and costly API protection

- Session cookies contain a random token; only an HMAC-SHA256 digest is stored in the database.
- `POST /api/auth/password` changes the password, invalidates every prior session, and rotates the current cookie.
- `POST /api/auth/sessions/revoke-all` invalidates every session for the current user.
- Cross-origin state-changing `/api/*` requests are rejected when an `Origin` header is present.
- ASR, summary, and proofread limits are tracked separately per authenticated user (or guest IP) with `COSTLY_API_RATE_LIMIT_REQUESTS` and `COSTLY_API_RATE_LIMIT_WINDOW_SECONDS`.
- Administrator demotion locks current administrator rows before enforcing the last-admin rule.

Password changes, session revocation, rate-limit rejection, and protected administrator operations emit audit-friendly server logs without including secrets.

### Initial administrator

HTTP administrator bootstrap is available only in development. Production deployments must create the first administrator from a trusted shell after applying migrations:

```bash
python -m server.cli.create_user \
  --email admin@example.com \
  --display-name Administrator \
  --admin
```

The command prompts for the password when it is omitted. Do not pass production passwords on a shared command line. The database serializes the one-time browser bootstrap used in development, so concurrent requests cannot create multiple initial administrators.

### Podman (rootless)

```bash
cp .env.example .env
# Set APP_SESSION_SECRET in .env to a unique value, e.g. generated with:
python3 -c 'import secrets; print(secrets.token_hex(32))'
./podman-run.sh
```

`podman-run.sh` builds with `--format docker` by default so the image `HEALTHCHECK` is preserved. Override with `PODMAN_BUILD_FORMAT=oci` only if you intentionally want OCI output and accept that Podman will ignore the healthcheck.

Container build behavior:

- `CONTAINER_BUILD_POLICY=missing` (default): build only when the image does not exist
- `CONTAINER_BUILD_POLICY=always`: rebuild every time
- `CONTAINER_BUILD_POLICY=never`: never build, require an existing local image

Container diarization dependency behavior:

- `CONTAINER_INSTALL_DIARIZATION=0` (default): do not install `torch` / `torchaudio` / `pyannote.audio`
- `CONTAINER_INSTALL_DIARIZATION=1`: include diarization dependencies in the image

The image runs as non-root UID/GID `10001`. `/app/data` is the only persistent writable
location; when using `--read-only`, mount `/app/data` as a volume and `/tmp` as a tmpfs:

```bash
docker run --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --mount type=volume,src=whistx-data,dst=/app/data \
  -e APP_SESSION_SECRET -e ASR_API_KEY -p 8005:8005 whistx:latest
```

TLS terminates at the trusted reverse proxy. The application emits CSP, frame, MIME-sniffing,
referrer, and permissions headers; the proxy must add HSTS only on HTTPS virtual hosts.
Runtime base and tool images use explicit release tags and are rebuilt by CI. Production
promotion should record the resolved image digest and deploy that digest; dependency update
PRs are the only place where these tags are advanced.

`start.sh`と`podman-run.sh`は、ローカルのbind mountを利用するためホストUID/GIDで
コンテナを実行し、アプリ起動前に同じimageで`alembic upgrade head`を実行します。
実行UIDを固定したい場合は`CONTAINER_USER=<uid>:<gid>`で上書きできます。

## Minimal Configuration

At minimum, set these in `.env`:

```env
APP_ENV=development
APP_DB_URL=postgresql+psycopg://whistx:whistx@localhost:5432/whistx
TZ=Asia/Tokyo
ASR_API_KEY=your_api_key
APP_SESSION_SECRET=replace-with-a-long-random-secret
```

`APP_ENV=production` では SQLite は許可されません。開発時のみ `APP_DB_URL` 未設定でローカル SQLite にフォールバックします。本番は PostgreSQL を前提にしてください。

### Database migration deployment step

アプリプロセスは起動時にmigrationを実行しません。新しいimageを起動する前に、同じimageと
`APP_DB_URL`を使うinit jobで次を1回だけ実行してください。

```bash
alembic upgrade head
```

`/api/health/live`はプロセスの生存だけを返し、`/api/health/ready`はDB revisionと必須ASR
providerを検査します。revision不一致ではアプリは診断可能なまま起動し、readinessが503を返します。

migrationが失敗した場合はアプリを新revisionへ切り替えず、DBバックアップとAlembicログを保全して
ください。DDLが未適用なら原因修正後に`alembic upgrade head`を再実行します。適用済みDDLを戻す必要が
ある場合だけ、対象migrationの`downgrade()`とデータ互換性を確認したうえで
`alembic downgrade <previous-revision>`を実行します。破壊的変更は原則roll-forwardで修復します。

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
- `APP_UI_BANNERS_TEXT`
- `APP_UI_BANNERS`
- `APP_PROMPT_TEMPLATES`
- `APP_ENV`
- `APP_DB_URL`
- `HISTORY_RETENTION_DAYS` default `7`
- `RUNTIME_TRANSCRIPT_RETENTION_HOURS`
- `DEBUG_CHUNKS_RETENTION_HOURS`
- `UNSAVED_RUNTIME_RETENTION_HOURS`

`APP_UI_BANNERS_TEXT` is the recommended format when you generate `.env` from `Makefile` or `sed`.

Single banner:

```env
APP_UI_BANNERS_TEXT=warning|注意|(社外)GPUを使うので社外秘情報を入力しないでください\nよろしくお願いいたします
```

Multiple banners:

```env
APP_UI_BANNERS_TEXT=warning|注意|社外秘情報を入力しないでください;;info|補足|録音前に共有音声を確認してください
```

Format:

```text
type|title|message
type|title|message|dismissible
```

- `type`: `info`, `warning`, `success`, `error`
- Multiple banners are separated by `;;`
- Line breaks can be written as `\n`

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

### Observability

- `LANGFUSE_ENABLED`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`
- `LANGFUSE_ENVIRONMENT`
- `LANGFUSE_RELEASE`

### Keycloak

- `KEYCLOAK_ENABLED`
- `KEYCLOAK_ISSUER`
- `KEYCLOAK_CLIENT_ID`
- `KEYCLOAK_CLIENT_SECRET`
- `KEYCLOAK_SCOPE`
- `KEYCLOAK_BUTTON_LABEL`
- `KEYCLOAK_REQUIRE_EMAIL_VERIFIED`

For rootless Podman, if Langfuse runs on the host machine, use:

- `LANGFUSE_HOST=http://host.containers.internal:3000`
- `PODMAN_NETWORK=slirp4netns:allow_host_loopback=true`

Start with [`.env.example`](./.env.example). It contains common settings only; [`.env.advanced.example`](./.env.advanced.example) is the commented reference for optional overrides. Generate a unique `APP_SESSION_SECRET` before startup; the template deliberately leaves it blank.

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
CONTAINER_INSTALL_DIARIZATION=1
DIARIZATION_HF_TOKEN=hf_xxx
```

3. Restart the app

Notes:

- Diarization is applied after recording stops
- Speaker labels are patched into stored transcript records after batch processing
- When diarization is disabled, speaker-count controls are hidden in the UI

## API Endpoints

### Health

- `GET /api/health/live`: process liveness
- `GET /api/health/ready`: DB/provider readiness
- `GET /api/health`: backward-compatible readiness alias

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

## ASR evaluation

Run the reproducible smoke evaluation and compare it with the checked-in Whisper baseline:

```bash
python scripts/eval_asr.py \
  --dataset tests/fixtures/asr_eval \
  --baseline tests/fixtures/asr_eval/baselines/whisper.json \
  --json-out build/asr-eval.json \
  --csv-out build/asr-eval.csv \
  --strict
```

The report combines WER, CER, named-entity accuracy, missing/duplicate/boundary rates, speaker error rate, end-to-end and finalize latency, API request count, estimated cost, timestamp validity, and classified failure counts. Store experiment toggles in each sample's metadata to compare VAD, chunk length, overlap, context prompt, retry, and silence-drop configurations.

Whisper and Qwen3 profile baselines live under `tests/fixtures/asr_eval/baselines/`. The checked-in dataset is a synthetic CI smoke test; production model decisions must use the same consented, anonymized private audio set for every profile.

## License

MIT. See [LICENSE](./LICENSE).

## Refactor Notes

- `server/app.py` is the only ASGI entrypoint; app wiring and lifespan live in `server/core/application.py`.
- `server/runtime.py` owns realtime resource composition but registers no FastAPI routes.
- Backend route registration lives in `server/core/application.py` and `server/api/routes/`.
- Runtime configuration has one implementation under `server/core/config/`. Add new env-backed settings and registry entries there first.
- `web/main.js` is now a thin module entrypoint that loads `web/src/app.js`.
- Dependency declarations remain in `requirements*.txt`; generated `requirements*.lock` files are the reproducible install inputs. `pyproject.toml` is metadata-only for now.

### Meeting workspace access

The meeting assistant requires an authenticated account; guest access covers transcription only. The UI displays this requirement above the question field. During recording, authenticated users can ask about recognized speech without stopping. The question field and Send button remain visible while the transcript and answers scroll independently.
