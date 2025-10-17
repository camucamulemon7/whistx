# whistx — Real-Time Transcription for Japanese Speech

whistx is a real-time transcription demo powered by NVIDIA NeMo’s Parakeet-CTC (Japanese) model. The browser captures audio, the server performs GPU inference, and results stream back as live captions plus finalized transcripts.

## Features
- **Browser client**: static HTML + AudioWorklet (16 kHz / PCM16, 200 ms frames)
- **Server**: FastAPI WebSocket pipeline with Silero/WebRTC VAD feeding Parakeet-CTC inference
- **Outputs**: text transcripts in JSONL, TXT, or SRT; audio buffers remain in memory only

## Requirements
- Windows + Chrome (current stable) for the UI
- NVIDIA GPU with current driver + CUDA runtime (required for the container workflow)

## Local Python Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # adapt for your shell
pip install -U pip
pip install -r requirements.txt
# For GPU use, ensure CUDA/cuDNN and the matching PyTorch build are installed beforehand.
```

### Run Locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8005
```
Open http://localhost:8005/ and connect to `ws://localhost:8005/ws/transcribe` for WebSocket streaming.

## Usage Notes
1. Start/stop capture with the header record button. Switch audio inputs via the sidebar toggles.
2. The header shows a waveform and timer during recording; the LIVE card displays partial captions.
3. Confirmed text accrues in the TRANSCRIPT card with export/copy/clear actions (`TXT` / `JSONL` / `SRT`).
4. Model/VAD status indicators live in the header; hotword and advanced VAD controls reside in collapsible panels.
5. Enable “Share audio” in your browser when capturing system audio.

Runtime parameters (VAD thresholds, language, etc.) are managed in `server/config.py`.

### English (en) Transcription
To enable the English Parakeet model, set the following environment variables before starting the server/container:

```bash
export PARAKEET_MODEL_ID_EN="nvidia/parakeet-ctc-1.1b-en"   # or your preferred model
export HF_AUTH_TOKEN="<your-huggingface-token>"
```

`HF_AUTH_TOKEN` is forwarded to the Hugging Face Hub so that private or gated NVIDIA models can be downloaded once and re-used from `./hf-home`. If the Parakeet model cannot be downloaded, the server automatically falls back to the Whisper backend so that transcription still works (albeit with different accuracy/performance characteristics).

## Logging
- Model downloads emit `[MODEL_DOWNLOAD] {"stage": "start|progress|complete|reuse", ...}`.
- During download, `stage: "progress"` adds a `percent` field in 10% steps. Tail logs with `podman logs -f <container>` to monitor progress.

## Container Workflow with Podman
Prerequisites: Podman with NVIDIA Container Toolkit (CDI) configured so `podman run --device nvidia.com/gpu=all` exposes the GPU.

1. Build the image
   ```bash
   podman build -t whistx:latest .
   ```
2. Start the container with GPU + HTTPS aware script
   ```bash
   ./podman-run.sh
   ```
   The script mounts `./data` and `./hf-home`, forwards port 8005 by default, and switches to HTTPS automatically when certificates are provided (see below).
3. Open http://localhost:8005/ (or the HTTPS port if enabled).

If you prefer a one-off command instead of the helper script:
```bash
podman run --rm -it \
  --device nvidia.com/gpu=all \
  --publish 8005:8005 \
  --security-opt label=disable \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/hf-home:/app/hf-home \
  whistx:latest
```

### Enabling HTTPS with Internal Certificates
1. Place PEM-encoded `server.crt`, `server.key`, and (optionally) `chain.crt` on the host; `run.sh` and `podman-run.sh` default to `./certs/`.
2. Override paths or the HTTPS port as needed:
   ```bash
   export TLS_CERT_HOST_PATH="/path/to/server.crt"
   export TLS_KEY_HOST_PATH="/path/to/server.key"
   export TLS_CHAIN_HOST_PATH="/path/to/chain.crt"  # optional
   export TLS_PORT=8443                              # optional
   ```
3. Launch `./podman-run.sh`. When certificate and key exist, the container serves `https://localhost:${TLS_PORT:-8443}` and upgrades WebSocket traffic to `wss://`. Without them, it falls back to HTTP on port 8005.
4. Reload the UI over HTTPS and confirm the browser reports a secure WebSocket handshake (`wss://`).
5. Rotate certificates by replacing the PEM files and rerunning the script.

## Licensing
The code is provided under the MIT License (see `LICENSE`). Review upstream licenses before repackaging or commercial use.

### Model
- NVIDIA Parakeet-CTC Japanese (`nvidia/parakeet-tdt_ctc-0.6b-ja`)

### Runtime Dependencies
- NVIDIA CUDA base image (`nvidia/cuda:12.4.1-runtime-ubuntu22.04`)
- PyTorch 2.4 with CUDA 12 wheels
- NVIDIA cuDNN (`nvidia-cudnn-cu12`)
- Silero-VAD (`snakers4/silero-vad`)
- webrtcvad (Python bindings)
- FastAPI / Uvicorn
- NumPy / SciPy
- RapidFuzz
- SoundFile (`pysoundfile`)

Consult each project’s documentation for detailed licensing terms.
