# Real-Time Transcription (Mic / Screen Share) — whistx (Parakeet‑CTC)

This is an accuracy‑first real‑time ASR prototype. The current setup uses NVIDIA NeMo's Parakeet‑CTC (Japanese) as the backend.
- Frontend: Static HTML + AudioWorklet (16 kHz / PCM16, 200 ms frames)
- Server: FastAPI WebSocket + VAD (Silero / WebRTC) + Parakeet-CTC (NeMo)
- Storage: Text only (JSONL/TXT/SRT). Audio is not persisted (memory only).

## Prerequisites
- Windows + Chrome (latest stable)
- NVIDIA A100 80 GB (CUDA environment)

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell, etc.
pip install -U pip
pip install -r requirements.txt
# For GPU usage, CUDA/cuDNN and PyTorch are required. See the Docker setup for details.
```

## Run (Local Python)
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

After startup, open:
- http://localhost:8000/  (UI)
- WS: `ws://localhost:8000/ws/transcribe`

## Usage
1) Choose "Microphone" or "Share system audio" → "Start".
2) Interim results appear in gray; finalized text is appended below.
3) After stopping, download `TXT`/`JSONL` from the link at the bottom-right.

## Key Parameters
- See `server/config.py` (VAD/silence thresholds, windows, language, etc.).

## Notes
- To capture audio during screen sharing, enable "Share audio" in the share dialog.
- English support: switch `WHISPER_LANGUAGE="en"` or extend the UI as needed.

### GPU Error Troubleshooting (libcublas.so.12 not found)
`RuntimeError: Library libcublas.so.12 is not found or cannot be loaded` occurs when the CUDA 12 runtime (cuBLAS) is not installed or detected.

1) Check GPU/WSL
```bash
nvidia-smi
ldconfig -p | grep cublas  # Check for libcublas.so.12
```

2) Install CUDA 12 Runtime (example for Ubuntu/WSL)
- Register the NVIDIA CUDA repository and install a 12.x toolkit such as `cuda-toolkit-12-4`.
- After installation, ensure `/usr/local/cuda/lib64/libcublas.so.12` exists.
```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
ldconfig -p | grep cublas
```

3) Temporary Workaround (validate on CPU)
```bash
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
(If GPU initialization fails, it automatically falls back to CPU (int8).)

## Run with Docker (CUDA‑enabled)
Prerequisites: NVIDIA driver + NVIDIA Container Toolkit installed, and `docker compose` can access the GPU.

1) Build
```bash
cd /home/remon1129/ai/whistx
docker compose build
```

2) Start (GPU enabled, port 8005)
```bash
docker compose up -d
# Tail logs
docker compose logs -f
```

3) Access
- Browser: http://localhost:8005/

4) Persistence
- Transcripts: `./data/transcripts/`
- Models/cache (Hugging Face): `./hf-home/`

Note: If Compose cannot allocate a GPU, you can run with this one‑liner:
```bash
docker build -t whistx:latest .
docker run --rm -it --gpus all -p 8005:8005 \
  -v $(pwd)/data:/app/data -v $(pwd)/hf-home:/app/hf-home \
  whistx:latest
```

## License / Credits (Commercial Use Notes)

This repository's code is released under the MIT License. See `LICENSE` for details.

This project is a sample implementation and follows the licenses of each component. For commercial use, please pay attention to credit display, distribution, and reuse conditions below. This section is a convenience summary and not legal advice. Always check the latest license/model card from the original sources.

1) Models
- ASR: NVIDIA Parakeet-CTC Japanese (`nvidia/parakeet-tdt_ctc-0.6b-ja`)
  - Sources: NVIDIA NeMo / NGC / Hugging Face, etc.
  - License: As stated in the model card (must verify). There may be requirements for attribution, allowed use (commercial or not), redistribution, and limitations of liability.
  - For commercial use: Follow the model card’s license terms and add credits/links to this README (or an in-app "Credits" page) as required.

2) Libraries / Code (examples)
- NVIDIA NeMo Toolkit (`nemo_toolkit[asr]`): Apache-2.0 (observe NOTICE retention and redistribution terms).
- PyTorch (`torch`): BSD-3-Clause.
- FastAPI / Uvicorn: MIT.
- NumPy / SciPy: BSD.
- RapidFuzz: MIT.
- webrtcvad (Python bindings): BSD-like.
- Silero-VAD (`snakers4/silero-vad`): MIT (see LICENSE in the repository).
- SpeechBrain (used for speaker embeddings options): Apache-2.0. Check each pretrained model’s model card/license.
- SoundFile (pysoundfile): BSD. Upstream `libsndfile` is LGPL-2.1+ (be mindful of dynamic linking terms).

3) Runtime (Docker / CUDA, etc.)
- NVIDIA CUDA base images (`nvidia/cuda:*`): Subject to NVIDIA Software License / NVIDIA Deep Learning Container License (pay attention to redistribution/commercial terms).
- cuDNN (`nvidia-cudnn-cu12`): NVIDIA license (conditions for redistribution/embedding apply).
- FFmpeg (from distributions): Distribution and linkage may be GPL/LGPL; if included in containers, follow the package’s license.

4) Example Credit Lines (when required)
- ASR model: “ASR model: NVIDIA Parakeet‑CTC (nvidia/parakeet‑tdt_ctc‑0.6b‑ja)”
- VAD: “VAD: Silero‑VAD (MIT, © contributors)” / “WebRTC VAD (BSD)”
- Speaker embedding: “Speaker embedding: SpeechBrain ECAPA (Apache‑2.0)”

Note: License names above are general. Actual conditions may differ by version or derivatives. Always follow the latest statements from the original sources.
