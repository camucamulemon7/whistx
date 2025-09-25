# Real-Time Transcription (Mic / Screen Share) — whistx (Parakeet‑CTC)

This is an accuracy‑first real‑time ASR prototype. The current setup uses NVIDIA NeMo's Parakeet‑CTC (Japanese) as the backend.
- Frontend: Static HTML + AudioWorklet (16 kHz / PCM16, 200 ms frames)
- Server: FastAPI WebSocket + VAD (Silero / WebRTC) + Parakeet-CTC (NeMo)
- Storage: Text only (JSONL/TXT/SRT). Audio is not persisted (memory only).

## Prerequisites
- Windows + Chrome (latest stable)
- NVIDIA GPU (CUDA environment)

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
uvicorn server.app:app --host 0.0.0.0 --port 8005
```

After startup, open:
- http://localhost:8005/  (UI)
- WS: `ws://localhost:8005/ws/transcribe`

## Usage
1) ヘッダー左の円形レコードボタンで開始/停止。音声ソースはサイドバーのセグメントトグルで切替できます。
2) 録音中はヘッダー内の波形/タイマーがアクティブになり、リアルタイム字幕は「LIVE」カード内に表示されます。
3) 確定テキストは「TRANSCRIPT」カードに蓄積され、ツールバーから `TXT` / `JSONL` / `SRT` ダウンロード、コピー、クリアが可能です。
4) ステータス（モデルロード/VAD）はヘッダーのメタ情報に集約され、ホットワード・VAD詳細設定はカード折りたたみで編集できます。

## Key Parameters
- See `server/config.py` (VAD/silence thresholds, windows, language, etc.).

## Notes
- To capture audio during screen sharing, enable "Share audio" in the share dialog.

## Logging
- Docker 実行中はモデルダウンロードの進行が `[MODEL_DOWNLOAD] {"stage": "start|progress|complete|reuse", ...}` 形式で標準出力に出力されます。
- `stage: "progress"` では `percent` フィールドが 10% 刻みで表示されるため、`docker logs -f <container>` で進行状況を確認できます。

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

Note: License names above are general. Actual conditions may differ by version or derivatives. Always follow the latest statements from the original sources.
