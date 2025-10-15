# CUDA 12 ランタイム（Ubuntu 22.04）
# 注意: PyTorch のホイールは cuDNN を同梱しているため、
# ベースイメージに含まれる cuDNN と衝突しないよう cudnn 無しの runtime を使用します。
FROM docker.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 一時的に nvidia/cuda の APT エントリを無効化
RUN set -eux; \
  for f in /etc/apt/sources.list.d/*cuda*.list /etc/apt/sources.list.d/*nvidia*.list; do \
    [ -f "$f" ] && sed -i 's/^\s*deb/# &/' "$f"; \
  done; \
  apt-get update; \
  apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
  update-ca-certificates

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf-home \
    TORCH_HOME=/app/hf-home/torch

WORKDIR /app

# APT ミラーを https に切替（環境で 80/tcp が遮断される場合の対策）
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

# OS 依存の開発ツール（webrtcvadビルドの保険）
RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential git \
    sox libsox-dev libsndfile1 ffmpeg \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 依存インストール
COPY requirements.txt /app/requirements.txt
ENV PIP_DEFAULT_TIMEOUT=120

RUN python3 -m pip install --upgrade pip && \
    # CUDA 12.4 対応の PyTorch を公式 index から取得
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.4.0 && \
    # PyTorch 2.4 + CUDA12 系は cuDNN を pip の nvidia-cudnn-cu12 から供給するのが安定
    python3 -m pip install --no-cache-dir "nvidia-cudnn-cu12>=9.1,<9.2" && \
    # Nemo の依存解決で sox のメタデータ生成に numpy が必要になるため先に入れる
    python3 -m pip install --no-cache-dir numpy==1.26.4 && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# PyTorch 同梱ライブラリを優先解決（system の古い cuDNN を誤って拾わないための保険）
ENV PYTORCH_TORCH_LIB=/usr/local/lib/python3.10/dist-packages/torch/lib
ENV NVIDIA_CUDNN_LIB=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
ENV LD_LIBRARY_PATH=${PYTORCH_TORCH_LIB}:${NVIDIA_CUDNN_LIB}:$LD_LIBRARY_PATH

# アプリ本体
COPY server /app/server
COPY web /app/web
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ボリューム（モデル/キャッシュ、書き出し用データ）
VOLUME ["/app/hf-home", "/app/data"]

EXPOSE 8005

CMD ["/app/start.sh"]
