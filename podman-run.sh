#!/usr/bin/env bash
# Launch whistx container with GPU using Podman.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME=${IMAGE_NAME:-whistx:latest}
CONTAINER_NAME=${CONTAINER_NAME:-whistx}
DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/data"}
HF_HOME_DIR=${HF_HOME_DIR:-"${SCRIPT_DIR}/hf-home"}

podman build -t whistx:latest .
podman run \
  --rm \
  --name "${CONTAINER_NAME}" \
  --publish 8005:8005 \
  --device nvidia.com/gpu=all \
  --security-opt label=disable \
  --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
  --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility} \
  --env ASR_BACKEND=${ASR_BACKEND:-parakeet} \
  --env ASR_DEVICE=${ASR_DEVICE:-cuda} \
  --env ASR_LANGUAGE=${ASR_LANGUAGE:-ja} \
  --env PARAKEET_MODEL_ID=${PARAKEET_MODEL_ID:-nvidia/parakeet-tdt_ctc-0.6b-ja} \
  --env VAD_BACKEND=${VAD_BACKEND:-silero} \
  --env SILERO_THRESHOLD=${SILERO_THRESHOLD:-0.5} \
  --env HF_HOME=/app/hf-home \
  --env TORCH_HOME=/app/hf-home/torch \
  --volume "${DATA_DIR}:/app/data" \
  --volume "${HF_HOME_DIR}:/app/hf-home" \
  "${IMAGE_NAME}" \
  /app/start.sh
