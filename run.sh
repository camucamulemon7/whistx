#!/usr/bin/env bash
# Launch whistx container with GPU using Docker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME=${IMAGE_NAME:-whistx:latest}
CONTAINER_NAME=${CONTAINER_NAME:-whistx}
DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/data"}
HF_HOME_DIR=${HF_HOME_DIR:-"${SCRIPT_DIR}/hf-home"}

# TLS configuration (host paths can be overridden via environment variables)
TLS_CERT_HOST_PATH=${TLS_CERT_HOST_PATH:-"${SCRIPT_DIR}/certs/server.crt"}
TLS_KEY_HOST_PATH=${TLS_KEY_HOST_PATH:-"${SCRIPT_DIR}/certs/server.key"}
TLS_CHAIN_HOST_PATH=${TLS_CHAIN_HOST_PATH:-"${SCRIPT_DIR}/certs/chain.crt"}
TLS_CONTAINER_CERT_PATH=${TLS_CONTAINER_CERT_PATH:-/app/certs/server.crt}
TLS_CONTAINER_KEY_PATH=${TLS_CONTAINER_KEY_PATH:-/app/certs/server.key}
TLS_CONTAINER_CHAIN_PATH=${TLS_CONTAINER_CHAIN_PATH:-/app/certs/chain.crt}
TLS_PORT=${TLS_PORT:-8443}
HTTP_PORT=${HTTP_PORT:-8005}

# Determine TLS activation based on actual file presence
TLS_ENABLED=false
if [[ -f "${TLS_CERT_HOST_PATH}" && -f "${TLS_KEY_HOST_PATH}" ]]; then
  TLS_ENABLED=true
fi

DOCKER_BUILD_ARGS=(build -t "${IMAGE_NAME}" .)
docker "${DOCKER_BUILD_ARGS[@]}"

DOCKER_RUN_ARGS=(
  run
  --rm
  --name "${CONTAINER_NAME}"
  --device nvidia.com/gpu=all
  --security-opt label=disable
  --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
  --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
  --env ASR_BACKEND=${ASR_BACKEND:-parakeet}
  --env ASR_DEVICE=${ASR_DEVICE:-cuda}
  --env ASR_LANGUAGE=${ASR_LANGUAGE:-ja}
  --env PARAKEET_MODEL_ID=${PARAKEET_MODEL_ID:-nvidia/parakeet-tdt_ctc-0.6b-ja}
  --env VAD_BACKEND=${VAD_BACKEND:-silero}
  --env SILERO_THRESHOLD=${SILERO_THRESHOLD:-0.5}
  --env HF_HOME=/app/hf-home
  --env TORCH_HOME=/app/hf-home/torch
  --env HTTP_PORT=${HTTP_PORT}
  --volume "${DATA_DIR}:/app/data"
  --volume "${HF_HOME_DIR}:/app/hf-home"
)

if [[ "${TLS_ENABLED}" == true ]]; then
  DOCKER_RUN_ARGS+=(
    --publish "${TLS_PORT}:${TLS_PORT}"
    --env TLS_CERT_FILE="${TLS_CONTAINER_CERT_PATH}"
    --env TLS_KEY_FILE="${TLS_CONTAINER_KEY_PATH}"
    --env TLS_PORT="${TLS_PORT}"
  )
  DOCKER_RUN_ARGS+=(--volume "${TLS_CERT_HOST_PATH}:${TLS_CONTAINER_CERT_PATH}:ro")
  DOCKER_RUN_ARGS+=(--volume "${TLS_KEY_HOST_PATH}:${TLS_CONTAINER_KEY_PATH}:ro")
  if [[ -f "${TLS_CHAIN_HOST_PATH}" ]]; then
    DOCKER_RUN_ARGS+=(
      --volume "${TLS_CHAIN_HOST_PATH}:${TLS_CONTAINER_CHAIN_PATH}:ro"
      --env TLS_CA_BUNDLE="${TLS_CONTAINER_CHAIN_PATH}"
    )
  fi
else
  DOCKER_RUN_ARGS+=(--publish "${HTTP_PORT}:${HTTP_PORT}")
fi

DOCKER_RUN_ARGS+=("${IMAGE_NAME}" /app/start.sh)

docker "${DOCKER_RUN_ARGS[@]}"
