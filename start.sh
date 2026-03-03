#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
IMAGE_NAME="${IMAGE_NAME:-whistx:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-whistx}"
PORT="${PORT:-8005}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${SCRIPT_DIR}/${DATA_DIR#./}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[start.sh] OPENAI_API_KEY を設定してください" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}/transcripts"

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8005" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  -e WHISPER_MODEL="${WHISPER_MODEL:-whisper-1}" \
  -e SUMMARY_API_KEY="${SUMMARY_API_KEY:-}" \
  -e SUMMARY_BASE_URL="${SUMMARY_BASE_URL:-}" \
  -e SUMMARY_MODEL="${SUMMARY_MODEL:-gpt-4o-mini}" \
  -e SUMMARY_TEMPERATURE="${SUMMARY_TEMPERATURE:-0.2}" \
  -e SUMMARY_INPUT_MAX_CHARS="${SUMMARY_INPUT_MAX_CHARS:-16000}" \
  -e DEFAULT_LANGUAGE="${DEFAULT_LANGUAGE:-ja}" \
  -e DEFAULT_PROMPT="${DEFAULT_PROMPT:-}" \
  -e DEFAULT_TEMPERATURE="${DEFAULT_TEMPERATURE:-0.0}" \
  -e CONTEXT_PROMPT_ENABLED="${CONTEXT_PROMPT_ENABLED:-1}" \
  -e CONTEXT_MAX_CHARS="${CONTEXT_MAX_CHARS:-1000}" \
  -e MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-8}" \
  -e MAX_CHUNK_BYTES="${MAX_CHUNK_BYTES:-12582912}" \
  -v "${DATA_DIR}:/app/data" \
  "${IMAGE_NAME}"
