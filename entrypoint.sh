#!/usr/bin/env bash
set -euo pipefail

HOST="${APP_HOST:-${HOST:-0.0.0.0}}"
PORT="${APP_PORT:-${PORT:-8005}}"
APP="${APP_ENTRYPOINT:-${APP:-server.app:app}}"
APP_DATA_DIR="${APP_DATA_DIR:-/app/data}"
APP_TRANSCRIPTS_DIR="${APP_TRANSCRIPTS_DIR:-${APP_DATA_DIR}/transcripts}"
APP_SESSION_SECRET="${APP_SESSION_SECRET:-change-me}"

mkdir -p "${APP_TRANSCRIPTS_DIR}"

echo "[entrypoint.sh] host=${HOST} port=${PORT} data_dir=${APP_DATA_DIR} transcripts=${APP_TRANSCRIPTS_DIR}" >&2

if [[ -z "${APP_SESSION_SECRET}" || ${#APP_SESSION_SECRET} -lt 32 ]]; then
  echo "[entrypoint.sh] APP_SESSION_SECRET には32文字以上のランダム値を設定してください" >&2
  exit 1
fi

exec uvicorn "${APP}" --host "${HOST}" --port "${PORT}"
