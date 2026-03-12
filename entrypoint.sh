#!/usr/bin/env bash
set -euo pipefail

HOST="${APP_HOST:-${HOST:-0.0.0.0}}"
PORT="${APP_PORT:-${PORT:-8005}}"
APP="${APP_ENTRYPOINT:-${APP:-server.app:app}}"
APP_DATA_DIR="${APP_DATA_DIR:-/app/data}"
APP_TRANSCRIPTS_DIR="${APP_TRANSCRIPTS_DIR:-${APP_DATA_DIR}/transcripts}"

mkdir -p "${APP_TRANSCRIPTS_DIR}"

exec uvicorn "${APP}" --host "${HOST}" --port "${PORT}"
