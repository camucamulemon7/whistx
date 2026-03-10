#!/usr/bin/env bash
set -euo pipefail

HOST="${APP_HOST:-${HOST:-0.0.0.0}}"
PORT="${APP_PORT:-${PORT:-8005}}"
APP="${APP_ENTRYPOINT:-${APP:-server.app:app}}"

exec uvicorn "${APP}" --host "${HOST}" --port "${PORT}"
