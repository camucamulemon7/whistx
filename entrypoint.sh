#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8005}"
APP="${APP:-server.app:app}"

exec uvicorn "${APP}" --host "${HOST}" --port "${PORT}"
