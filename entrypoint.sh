#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/runtime_common.sh"

resolve_app_runtime_env "${SCRIPT_DIR}"
require_session_secret "entrypoint.sh" "${APP_SESSION_SECRET:-change-me}"

mkdir -p "${APP_TRANSCRIPTS_DIR}"

echo "[entrypoint.sh] host=${APP_HOST} port=${APP_PORT} data_dir=${APP_DATA_DIR} transcripts=${APP_TRANSCRIPTS_DIR}" >&2

exec uvicorn "${APP_ENTRYPOINT}" --host "${APP_HOST}" --port "${APP_PORT}"
