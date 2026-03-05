#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
VENV_DIR="${DEV_VENV_DIR:-${VENV_DIR:-${SCRIPT_DIR}/.venv}}"
HOST="${APP_HOST:-${HOST:-0.0.0.0}}"
PORT="${APP_PORT:-${PORT:-8005}}"
APP="${APP_ENTRYPOINT:-${APP:-server.app:app}}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${SCRIPT_DIR}/${VENV_DIR#./}"
fi

if [[ -z "${ASR_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "[run.sh] ASR_API_KEY（または OPENAI_API_KEY）を設定してください" >&2
  exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[run.sh] Python仮想環境を作成します: ${VENV_DIR}" >&2
  rm -rf "${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ "${DEV_SKIP_PIP_INSTALL:-${SKIP_PIP_INSTALL:-0}}" != "1" ]]; then
  echo "[run.sh] 依存関係を更新します" >&2
  python -m pip install -U pip
  pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

APP_DATA_DIR="${APP_DATA_DIR:-${DATA_DIR:-${SCRIPT_DIR}/data}}"
APP_TRANSCRIPTS_DIR="${APP_TRANSCRIPTS_DIR:-${APP_DATA_DIR}/transcripts}"

if [[ "${APP_DATA_DIR}" != /* ]]; then
  APP_DATA_DIR="${SCRIPT_DIR}/${APP_DATA_DIR#./}"
fi

if [[ "${APP_TRANSCRIPTS_DIR}" != /* ]]; then
  APP_TRANSCRIPTS_DIR="${SCRIPT_DIR}/${APP_TRANSCRIPTS_DIR#./}"
fi

mkdir -p "${APP_TRANSCRIPTS_DIR}"
export APP_DATA_DIR APP_TRANSCRIPTS_DIR

exec uvicorn "${APP}" --host "${HOST}" --port "${PORT}"
