#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/runtime_common.sh"

load_project_env "${SCRIPT_DIR}"
resolve_app_runtime_env "${SCRIPT_DIR}"
require_asr_api_key "run.sh"

VENV_DIR="${DEV_VENV_DIR:-${VENV_DIR:-${SCRIPT_DIR}/.venv}}"
VENV_DIR="$(resolve_path_from_root "${SCRIPT_DIR}" "${VENV_DIR}")"
PYTHON_BIN="${DEV_PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[run.sh] Python 3.12以上が必要です: ${PYTHON_BIN} が見つかりません" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "[run.sh] Python 3.12以上が必要です: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[run.sh] Python仮想環境を作成します: ${VENV_DIR}" >&2
  rm -rf "${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "[run.sh] 既存の仮想環境がPython 3.12未満です。DEV_VENV_DIRを変更するか仮想環境を再作成してください" >&2
  exit 1
fi

if [[ "${DEV_SKIP_PIP_INSTALL:-${SKIP_PIP_INSTALL:-0}}" != "1" ]]; then
  echo "[run.sh] 依存関係を更新します" >&2
  python -m pip install -U pip
  pip install -r "${SCRIPT_DIR}/requirements.txt"
  if [[ "${DIARIZATION_ENABLED:-0}" == "1" || "${INSTALL_DIARIZATION_DEPS:-0}" == "1" ]]; then
    echo "[run.sh] diarization 依存関係を追加インストールします" >&2
    pip install -r "${SCRIPT_DIR}/requirements-diarization.txt"
  fi
fi

mkdir -p "${APP_TRANSCRIPTS_DIR}"
export APP_DATA_DIR APP_TRANSCRIPTS_DIR APP_HOST APP_PORT APP_ENTRYPOINT APP_WS_PATH

exec uvicorn "${APP_ENTRYPOINT}" --host "${APP_HOST}" --port "${APP_PORT}"
