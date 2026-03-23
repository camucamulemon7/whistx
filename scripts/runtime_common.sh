#!/usr/bin/env bash
set -euo pipefail

load_project_env() {
  local script_dir="$1"
  local env_file="${ENV_FILE:-${script_dir}/.env}"

  # shellcheck disable=SC1091
  source "${script_dir}/scripts/load_env.sh"
  if [[ -f "${env_file}" ]]; then
    load_env_file "${env_file}"
  fi
}

resolve_path_from_root() {
  local script_dir="$1"
  local raw_path="$2"
  if [[ "${raw_path}" == /* ]]; then
    printf '%s\n' "${raw_path}"
    return
  fi
  printf '%s\n' "${script_dir}/${raw_path#./}"
}

resolve_app_runtime_env() {
  local script_dir="$1"
  APP_HOST="${APP_HOST:-${HOST:-0.0.0.0}}"
  APP_PORT="${APP_PORT:-${PORT:-8005}}"
  APP_ENTRYPOINT="${APP_ENTRYPOINT:-${APP:-server.app:app}}"
  APP_WS_PATH="${APP_WS_PATH:-${WS_PATH:-/ws/transcribe}}"
  APP_DATA_DIR="${APP_DATA_DIR:-${DATA_DIR:-${script_dir}/data}}"
  APP_DATA_DIR="$(resolve_path_from_root "${script_dir}" "${APP_DATA_DIR}")"
  APP_TRANSCRIPTS_DIR="${APP_TRANSCRIPTS_DIR:-${APP_DATA_DIR}/transcripts}"
  APP_TRANSCRIPTS_DIR="$(resolve_path_from_root "${script_dir}" "${APP_TRANSCRIPTS_DIR}")"
}

require_asr_api_key() {
  local caller="$1"
  if [[ -z "${ASR_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
    echo "[${caller}] ASR_API_KEY（または OPENAI_API_KEY）を設定してください" >&2
    exit 1
  fi
}

require_session_secret() {
  local caller="$1"
  local session_secret="$2"
  if [[ -z "${session_secret}" || ${#session_secret} -lt 32 ]]; then
    echo "[${caller}] APP_SESSION_SECRET には32文字以上のランダム値を設定してください" >&2
    exit 1
  fi
}
