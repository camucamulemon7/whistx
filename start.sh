#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/container_common.sh"

load_project_env "${SCRIPT_DIR}"
resolve_container_runtime_env "${SCRIPT_DIR}"
ensure_container_prerequisites "start.sh"
build_common_container_env

case "${CONTAINER_BUILD_POLICY}" in
  always)
    echo "[start.sh] イメージを毎回ビルドします: ${CONTAINER_IMAGE_NAME}" >&2
    docker build --build-arg INSTALL_DIARIZATION="${CONTAINER_INSTALL_DIARIZATION}" -t "${CONTAINER_IMAGE_NAME}" "${SCRIPT_DIR}"
    ;;
  missing)
    if docker image inspect "${CONTAINER_IMAGE_NAME}" >/dev/null 2>&1; then
      echo "[start.sh] 既存イメージを再利用します: ${CONTAINER_IMAGE_NAME}" >&2
    else
      echo "[start.sh] イメージがないためビルドします: ${CONTAINER_IMAGE_NAME}" >&2
      docker build --build-arg INSTALL_DIARIZATION="${CONTAINER_INSTALL_DIARIZATION}" -t "${CONTAINER_IMAGE_NAME}" "${SCRIPT_DIR}"
    fi
    ;;
  never)
    if ! docker image inspect "${CONTAINER_IMAGE_NAME}" >/dev/null 2>&1; then
      echo "[start.sh] BUILD_POLICY=never ですがイメージが存在しません: ${CONTAINER_IMAGE_NAME}" >&2
      exit 1
    fi
    echo "[start.sh] 既存イメージを使用します: ${CONTAINER_IMAGE_NAME}" >&2
    ;;
  *)
    echo "[start.sh] CONTAINER_BUILD_POLICY は always / missing / never のいずれかにしてください" >&2
    exit 1
    ;;
esac

docker run --rm \
  --name "${CONTAINER_NAME}" \
  -p "${APP_PORT}:${APP_PORT}" \
  "${COMMON_CONTAINER_ENV[@]}" \
  -v "${APP_DATA_DIR}:/app/data" \
  "${CONTAINER_IMAGE_NAME}"
