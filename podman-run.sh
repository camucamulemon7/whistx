#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/container_common.sh"

load_project_env "${SCRIPT_DIR}"
resolve_container_runtime_env "${SCRIPT_DIR}"
ensure_container_prerequisites "podman-run.sh"
# Bridge networking needs the host gateway for locally hosted API services.
if [[ "${PODMAN_NETWORK:-slirp4netns:allow_host_loopback=true}" != "host" ]]; then
  for endpoint_key in ASR_BASE_URL SUMMARY_BASE_URL PROOFREAD_BASE_URL LANGFUSE_HOST; do
    endpoint_value="${!endpoint_key:-}"
    endpoint_value="${endpoint_value/#http:\/\/localhost:/http:\/\/host.containers.internal:}"
    endpoint_value="${endpoint_value/#http:\/\/127.0.0.1:/http:\/\/host.containers.internal:}"
    printf -v "${endpoint_key}" '%s' "${endpoint_value}"
  done
fi
build_common_container_env

PODMAN_USERNS="${PODMAN_USERNS:-keep-id}"
PODMAN_VOLUME_OPTS="${PODMAN_VOLUME_OPTS:-Z}"
PODMAN_NETWORK="${PODMAN_NETWORK:-slirp4netns:allow_host_loopback=true}"
PODMAN_BUILD_FORMAT="${PODMAN_BUILD_FORMAT:-docker}"

podman_cmd=(podman)
if [[ -n "${PODMAN_CGROUP_MANAGER:-}" ]]; then
  podman_cmd+=(--cgroup-manager "${PODMAN_CGROUP_MANAGER}")
fi

case "${CONTAINER_BUILD_POLICY}" in
  always)
    echo "[podman-run.sh] イメージを毎回ビルドします: ${CONTAINER_IMAGE_NAME}" >&2
    "${podman_cmd[@]}" build --format "${PODMAN_BUILD_FORMAT}" --build-arg INSTALL_DIARIZATION="${CONTAINER_INSTALL_DIARIZATION}" -t "${CONTAINER_IMAGE_NAME}" "${SCRIPT_DIR}"
    ;;
  missing)
    if "${podman_cmd[@]}" image exists "${CONTAINER_IMAGE_NAME}"; then
      echo "[podman-run.sh] 既存イメージを再利用します: ${CONTAINER_IMAGE_NAME}" >&2
    else
      echo "[podman-run.sh] イメージがないためビルドします: ${CONTAINER_IMAGE_NAME}" >&2
      "${podman_cmd[@]}" build --format "${PODMAN_BUILD_FORMAT}" --build-arg INSTALL_DIARIZATION="${CONTAINER_INSTALL_DIARIZATION}" -t "${CONTAINER_IMAGE_NAME}" "${SCRIPT_DIR}"
    fi
    ;;
  never)
    if ! "${podman_cmd[@]}" image exists "${CONTAINER_IMAGE_NAME}"; then
      echo "[podman-run.sh] BUILD_POLICY=never ですがイメージが存在しません: ${CONTAINER_IMAGE_NAME}" >&2
      exit 1
    fi
    echo "[podman-run.sh] 既存イメージを使用します: ${CONTAINER_IMAGE_NAME}" >&2
    ;;
  *)
    echo "[podman-run.sh] CONTAINER_BUILD_POLICY は always / missing / never のいずれかにしてください" >&2
    exit 1
    ;;
esac

volume_spec="${APP_DATA_DIR}:/app/data"
if [[ -n "${PODMAN_VOLUME_OPTS}" ]]; then
  volume_spec="${volume_spec}:${PODMAN_VOLUME_OPTS}"
fi

run_args=(
  --rm
  --name "${CONTAINER_NAME}"
  --user "${CONTAINER_USER}"
  -p "${APP_PORT}:${APP_PORT}"
)

if [[ -n "${PODMAN_USERNS}" ]]; then
  run_args+=(--userns "${PODMAN_USERNS}")
fi

if [[ -n "${PODMAN_NETWORK}" ]]; then
  run_args+=(--network "${PODMAN_NETWORK}")
fi

run_args+=(
  "${COMMON_CONTAINER_ENV[@]}"
  -v "${volume_spec}"
)

echo "[podman-run.sh] DB migrationを実行します" >&2
migration_args=(
  --rm
  --user "${CONTAINER_USER}"
)
if [[ -n "${PODMAN_USERNS}" ]]; then
  migration_args+=(--userns "${PODMAN_USERNS}")
fi
if [[ -n "${PODMAN_NETWORK}" ]]; then
  migration_args+=(--network "${PODMAN_NETWORK}")
fi
migration_args+=(
  "${COMMON_CONTAINER_ENV[@]}"
  -v "${volume_spec}"
  "${CONTAINER_IMAGE_NAME}"
  alembic upgrade head
)
"${podman_cmd[@]}" run "${migration_args[@]}"

"${podman_cmd[@]}" run "${run_args[@]}" "${CONTAINER_IMAGE_NAME}"
