#!/usr/bin/env bash
set -euo pipefail

TLS_CERT_FILE=${TLS_CERT_FILE:-}
TLS_KEY_FILE=${TLS_KEY_FILE:-}
TLS_CA_BUNDLE=${TLS_CA_BUNDLE:-}
TLS_PORT=${TLS_PORT:-8443}
HTTP_PORT=${HTTP_PORT:-8005}
TLS_ENABLED=false
if [[ -n "${TLS_CERT_FILE}" || -n "${TLS_KEY_FILE}" ]]; then
  if [[ -z "${TLS_CERT_FILE}" || -z "${TLS_KEY_FILE}" ]]; then
    echo "[start.sh] TLS_CERT_FILE と TLS_KEY_FILE が揃っていません" >&2
    exit 1
  fi
  if [[ ! -f "${TLS_CERT_FILE}" ]]; then
    echo "[start.sh] TLS_CERT_FILE='${TLS_CERT_FILE}' が見つかりません" >&2
    exit 1
  fi
  if [[ ! -f "${TLS_KEY_FILE}" ]]; then
    echo "[start.sh] TLS_KEY_FILE='${TLS_KEY_FILE}' が見つかりません" >&2
    exit 1
  fi
  if [[ -n "${TLS_CA_BUNDLE}" && ! -f "${TLS_CA_BUNDLE}" ]]; then
    echo "[start.sh] TLS_CA_BUNDLE='${TLS_CA_BUNDLE}' が見つかりません" >&2
    exit 1
  fi
  TLS_ENABLED=true
fi

# Torch / NVIDIA pip 配布ライブラリを優先（cuDNN を確実に解決）
PY_TORCH_LIB="$(python3 - <<'PY'
import os
try:
    import torch, os.path
    p = os.path.join(os.path.dirname(torch.__file__), 'lib')
    print(p)
except Exception:
    print("")
PY
)"
if [[ -n "${PY_TORCH_LIB}" && -d "${PY_TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${PY_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

PY_CUDNN_LIB="$(python3 - <<'PY'
import os
try:
    import nvidia.cudnn, os.path
    base = os.path.dirname(nvidia.cudnn.__file__)
    lib = os.path.join(base, 'lib')
    print(lib)
except Exception:
    print('')
PY
)"
if [[ -n "${PY_CUDNN_LIB}" && -d "${PY_CUDNN_LIB}" ]]; then
  export LD_LIBRARY_PATH="${PY_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

UVICORN_PORT=${HTTP_PORT}
declare -a UVICORN_ARGS
if [[ "${TLS_ENABLED}" == true ]]; then
  UVICORN_PORT=${TLS_PORT}
  echo "[start.sh] Starting uvicorn with HTTPS on port ${UVICORN_PORT}" >&2
  UVICORN_ARGS+=(--ssl-certfile "${TLS_CERT_FILE}" --ssl-keyfile "${TLS_KEY_FILE}")
  if [[ -n "${TLS_CA_BUNDLE}" ]]; then
    UVICORN_ARGS+=(--ssl-ca-certs "${TLS_CA_BUNDLE}")
  fi
else
  echo "[start.sh] Starting uvicorn with HTTP on port ${UVICORN_PORT}" >&2
fi

exec uvicorn server.app:app --host 0.0.0.0 --port "${UVICORN_PORT}" "${UVICORN_ARGS[@]}"
