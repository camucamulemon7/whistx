#!/usr/bin/env bash
set -euo pipefail

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

exec uvicorn server.app:app --host 0.0.0.0 --port 8005
