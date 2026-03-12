#!/usr/bin/env bash
set -euo pipefail

trim_whitespace() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

load_env_file() {
  local env_file="${1:-}"
  [[ -n "${env_file}" && -f "${env_file}" ]] || return 0

  local raw line key value quote
  while IFS= read -r raw || [[ -n "${raw}" ]]; do
    line="$(trim_whitespace "${raw}")"
    [[ -z "${line}" || "${line:0:1}" == "#" ]] && continue

    if [[ "${line}" =~ ^(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      key="${BASH_REMATCH[2]}"
      value="$(trim_whitespace "${BASH_REMATCH[3]}")"

      if [[ ${#value} -ge 2 ]]; then
        quote="${value:0:1}"
        if [[ "${quote}" == "'" || "${quote}" == '"' ]] && [[ "${value: -1}" == "${quote}" ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi

      printf -v "${key}" '%s' "${value}"
      export "${key}"
    fi
  done < "${env_file}"
}
