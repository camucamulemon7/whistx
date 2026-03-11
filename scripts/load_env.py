#!/usr/bin/env python3
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path


LINE_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def parse_env_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        pairs.append((key, value))
    return pairs


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: load_env.py <env-file>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        return 0
    for key, value in parse_env_file(path):
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
