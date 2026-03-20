from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def env_first_non_empty(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return value
    return None


def to_int_alias(default: int, *names: str) -> int:
    raw = env_first_non_empty(*names)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def to_float_alias(default: float, *names: str) -> float:
    raw = env_first_non_empty(*names)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def to_bool_alias(default: bool, *names: str) -> bool:
    raw = env_first_non_empty(*names)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def to_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def to_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def to_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def to_bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def sanitize_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", (value or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or fallback


def decode_env_text(value: str) -> str:
    return value.replace("\\n", "\n")


def clean_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text[:limit]


def normalize_banner_type(value: Any) -> str:
    lowered = str(value or "info").strip().lower()
    if lowered in {"success", "warning", "error", "info"}:
        return lowered
    return "info"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_json(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
