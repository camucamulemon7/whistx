from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import settings


def glossary_path() -> Path:
    return settings.app_data_dir / "shared_glossary.json"


def load_shared_glossary() -> dict[str, Any]:
    path = glossary_path()
    if not path.exists():
        return {"text": "", "updatedAt": None, "updatedBy": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"text": "", "updatedAt": None, "updatedBy": None}
    if not isinstance(payload, dict):
        return {"text": "", "updatedAt": None, "updatedBy": None}
    return {
        "text": str(payload.get("text") or "").strip(),
        "updatedAt": str(payload.get("updatedAt") or "").strip() or None,
        "updatedBy": str(payload.get("updatedBy") or "").strip() or None,
    }


def save_shared_glossary(*, text: str, updated_by: str | None) -> dict[str, Any]:
    clean_text = str(text or "").strip()
    payload = {
        "text": clean_text,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "updatedBy": str(updated_by or "").strip() or None,
    }
    path = glossary_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = path.parent
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=temp_dir, delete=False, prefix=".glossary.", suffix=".json") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)
    return payload


def parse_shared_glossary_replacements(text: str) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        source = ""
        target = ""
        for delimiter in ("=>", "->", "→", "="):
            if delimiter not in line:
                continue
            left, right = line.split(delimiter, 1)
            source = left.strip()
            target = right.strip()
            break

        if not source or not target:
            continue

        aliases = [item.strip() for item in source.replace("、", ",").split(",")]
        for alias in aliases:
            if alias and alias != target:
                replacements.append((alias, target))

    replacements.sort(key=lambda item: (len(item[0]), len(item[1])), reverse=True)
    return replacements


def apply_shared_glossary_replacements(text: str, glossary_text: str) -> str:
    result = str(text or "")
    if not result:
        return result

    for source, target in parse_shared_glossary_replacements(glossary_text):
        result = result.replace(source, target)
    return result
