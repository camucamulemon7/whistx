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
