"""Authorized, versioned meeting snapshots shared by recap and the assistant.

The existing runtime session is the meeting identity before it is saved to
history. Insights live next to its transcript and travel with history artifacts.
No transcript or media URL supplied by a client is trusted as a source.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..core.config import settings
from ..core.security import runtime_access_allowed
from ..db import db_session
from ..repositories.history_repository import get_history_for_user
from ..transcript_store import read_jsonl_records, resolve_transcript_path

SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class MeetingError(Exception):
    def __init__(self, code: str, status_code: int = 400):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Readers see either the previous complete snapshot or the new one."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, separators=(",", ":"))
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@dataclass(frozen=True)
class MeetingSnapshot:
    key: str
    transcript_path: Path
    insight_path: Path
    segments: list[dict[str, Any]]
    images: list[dict[str, Any]]
    revision: str
    through_ms: int
    finalized: bool

    def public(self) -> dict[str, Any]:
        return {
            "sourceKey": self.key,
            "revision": self.revision,
            "throughMs": self.through_ms,
            "finalized": self.finalized,
            "segmentCount": len(self.segments),
            "images": self.images,
        }


def _filename(raw: Any) -> str:
    name = Path(urlsplit(str(raw or "")).path).name
    return name if SAFE_FILENAME.fullmatch(name) else ""


def _integer(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def load_meeting(*, user_id: int, runtime_session_id: str = "", history_id: str = "") -> MeetingSnapshot:
    if bool(runtime_session_id) == bool(history_id):
        raise MeetingError("one_meeting_source_required")
    live: dict[str, Any] = {}
    if history_id:
        with db_session() as db:
            history = get_history_for_user(db, user_id=user_id, history_id=history_id)
            if history is None or not history.jsonl_path:
                raise MeetingError("meeting_not_found", 404)
            path = (settings.history_dir / history.jsonl_path).resolve()
            if not path.is_relative_to(settings.history_dir.resolve()) or not path.is_file():
                raise MeetingError("meeting_not_found", 404)
        key = f"history:{history_id}"
        media_prefix = f"/api/history/{history_id}"
        finalized = True
    else:
        if not runtime_access_allowed(
            transcripts_dir=settings.transcripts_dir,
            session_id=runtime_session_id,
            user_id=user_id,
            guest_grant_id=None,
        ):
            raise MeetingError("meeting_not_found", 404)
        path = resolve_transcript_path(settings.transcripts_dir, runtime_session_id, "jsonl")
        if path is None or not path.is_file():
            raise MeetingError("meeting_not_found", 404)
        live = read_json(path.with_suffix(".live.json"))
        finalized = bool(read_json(path.with_suffix(".meta.json")).get("finalized"))
        key = f"runtime:{runtime_session_id}"
        media_prefix = f"/api/transcripts/{runtime_session_id}"

    records = read_jsonl_records(path)
    # A growing, agreed prefix is available to QA while its utterance is still
    # open. Never use the unstable hypothesis as evidence.
    for stable in live.get("stableSegments", []):
        if isinstance(stable, dict) and stable.get("text"):
            records = [*records, {**stable, "type": "final"}]

    segments: dict[str, dict[str, Any]] = {}
    images: dict[str, dict[str, Any]] = {}
    for row in records:
        seq = _integer(row.get("seq"))
        start = _integer(row.get("tsStart", row.get("offsetMs")))
        end = max(start, _integer(row.get("tsEnd", start)))
        image_name = _filename(row.get("screenshotPath"))
        if image_name:
            images.setdefault(image_name, {
                "id": image_name,
                "timeMs": start,
                "url": f"{media_prefix}/screenshots/{image_name}",
            })
        if row.get("type") != "final" or not str(row.get("text") or "").strip():
            continue
        segment_id = str(row.get("segmentId") or f"{seq:06d}")
        audio = _filename(row.get("rawAudioPath") or row.get("audioPath"))
        segments[segment_id] = {
            "id": segment_id,
            "seq": seq,
            "text": str(row["text"]).strip(),
            "startMs": start,
            "endMs": end,
            "speaker": str(row.get("speaker") or ""),
            "imageId": image_name or None,
            "audioUrl": f"{media_prefix}/audio/{audio}" if audio else None,
        }
    rows = sorted(segments.values(), key=lambda row: (row["startMs"], row["seq"]))
    image_rows = sorted(images.values(), key=lambda row: row["timeMs"])
    # URLs change when saved to history; source identity does not.
    canonical = [{k: row[k] for k in ("id", "text", "startMs", "endMs", "speaker")} for row in rows]
    digest = hashlib.sha256(json.dumps([canonical, [(i["id"], i["timeMs"]) for i in image_rows]], ensure_ascii=False).encode()).hexdigest()[:24]
    return MeetingSnapshot(
        key=key,
        transcript_path=path,
        insight_path=path.with_suffix(".meeting.json"),
        segments=rows,
        images=image_rows,
        revision=digest,
        through_ms=max((row["endMs"] for row in rows), default=0),
        finalized=finalized,
    )
