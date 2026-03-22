from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,95})$")


@dataclass(slots=True)
class TranscriptRecord:
    type: str
    segmentId: str
    seq: int
    text: str
    tsStart: int
    tsEnd: int
    chunkOffsetMs: int
    chunkDurationMs: int
    language: str
    createdAt: str
    speaker: str | None = None
    screenshotPath: str | None = None
    rawAudioPath: str | None = None
    audioPath: str | None = None


class TranscriptStore:
    def __init__(self, root_dir: Path, session_id: str):
        self.root_dir = root_dir
        self.session_id = session_id
        self.base_path = build_runtime_transcript_base_path(root_dir, session_id)
        self.jsonl_path = self.base_path.with_suffix(".jsonl")
        self.txt_path = self.base_path.with_suffix(".txt")
        self.metadata_path = self.base_path.with_suffix(".meta.json")
        self.chunks_dir = self.root_dir / "_chunks" / session_id
        self.screenshots_dir = build_runtime_screenshots_dir(self.root_dir, session_id)

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)
        self.txt_path.touch(exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, payload: dict) -> None:
        self.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        try:
            parsed = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def is_finalized(self) -> bool:
        return is_runtime_transcript_finalized(
            self.read_metadata(),
            txt_path=self.txt_path,
            jsonl_path=self.jsonl_path,
            metadata_path=self.metadata_path,
        )

    def append_final(self, record: TranscriptRecord) -> None:
        payload = asdict(record)

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        with self.txt_path.open("a", encoding="utf-8") as handle:
            line = _render_txt_line(payload)
            if line:
                handle.write(line + "\n")

    def save_audio_chunk(self, *, seq: int, mime_type: str, audio_bytes: bytes) -> Path:
        ext = _ext_from_mime(mime_type)
        path = self.chunks_dir / f"{seq:06d}{ext}"
        path.write_bytes(audio_bytes)
        return path

    def save_screenshot(self, *, seq: int, mime_type: str, image_bytes: bytes) -> str:
        ext = _image_ext_from_mime(mime_type)
        filename = f"{seq:06d}{ext}"
        path = self.screenshots_dir / filename
        path.write_bytes(image_bytes)
        return filename

    def rewrite_records(self, records: Iterable[dict]) -> None:
        rows = [row for row in records if isinstance(row, dict)]
        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        with self.txt_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if row.get("type") != "final":
                    continue
                line = _render_txt_line(row)
                if line:
                    handle.write(line + "\n")

    def cleanup_chunks(self) -> None:
        shutil.rmtree(self.chunks_dir, ignore_errors=True)

    @staticmethod
    def make_runtime_session_id(base_session_id: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        token = uuid.uuid4().hex[:4]
        return f"{base_session_id}_{stamp}_{token}"

    @staticmethod
    def sanitize_or_generate(raw: str | None) -> str:
        value = (raw or "").strip()
        if SESSION_ID_RE.fullmatch(value):
            return value
        return f"sess-{uuid.uuid4().hex[:12]}"



def resolve_transcript_path(root_dir: Path, session_id: str, ext: str) -> Path | None:
    if not SESSION_ID_RE.fullmatch(session_id):
        return None

    safe_ext = ext.lower().strip(".")
    if safe_ext not in {"txt", "jsonl"}:
        return None

    for path in _candidate_transcript_paths(root_dir, session_id, safe_ext):
        if _is_path_inside(root_dir, path):
            return path
    return None


def resolve_screenshot_path(root_dir: Path, session_id: str, filename: str) -> Path | None:
    if not SESSION_ID_RE.fullmatch(session_id):
        return None
    if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", filename or ""):
        return None

    for path in _candidate_screenshot_paths(root_dir, session_id, filename):
        if _is_path_inside(root_dir / "_screenshots", path):
            return path
    return None


def resolve_debug_audio_path(root_dir: Path, session_id: str, filename: str) -> Path | None:
    if not SESSION_ID_RE.fullmatch(session_id):
        return None
    if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", filename or ""):
        return None

    for path in _candidate_debug_audio_paths(root_dir, session_id, filename):
        if _is_path_inside(root_dir, path):
            return path
    return None


def build_runtime_transcript_base_path(root_dir: Path, session_id: str, *, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return root_dir / stamp.strftime("%Y") / stamp.strftime("%m") / stamp.strftime("%d") / session_id


def build_runtime_screenshots_dir(root_dir: Path, session_id: str, *, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return (
        root_dir
        / "_screenshots"
        / stamp.strftime("%Y")
        / stamp.strftime("%m")
        / stamp.strftime("%d")
        / session_id
    )


def build_debug_chunks_dir(root_dir: Path, session_id: str, *, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return root_dir / stamp.strftime("%Y") / stamp.strftime("%m") / stamp.strftime("%d") / session_id


def iter_runtime_transcript_paths(root_dir: Path, session_id: str) -> list[Path]:
    return _candidate_transcript_paths(root_dir, session_id, "txt")


def iter_runtime_screenshot_dirs(root_dir: Path, session_id: str) -> list[Path]:
    screenshot_root = root_dir / "_screenshots"
    matches = sorted(screenshot_root.glob(f"*/" f"*/" f"*/{session_id}"), reverse=True)
    return [*matches, screenshot_root / session_id]


def iter_debug_chunk_dirs(root_dir: Path, session_id: str) -> list[Path]:
    matches = sorted(root_dir.glob(f"*/" f"*/" f"*/{session_id}"), reverse=True)
    return [*matches, root_dir / session_id]



def read_jsonl_records(path: Path, *, strict: bool = False) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(f"invalid_jsonl_line:{exc.lineno}") from exc
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def is_runtime_transcript_finalized(
    metadata: dict,
    *,
    txt_path: Path,
    jsonl_path: Path,
    metadata_path: Path | None = None,
) -> bool:
    explicit = _read_finalized_flag(metadata)
    if explicit is not None:
        return explicit

    if not txt_path.exists() or not jsonl_path.exists():
        return False

    try:
        txt_stat_before = txt_path.stat()
        jsonl_stat_before = jsonl_path.stat()
        metadata_stat_before = metadata_path.stat() if metadata_path and metadata_path.exists() else None
        txt_path.read_text(encoding="utf-8")
        jsonl_path.read_text(encoding="utf-8")
        txt_stat_after = txt_path.stat()
        jsonl_stat_after = jsonl_path.stat()
    except OSError:
        return False

    if (
        txt_stat_before.st_size != txt_stat_after.st_size
        or jsonl_stat_before.st_size != jsonl_stat_after.st_size
        or txt_stat_before.st_mtime != txt_stat_after.st_mtime
        or jsonl_stat_before.st_mtime != jsonl_stat_after.st_mtime
    ):
        return False

    newest_mtime = max(txt_stat_after.st_mtime, jsonl_stat_after.st_mtime)
    if metadata_stat_before is not None:
        newest_mtime = max(newest_mtime, metadata_stat_before.st_mtime)

    age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - newest_mtime)
    return age_seconds >= 2.0


def _render_txt_line(rec: dict) -> str:
    text = str(rec.get("text", "")).strip()
    if not text:
        return ""
    speaker = str(rec.get("speaker", "")).strip()
    if not speaker:
        return text
    return f"[{speaker}] {text}"


def _read_finalized_flag(metadata: dict) -> bool | None:
    if not isinstance(metadata, dict):
        return None

    for key in ("finalized", "isFinalized", "finalizedAt", "finishedAt", "stoppedAt", "endedAt", "closedAt"):
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if value is not None:
            return True

    status = metadata.get("status")
    if isinstance(status, str):
        lowered = status.strip().lower()
        if lowered in {"finalized", "finished", "complete", "completed", "stopped", "closed", "done"}:
            return True
        if lowered in {"active", "running", "recording", "pending", "in_progress", "in-progress", "open"}:
            return False

    return None


def _ext_from_mime(mime_type: str) -> str:
    lowered = (mime_type or "").lower()
    if "wav" in lowered:
        return ".wav"
    if "webm" in lowered:
        return ".webm"
    if "ogg" in lowered or "opus" in lowered:
        return ".ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return ".m4a"
    if "mpeg" in lowered or "mp3" in lowered:
        return ".mp3"
    return ".bin"


def _image_ext_from_mime(mime_type: str) -> str:
    lowered = (mime_type or "").lower()
    if "webp" in lowered:
        return ".webp"
    if "png" in lowered:
        return ".png"
    if "jpeg" in lowered or "jpg" in lowered:
        return ".jpg"
    return ".bin"



def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _candidate_transcript_paths(root_dir: Path, session_id: str, ext: str) -> list[Path]:
    matches = sorted(root_dir.glob(f"*/" f"*/" f"*/{session_id}.{ext}"), reverse=True)
    return [*matches, (root_dir / session_id).with_suffix(f".{ext}")]


def _candidate_screenshot_paths(root_dir: Path, session_id: str, filename: str) -> list[Path]:
    screenshot_root = root_dir / "_screenshots"
    matches = sorted(screenshot_root.glob(f"*/" f"*/" f"*/{session_id}/{filename}"), reverse=True)
    return [*matches, screenshot_root / session_id / filename]


def _candidate_debug_audio_paths(root_dir: Path, session_id: str, filename: str) -> list[Path]:
    matches = sorted(root_dir.glob(f"*/" f"*/" f"*/{session_id}/{filename}"), reverse=True)
    return [*matches, root_dir / session_id / filename]


def _is_path_inside(root_dir: Path, path: Path) -> bool:
    try:
        resolved_root = root_dir.resolve()
        resolved_path = path.resolve()
    except Exception:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents
