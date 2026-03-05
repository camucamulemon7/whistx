from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
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


class TranscriptStore:
    def __init__(self, root_dir: Path, session_id: str):
        self.root_dir = root_dir
        self.session_id = session_id
        self.base_path = root_dir / session_id
        self.jsonl_path = self.base_path.with_suffix(".jsonl")
        self.txt_path = self.base_path.with_suffix(".txt")
        self.chunks_dir = self.root_dir / "_chunks" / session_id

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)
        self.txt_path.touch(exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

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
    if safe_ext not in {"txt", "jsonl", "srt"}:
        return None

    path = (root_dir / session_id).with_suffix(f".{safe_ext}")
    try:
        resolved_root = root_dir.resolve()
        resolved_path = path.resolve()
    except Exception:
        return None

    if resolved_root != resolved_path.parent:
        return None
    return path



def read_jsonl_records(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out



def format_srt(records: Iterable[dict]) -> str:
    lines: list[str] = []
    index = 1

    for rec in records:
        if rec.get("type") != "final":
            continue

        text = str(rec.get("text", "")).strip()
        if not text:
            continue

        ts_start = _as_int(rec.get("tsStart"), 0)
        ts_end = _as_int(rec.get("tsEnd"), ts_start)
        if ts_end < ts_start:
            ts_end = ts_start

        lines.append(str(index))
        lines.append(f"{_fmt_srt_time(ts_start)} --> {_fmt_srt_time(ts_end)}")
        speaker = str(rec.get("speaker", "")).strip()
        if speaker:
            lines.append(f"[{speaker}] {text}")
        else:
            lines.append(text)
        lines.append("")
        index += 1

    return "\n".join(lines)


def _render_txt_line(rec: dict) -> str:
    text = str(rec.get("text", "")).strip()
    if not text:
        return ""
    speaker = str(rec.get("speaker", "")).strip()
    if not speaker:
        return text
    return f"[{speaker}] {text}"


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



def _fmt_srt_time(ms: int) -> str:
    ms = max(0, int(ms))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1_000
    ms -= s * 1_000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"



def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
