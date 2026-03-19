from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..models import TranscriptHistory, TranscriptSegment, User
from ..transcript_store import read_jsonl_records, resolve_transcript_path


class HistoryError(Exception):
    status_code = 400
    code = "history_error"

    def __init__(self, code: str, status_code: int = 400):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass(slots=True)
class RuntimeTranscriptSnapshot:
    runtime_session_id: str
    txt_path: Path
    jsonl_path: Path
    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    segments: list[dict[str, Any]]


def utcnow() -> datetime:
    return datetime.utcnow()


def load_runtime_snapshot(runtime_session_id: str) -> RuntimeTranscriptSnapshot:
    txt_path = resolve_transcript_path(settings.transcripts_dir, runtime_session_id, "txt")
    jsonl_path = resolve_transcript_path(settings.transcripts_dir, runtime_session_id, "jsonl")
    if txt_path is None or jsonl_path is None or not txt_path.exists() or not jsonl_path.exists():
        raise HistoryError("runtime_session_not_found", 404)

    metadata_path = settings.transcripts_dir / f"{runtime_session_id}.meta.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, json.JSONDecodeError):
            metadata = {}

    records = read_jsonl_records(jsonl_path)
    segments = [row for row in records if isinstance(row, dict) and row.get("type") == "final"]
    if not segments:
        raise HistoryError("empty_transcript", 400)

    return RuntimeTranscriptSnapshot(
        runtime_session_id=runtime_session_id,
        txt_path=txt_path,
        jsonl_path=jsonl_path,
        metadata=metadata,
        records=records,
        segments=segments,
    )


def generate_history_id() -> str:
    return f"hist_{uuid.uuid4().hex[:16]}"


def default_title(snapshot: RuntimeTranscriptSnapshot) -> str:
    first_text = ""
    for row in snapshot.segments:
        value = str(row.get("text") or "").strip()
        if value:
            first_text = value
            break
    if first_text:
        return first_text[:30]
    stamp = utcnow().astimezone().strftime("%Y-%m-%d %H:%M")
    language = str(snapshot.metadata.get("language") or "").strip()
    if language:
        return f"{language} {stamp}"
    return f"Transcript {stamp}"


def save_history(
    db: Session,
    *,
    user: User,
    runtime_session_id: str,
    runtime_session_token: str,
    title: str | None,
    summary_text: str | None,
    proofread_text: str | None,
) -> TranscriptHistory:
    existing = db.scalar(
        select(TranscriptHistory).where(
            TranscriptHistory.user_id == user.id,
            TranscriptHistory.runtime_session_id == runtime_session_id,
        )
    )
    if existing is not None:
        raise HistoryError("history_already_saved", 409)

    snapshot = load_runtime_snapshot(runtime_session_id)
    required_token = str(snapshot.metadata.get("accessToken") or "").strip()
    if required_token and required_token != runtime_session_token.strip():
        raise HistoryError("runtime_session_not_found", 404)
    history_id = generate_history_id()
    artifact_dir = settings.history_dir / str(user.id) / history_id
    screenshots_dir = artifact_dir / "screenshots"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    summary_value = (summary_text or "").strip() or None
    proofread_value = (proofread_text or "").strip() or None
    plain_text = snapshot.txt_path.read_text(encoding="utf-8").strip()
    if not plain_text:
        plain_text = "\n".join(str(row.get("text") or "").strip() for row in snapshot.segments).strip()
    if not plain_text:
        raise HistoryError("empty_transcript", 400)

    saved_at = utcnow()
    language = str(snapshot.metadata.get("language") or snapshot.segments[0].get("language") or "").strip() or None
    audio_source = str(snapshot.metadata.get("audioSource") or "").strip() or None
    final_title = (title or "").strip() or default_title(snapshot)
    has_diarization = any(str(row.get("speaker") or "").strip() for row in snapshot.segments)

    copied_records: list[dict[str, Any]] = []
    db_segments: list[TranscriptSegment] = []

    for row in snapshot.records:
        copied = dict(row)
        screenshot_filename = copy_runtime_screenshot(runtime_session_id, copied, screenshots_dir)
        if screenshot_filename:
            copied["screenshotPath"] = f"/api/history/{history_id}/screenshots/{screenshot_filename}"
        elif "screenshotPath" in copied:
            copied["screenshotPath"] = None
        copied_records.append(copied)

        if copied.get("type") != "final":
            continue

        db_segments.append(
            TranscriptSegment(
                seq=_as_int(copied.get("seq"), 0),
                segment_id=_as_str(copied.get("segmentId")) or None,
                text=_as_str(copied.get("text")),
                ts_start=_as_int(copied.get("tsStart"), 0),
                ts_end=_as_int(copied.get("tsEnd"), 0),
                chunk_offset_ms=_nullable_int(copied.get("chunkOffsetMs")),
                chunk_duration_ms=_nullable_int(copied.get("chunkDurationMs")),
                language=_as_str(copied.get("language")) or None,
                speaker=_as_str(copied.get("speaker")) or None,
                screenshot_path=screenshot_filename,
                created_at=_nullable_datetime(copied.get("createdAt")),
            )
        )

    txt_path = artifact_dir / "transcript.txt"
    jsonl_path = artifact_dir / "transcript.jsonl"
    metadata_path = artifact_dir / "metadata.json"
    zip_path = artifact_dir / "transcript.zip"

    txt_path.write_text(plain_text + ("\n" if plain_text else ""), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in copied_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "historyId": history_id,
        "runtimeSessionId": runtime_session_id,
        "title": final_title,
        "savedAt": saved_at.isoformat(),
        "language": language,
        "audioSource": audio_source,
        "hasDiarization": has_diarization,
        "segmentCount": len(db_segments),
        "summaryText": summary_value,
        "proofreadText": proofread_value,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if summary_value:
        (artifact_dir / "summary.txt").write_text(summary_value + "\n", encoding="utf-8")
    if proofread_value:
        (artifact_dir / "proofread.txt").write_text(proofread_value + "\n", encoding="utf-8")

    build_history_zip(
        zip_path=zip_path,
        artifact_dir=artifact_dir,
        include_summary=bool(summary_value),
        include_proofread=bool(proofread_value),
    )

    history = TranscriptHistory(
        id=history_id,
        user_id=user.id,
        runtime_session_id=runtime_session_id,
        title=final_title,
        language=language,
        audio_source=audio_source,
        segment_count=len(db_segments),
        plain_text=plain_text,
        summary_text=summary_value,
        proofread_text=proofread_value,
        has_diarization=has_diarization,
        artifact_dir=str(artifact_dir.relative_to(settings.history_dir)),
        txt_path=str(txt_path.relative_to(settings.history_dir)),
        jsonl_path=str(jsonl_path.relative_to(settings.history_dir)),
        zip_path=str(zip_path.relative_to(settings.history_dir)),
        created_at=saved_at,
        updated_at=saved_at,
        saved_at=saved_at,
        segments=db_segments,
    )
    db.add(history)
    db.flush()
    return history


def build_history_zip(*, zip_path: Path, artifact_dir: Path, include_summary: bool, include_proofread: bool) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(artifact_dir / "transcript.txt", arcname="transcript.txt")
        archive.write(artifact_dir / "transcript.jsonl", arcname="transcript.jsonl")
        archive.write(artifact_dir / "metadata.json", arcname="metadata.json")
        if include_summary:
            archive.write(artifact_dir / "summary.txt", arcname="summary.txt")
        if include_proofread:
            archive.write(artifact_dir / "proofread.txt", arcname="proofread.txt")
        screenshots_dir = artifact_dir / "screenshots"
        if screenshots_dir.exists():
            for path in sorted(screenshots_dir.iterdir()):
                if path.is_file():
                    archive.write(path, arcname=f"screenshots/{path.name}")


def copy_runtime_screenshot(runtime_session_id: str, row: dict[str, Any], target_dir: Path) -> str | None:
    raw_path = _as_str(row.get("screenshotPath"))
    screenshots_root = settings.transcripts_dir / "_screenshots" / runtime_session_id
    filename = Path(raw_path).name if raw_path else ""
    src = screenshots_root / filename if filename else None
    if not src or not src.exists():
        seq = _as_int(row.get("seq"), -1)
        if seq >= 0:
            matches = sorted(screenshots_root.glob(f"{seq:06d}.*"))
            if matches:
                src = matches[0]
                filename = src.name
    if not src or not src.exists() or not filename:
        return None
    if not src.exists():
        return None
    shutil.copy2(src, target_dir / filename)
    return filename


def history_query_for_user(user: User, query: str | None = None) -> Select[tuple[TranscriptHistory]]:
    stmt = (
        select(TranscriptHistory)
        .where(TranscriptHistory.user_id == user.id)
        .order_by(TranscriptHistory.saved_at.desc())
    )
    clean_query = (query or "").strip()
    if clean_query:
        like = f"%{clean_query}%"
        stmt = stmt.where(
            TranscriptHistory.title.ilike(like) | TranscriptHistory.plain_text.ilike(like)
        )
    return stmt


def count_histories(db: Session, *, user: User, query: str | None = None) -> int:
    stmt = select(func.count()).select_from(history_query_for_user(user, query).subquery())
    return int(db.scalar(stmt) or 0)


def list_histories(
    db: Session,
    *,
    user: User,
    limit: int,
    offset: int,
    query: str | None = None,
) -> list[TranscriptHistory]:
    stmt = history_query_for_user(user, query).limit(limit).offset(offset)
    return list(db.scalars(stmt).all())


def get_history_for_user(db: Session, *, user: User, history_id: str) -> TranscriptHistory | None:
    stmt = (
        select(TranscriptHistory)
        .options(selectinload(TranscriptHistory.segments))
        .where(TranscriptHistory.id == history_id, TranscriptHistory.user_id == user.id)
    )
    return db.scalar(stmt)


def get_history_file_path(history: TranscriptHistory, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    path = settings.history_dir / relative_path
    try:
        resolved_root = settings.history_dir.resolve()
        resolved_path = path.resolve()
    except Exception:
        return None
    if resolved_root not in resolved_path.parents and resolved_root != resolved_path.parent:
        return None
    return resolved_path


def resolve_history_screenshot_path(history: TranscriptHistory, filename: str) -> Path | None:
    if not history.artifact_dir:
        return None
    if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", filename or ""):
        return None

    screenshots_dir = settings.history_dir / history.artifact_dir / "screenshots"
    path = screenshots_dir / filename
    try:
        resolved_dir = screenshots_dir.resolve()
        resolved_path = path.resolve()
    except Exception:
        return None
    if resolved_path.parent != resolved_dir:
        return None
    return resolved_path


def build_history_detail_payload(history: TranscriptHistory) -> dict[str, Any]:
    return {
        "id": history.id,
        "title": history.title,
        "savedAt": history.saved_at.isoformat(),
        "language": history.language,
        "audioSource": history.audio_source,
        "plainText": history.plain_text,
        "summaryText": history.summary_text,
        "proofreadText": history.proofread_text,
        "hasDiarization": history.has_diarization,
        "segmentCount": history.segment_count,
        "segments": [
            {
                "seq": segment.seq,
                "segmentId": segment.segment_id,
                "text": segment.text,
                "tsStart": segment.ts_start,
                "tsEnd": segment.ts_end,
                "chunkOffsetMs": segment.chunk_offset_ms,
                "chunkDurationMs": segment.chunk_duration_ms,
                "language": segment.language,
                "speaker": segment.speaker,
                "screenshotUrl": (
                    f"/api/history/{history.id}/screenshots/{segment.screenshot_path}"
                    if segment.screenshot_path
                    else None
                ),
            }
            for segment in history.segments
        ],
    }


def build_history_list_item(history: TranscriptHistory) -> dict[str, Any]:
    preview = history.plain_text.strip().replace("\r", "")
    preview = "\n".join(preview.splitlines()[:2]).strip()
    preview = preview[:180]
    return {
        "id": history.id,
        "title": history.title,
        "savedAt": history.saved_at.isoformat(),
        "language": history.language,
        "segmentCount": history.segment_count,
        "hasDiarization": history.has_diarization,
        "preview": preview,
    }


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _nullable_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _nullable_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
