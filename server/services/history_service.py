from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import settings
from ..models import TranscriptHistory, TranscriptSegment, User
from ..repositories import history_repository
from ..transcript_store import (
    is_runtime_transcript_finalized,
    read_jsonl_records,
    resolve_transcript_path,
)


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

    try:
        records = read_jsonl_records(jsonl_path, strict=True)
    except ValueError as exc:
        raise HistoryError("transcript_parse_failed", 500) from exc
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
    existing = history_repository.get_history_by_runtime_session(
        db,
        user_id=user.id,
        runtime_session_id=runtime_session_id,
    )
    if existing is not None:
        raise HistoryError("history_already_saved", 409)

    snapshot = load_runtime_snapshot(runtime_session_id)
    if not is_runtime_transcript_finalized(
        snapshot.metadata,
        txt_path=snapshot.txt_path,
        jsonl_path=snapshot.jsonl_path,
    ):
        raise HistoryError("runtime_session_not_finalized", 409)
    required_token = str(snapshot.metadata.get("accessToken") or "").strip()
    if required_token and required_token != runtime_session_token.strip():
        raise HistoryError("runtime_session_not_found", 404)
    history_id = generate_history_id()
    user_history_dir = settings.history_dir / str(user.id)
    user_history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = user_history_dir / history_id
    temp_artifact_dir = Path(tempfile.mkdtemp(prefix=f".{history_id}.", dir=str(user_history_dir)))
    screenshots_dir = temp_artifact_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    summary_value = (summary_text or "").strip() or None
    proofread_value = (proofread_text or "").strip() or None
    try:
        plain_text = snapshot.txt_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise HistoryError("transcript_read_failed", 500) from exc
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

    txt_path = temp_artifact_dir / "transcript.txt"
    jsonl_path = temp_artifact_dir / "transcript.jsonl"
    metadata_path = temp_artifact_dir / "metadata.json"
    zip_path = temp_artifact_dir / "transcript.zip"

    try:
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
            (temp_artifact_dir / "summary.txt").write_text(summary_value + "\n", encoding="utf-8")
        if proofread_value:
            (temp_artifact_dir / "proofread.txt").write_text(proofread_value + "\n", encoding="utf-8")

        build_history_zip(
            zip_path=zip_path,
            artifact_dir=temp_artifact_dir,
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
            txt_path=str((artifact_dir / "transcript.txt").relative_to(settings.history_dir)),
            jsonl_path=str((artifact_dir / "transcript.jsonl").relative_to(settings.history_dir)),
            zip_path=str((artifact_dir / "transcript.zip").relative_to(settings.history_dir)),
            created_at=saved_at,
            updated_at=saved_at,
            saved_at=saved_at,
            segments=db_segments,
        )
        db.add(history)
        db.flush()
        if artifact_dir.exists():
            raise HistoryError("history_artifact_conflict", 409)
        temp_artifact_dir.rename(artifact_dir)
        return history
    except Exception:
        shutil.rmtree(temp_artifact_dir, ignore_errors=True)
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


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
            matches = sorted(screenshots_root.glob(f"{seq:06d}.*"), key=_screenshot_match_sort_key)
            if matches:
                src = matches[0]
                filename = src.name
    if not src or not src.exists() or not filename:
        return None
    if not src.exists():
        return None
    shutil.copy2(src, target_dir / filename)
    return filename



def count_histories(db: Session, *, user: User, query: str | None = None) -> int:
    return history_repository.count_histories_for_user(db, user_id=user.id, query=query)


def list_histories(
    db: Session,
    *,
    user: User,
    limit: int,
    offset: int,
    query: str | None = None,
) -> list[TranscriptHistory]:
    return history_repository.list_histories_for_user(
        db,
        user_id=user.id,
        limit=limit,
        offset=offset,
        query=query,
    )


def get_history_for_user(db: Session, *, user: User, history_id: str) -> TranscriptHistory | None:
    return history_repository.get_history_for_user(db, user_id=user.id, history_id=history_id, with_segments=True)


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


def _history_segment_screenshot_filename(history: TranscriptHistory, seq: int) -> str | None:
    if seq < 0 or not history.artifact_dir:
        return None
    screenshots_dir = settings.history_dir / history.artifact_dir / "screenshots"
    matches = sorted(screenshots_dir.glob(f"{seq:06d}.*"), key=_screenshot_match_sort_key)
    if not matches:
        return None
    return matches[0].name


def _screenshot_match_sort_key(path: Path) -> tuple[int, str]:
    suffix_order = {".webp": 0, ".png": 1, ".jpg": 2, ".jpeg": 3}
    return (suffix_order.get(path.suffix.lower(), 9), path.name)


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
                    f"/api/history/{history.id}/screenshots/{(segment.screenshot_path or _history_segment_screenshot_filename(history, segment.seq))}"
                    if (segment.screenshot_path or _history_segment_screenshot_filename(history, segment.seq))
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



def create_history_from_payload(db: Session, *, user: User, payload: Any) -> TranscriptHistory:
    try:
        history = save_history(
            db,
            user=user,
            runtime_session_id=str(payload.runtimeSessionId).strip(),
            runtime_session_token=str(payload.runtimeSessionToken),
            title=payload.title,
            summary_text=payload.summaryText,
            proofread_text=payload.proofreadText,
        )
        db.commit()
        return history
    except HistoryError:
        db.rollback()
        raise
    except IntegrityError as exc:
        db.rollback()
        raise HistoryError('history_already_saved', 409) from exc


def build_history_create_payload(history: TranscriptHistory) -> dict[str, Any]:
    return {
        'ok': True,
        'history': {
            'id': history.id,
            'title': history.title,
            'savedAt': history.saved_at.isoformat(),
            'segmentCount': history.segment_count,
        },
    }


def build_history_list_payload(
    db: Session,
    *,
    user: User,
    limit: int,
    offset: int,
    query: str | None = None,
) -> dict[str, Any]:
    items = list_histories(db, user=user, limit=limit, offset=offset, query=query)
    total = count_histories(db, user=user, query=query)
    return {
        'items': [build_history_list_item(item) for item in items],
        'total': total,
        'limit': limit,
        'offset': offset,
    }


def build_history_detail_payload_for_user(db: Session, *, user: User, history_id: str) -> dict[str, Any]:
    history = get_history_for_user(db, user=user, history_id=history_id)
    if history is None:
        raise HistoryError('history_not_found', 404)
    return build_history_detail_payload(history)


def get_history_download_response(
    db: Session,
    *,
    user: User,
    history_id: str,
    kind: str,
) -> FileResponse | None:
    history = get_history_for_user(db, user=user, history_id=history_id)
    if history is None:
        return None
    path_map = {
        'txt': (history.txt_path, 'text/plain', f'{history.id}.txt'),
        'jsonl': (history.jsonl_path, 'application/x-ndjson', f'{history.id}.jsonl'),
        'zip': (history.zip_path, 'application/zip', f'{history.id}.zip'),
    }
    relative_path, media_type, filename = path_map[kind]
    path = get_history_file_path(history, relative_path)
    if path is None or not path.exists():
        return None
    return FileResponse(str(path), media_type=media_type, filename=filename)


def get_history_screenshot_response(db: Session, *, user: User, history_id: str, filename: str) -> FileResponse | None:
    history = get_history_for_user(db, user=user, history_id=history_id)
    if history is None:
        return None
    path = resolve_history_screenshot_path(history, filename)
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    media_type = 'application/octet-stream'
    if suffix == '.webp':
        media_type = 'image/webp'
    elif suffix in {'.jpg', '.jpeg'}:
        media_type = 'image/jpeg'
    elif suffix == '.png':
        media_type = 'image/png'
    return FileResponse(str(path), media_type=media_type)
