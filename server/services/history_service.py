from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from ..config import settings
from ..models import TranscriptHistory, TranscriptSegment, User
from ..repositories import history_repository
from ..transcript_store import is_runtime_transcript_finalized, read_jsonl_records, resolve_transcript_path
from . import artifact_storage as artifact_storage_module
from .artifact_storage import (
    LocalArtifactStorage,
    StagedHistoryArtifacts,
    StoredHistoryArtifacts,
    build_history_zip as build_history_zip_file,
)

artifact_storage = LocalArtifactStorage()


def _storage() -> LocalArtifactStorage:
    artifact_storage_module.settings = settings
    return artifact_storage


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


@dataclass(slots=True)
class PendingHistorySave:
    history: TranscriptHistory
    staged_artifacts: StagedHistoryArtifacts


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def serialize_app_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=timezone.utc)
    return normalized.astimezone().isoformat()


def load_runtime_snapshot(runtime_session_id: str) -> RuntimeTranscriptSnapshot:
    txt_path = resolve_transcript_path(settings.transcripts_dir, runtime_session_id, "txt")
    jsonl_path = resolve_transcript_path(settings.transcripts_dir, runtime_session_id, "jsonl")
    if txt_path is None or jsonl_path is None or not txt_path.exists() or not jsonl_path.exists():
        raise HistoryError("runtime_session_not_found", 404)

    metadata_path = txt_path.with_suffix(".meta.json")
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
) -> PendingHistorySave:
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
        metadata_path=snapshot.txt_path.with_suffix(".meta.json"),
    ):
        raise HistoryError("runtime_session_not_finalized", 409)
    required_token = str(snapshot.metadata.get("accessToken") or "").strip()
    if required_token and required_token != runtime_session_token.strip():
        raise HistoryError("runtime_session_not_found", 404)

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

    history_id = generate_history_id()
    saved_at = utcnow()
    language = str(snapshot.metadata.get("language") or snapshot.segments[0].get("language") or "").strip() or None
    audio_source = str(snapshot.metadata.get("audioSource") or "").strip() or None
    final_title = (title or "").strip() or default_title(snapshot)
    has_diarization = any(str(row.get("speaker") or "").strip() for row in snapshot.segments)

    staged_artifacts = _storage().stage_history_artifacts(
        history_id=history_id,
        user_id=user.id,
        saved_at=saved_at,
        plain_text=plain_text,
        records=[],
        metadata={
            "historyId": history_id,
            "runtimeSessionId": runtime_session_id,
            "title": final_title,
            "savedAt": serialize_app_datetime(saved_at),
            "language": language,
            "audioSource": audio_source,
            "hasDiarization": has_diarization,
            "segmentCount": 0,
            "summaryText": summary_value,
            "proofreadText": proofread_value,
        },
        summary_text=summary_value,
        proofread_text=proofread_value,
    )
    screenshots_dir = staged_artifacts.temp_dir / "screenshots"
    audio_dir = staged_artifacts.temp_dir / "audio"

    copied_records: list[dict[str, Any]] = []
    db_segments: list[TranscriptSegment] = []

    try:
        for row in snapshot.records:
            copied = dict(row)
            screenshot_filename = copy_runtime_screenshot(runtime_session_id, copied, screenshots_dir)
            raw_audio_filename = copy_runtime_debug_audio(runtime_session_id, copied, audio_dir, key="rawAudioPath")
            asr_audio_filename = copy_runtime_debug_audio(runtime_session_id, copied, audio_dir, key="audioPath")
            if screenshot_filename:
                copied["screenshotPath"] = f"/api/history/{history_id}/screenshots/{screenshot_filename}"
            elif "screenshotPath" in copied:
                copied["screenshotPath"] = None
            if raw_audio_filename:
                copied["rawAudioPath"] = f"/api/history/{history_id}/audio/{raw_audio_filename}"
            elif "rawAudioPath" in copied:
                copied["rawAudioPath"] = None
            if asr_audio_filename:
                copied["audioPath"] = f"/api/history/{history_id}/audio/{asr_audio_filename}"
            elif "audioPath" in copied:
                copied["audioPath"] = None
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

        metadata_path = staged_artifacts.temp_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "historyId": history_id,
                    "runtimeSessionId": runtime_session_id,
                    "title": final_title,
                    "savedAt": serialize_app_datetime(saved_at),
                    "language": language,
                    "audioSource": audio_source,
                    "hasDiarization": has_diarization,
                    "segmentCount": len(db_segments),
                    "summaryText": summary_value,
                    "proofreadText": proofread_value,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with (staged_artifacts.temp_dir / "transcript.jsonl").open("w", encoding="utf-8") as handle:
            for row in copied_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

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
            artifact_dir=staged_artifacts.artifact_key,
            txt_path=staged_artifacts.txt_key,
            jsonl_path=staged_artifacts.jsonl_key,
            zip_path=staged_artifacts.zip_key,
            created_at=saved_at,
            updated_at=saved_at,
            saved_at=saved_at,
            segments=db_segments,
        )
        db.add(history)
        db.flush()
        return PendingHistorySave(history=history, staged_artifacts=staged_artifacts)
    except Exception:
        shutil.rmtree(staged_artifacts.temp_dir, ignore_errors=True)
        shutil.rmtree(staged_artifacts.final_dir, ignore_errors=True)
        raise


def build_history_zip_on_demand(history: TranscriptHistory) -> tuple[Path, str, str, bool] | None:
    return _storage().open_download(history, "zip")


def build_history_zip(*, zip_path: Path, artifact_dir: Path, include_summary: bool, include_proofread: bool) -> None:
    return build_history_zip_file(
        zip_path=zip_path,
        artifact_dir=artifact_dir,
        include_summary=include_summary,
        include_proofread=include_proofread,
    )


def copy_runtime_screenshot(runtime_session_id: str, row: dict[str, Any], target_dir: Path) -> str | None:
    return _storage().copy_runtime_screenshot(runtime_session_id, row, target_dir)


def copy_runtime_debug_audio(runtime_session_id: str, row: dict[str, Any], target_dir: Path, *, key: str) -> str | None:
    return _storage().copy_runtime_debug_audio(runtime_session_id, row, target_dir, key=key)


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
    return history_repository.list_histories_for_user(db, user_id=user.id, limit=limit, offset=offset, query=query)


def get_history_for_user(db: Session, *, user: User, history_id: str) -> TranscriptHistory | None:
    return history_repository.get_history_for_user(db, user_id=user.id, history_id=history_id, with_segments=True)


def delete_history_for_user(db: Session, *, user: User, history_id: str) -> bool:
    history = get_history_for_user(db, user=user, history_id=history_id)
    if history is None:
        return False

    history_repository.delete_history(db, history)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    _storage().delete_history_artifacts(
        StoredHistoryArtifacts(
            artifact_key=history.artifact_dir or "",
            txt_key=history.txt_path,
            jsonl_key=history.jsonl_path,
            zip_key=history.zip_path,
        )
    )
    return True


def get_history_file_path(history: TranscriptHistory, relative_path: str | None) -> Path | None:
    del history
    if not relative_path:
        return None
    return _storage()._resolve_history_key(relative_path)


def _history_segment_screenshot_filename(history: TranscriptHistory, seq: int) -> str | None:
    if seq < 0:
        return None
    artifact_dir = _storage()._resolve_history_dir(history)
    if artifact_dir is None:
        return None
    screenshots_dir = artifact_dir / "screenshots"
    matches = sorted(screenshots_dir.glob(f"{seq:06d}.*"), key=_screenshot_match_sort_key)
    if not matches:
        return None
    return matches[0].name


def resolve_history_screenshot_path(history: TranscriptHistory, filename: str) -> Path | None:
    return _storage().resolve_media(history, category="screenshots", filename=filename)


def resolve_history_audio_path(history: TranscriptHistory, filename: str) -> Path | None:
    return _storage().resolve_media(history, category="audio", filename=filename)


def build_history_detail_payload(history: TranscriptHistory) -> dict[str, Any]:
    audio_map: dict[int, dict[str, str | None]] = {}
    jsonl_path = get_history_file_path(history, history.jsonl_path)
    if jsonl_path is not None and jsonl_path.exists():
        try:
            for row in read_jsonl_records(jsonl_path):
                if not isinstance(row, dict) or row.get("type") != "final":
                    continue
                seq = _as_int(row.get("seq"), -1)
                if seq < 0:
                    continue
                audio_map[seq] = {
                    "rawAudioPath": _as_str(row.get("rawAudioPath")) or None,
                    "audioPath": _as_str(row.get("audioPath")) or None,
                }
        except Exception:
            audio_map = {}
    return {
        "id": history.id,
        "title": history.title,
        "savedAt": serialize_app_datetime(history.saved_at),
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
                "rawAudioUrl": audio_map.get(segment.seq, {}).get("rawAudioPath"),
                "audioUrl": audio_map.get(segment.seq, {}).get("audioPath"),
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
        "savedAt": serialize_app_datetime(history.saved_at),
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


def _screenshot_match_sort_key(path: Path) -> tuple[int, str]:
    suffix_order = {".webp": 0, ".png": 1, ".jpg": 2, ".jpeg": 3}
    return (suffix_order.get(path.suffix.lower(), 9), path.name)


def create_history_from_payload(db: Session, *, user: User, payload: Any) -> TranscriptHistory:
    pending: PendingHistorySave | None = None
    try:
        pending = save_history(
            db,
            user=user,
            runtime_session_id=str(payload.runtimeSessionId).strip(),
            runtime_session_token=str(payload.runtimeSessionToken),
            title=payload.title,
            summary_text=payload.summaryText,
            proofread_text=payload.proofreadText,
        )
        db.commit()
        _storage().finalize_history_artifacts(pending.staged_artifacts)
        return pending.history
    except HistoryError:
        db.rollback()
        if pending is not None:
            shutil.rmtree(pending.staged_artifacts.temp_dir, ignore_errors=True)
        raise
    except IntegrityError as exc:
        db.rollback()
        if pending is not None:
            shutil.rmtree(pending.staged_artifacts.temp_dir, ignore_errors=True)
        raise HistoryError("history_already_saved", 409) from exc
    except Exception:
        db.rollback()
        if pending is not None:
            shutil.rmtree(pending.staged_artifacts.temp_dir, ignore_errors=True)
            shutil.rmtree(pending.staged_artifacts.final_dir, ignore_errors=True)
        raise


def build_history_create_payload(history: TranscriptHistory) -> dict[str, Any]:
    return {
        "ok": True,
        "history": {
            "id": history.id,
            "title": history.title,
            "savedAt": serialize_app_datetime(history.saved_at),
            "segmentCount": history.segment_count,
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
    return {"items": [build_history_list_item(item) for item in items], "total": total, "limit": limit, "offset": offset}


def build_history_detail_payload_for_user(db: Session, *, user: User, history_id: str) -> dict[str, Any]:
    history = get_history_for_user(db, user=user, history_id=history_id)
    if history is None:
        raise HistoryError("history_not_found", 404)
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
    resolved = _storage().open_download(history, kind)
    if resolved is None:
        return None
    path, media_type, filename, cleanup_after = resolved
    background = BackgroundTask(path.unlink, missing_ok=True) if cleanup_after else None
    return FileResponse(str(path), media_type=media_type, filename=filename, background=background)


def get_history_screenshot_response(db: Session, *, user: User, history_id: str, filename: str) -> FileResponse | None:
    history = get_history_for_user(db, user=user, history_id=history_id)
    if history is None:
        return None
    path = resolve_history_screenshot_path(history, filename)
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    media_type = "application/octet-stream"
    if suffix == ".webp":
        media_type = "image/webp"
    elif suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"
    return FileResponse(str(path), media_type=media_type)


def get_history_audio_response(db: Session, *, user: User, history_id: str, filename: str) -> FileResponse | None:
    history = get_history_for_user(db, user=user, history_id=history_id)
    if history is None:
        return None
    path = resolve_history_audio_path(history, filename)
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    media_type = "application/octet-stream"
    if suffix == ".wav":
        media_type = "audio/wav"
    elif suffix == ".webm":
        media_type = "audio/webm"
    elif suffix == ".ogg":
        media_type = "audio/ogg"
    elif suffix in {".mp4", ".m4a"}:
        media_type = "audio/mp4"
    elif suffix == ".mp3":
        media_type = "audio/mpeg"
    return FileResponse(str(path), media_type=media_type)


def cleanup_expired_runtime_data(db: Session) -> None:
    cleanup_expired_histories(db)
    protected_ids = set(history_repository.list_runtime_session_ids(db))
    _storage().cleanup_expired_runtime_data(protected_runtime_session_ids=protected_ids)


def cleanup_expired_histories(db: Session) -> int:
    cutoff = utcnow() - timedelta(days=settings.history_retention_days)
    expired_histories = history_repository.list_histories_saved_before(db, cutoff=cutoff)
    if not expired_histories:
        return 0

    stored_artifacts = [
        StoredHistoryArtifacts(
            artifact_key=history.artifact_dir or "",
            txt_key=history.txt_path,
            jsonl_key=history.jsonl_path,
            zip_key=history.zip_path,
        )
        for history in expired_histories
    ]
    for history in expired_histories:
        history_repository.delete_history(db, history)
    db.flush()

    for stored in stored_artifacts:
        _storage().delete_history_artifacts(stored)
    return len(expired_histories)
