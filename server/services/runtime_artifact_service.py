from __future__ import annotations

import json
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import BoundedSemaphore

from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.background import BackgroundTask

from ..core.config import settings
from ..core.security import runtime_access_allowed as security_runtime_access_allowed
from ..transcript_store import (
    iter_runtime_screenshot_dirs,
    read_jsonl_records,
    resolve_debug_audio_path,
    resolve_screenshot_path,
    resolve_transcript_path,
)

_ZIP_WORKERS = BoundedSemaphore(settings.artifact_worker_concurrency)


def _runtime_access_allowed(
    session_id: str,
    *,
    user_id: int | None,
    guest_grant_id: str | None,
) -> bool:
    return security_runtime_access_allowed(
        transcripts_dir=settings.transcripts_dir,
        session_id=session_id,
        user_id=user_id,
        guest_grant_id=guest_grant_id,
    )


def get_txt(
    session_id: str, *, user_id: int | None, guest_grant_id: str | None
) -> Response:
    if not _runtime_access_allowed(
        session_id, user_id=user_id, guest_grant_id=guest_grant_id
    ):
        return HTMLResponse(status_code=404, content="not found")
    path = resolve_transcript_path(settings.transcripts_dir, session_id, "txt")
    if not path or not path.exists():
        return HTMLResponse(status_code=404, content="not found")
    return FileResponse(str(path), media_type="text/plain")


def _qwen_jsonl_snapshot(path: Path) -> str | None:
    from .meeting_source import read_json
    if read_json(path.with_suffix(".meta.json")).get("asrBackend") != "qwen3_vllm":
        return None
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in read_jsonl_records(path))


def get_jsonl(
    session_id: str, *, user_id: int | None, guest_grant_id: str | None
) -> Response:
    if not _runtime_access_allowed(
        session_id, user_id=user_id, guest_grant_id=guest_grant_id
    ):
        return HTMLResponse(status_code=404, content="not found")
    path = resolve_transcript_path(settings.transcripts_dir, session_id, "jsonl")
    if not path or not path.exists():
        return HTMLResponse(status_code=404, content="not found")
    snapshot = _qwen_jsonl_snapshot(path)
    if snapshot is not None:
        return Response(content=snapshot, media_type="application/x-ndjson")
    return FileResponse(str(path), media_type="application/x-ndjson")


def get_zip(
    session_id: str, *, user_id: int | None, guest_grant_id: str | None
) -> Response:
    if not _runtime_access_allowed(
        session_id, user_id=user_id, guest_grant_id=guest_grant_id
    ):
        return HTMLResponse(status_code=404, content="not found")
    txt_path = resolve_transcript_path(settings.transcripts_dir, session_id, "txt")
    jsonl_path = resolve_transcript_path(settings.transcripts_dir, session_id, "jsonl")
    if (
        not txt_path
        or not jsonl_path
        or not txt_path.exists()
        or not jsonl_path.exists()
    ):
        return HTMLResponse(status_code=404, content="not found")

    acquired = _ZIP_WORKERS.acquire(
        timeout=settings.blocking_worker_queue_timeout_seconds
    )
    if not acquired:
        return Response(
            status_code=503,
            content="artifact worker queue timeout",
            media_type="text/plain",
        )
    try:
        temp_dir = settings.app_data_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            prefix=f"{session_id}-",
            suffix=".zip",
            dir=temp_dir,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            with zipfile.ZipFile(
                temp_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.write(txt_path, arcname=f"{session_id}.txt")
                snapshot = _qwen_jsonl_snapshot(jsonl_path)
                if snapshot is None:
                    archive.write(jsonl_path, arcname=f"{session_id}.jsonl")
                else:
                    archive.writestr(f"{session_id}.jsonl", snapshot)
                    journal = jsonl_path.with_suffix(".revisions.jsonl")
                    archive.write(journal if journal.exists() else jsonl_path, arcname=f"{session_id}.revisions.jsonl")

                seen_screenshots: set[str] = set()
                for screenshots_dir in iter_runtime_screenshot_dirs(
                    settings.transcripts_dir, session_id
                ):
                    if not screenshots_dir.exists():
                        continue
                    for screenshot_path in sorted(screenshots_dir.iterdir()):
                        if not screenshot_path.is_file():
                            continue
                        if screenshot_path.name in seen_screenshots:
                            continue
                        seen_screenshots.add(screenshot_path.name)
                        archive.write(
                            screenshot_path,
                            arcname=f"{session_id}/screenshots/{screenshot_path.name}",
                        )
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    finally:
        _ZIP_WORKERS.release()

    headers = {"Content-Disposition": f'attachment; filename="{session_id}.zip"'}
    return FileResponse(
        str(temp_path),
        media_type="application/zip",
        headers=headers,
        background=BackgroundTask(_remove_temp_file, temp_path),
    )


def get_screenshot(
    session_id: str,
    filename: str,
    *,
    user_id: int | None,
    guest_grant_id: str | None,
) -> Response:
    if not _runtime_access_allowed(
        session_id, user_id=user_id, guest_grant_id=guest_grant_id
    ):
        return HTMLResponse(status_code=404, content="not found")
    path = resolve_screenshot_path(settings.transcripts_dir, session_id, filename)
    if not path or not path.exists():
        return HTMLResponse(status_code=404, content="not found")

    suffix = path.suffix.lower()
    media_type = "application/octet-stream"
    if suffix == ".webp":
        media_type = "image/webp"
    elif suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"

    return FileResponse(str(path), media_type=media_type)


def get_debug_audio(
    session_id: str,
    filename: str,
    *,
    user_id: int | None,
    guest_grant_id: str | None,
) -> Response:
    if not _runtime_access_allowed(
        session_id, user_id=user_id, guest_grant_id=guest_grant_id
    ):
        return HTMLResponse(status_code=404, content="not found")
    path = resolve_debug_audio_path(settings.debug_chunks_dir, session_id, filename)
    if not path or not path.exists():
        return HTMLResponse(status_code=404, content="not found")

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


def _remove_temp_file(path: Path) -> None:
    path.unlink(missing_ok=True)
