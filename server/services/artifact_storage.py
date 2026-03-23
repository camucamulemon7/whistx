from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import settings
from ..models import TranscriptHistory
from ..transcript_store import (
    iter_debug_chunk_dirs,
    iter_runtime_screenshot_dirs,
    resolve_debug_audio_path,
    resolve_screenshot_path,
)


@dataclass(slots=True)
class StagedHistoryArtifacts:
    temp_dir: Path
    final_dir: Path
    artifact_key: str
    txt_key: str
    jsonl_key: str
    zip_key: str | None


@dataclass(slots=True)
class StoredHistoryArtifacts:
    artifact_key: str
    txt_key: str | None
    jsonl_key: str | None
    zip_key: str | None


class LocalArtifactStorage:
    def __init__(self) -> None:
        pass

    @property
    def history_root(self) -> Path:
        return settings.history_dir

    @property
    def transcripts_root(self) -> Path:
        return settings.transcripts_dir

    @property
    def debug_chunks_root(self) -> Path:
        return settings.debug_chunks_dir

    def stage_history_artifacts(
        self,
        *,
        history_id: str,
        user_id: int,
        saved_at: datetime,
        plain_text: str,
        records: list[dict[str, Any]],
        metadata: dict[str, Any],
        summary_text: str | None,
        proofread_text: str | None,
    ) -> StagedHistoryArtifacts:
        final_dir = self._history_artifact_dir(user_id=user_id, history_id=history_id, saved_at=saved_at)
        temp_parent = final_dir.parent
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{history_id}.", dir=str(temp_parent)))
        (temp_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (temp_dir / "audio").mkdir(parents=True, exist_ok=True)

        (temp_dir / "transcript.txt").write_text(plain_text + ("\n" if plain_text else ""), encoding="utf-8")
        with (temp_dir / "transcript.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (temp_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if summary_text:
            (temp_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
        if proofread_text:
            (temp_dir / "proofread.txt").write_text(proofread_text + "\n", encoding="utf-8")

        artifact_key = str(final_dir.relative_to(self.history_root))
        return StagedHistoryArtifacts(
            temp_dir=temp_dir,
            final_dir=final_dir,
            artifact_key=artifact_key,
            txt_key=f"{artifact_key}/transcript.txt",
            jsonl_key=f"{artifact_key}/transcript.jsonl",
            zip_key=None,
        )

    def finalize_history_artifacts(self, staged: StagedHistoryArtifacts) -> StoredHistoryArtifacts:
        if staged.final_dir.exists():
            raise FileExistsError(staged.final_dir)
        staged.temp_dir.rename(staged.final_dir)
        return StoredHistoryArtifacts(
            artifact_key=staged.artifact_key,
            txt_key=staged.txt_key,
            jsonl_key=staged.jsonl_key,
            zip_key=staged.zip_key,
        )

    def delete_history_artifacts(self, stored: StoredHistoryArtifacts) -> None:
        for path in self._candidate_history_dirs(stored):
            shutil.rmtree(path, ignore_errors=True)

    def open_download(self, history: TranscriptHistory, kind: str) -> tuple[Path, str, str, bool] | None:
        if kind == "txt":
            path = self._resolve_history_key(history.txt_path)
            if path and path.exists():
                return path, "text/plain", f"{history.id}.txt", False
            return None
        if kind == "jsonl":
            path = self._resolve_history_key(history.jsonl_path)
            if path and path.exists():
                return path, "application/x-ndjson", f"{history.id}.jsonl", False
            return None
        if kind != "zip":
            return None

        existing_zip = self._resolve_history_key(history.zip_path)
        if existing_zip and existing_zip.exists():
            return existing_zip, "application/zip", f"{history.id}.zip", False

        artifact_dir = self._resolve_history_dir(history)
        if artifact_dir is None or not artifact_dir.exists():
            return None

        fd, temp_zip_name = tempfile.mkstemp(prefix=f"{history.id}.", suffix=".zip")
        temp_zip = Path(temp_zip_name)
        os.close(fd)
        build_history_zip(
            zip_path=temp_zip,
            artifact_dir=artifact_dir,
            include_summary=bool(history.summary_text),
            include_proofread=bool(history.proofread_text),
        )
        return temp_zip, "application/zip", f"{history.id}.zip", True

    def resolve_media(self, history: TranscriptHistory, *, category: str, filename: str) -> Path | None:
        if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", filename or ""):
            return None
        artifact_dir = self._resolve_history_dir(history)
        if artifact_dir is None:
            return None
        media_dir = artifact_dir / category
        path = media_dir / filename
        if not _is_path_inside(media_dir, path):
            return None
        return path

    def copy_runtime_screenshot(self, runtime_session_id: str, row: dict[str, Any], target_dir: Path) -> str | None:
        raw_path = _as_str(row.get("screenshotPath"))
        filename = Path(raw_path.split("?", 1)[0]).name if raw_path else ""
        src = resolve_screenshot_path(self.transcripts_root, runtime_session_id, filename) if filename else None
        if not src or not src.exists():
            seq = _as_int(row.get("seq"), -1)
            if seq >= 0:
                for screenshots_dir in iter_runtime_screenshot_dirs(self.transcripts_root, runtime_session_id):
                    matches = sorted(screenshots_dir.glob(f"{seq:06d}.*"), key=_screenshot_match_sort_key)
                    if matches:
                        src = matches[0]
                        filename = src.name
                        break
        if not src or not src.exists() or not filename:
            return None
        shutil.copy2(src, target_dir / filename)
        return filename

    def copy_runtime_debug_audio(
        self,
        runtime_session_id: str,
        row: dict[str, Any],
        target_dir: Path,
        *,
        key: str,
    ) -> str | None:
        raw_path = _as_str(row.get(key))
        filename = Path(raw_path.split("?", 1)[0]).name if raw_path else ""
        src = resolve_debug_audio_path(self.debug_chunks_root, runtime_session_id, filename) if filename else None
        if (not src or not src.exists()) and _as_int(row.get("seq"), -1) >= 0:
            prefix = "raw" if key == "rawAudioPath" else "asr"
            for debug_dir in iter_debug_chunk_dirs(self.debug_chunks_root, runtime_session_id):
                matches = sorted(debug_dir.glob(f"{prefix}-{_as_int(row.get('seq'), -1):06d}.*"))
                if matches:
                    src = matches[0]
                    filename = src.name
                    break
        if not src or not src.exists() or not filename:
            return None
        shutil.copy2(src, target_dir / filename)
        return filename

    def cleanup_expired_runtime_data(self, *, protected_runtime_session_ids: set[str]) -> None:
        now = datetime.now(timezone.utc)
        self._cleanup_runtime_transcripts(now=now, protected_runtime_session_ids=protected_runtime_session_ids)
        self._cleanup_screenshots(now=now)
        self._cleanup_debug_chunks(now=now)

    def _cleanup_runtime_transcripts(self, *, now: datetime, protected_runtime_session_ids: set[str]) -> None:
        keep_saved_cutoff = now - timedelta(hours=settings.runtime_transcript_retention_hours)
        keep_unsaved_cutoff = now - timedelta(hours=settings.unsaved_runtime_retention_hours)
        candidates = list(self.transcripts_root.glob("*/*/*/*.txt")) + list(self.transcripts_root.glob("*.txt"))
        for txt_path in candidates:
            base_path = txt_path.with_suffix("")
            session_id = base_path.name
            if not session_id:
                continue
            cutoff = keep_saved_cutoff if session_id in protected_runtime_session_ids else keep_unsaved_cutoff
            if _path_mtime(base_path.with_suffix(".txt")) > cutoff.timestamp():
                continue
            self._delete_runtime_session(session_id)

    def _cleanup_screenshots(self, *, now: datetime) -> None:
        cutoff = now - timedelta(hours=settings.runtime_transcript_retention_hours)
        roots = list((self.transcripts_root / "_screenshots").glob("*/*/*/*")) + list(
            (self.transcripts_root / "_screenshots").glob("*")
        )
        for directory in roots:
            if not directory.is_dir():
                continue
            if _path_mtime(directory) > cutoff.timestamp():
                continue
            shutil.rmtree(directory, ignore_errors=True)

    def _cleanup_debug_chunks(self, *, now: datetime) -> None:
        cutoff = now - timedelta(hours=settings.debug_chunks_retention_hours)
        roots = list(self.debug_chunks_root.glob("*/*/*/*")) + list(self.debug_chunks_root.glob("*"))
        for directory in roots:
            if not directory.is_dir():
                continue
            if _path_mtime(directory) > cutoff.timestamp():
                continue
            shutil.rmtree(directory, ignore_errors=True)

    def _delete_runtime_session(self, session_id: str) -> None:
        for txt_path in list(self.transcripts_root.glob(f"*/*/*/{session_id}.txt")) + [
            (self.transcripts_root / session_id).with_suffix(".txt")
        ]:
            for path in [txt_path, txt_path.with_suffix(".jsonl"), txt_path.with_suffix(".meta.json")]:
                path.unlink(missing_ok=True)

    def _history_artifact_dir(self, *, user_id: int, history_id: str, saved_at: datetime) -> Path:
        stamp = saved_at.astimezone(timezone.utc)
        return self.history_root / str(user_id) / stamp.strftime("%Y") / stamp.strftime("%m") / history_id

    def _resolve_history_key(self, key: str | None) -> Path | None:
        if not key:
            return None
        path = self.history_root / key
        if _is_path_inside(self.history_root, path):
            return path
        return None

    def _resolve_history_dir(self, history: TranscriptHistory) -> Path | None:
        if history.artifact_dir:
            path = self._resolve_history_key(history.artifact_dir)
            if path:
                return path
        if history.txt_path:
            path = self._resolve_history_key(history.txt_path)
            if path:
                return path.parent
        return None

    def _candidate_history_dirs(self, stored: StoredHistoryArtifacts) -> list[Path]:
        candidates: list[Path] = []
        resolved = self._resolve_history_key(stored.artifact_key)
        if resolved:
            candidates.append(resolved)
            parts = Path(stored.artifact_key).parts
            if len(parts) >= 4:
                legacy = self.history_root / parts[0] / parts[-1]
                if legacy != resolved:
                    candidates.append(legacy)
        return candidates


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
        audio_dir = artifact_dir / "audio"
        if audio_dir.exists():
            for path in sorted(audio_dir.iterdir()):
                if path.is_file():
                    archive.write(path, arcname=f"audio/{path.name}")


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_path_inside(root_dir: Path, path: Path) -> bool:
    try:
        resolved_root = root_dir.resolve()
        resolved_path = path.resolve()
    except Exception:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _screenshot_match_sort_key(path: Path) -> tuple[int, str]:
    suffix_order = {".webp": 0, ".png": 1, ".jpg": 2, ".jpeg": 3}
    return (suffix_order.get(path.suffix.lower(), 9), path.name)


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
