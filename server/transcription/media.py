from __future__ import annotations

import logging
import wave
from io import BytesIO

from ..core.config import settings
from ..core.logging import is_debug_logging_enabled
from ..transcript_store import build_debug_chunks_dir, iter_debug_chunk_dirs
from .session import ChunkMessage, LiveSession

logger = logging.getLogger(__name__)


def _store_screenshot_for_chunk(session: LiveSession, item: ChunkMessage) -> str | None:
    if not item.screenshot_bytes or not item.screenshot_mime_type:
        return None
    try:
        filename = session.store.save_screenshot(
            seq=item.seq,
            mime_type=item.screenshot_mime_type,
            image_bytes=item.screenshot_bytes,
        )
        return f"/api/transcripts/{session.session_id}/screenshots/{filename}"
    except Exception:  # noqa: BLE001
        logger.warning(
            "Screenshot save failed: session=%s seq=%s",
            session.session_id,
            item.seq,
            exc_info=True,
        )
        return None


def _save_debug_audio_chunk(session: LiveSession, item: ChunkMessage) -> None:
    if not is_debug_logging_enabled():
        return
    if not item.audio_bytes:
        return
    try:
        ext = _debug_audio_ext_from_mime(item.mime_type)
        debug_dir = build_debug_chunks_dir(
            settings.debug_chunks_dir, session.session_id
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"raw-{item.seq:06d}{ext}"
        path.write_bytes(item.audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Debug chunk save failed: session=%s seq=%s err=%s",
            session.session_id,
            item.seq,
            exc,
        )


def _store_debug_raw_audio_for_final(
    session: LiveSession, item: ChunkMessage
) -> str | None:
    if not item.audio_bytes:
        return None
    try:
        ext = _debug_audio_ext_from_mime(item.mime_type)
        debug_dir = build_debug_chunks_dir(
            settings.debug_chunks_dir, session.session_id
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = f"raw-{item.seq:06d}{ext}"
        path = debug_dir / filename
        if not path.exists():
            path.write_bytes(item.audio_bytes)
        return f"/api/transcripts/{session.session_id}/audio/{filename}"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Debug raw audio save failed: session=%s seq=%s err=%s",
            session.session_id,
            item.seq,
            exc,
        )
        return None


def _store_debug_audio_for_final(
    session: LiveSession, seq: int, audio_bytes: bytes
) -> str | None:
    if not audio_bytes:
        return None
    try:
        debug_dir = build_debug_chunks_dir(
            settings.debug_chunks_dir, session.session_id
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = f"asr-{seq:06d}.wav"
        (debug_dir / filename).write_bytes(audio_bytes)
        return f"/api/transcripts/{session.session_id}/audio/{filename}"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Debug ASR audio save failed: session=%s seq=%s err=%s",
            session.session_id,
            seq,
            exc,
        )
        return None


def _debug_audio_ext_from_mime(mime_type: str) -> str:
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


def _resolve_existing_debug_audio_url(
    session: LiveSession, *, prefix: str, seq: int
) -> str | None:
    for debug_dir in iter_debug_chunk_dirs(
        settings.debug_chunks_dir, session.session_id
    ):
        if not debug_dir.exists():
            continue
        matches = sorted(debug_dir.glob(f"{prefix}-{seq:06d}.*"))
        if matches:
            return f"/api/transcripts/{session.session_id}/audio/{matches[0].name}"
    return None


def _merge_wav_chunks(chunks: list[bytes]) -> bytes:
    frames: list[bytes] = []
    sample_rate = settings.asr_preprocess_sample_rate
    for chunk in chunks:
        if not chunk:
            continue
        with wave.open(BytesIO(chunk), "rb") as wav_in:
            sample_rate = wav_in.getframerate() or sample_rate
            frames.append(wav_in.readframes(wav_in.getnframes()))

    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            for frame in frames:
                wav_out.writeframes(frame)
        return buffer.getvalue()
