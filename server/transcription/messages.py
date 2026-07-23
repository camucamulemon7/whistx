from __future__ import annotations

import base64
from typing import Any, Protocol

from .session import ChunkMessage


class ChunkOrderState(Protocol):
    last_chunk_seq: int
    last_chunk_offset_ms: int


def as_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def normalize_asr_language(value: str) -> str | None:
    lowered = (value or "").strip().lower()
    if not lowered or lowered == "auto":
        return None
    return lowered


def normalize_audio_source(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in {"display", "both"}:
        return lowered
    return "mic"


def parse_chunk_message(payload: dict[str, Any]) -> ChunkMessage | None:
    audio_b64 = as_str(payload.get("audio"))
    if not audio_b64:
        return None
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except (ValueError, TypeError):
        return None

    screenshot_bytes: bytes | None = None
    screenshot_b64 = as_str(payload.get("screenshot"))
    if screenshot_b64:
        try:
            screenshot_bytes = base64.b64decode(screenshot_b64, validate=True)
        except (ValueError, TypeError):
            screenshot_bytes = None

    duration_ms = max(200, as_int(payload.get("durationMs"), 2000))
    return ChunkMessage(
        seq=as_int(payload.get("seq"), 0),
        offset_ms=max(0, as_int(payload.get("offsetMs"), 0)),
        duration_ms=duration_ms,
        mime_type=as_str(payload.get("mimeType")) or "audio/webm",
        audio_bytes=audio_bytes,
        speech_ratio=max(0.0, min(1.0, as_float(payload.get("speechRatio"), 1.0))),
        active_ms=max(0, as_int(payload.get("activeMs"), duration_ms)),
        silence_ms=max(0, as_int(payload.get("silenceMs"), 0)),
        screenshot_mime_type=as_str(payload.get("screenshotMimeType")) or None,
        screenshot_bytes=screenshot_bytes,
    )


def validate_chunk_order(state: ChunkOrderState, chunk: ChunkMessage) -> dict[str, Any] | None:
    if chunk.seq < 0:
        return {
            "type": "error",
            "message": "invalid_chunk_sequence",
            "detail": "seq_must_be_non_negative",
            "seq": chunk.seq,
        }
    if state.last_chunk_seq >= 0 and chunk.seq <= state.last_chunk_seq:
        return {
            "type": "error",
            "message": "invalid_chunk_sequence",
            "detail": "seq_must_strictly_increase",
            "seq": chunk.seq,
            "previousSeq": state.last_chunk_seq,
        }
    if state.last_chunk_offset_ms >= 0 and chunk.offset_ms < state.last_chunk_offset_ms:
        return {
            "type": "error",
            "message": "invalid_chunk_offset",
            "detail": "offset_ms_must_be_monotonic",
            "offsetMs": chunk.offset_ms,
            "previousOffsetMs": state.last_chunk_offset_ms,
        }
    return None
