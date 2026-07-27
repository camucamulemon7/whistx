from __future__ import annotations

import base64
import json
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


def validate_start_message(
    payload: dict[str, Any],
    *,
    prompt_max_chars: int,
    vocabulary_max_chars: int,
) -> str | None:
    if len(as_str(payload.get("prompt"))) > prompt_max_chars:
        return "prompt_too_large"
    if len(as_str(payload.get("sharedVocabulary"))) > vocabulary_max_chars:
        return "vocabulary_too_large"
    return None


def validate_telemetry_message(payload: dict[str, Any], *, max_chars: int) -> str | None:
    if len(as_str(payload.get("event"))) > 128:
        return "telemetry_event_too_large"
    try:
        serialized = json.dumps(payload.get("detail"), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "invalid_telemetry"
    if len(serialized) > max_chars:
        return "telemetry_too_large"
    return None


def chunk_payload_size_error(
    payload: dict[str, Any],
    *,
    max_audio_bytes: int,
    max_screenshot_bytes: int,
) -> str | None:
    audio_b64 = as_str(payload.get("audio"))
    if (
        len(audio_b64) > _max_base64_length(max_audio_bytes)
        or _estimated_base64_decoded_length(audio_b64) > max_audio_bytes
    ):
        return "chunk_too_large"
    screenshot_b64 = as_str(payload.get("screenshot"))
    if screenshot_b64 and (
        len(screenshot_b64) > _max_base64_length(max_screenshot_bytes)
        or _estimated_base64_decoded_length(screenshot_b64) > max_screenshot_bytes
    ):
        return "screenshot_too_large"
    return None


def parse_chunk_message(
    payload: dict[str, Any],
    *,
    max_audio_bytes: int | None = None,
    max_screenshot_bytes: int | None = None,
) -> ChunkMessage | None:
    audio_b64 = as_str(payload.get("audio"))
    if not audio_b64:
        return None
    if max_audio_bytes is not None and (
        len(audio_b64) > _max_base64_length(max_audio_bytes)
        or _estimated_base64_decoded_length(audio_b64) > max_audio_bytes
    ):
        return None
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except (ValueError, TypeError):
        return None
    if max_audio_bytes is not None and len(audio_bytes) > max_audio_bytes:
        return None

    screenshot_bytes: bytes | None = None
    screenshot_b64 = as_str(payload.get("screenshot"))
    if screenshot_b64:
        if max_screenshot_bytes is not None and (
            len(screenshot_b64) > _max_base64_length(max_screenshot_bytes)
            or _estimated_base64_decoded_length(screenshot_b64) > max_screenshot_bytes
        ):
            return None
        try:
            screenshot_bytes = base64.b64decode(screenshot_b64, validate=True)
        except (ValueError, TypeError):
            return None
        if max_screenshot_bytes is not None and len(screenshot_bytes) > max_screenshot_bytes:
            return None
        screenshot_mime_type = as_str(payload.get("screenshotMimeType"))
        if not _image_matches_mime_type(screenshot_bytes, screenshot_mime_type):
            return None

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


def _max_base64_length(max_decoded_bytes: int) -> int:
    return ((max(0, max_decoded_bytes) + 2) // 3) * 4


def _estimated_base64_decoded_length(value: str) -> int:
    clean_length = len(value)
    if clean_length == 0:
        return 0
    padding = 2 if value.endswith("==") else 1 if value.endswith("=") else 0
    return max(0, (clean_length // 4) * 3 - padding)


def _image_matches_mime_type(image_bytes: bytes, mime_type: str) -> bool:
    normalized = mime_type.strip().lower()
    if normalized == "image/png":
        return image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if normalized in {"image/jpeg", "image/jpg"}:
        return image_bytes.startswith(b"\xff\xd8\xff")
    if normalized == "image/webp":
        return len(image_bytes) >= 12 and image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"
    return False


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
