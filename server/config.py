from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    ws_path: str
    transcripts_dir: Path
    openai_api_key: str
    openai_base_url: str | None
    whisper_model: str
    summary_api_key: str
    summary_base_url: str | None
    summary_model: str
    summary_temperature: float
    summary_input_max_chars: int
    proofread_api_key: str
    proofread_base_url: str | None
    proofread_model: str
    proofread_temperature: float
    proofread_input_max_chars: int
    diarization_enabled: bool
    diarization_hf_token: str
    diarization_model: str
    diarization_device: str
    diarization_sample_rate: int
    diarization_num_speakers: int
    diarization_min_speakers: int
    diarization_max_speakers: int
    diarization_work_dir: Path
    diarization_keep_chunks: bool
    ffmpeg_bin: str
    default_language: str
    default_prompt: str
    default_temperature: float
    context_prompt_enabled: bool
    context_max_chars: int
    max_queue_size: int
    max_chunk_bytes: int
    ui_banners: tuple[dict[str, Any], ...]



def _to_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default



def _to_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default



def _to_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _to_bool_value(value: Any, default: bool) -> bool:
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


def _sanitize_banner_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", (value or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or fallback


def _normalize_banner_type(value: Any) -> str:
    lowered = str(value or "info").strip().lower()
    if lowered in {"success", "warning", "error", "info"}:
        return lowered
    return "info"


def _clean_banner_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text[:limit]


def _parse_ui_banners() -> tuple[dict[str, Any], ...]:
    raw = os.getenv("UI_BANNERS", "").strip() or os.getenv("WEBUI_BANNERS", "").strip()
    if not raw:
        return ()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        message = _clean_banner_text(raw, 2000)
        if not message:
            return ()
        return (
            {
                "id": "banner-1",
                "type": "info",
                "title": "",
                "message": message,
                "dismissible": True,
            },
        )

    items: list[Any]
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = [parsed]
    else:
        return ()

    banners: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        fallback_id = f"banner-{index}"

        if isinstance(item, str):
            message = _clean_banner_text(item, 2000)
            if not message:
                continue
            banners.append(
                {
                    "id": fallback_id,
                    "type": "info",
                    "title": "",
                    "message": message,
                    "dismissible": True,
                }
            )
            continue

        if not isinstance(item, dict):
            continue
        if not _to_bool_value(item.get("enabled"), True):
            continue

        title = _clean_banner_text(item.get("title"), 200)
        message = _clean_banner_text(
            item.get("message") or item.get("content") or item.get("text") or item.get("body"),
            2000,
        )
        if not message:
            continue

        banners.append(
            {
                "id": _sanitize_banner_id(str(item.get("id") or ""), fallback_id),
                "type": _normalize_banner_type(item.get("type") or item.get("level")),
                "title": title,
                "message": message,
                "dismissible": _to_bool_value(item.get("dismissible"), True),
            }
        )

    return tuple(banners)



def load_settings() -> Settings:
    transcripts_dir = Path(os.getenv("DATA_DIR", "data/transcripts"))
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    diarization_work_dir = Path(os.getenv("DIARIZATION_WORK_DIR", "data/diarization"))
    diarization_work_dir.mkdir(parents=True, exist_ok=True)

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    summary_api_key = os.getenv("SUMMARY_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    summary_base_url = os.getenv("SUMMARY_BASE_URL", "").strip() or base_url
    proofread_api_key = (
        os.getenv("PROOFREAD_API_KEY", "").strip()
        or summary_api_key
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    proofread_base_url = os.getenv("PROOFREAD_BASE_URL", "").strip() or summary_base_url or base_url

    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_to_int("PORT", 8005),
        ws_path=os.getenv("WS_PATH", "/ws/transcribe"),
        transcripts_dir=transcripts_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=base_url,
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-1").strip() or "whisper-1",
        summary_api_key=summary_api_key,
        summary_base_url=summary_base_url,
        summary_model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        summary_temperature=_to_float("SUMMARY_TEMPERATURE", 0.2),
        summary_input_max_chars=max(2_000, _to_int("SUMMARY_INPUT_MAX_CHARS", 16_000)),
        proofread_api_key=proofread_api_key,
        proofread_base_url=proofread_base_url,
        proofread_model=os.getenv("PROOFREAD_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        proofread_temperature=_to_float("PROOFREAD_TEMPERATURE", 0.0),
        proofread_input_max_chars=max(2_000, _to_int("PROOFREAD_INPUT_MAX_CHARS", 24_000)),
        diarization_enabled=_to_bool("DIARIZATION_ENABLED", False),
        diarization_hf_token=os.getenv("DIARIZATION_HF_TOKEN", "").strip(),
        diarization_model=(
            os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1").strip()
            or "pyannote/speaker-diarization-3.1"
        ),
        diarization_device=os.getenv("DIARIZATION_DEVICE", "auto").strip() or "auto",
        diarization_sample_rate=max(8_000, _to_int("DIARIZATION_SAMPLE_RATE", 16_000)),
        diarization_num_speakers=max(0, _to_int("DIARIZATION_NUM_SPEAKERS", 0)),
        diarization_min_speakers=max(0, _to_int("DIARIZATION_MIN_SPEAKERS", 0)),
        diarization_max_speakers=max(0, _to_int("DIARIZATION_MAX_SPEAKERS", 0)),
        diarization_work_dir=diarization_work_dir,
        diarization_keep_chunks=_to_bool("DIARIZATION_KEEP_CHUNKS", False),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg",
        default_language=os.getenv("DEFAULT_LANGUAGE", "ja").strip() or "ja",
        default_prompt=os.getenv("DEFAULT_PROMPT", "").strip(),
        default_temperature=_to_float("DEFAULT_TEMPERATURE", 0.0),
        context_prompt_enabled=_to_bool("CONTEXT_PROMPT_ENABLED", True),
        context_max_chars=max(0, _to_int("CONTEXT_MAX_CHARS", 1000)),
        max_queue_size=max(1, _to_int("MAX_QUEUE_SIZE", 8)),
        max_chunk_bytes=max(1024, _to_int("MAX_CHUNK_BYTES", 12 * 1024 * 1024)),
        ui_banners=_parse_ui_banners(),
    )


settings = load_settings()
