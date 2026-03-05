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
    asr_model: str
    summary_api_key: str
    summary_base_url: str | None
    summary_model: str
    summary_temperature: float
    summary_input_max_chars: int
    summary_system_prompt: str
    summary_prompt_template: str
    proofread_api_key: str
    proofread_base_url: str | None
    proofread_model: str
    proofread_temperature: float
    proofread_input_max_chars: int
    proofread_system_prompt: str
    proofread_prompt_template: str
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
    app_brand_title: str
    app_brand_tagline: str
    ui_banners: tuple[dict[str, Any], ...]


def _env_first_non_empty(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return value
    return None


def _to_int_alias(default: int, *names: str) -> int:
    raw = _env_first_non_empty(*names)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _to_float_alias(default: float, *names: str) -> float:
    raw = _env_first_non_empty(*names)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _to_bool_alias(default: bool, *names: str) -> bool:
    raw = _env_first_non_empty(*names)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}



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


def _decode_env_text(value: str) -> str:
    # Allow writing newline as \n in .env while keeping plain values readable.
    return value.replace("\\n", "\n")


def _parse_ui_banners() -> tuple[dict[str, Any], ...]:
    raw = (
        _env_first_non_empty("APP_UI_BANNERS", "UI_BANNERS", "WEBUI_BANNERS")
        or ""
    )
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
    app_data_dir_raw = _env_first_non_empty("APP_DATA_DIR", "DATA_DIR")
    if app_data_dir_raw:
        app_data_dir = Path(app_data_dir_raw)
    else:
        app_data_dir = Path("data")

    transcripts_dir = Path(
        _env_first_non_empty("APP_TRANSCRIPTS_DIR")
        or str(app_data_dir / "transcripts")
    )
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    diarization_work_dir = Path(
        _env_first_non_empty("DIARIZATION_WORK_DIR")
        or str(app_data_dir / "diarization")
    )
    diarization_work_dir.mkdir(parents=True, exist_ok=True)

    asr_api_key = _env_first_non_empty("ASR_API_KEY", "OPENAI_API_KEY") or ""
    base_url = _env_first_non_empty("ASR_BASE_URL", "OPENAI_BASE_URL")
    summary_api_key = _env_first_non_empty("SUMMARY_API_KEY") or asr_api_key
    summary_base_url = _env_first_non_empty("SUMMARY_BASE_URL") or base_url
    summary_system_prompt = _decode_env_text(_env_first_non_empty("SUMMARY_SYSTEM_PROMPT") or "")
    summary_prompt_template = _decode_env_text(_env_first_non_empty("SUMMARY_PROMPT_TEMPLATE") or "")
    proofread_api_key = (
        _env_first_non_empty("PROOFREAD_API_KEY")
        or summary_api_key
        or asr_api_key
    )
    proofread_base_url = _env_first_non_empty("PROOFREAD_BASE_URL") or summary_base_url or base_url
    proofread_system_prompt = _decode_env_text(_env_first_non_empty("PROOFREAD_SYSTEM_PROMPT") or "")
    proofread_prompt_template = _decode_env_text(_env_first_non_empty("PROOFREAD_PROMPT_TEMPLATE") or "")

    asr_model = (
        _env_first_non_empty("ASR_MODEL")
        or _env_first_non_empty("WHISPER_MODEL")
        or "mistralai/Voxtral-Mini-4B-Realtime-2602"
    )
    app_brand_title = _decode_env_text(_env_first_non_empty("APP_BRAND_TITLE") or "whistx")
    app_brand_tagline = _decode_env_text(
        _env_first_non_empty("APP_BRAND_TAGLINE")
        or "高精度リアルタイム文字起こし"
    )

    return Settings(
        host=_env_first_non_empty("APP_HOST", "HOST") or "0.0.0.0",
        port=_to_int_alias(8005, "APP_PORT", "PORT"),
        ws_path=_env_first_non_empty("APP_WS_PATH", "WS_PATH") or "/ws/transcribe",
        transcripts_dir=transcripts_dir,
        openai_api_key=asr_api_key,
        openai_base_url=base_url,
        asr_model=asr_model,
        summary_api_key=summary_api_key,
        summary_base_url=summary_base_url,
        summary_model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        summary_temperature=_to_float("SUMMARY_TEMPERATURE", 0.2),
        summary_input_max_chars=max(2_000, _to_int("SUMMARY_INPUT_MAX_CHARS", 16_000)),
        summary_system_prompt=summary_system_prompt,
        summary_prompt_template=summary_prompt_template,
        proofread_api_key=proofread_api_key,
        proofread_base_url=proofread_base_url,
        proofread_model=os.getenv("PROOFREAD_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        proofread_temperature=_to_float("PROOFREAD_TEMPERATURE", 0.0),
        proofread_input_max_chars=max(2_000, _to_int("PROOFREAD_INPUT_MAX_CHARS", 24_000)),
        proofread_system_prompt=proofread_system_prompt,
        proofread_prompt_template=proofread_prompt_template,
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
        ffmpeg_bin=(
            _env_first_non_empty("DIARIZATION_FFMPEG_BIN", "FFMPEG_BIN")
            or "ffmpeg"
        ),
        default_language=_env_first_non_empty("ASR_DEFAULT_LANGUAGE", "DEFAULT_LANGUAGE") or "ja",
        default_prompt=_env_first_non_empty("ASR_DEFAULT_PROMPT", "DEFAULT_PROMPT") or "",
        default_temperature=_to_float_alias(0.0, "ASR_DEFAULT_TEMPERATURE", "DEFAULT_TEMPERATURE"),
        context_prompt_enabled=_to_bool_alias(True, "ASR_CONTEXT_PROMPT_ENABLED", "CONTEXT_PROMPT_ENABLED"),
        context_max_chars=max(0, _to_int_alias(1000, "ASR_CONTEXT_MAX_CHARS", "CONTEXT_MAX_CHARS")),
        max_queue_size=max(1, _to_int_alias(8, "ASR_MAX_QUEUE_SIZE", "MAX_QUEUE_SIZE")),
        max_chunk_bytes=max(
            1024,
            _to_int_alias(12 * 1024 * 1024, "ASR_MAX_CHUNK_BYTES", "MAX_CHUNK_BYTES"),
        ),
        app_brand_title=app_brand_title,
        app_brand_tagline=app_brand_tagline,
        ui_banners=_parse_ui_banners(),
    )


settings = load_settings()
