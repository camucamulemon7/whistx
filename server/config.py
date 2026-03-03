from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
    default_language: str
    default_prompt: str
    default_temperature: float
    context_prompt_enabled: bool
    context_max_chars: int
    max_queue_size: int
    max_chunk_bytes: int



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



def load_settings() -> Settings:
    transcripts_dir = Path(os.getenv("DATA_DIR", "data/transcripts"))
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    summary_api_key = os.getenv("SUMMARY_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    summary_base_url = os.getenv("SUMMARY_BASE_URL", "").strip() or base_url

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
        default_language=os.getenv("DEFAULT_LANGUAGE", "ja").strip() or "ja",
        default_prompt=os.getenv("DEFAULT_PROMPT", "").strip(),
        default_temperature=_to_float("DEFAULT_TEMPERATURE", 0.0),
        context_prompt_enabled=_to_bool("CONTEXT_PROMPT_ENABLED", True),
        context_max_chars=max(0, _to_int("CONTEXT_MAX_CHARS", 1000)),
        max_queue_size=max(1, _to_int("MAX_QUEUE_SIZE", 8)),
        max_chunk_bytes=max(1024, _to_int("MAX_CHUNK_BYTES", 12 * 1024 * 1024)),
    )


settings = load_settings()
