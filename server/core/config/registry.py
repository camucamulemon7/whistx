"""Canonical environment-variable registry.

Keep this registry aligned with the config loaders, ``.env.example`` and
``scripts/container_common.sh``. Tests enforce that contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    name: str
    kind: str
    default: str
    allowed: str = ""
    secret: bool = False
    aliases: tuple[str, ...] = ()
    deprecated_aliases: tuple[str, ...] = ()


def spec(
    name: str,
    kind: str,
    default: str,
    allowed: str = "",
    *,
    secret: bool = False,
    aliases: tuple[str, ...] = (),
) -> EnvSpec:
    return EnvSpec(name, kind, default, allowed, secret, aliases, aliases)


ENV_REGISTRY = (
    spec("APP_ENV", "enum", "development", "development|production"),
    spec("APP_HOST", "string", "0.0.0.0", aliases=("HOST",)),
    spec("APP_PORT", "integer", "8005", "1..65535", aliases=("PORT",)),
    spec("APP_WS_PATH", "path", "/ws/transcribe", aliases=("WS_PATH",)),
    spec("APP_DATA_DIR", "path", "data", aliases=("DATA_DIR",)),
    spec("APP_TRANSCRIPTS_DIR", "path", "", "defaults to APP_DATA_DIR/transcripts"),
    spec("APP_HISTORY_DIR", "path", "", "defaults to APP_DATA_DIR/history"),
    spec("APP_DEBUG_CHUNKS_DIR", "path", "", "defaults to APP_DATA_DIR/debug_chunks"),
    spec("APP_LOG_LEVEL", "enum", "INFO", "DEBUG|INFO"),
    spec("APP_DB_URL", "url", "", "development default: sqlite under APP_DATA_DIR", secret=True),
    spec("APP_SESSION_SECRET", "string", "", "at least 32 characters", secret=True),
    spec("APP_SESSION_DAYS", "integer", "7", ">=1"),
    spec("ENABLE_SELF_SIGNUP", "boolean", "0"),
    spec("ALLOW_GUEST_TRANSCRIPTION", "boolean", "0"),
    spec("GUEST_WS_MAX_PER_IP", "integer", "2", ">=1"),
    spec("GUEST_WS_MAX_CONNECTIONS", "integer", "10", ">=1"),
    spec("GUEST_WS_MAX_DURATION_SECONDS", "integer", "900", ">=1"),
    spec("GUEST_WS_MAX_AUDIO_BYTES", "integer", "104857600", ">=1024"),
    spec("GUEST_WS_MAX_ASR_REQUESTS", "integer", "120", ">=1"),
    spec("HISTORY_RETENTION_DAYS", "integer", "7", ">=1"),
    spec("RUNTIME_TRANSCRIPT_RETENTION_HOURS", "integer", "24", ">=1"),
    spec("DEBUG_CHUNKS_RETENTION_HOURS", "integer", "24", ">=1"),
    spec("UNSAVED_RUNTIME_RETENTION_HOURS", "integer", "24", ">=1"),
    spec("ASR_API_KEY", "string", "", secret=True, aliases=("OPENAI_API_KEY",)),
    spec("ASR_BASE_URL", "url", "", aliases=("OPENAI_BASE_URL",)),
    spec("ASR_MODEL", "string", "whisper-1", aliases=("WHISPER_MODEL",)),
    spec("ASR_API_TIMEOUT_SECONDS", "float", "60", ">=1"),
    spec("SUMMARY_API_TIMEOUT_SECONDS", "float", "60", ">=1"),
    spec("PROOFREAD_API_TIMEOUT_SECONDS", "float", "60", ">=1"),
    spec("FFMPEG_TIMEOUT_SECONDS", "float", "30", ">=1"),
    spec("ASR_PREPROCESS_ENABLED", "boolean", "1"),
    spec("ASR_PREPROCESS_SAMPLE_RATE", "integer", "16000", ">=8000"),
    spec("ASR_OVERLAP_MS", "integer", "3500", ">=0"),
    spec("ASR_VAD_DROP_ENABLED", "boolean", "1"),
    spec("ASR_VAD_SPEECH_RATIO_MIN", "float", "0.02", "0..1"),
    spec("ASR_RETRY_MAX_ATTEMPTS", "integer", "3", ">=1"),
    spec("ASR_RETRY_BASE_DELAY_MS", "integer", "1000", ">=0"),
    spec("ASR_MULTI_PASS_ENABLED", "boolean", "1"),
    spec("ASR_RESCUE_RETRY_ENABLED", "boolean", "1"),
    spec("ASR_RESCUE_RETRY_TEMPERATURE", "float", "0.25", "0..1"),
    spec("ASR_LIGHT_PROOFREAD_ENABLED", "boolean", "1"),
    spec("ASR_DEFAULT_LANGUAGE", "string", "ja", aliases=("DEFAULT_LANGUAGE",)),
    spec("ASR_DEFAULT_PROMPT", "string", "", aliases=("DEFAULT_PROMPT",)),
    spec("ASR_DEFAULT_TEMPERATURE", "float", "0.0", aliases=("DEFAULT_TEMPERATURE",)),
    spec("ASR_CONTEXT_PROMPT_ENABLED", "boolean", "1", aliases=("CONTEXT_PROMPT_ENABLED",)),
    spec("ASR_CONTEXT_MAX_CHARS", "integer", "2200", ">=0", aliases=("CONTEXT_MAX_CHARS",)),
    spec("ASR_CONTEXT_RECENT_LINES", "integer", "4", ">=1"),
    spec("ASR_CONTEXT_TERM_LIMIT", "integer", "80", ">=8"),
    spec("ASR_MAX_QUEUE_SIZE", "integer", "8", ">=1", aliases=("MAX_QUEUE_SIZE",)),
    spec("ASR_MAX_CHUNK_BYTES", "integer", "12582912", ">=1024", aliases=("MAX_CHUNK_BYTES",)),
    spec("SUMMARY_API_KEY", "string", "", "falls back to ASR_API_KEY", secret=True),
    spec("SUMMARY_BASE_URL", "url", "", "falls back to ASR_BASE_URL"),
    spec("SUMMARY_MODEL", "string", "gpt-4o-mini"),
    spec("SUMMARY_TEMPERATURE", "float", "0.2"),
    spec("SUMMARY_INPUT_MAX_CHARS", "integer", "16000", ">=2000"),
    spec("SUMMARY_SYSTEM_PROMPT", "string", ""),
    spec("SUMMARY_PROMPT_TEMPLATE", "string", ""),
    spec("PROOFREAD_API_KEY", "string", "", "falls back to SUMMARY_API_KEY", secret=True),
    spec("PROOFREAD_BASE_URL", "url", "", "falls back to SUMMARY_BASE_URL"),
    spec("PROOFREAD_MODEL", "string", "gpt-4o-mini"),
    spec("PROOFREAD_TEMPERATURE", "float", "0.0"),
    spec("PROOFREAD_INPUT_MAX_CHARS", "integer", "24000", ">=2000"),
    spec("PROOFREAD_SYSTEM_PROMPT", "string", ""),
    spec("PROOFREAD_PROMPT_TEMPLATE", "string", ""),
    spec("APP_UI_BANNERS_TEXT", "string", ""),
    spec("APP_UI_BANNERS", "json", "", aliases=("UI_BANNERS", "WEBUI_BANNERS")),
    spec("APP_BRAND_TITLE", "string", "whistx"),
    spec("APP_BRAND_TAGLINE", "string", "高精度リアルタイム文字起こし"),
    spec("APP_PROMPT_TEMPLATES", "json", "", aliases=("APP_SOC_PROMPT_TEMPLATE",)),
    spec("DIARIZATION_ENABLED", "boolean", "0"),
    spec("DIARIZATION_HF_TOKEN", "string", "", secret=True),
    spec("DIARIZATION_MODEL", "string", "pyannote/speaker-diarization-3.1"),
    spec("DIARIZATION_DEVICE", "string", "auto"),
    spec("DIARIZATION_SAMPLE_RATE", "integer", "16000", ">=8000"),
    spec("DIARIZATION_NUM_SPEAKERS", "integer", "0", ">=0"),
    spec("DIARIZATION_MIN_SPEAKERS", "integer", "0", ">=0"),
    spec("DIARIZATION_MAX_SPEAKERS", "integer", "0", ">=0"),
    spec("DIARIZATION_WORK_DIR", "path", "", "defaults to APP_DATA_DIR/diarization"),
    spec("DIARIZATION_KEEP_CHUNKS", "boolean", "0"),
    spec("DIARIZATION_FFMPEG_BIN", "path", "ffmpeg", aliases=("FFMPEG_BIN",)),
    spec("KEYCLOAK_ENABLED", "boolean", "0"),
    spec("KEYCLOAK_ISSUER", "url", ""),
    spec("KEYCLOAK_CLIENT_ID", "string", ""),
    spec("KEYCLOAK_CLIENT_SECRET", "string", "", secret=True),
    spec("KEYCLOAK_SCOPE", "string", "openid profile email"),
    spec("KEYCLOAK_BUTTON_LABEL", "string", "Keycloakでログイン"),
    spec("KEYCLOAK_REQUIRE_EMAIL_VERIFIED", "boolean", "1"),
    spec("LANGFUSE_ENABLED", "boolean", "1"),
    spec("LANGFUSE_PUBLIC_KEY", "string", "", secret=True),
    spec("LANGFUSE_SECRET_KEY", "string", "", secret=True),
    spec("LANGFUSE_HOST", "url", ""),
    spec("LANGFUSE_ENVIRONMENT", "string", "", aliases=("LANGFUSE_ENV",)),
    spec("LANGFUSE_RELEASE", "string", ""),
)

ENV_BY_NAME = {item.name: item for item in ENV_REGISTRY}
