from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import env_first_non_empty, ensure_dir, to_bool, to_int, to_int_alias

MIN_SESSION_SECRET_LENGTH = 32


@dataclass(frozen=True)
class AppConfig:
    app_env: str
    host: str
    port: int
    ws_path: str
    app_data_dir: Path
    transcripts_dir: Path
    history_dir: Path
    app_log_level: str
    debug_chunks_dir: Path
    app_db_url: str
    app_session_secret: str
    app_session_days: int
    app_public_url: str | None
    app_trust_proxy_headers: bool
    app_trusted_proxy_ips: tuple[str, ...]
    app_allowed_hosts: tuple[str, ...]
    enable_self_signup: bool
    allow_guest_transcription: bool
    guest_ws_max_per_ip: int
    guest_ws_max_connections: int
    guest_ws_max_duration_seconds: int
    guest_ws_max_audio_bytes: int
    guest_ws_max_asr_requests: int
    costly_api_rate_limit_requests: int
    costly_api_rate_limit_window_seconds: int
    history_retention_days: int
    runtime_transcript_retention_hours: int
    debug_chunks_retention_hours: int
    unsaved_runtime_retention_hours: int


def load_app_config() -> AppConfig:
    app_data_dir_raw = env_first_non_empty("APP_DATA_DIR", "DATA_DIR")
    app_data_dir = Path(app_data_dir_raw) if app_data_dir_raw else Path("data")
    app_env = (env_first_non_empty("APP_ENV") or "development").strip().lower()
    if app_env not in {"development", "production"}:
        app_env = "development"
    transcripts_dir = ensure_dir(Path(env_first_non_empty("APP_TRANSCRIPTS_DIR") or str(app_data_dir / "transcripts")))
    history_dir = ensure_dir(Path(env_first_non_empty("APP_HISTORY_DIR") or str(app_data_dir / "history")))
    debug_chunks_dir = ensure_dir(Path(env_first_non_empty("APP_DEBUG_CHUNKS_DIR") or str(app_data_dir / "debug_chunks")))
    app_log_level = (env_first_non_empty("APP_LOG_LEVEL") or "INFO").strip().upper()
    if app_log_level not in {"DEBUG", "INFO"}:
        app_log_level = "INFO"
    app_session_secret = env_first_non_empty("APP_SESSION_SECRET") or "change-me"
    if app_session_secret.strip() == "change-me" or len(app_session_secret.strip()) < MIN_SESSION_SECRET_LENGTH:
        raise RuntimeError(
            f"APP_SESSION_SECRET must be set to a non-default value with at least {MIN_SESSION_SECRET_LENGTH} characters"
        )
    app_db_url = env_first_non_empty("APP_DB_URL") or f"sqlite:///{app_data_dir / 'app.db'}"
    if app_env == "production" and app_db_url.startswith("sqlite"):
        raise RuntimeError("APP_ENV=production requires PostgreSQL-compatible APP_DB_URL; SQLite is not allowed")

    return AppConfig(
        app_env=app_env,
        host=env_first_non_empty("APP_HOST", "HOST") or "0.0.0.0",
        port=to_int_alias(8005, "APP_PORT", "PORT"),
        ws_path=env_first_non_empty("APP_WS_PATH", "WS_PATH") or "/ws/transcribe",
        app_data_dir=app_data_dir,
        transcripts_dir=transcripts_dir,
        history_dir=history_dir,
        app_log_level=app_log_level,
        debug_chunks_dir=debug_chunks_dir,
        app_db_url=app_db_url,
        app_session_secret=app_session_secret,
        app_session_days=max(1, to_int("APP_SESSION_DAYS", 7)),
        app_public_url=env_first_non_empty("APP_PUBLIC_URL"),
        app_trust_proxy_headers=to_bool("APP_TRUST_PROXY_HEADERS", False),
        app_trusted_proxy_ips=tuple(
            item.strip() for item in (env_first_non_empty("APP_TRUSTED_PROXY_IPS") or "").split(",") if item.strip()
        ),
        app_allowed_hosts=tuple(
            item.strip()
            for item in (env_first_non_empty("APP_ALLOWED_HOSTS") or "localhost,127.0.0.1,[::1],testserver").split(",")
            if item.strip()
        ),
        enable_self_signup=to_bool("ENABLE_SELF_SIGNUP", False),
        allow_guest_transcription=to_bool("ALLOW_GUEST_TRANSCRIPTION", False),
        guest_ws_max_per_ip=max(1, to_int("GUEST_WS_MAX_PER_IP", 2)),
        guest_ws_max_connections=max(1, to_int("GUEST_WS_MAX_CONNECTIONS", 10)),
        guest_ws_max_duration_seconds=max(1, to_int("GUEST_WS_MAX_DURATION_SECONDS", 900)),
        guest_ws_max_audio_bytes=max(1024, to_int("GUEST_WS_MAX_AUDIO_BYTES", 100 * 1024 * 1024)),
        guest_ws_max_asr_requests=max(1, to_int("GUEST_WS_MAX_ASR_REQUESTS", 120)),
        costly_api_rate_limit_requests=max(1, to_int("COSTLY_API_RATE_LIMIT_REQUESTS", 60)),
        costly_api_rate_limit_window_seconds=max(1, to_int("COSTLY_API_RATE_LIMIT_WINDOW_SECONDS", 60)),
        history_retention_days=max(1, to_int("HISTORY_RETENTION_DAYS", 7)),
        runtime_transcript_retention_hours=max(1, to_int("RUNTIME_TRANSCRIPT_RETENTION_HOURS", 24)),
        debug_chunks_retention_hours=max(1, to_int("DEBUG_CHUNKS_RETENTION_HOURS", 24)),
        unsaved_runtime_retention_hours=max(1, to_int("UNSAVED_RUNTIME_RETENTION_HOURS", 24)),
    )
