from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import env_first_non_empty, ensure_dir, to_bool, to_int, to_int_alias


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    ws_path: str
    app_data_dir: Path
    transcripts_dir: Path
    history_dir: Path
    app_db_url: str
    app_session_secret: str
    app_session_days: int
    enable_self_signup: bool


def load_app_config() -> AppConfig:
    app_data_dir_raw = env_first_non_empty("APP_DATA_DIR", "DATA_DIR")
    app_data_dir = Path(app_data_dir_raw) if app_data_dir_raw else Path("data")
    transcripts_dir = ensure_dir(Path(env_first_non_empty("APP_TRANSCRIPTS_DIR") or str(app_data_dir / "transcripts")))
    history_dir = ensure_dir(Path(env_first_non_empty("APP_HISTORY_DIR") or str(app_data_dir / "history")))
    return AppConfig(
        host=env_first_non_empty("APP_HOST", "HOST") or "0.0.0.0",
        port=to_int_alias(8005, "APP_PORT", "PORT"),
        ws_path=env_first_non_empty("APP_WS_PATH", "WS_PATH") or "/ws/transcribe",
        app_data_dir=app_data_dir,
        transcripts_dir=transcripts_dir,
        history_dir=history_dir,
        app_db_url=env_first_non_empty("APP_DB_URL") or f"sqlite:///{app_data_dir / 'app.db'}",
        app_session_secret=env_first_non_empty("APP_SESSION_SECRET") or "change-me",
        app_session_days=max(1, to_int("APP_SESSION_DAYS", 7)),
        enable_self_signup=to_bool("ENABLE_SELF_SIGNUP", False),
    )
