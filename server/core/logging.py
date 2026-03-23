from __future__ import annotations

import logging
from datetime import datetime, timezone
import sys


APP_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_APP_LOG_LEVEL = logging.INFO


def normalize_log_level(value: str | int | None) -> int:
    if isinstance(value, int):
        return value
    name = str(value or "INFO").strip().upper()
    return logging.DEBUG if name == "DEBUG" else logging.INFO


def configure_application_logging(level: int | str = logging.INFO) -> None:
    global _APP_LOG_LEVEL
    normalized_level = normalize_log_level(level)
    _APP_LOG_LEVEL = normalized_level
    logger = logging.getLogger("server")
    logger.setLevel(normalized_level)

    has_stdout_handler = False
    for handler in logger.handlers:
        stream = getattr(handler, "stream", None)
        if stream is sys.stdout or stream is sys.stderr:
            has_stdout_handler = True
            handler.setLevel(normalized_level)
            if handler.formatter is None:
                handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))

    if not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(normalized_level)
        handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))
        logger.addHandler(handler)

    logger.propagate = False


def emit_container_log(name: str, level: str, message: str, *args) -> None:
    numeric_level = normalize_log_level(level)
    if numeric_level < _APP_LOG_LEVEL:
        return
    rendered = message % args if args else message
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"{timestamp} [{level.upper()}] {name}: {rendered}", file=sys.stdout, flush=True)


def is_debug_logging_enabled() -> bool:
    return _APP_LOG_LEVEL <= logging.DEBUG
