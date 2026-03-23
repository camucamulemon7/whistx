from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .. import legacy_app as legacy
from ..api.routes import admin, auth, glossary, health, history, summary, transcript
from ..api.ws.transcribe import router as transcribe_router
from .logging import configure_application_logging, emit_container_log

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_application_logging(legacy.settings.app_log_level)
    app = FastAPI(title="whistx", version="2.0.0")

    @app.middleware("http")
    async def log_http_requests(request: Request, call_next):
        started_at = time.perf_counter()
        emit_container_log(__name__, "debug", "http request: method=%s path=%s", request.method, request.url.path)
        logger.debug("http request: method=%s path=%s", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started_at) * 1000)
            emit_container_log(__name__, "info", "http error: method=%s path=%s duration_ms=%s", request.method, request.url.path, duration_ms)
            logger.exception(
                "http error: method=%s path=%s duration_ms=%s",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000)
        emit_container_log(__name__, "debug", "http response: method=%s path=%s status=%s duration_ms=%s", request.method, request.url.path, response.status_code, duration_ms)
        logger.debug(
            "http response: method=%s path=%s status=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    app.add_event_handler("startup", legacy.on_startup)
    app.add_event_handler("shutdown", legacy.on_shutdown)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(glossary.router)
    app.include_router(history.router)
    app.include_router(summary.router)
    app.include_router(transcript.router)
    app.include_router(transcribe_router)

    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
    return app
