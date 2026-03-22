from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .. import legacy_app as legacy
from ..api.routes import admin, auth, glossary, health, history, summary, transcript
from ..api.ws.transcribe import router as transcribe_router


def create_app() -> FastAPI:
    app = FastAPI(title="whistx", version="2.0.0")
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

    web_dir = Path("web")
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
    return app
