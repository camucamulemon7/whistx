from __future__ import annotations

from fastapi import APIRouter, WebSocket
from sqlalchemy.orm import Session

from ...core.config import settings
from ... import legacy_app as legacy
from ...db import SessionLocal
from ...services.auth_service import get_optional_user_from_request

router = APIRouter()


@router.websocket(settings.ws_path)
async def ws_transcribe(ws: WebSocket) -> None:
    db: Session = SessionLocal()
    try:
        user = get_optional_user_from_request(ws, db)
    finally:
        db.close()
    if user is None:
        await ws.close(code=4401)
        return
    await legacy.ws_transcribe(ws)
