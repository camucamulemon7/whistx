from __future__ import annotations

from fastapi import APIRouter, WebSocket

from ...core.config import settings
from ... import legacy_app as legacy

router = APIRouter()


@router.websocket(settings.ws_path)
async def ws_transcribe(ws: WebSocket) -> None:
    await legacy.ws_transcribe(ws)
