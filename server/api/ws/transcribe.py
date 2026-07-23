from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, WebSocket

from ... import legacy_app as legacy
from ...core.config import settings
from ...db import db_session
from ...services.auth_service import get_optional_user_from_request

router = APIRouter()
_GUEST_CONNECTIONS_BY_IP: Counter[str] = Counter()
_GUEST_CONNECTIONS_TOTAL = 0


def _client_ip(ws: WebSocket) -> str:
    return ws.client.host if ws.client and ws.client.host else "unknown"


async def _close_safely(ws: WebSocket, code: int, reason: str) -> None:
    try:
        await ws.close(code=code, reason=reason)
    except RuntimeError:
        pass


@router.websocket(settings.ws_path)
async def ws_transcribe(ws: WebSocket) -> None:
    global _GUEST_CONNECTIONS_TOTAL

    with db_session() as db:
        user = get_optional_user_from_request(ws, db)
    if user is not None and not user.is_active:
        await _close_safely(ws, 4403, "inactive_user")
        return
    if user is not None:
        ws.state.authenticated_user_id = user.id
        await legacy.ws_transcribe(ws)
        return

    if not settings.allow_guest_transcription:
        await _close_safely(ws, 4401, "authentication_required")
        return

    client_ip = _client_ip(ws)
    if (
        _GUEST_CONNECTIONS_TOTAL >= settings.guest_ws_max_connections
        or _GUEST_CONNECTIONS_BY_IP[client_ip] >= settings.guest_ws_max_per_ip
    ):
        await _close_safely(ws, 4429, "guest_connection_limit")
        return

    _GUEST_CONNECTIONS_TOTAL += 1
    _GUEST_CONNECTIONS_BY_IP[client_ip] += 1
    ws.state.is_guest = True
    try:
        await asyncio.wait_for(legacy.ws_transcribe(ws), timeout=settings.guest_ws_max_duration_seconds)
    except TimeoutError:
        await _close_safely(ws, 4408, "guest_duration_limit")
    finally:
        _GUEST_CONNECTIONS_TOTAL -= 1
        _GUEST_CONNECTIONS_BY_IP[client_ip] -= 1
        if _GUEST_CONNECTIONS_BY_IP[client_ip] <= 0:
            del _GUEST_CONNECTIONS_BY_IP[client_ip]
