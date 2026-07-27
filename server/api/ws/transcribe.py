from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket

from ... import runtime
from ...core.config import settings
from ...core.security import (
    client_ip,
    digest_guest_artifact_grant,
    origin_is_allowed,
    read_guest_artifact_grant,
)
from ...db import db_session
from ...repositories import quota_repository
from ...services.auth_service import get_optional_user_from_request

router = APIRouter()


def _client_ip(ws: WebSocket) -> str:
    return client_ip(ws)


async def _close_safely(ws: WebSocket, code: int, reason: str) -> None:
    try:
        await ws.close(code=code, reason=reason)
    except RuntimeError:
        pass


@router.websocket(settings.ws_path)
async def ws_transcribe(ws: WebSocket) -> None:
    if not origin_is_allowed(ws):
        await _close_safely(ws, 4403, "origin_not_allowed")
        return

    with db_session() as db:
        user = get_optional_user_from_request(ws, db)
    if user is not None and not user.is_active:
        await _close_safely(ws, 4403, "inactive_user")
        return
    if user is not None:
        lease_id = await asyncio.to_thread(
            quota_repository.acquire_connection_lease,
            subject=f"user:{user.id}",
            is_guest=False,
            ttl_seconds=24 * 60 * 60,
        )
        if lease_id is None:
            await _close_safely(ws, 4429, "connection_limit")
            return
        ws.state.authenticated_user_id = user.id
        ws.state.rate_limit_subject = f"user:{user.id}"
        try:
            await runtime.ws_transcribe(ws)
        finally:
            await asyncio.to_thread(quota_repository.release_connection_lease, lease_id)
        return

    if not settings.allow_guest_transcription:
        await _close_safely(ws, 4401, "authentication_required")
        return

    guest_grant_id = read_guest_artifact_grant(ws)
    if not guest_grant_id:
        await _close_safely(ws, 4401, "guest_grant_required")
        return

    client_ip = _client_ip(ws)
    lease_id = await asyncio.to_thread(
        quota_repository.acquire_connection_lease,
        subject=f"ip:{client_ip}",
        is_guest=True,
        ttl_seconds=settings.guest_ws_max_duration_seconds + 60,
        guest_total_limit=settings.guest_ws_max_connections,
        guest_subject_limit=settings.guest_ws_max_per_ip,
    )
    if lease_id is None:
        await _close_safely(ws, 4429, "guest_connection_limit")
        return

    ws.state.is_guest = True
    ws.state.rate_limit_subject = f"ip:{client_ip}"
    ws.state.guest_artifact_grant_digest = digest_guest_artifact_grant(guest_grant_id)
    try:
        await asyncio.wait_for(runtime.ws_transcribe(ws), timeout=settings.guest_ws_max_duration_seconds)
    except TimeoutError:
        await _close_safely(ws, 4408, "guest_duration_limit")
    finally:
        await asyncio.to_thread(quota_repository.release_connection_lease, lease_id)
