from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ... import runtime
from ...core.config import settings
from ...core.logging import emit_container_log
from ...db import schema_revision_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/api/health/live")
def liveness() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.get("/api/health/ready")
async def readiness() -> JSONResponse:
    return await _readiness_response()


@router.get("/api/health")
async def health() -> JSONResponse:
    """Backward-compatible readiness endpoint."""
    return await _readiness_response()


async def _readiness_response() -> JSONResponse:
    emit_container_log(__name__, "debug", "readiness requested")
    logger.debug("readiness requested")
    schema = await asyncio.to_thread(schema_revision_status)
    if not schema.ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "database": {
                    "ready": False,
                    "currentRevisions": list(schema.current_revisions),
                    "expectedRevisions": list(schema.expected_revisions),
                    "error": schema.error,
                },
            },
        )
    response = await runtime.health()
    provider_required = bool(settings.openai_api_key)
    provider_ready = runtime.TRANSCRIBER_FACTORY is not None
    payload = json.loads(response.body)
    payload.update(
        {
            "status": "ok" if (not provider_required or provider_ready) else "not_ready",
            "database": {
                "ready": True,
                "currentRevisions": list(schema.current_revisions),
                "expectedRevisions": list(schema.expected_revisions),
            },
            "asr": {
                "required": provider_required,
                "ready": provider_ready,
            },
        }
    )
    if provider_required and not provider_ready:
        return JSONResponse(status_code=503, content=payload)
    return JSONResponse(
        payload,
        headers={"X-Database-Revision": ",".join(schema.current_revisions)},
    )
