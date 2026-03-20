from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ... import legacy_app as legacy

router = APIRouter()


@router.get("/api/health")
async def health() -> JSONResponse:
    return await legacy.health()
