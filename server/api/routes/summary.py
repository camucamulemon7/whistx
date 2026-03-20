from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from ... import legacy_app as legacy

router = APIRouter()


@router.post("/api/summarize")
async def summarize(payload: legacy.SummarizeRequest) -> JSONResponse:
    return await legacy.summarize(payload)


@router.post("/api/proofread")
async def proofread(payload: legacy.ProofreadRequest) -> JSONResponse:
    return await legacy.proofread(payload)


@router.post("/api/proofread/stream")
async def proofread_stream(payload: legacy.ProofreadRequest) -> Response:
    return await legacy.proofread_stream(payload)
