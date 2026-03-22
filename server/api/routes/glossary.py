from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...core.logging import emit_container_log
from ...deps import get_current_user
from ...models import User
from ...services.glossary_service import load_shared_glossary, save_shared_glossary

router = APIRouter()
logger = logging.getLogger(__name__)


class SharedGlossaryUpdateRequest(BaseModel):
    text: str = Field(default="", max_length=8000)


@router.get("/api/glossary/shared")
async def get_shared_glossary() -> dict[str, object]:
    emit_container_log(__name__, "debug", "shared glossary requested")
    logger.debug("shared glossary requested")
    payload = load_shared_glossary()
    return {"items": payload["text"], "updatedAt": payload["updatedAt"], "updatedBy": payload["updatedBy"]}


@router.put("/api/glossary/shared")
async def put_shared_glossary(
    payload: SharedGlossaryUpdateRequest,
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    emit_container_log(__name__, "debug", "shared glossary update requested: user=%s chars=%s", user.email, len(payload.text or ""))
    logger.debug("shared glossary update requested: user=%s chars=%s", user.email, len(payload.text or ""))
    record = save_shared_glossary(text=payload.text, updated_by=user.email)
    return {"ok": True, "items": record["text"], "updatedAt": record["updatedAt"], "updatedBy": record["updatedBy"]}
