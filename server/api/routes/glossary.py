from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...deps import get_current_user
from ...models import User
from ...services.glossary_service import load_shared_glossary, save_shared_glossary

router = APIRouter()


class SharedGlossaryUpdateRequest(BaseModel):
    text: str = Field(default="", max_length=8000)


@router.get("/api/glossary/shared")
async def get_shared_glossary() -> dict[str, object]:
    payload = load_shared_glossary()
    return {"items": payload["text"], "updatedAt": payload["updatedAt"], "updatedBy": payload["updatedBy"]}


@router.put("/api/glossary/shared")
async def put_shared_glossary(
    payload: SharedGlossaryUpdateRequest,
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    record = save_shared_glossary(text=payload.text, updated_by=user.email)
    return {"ok": True, "items": record["text"], "updatedAt": record["updatedAt"], "updatedBy": record["updatedBy"]}
