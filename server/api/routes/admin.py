from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from ...db import get_db
from ...deps import get_current_admin
from ...models import User
from ... import legacy_app as legacy

router = APIRouter()


@router.get("/api/admin/pending-users")
async def admin_pending_users(user: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> JSONResponse:
    return await legacy.admin_pending_users(user, db)


@router.post("/api/admin/pending-users/{user_id}/approve")
async def admin_approve_pending_user(
    user_id: int,
    user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return await legacy.admin_approve_pending_user(user_id, user, db)


@router.get("/api/admin/users")
async def admin_users(user: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> JSONResponse:
    return await legacy.admin_users(user, db)


@router.post("/api/admin/users/{user_id}/role")
async def admin_update_user_role(
    user_id: int,
    request: Request,
    user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return await legacy.admin_update_user_role(user_id, request, user, db)


@router.get("/admin", response_model=None)
async def admin_page(user: User = Depends(get_current_admin)) -> Response:
    return await legacy.admin_page(user)
