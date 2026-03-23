from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from ... import legacy_app as legacy
from ...db import get_db
from ...deps import get_current_admin
from ...models import User
from ...services.admin_service import AdminServiceError, approve_pending_user, list_pending_users_payload, list_users_payload, update_user_role

router = APIRouter()


@router.get('/api/admin/pending-users')
async def admin_pending_users(user: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> JSONResponse:
    return JSONResponse(list_pending_users_payload(db))


@router.post('/api/admin/pending-users/{user_id}/approve')
async def admin_approve_pending_user(
    user_id: int,
    user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    try:
        return JSONResponse(approve_pending_user(db, pending_user_id=user_id, admin=user))
    except AdminServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content={'error': exc.code})


@router.get('/api/admin/users')
async def admin_users(user: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> JSONResponse:
    return JSONResponse(list_users_payload(db))


@router.post('/api/admin/users/{user_id}/role')
async def admin_update_user_role(
    user_id: int,
    request: Request,
    user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={'error': 'invalid_json'})
    try:
        return JSONResponse(update_user_role(db, user_id=user_id, role=payload.get('role')))
    except AdminServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content={'error': exc.code})


@router.get('/admin', response_model=None)
async def admin_page(user: User = Depends(get_current_admin)) -> Response:
    return await legacy.admin_page(user)
