from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from ... import legacy_app as legacy
from ...core.security import clear_session_cookie, set_session_cookie, serialize_user
from ...db import get_db
from ...schemas import BootstrapAdminRequest, LoginRequest, RegisterRequest
from ...services.auth_service import (
    AuthServiceError,
    build_auth_me_payload,
    bootstrap_admin,
    login_user,
    logout_user,
    register_user,
)

router = APIRouter()


@router.get('/api/auth/me')
async def auth_me(request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    return JSONResponse(build_auth_me_payload(request, db))


@router.post('/api/auth/login')
async def auth_login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    try:
        result = login_user(payload, request, db)
    except AuthServiceError as exc:
        content = {'error': exc.code}
        if exc.retry_after_sec is not None:
            content['retryAfterSec'] = exc.retry_after_sec
        return JSONResponse(status_code=exc.status_code, content=content)
    response = JSONResponse({'ok': True, 'user': serialize_user(result.user)})
    set_session_cookie(response=response, request=request, cookie_name='whistx_session', session_id=result.session_id)
    return response


@router.post('/api/auth/bootstrap-admin')
async def auth_bootstrap_admin(
    payload: BootstrapAdminRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> JSONResponse:
    try:
        result = bootstrap_admin(payload, request, db)
    except AuthServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content={'error': exc.code})
    response = JSONResponse({'ok': True, 'user': serialize_user(result.user)})
    set_session_cookie(response=response, request=request, cookie_name='whistx_session', session_id=result.session_id)
    return response


@router.get('/api/auth/keycloak/login')
async def auth_keycloak_login(request: Request) -> Response:
    return await legacy.auth_keycloak_login(request)


@router.get('/api/auth/keycloak/callback')
async def auth_keycloak_callback(request: Request, db: Session = Depends(get_db)) -> Response:
    return await legacy.auth_keycloak_callback(request, db)


@router.post('/api/auth/register')
async def auth_register(payload: RegisterRequest, db: Session = Depends(get_db)) -> JSONResponse:
    try:
        result = register_user(payload, db)
    except AuthServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content={'error': exc.code})
    return JSONResponse({'ok': True, 'pending': result.pending, 'user': serialize_user(result.user)})


@router.post('/api/auth/logout')
async def auth_logout(request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    logout_user(request, db)
    response = JSONResponse({'ok': True})
    clear_session_cookie(response=response, request=request, cookie_name='whistx_session')
    return response
