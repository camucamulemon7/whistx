from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from ...db import get_db
from ...schemas import BootstrapAdminRequest, LoginRequest, RegisterRequest
from ... import legacy_app as legacy

router = APIRouter()


@router.get("/api/auth/me")
async def auth_me(request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    return await legacy.auth_me(request, db)


@router.post("/api/auth/login")
async def auth_login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    return await legacy.auth_login(payload, request, db)


@router.post("/api/auth/bootstrap-admin")
async def auth_bootstrap_admin(
    payload: BootstrapAdminRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> JSONResponse:
    return await legacy.auth_bootstrap_admin(payload, request, db)


@router.get("/api/auth/keycloak/login")
async def auth_keycloak_login(request: Request) -> Response:
    return await legacy.auth_keycloak_login(request)


@router.get("/api/auth/keycloak/callback")
async def auth_keycloak_callback(request: Request, db: Session = Depends(get_db)) -> Response:
    return await legacy.auth_keycloak_callback(request, db)


@router.post("/api/auth/register")
async def auth_register(payload: RegisterRequest, db: Session = Depends(get_db)) -> JSONResponse:
    return await legacy.auth_register(payload, db)


@router.post("/api/auth/logout")
async def auth_logout(request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    return await legacy.auth_logout(request, db)
