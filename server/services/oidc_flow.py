from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

from ..auth import SESSION_COOKIE_NAME, create_user_session
from ..core.config import settings
from ..core.security import (
    external_url_for as security_external_url_for, client_ip as security_client_ip,
    set_oidc_state_cookie as security_set_oidc_state_cookie,
    read_oidc_state_cookie as security_read_oidc_state_cookie,
    clear_oidc_state_cookie as security_clear_oidc_state_cookie,
    set_session_cookie as security_set_session_cookie,
)
from ..db import get_db
from ..models import User
from .auth_service import map_keycloak_auth_error as auth_service_map_keycloak_auth_error, upsert_keycloak_user as auth_service_upsert_keycloak_user
from .oidc_service import (
    fetch_json as oidc_fetch_json, fetch_discovery as oidc_fetch_discovery,
    build_authorization_url as oidc_build_authorization_url, exchange_code as oidc_exchange_code,
    fetch_userinfo as oidc_fetch_userinfo,
)

logger = logging.getLogger(__name__)
OIDC_STATE_COOKIE_NAME = 'whistx_oidc_state'
@dataclass
class OIDCFlow:
    discovery_cache: dict[str, Any] | None = None


async def auth_keycloak_login(request: Request, *, flow: OIDCFlow) -> Response:
    if not _keycloak_login_enabled():
        return JSONResponse(status_code=404, content={"error": "keycloak_disabled"})

    discovery = await asyncio.to_thread(_get_keycloak_discovery, flow)
    state = secrets.token_urlsafe(24)
    code_verifier = secrets.token_urlsafe(48)
    code_challenge = _pkce_code_challenge(code_verifier)
    redirect_uri = security_external_url_for(request, "auth_keycloak_callback")
    authorization_url = _build_keycloak_authorization_url(
        discovery=discovery,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )

    response = Response(status_code=302)
    response.headers["Location"] = authorization_url
    _set_oidc_state_cookie(
        response,
        request,
        {"state": state, "code_verifier": code_verifier, "redirect_uri": redirect_uri},
    )
    return response



async def auth_keycloak_callback(
    request: Request,
    db: Session = Depends(get_db),
    *, flow: OIDCFlow,
) -> Response:
    if not _keycloak_login_enabled():
        return HTMLResponse(status_code=404, content="not found")

    state_payload = _read_oidc_state_cookie(request)
    _response = Response(status_code=302)
    _clear_oidc_state_cookie(_response, request)

    state = request.query_params.get("state") or ""
    code = request.query_params.get("code") or ""
    if not state_payload or not code or state_payload.get("state") != state:
        _response.headers["Location"] = "/?authError=keycloak_state"
        return _response

    try:
        discovery = await asyncio.to_thread(_get_keycloak_discovery, flow)
        token_payload = await asyncio.to_thread(
            _exchange_keycloak_code,
            discovery,
            code,
            str(state_payload.get("redirect_uri") or ""),
            str(state_payload.get("code_verifier") or ""),
        )
        userinfo = await asyncio.to_thread(
            _fetch_keycloak_userinfo,
            discovery,
            str(token_payload.get("access_token") or ""),
        )
        user = _upsert_keycloak_user(db, userinfo)
        user.last_login_at = datetime.now(timezone.utc)
        session_id = create_user_session(
            db,
            user=user,
            user_agent=request.headers.get("user-agent"),
            ip_address=security_client_ip(request),
        )
        db.commit()
    except PermissionError:
        db.rollback()
        _response.headers["Location"] = "/?authError=approval_required"
        return _response
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("keycloak login failed: %s", exc)
        _response.headers["Location"] = (
            f"/?authError={auth_service_map_keycloak_auth_error(exc)}"
        )
        return _response

    _set_session_cookie(_response, request, session_id)
    _response.headers["Location"] = "/"
    return _response



def _keycloak_login_enabled() -> bool:
    return bool(
        settings.keycloak_enabled
        and settings.keycloak_issuer
        and settings.keycloak_client_id
    )



def _pkce_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")



def _set_oidc_state_cookie(
    response: Response, request: Request, payload: dict[str, Any]
) -> None:
    security_set_oidc_state_cookie(
        response=response,
        request=request,
        cookie_name=OIDC_STATE_COOKIE_NAME,
        payload=payload,
    )



def _read_oidc_state_cookie(request: Request) -> dict[str, Any] | None:
    return security_read_oidc_state_cookie(
        request=request, cookie_name=OIDC_STATE_COOKIE_NAME
    )



def _clear_oidc_state_cookie(response: Response, request: Request) -> None:
    security_clear_oidc_state_cookie(
        response=response, request=request, cookie_name=OIDC_STATE_COOKIE_NAME
    )



def _fetch_json(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return oidc_fetch_json(url, method=method, data=data, headers=headers)



def _get_keycloak_discovery(flow: OIDCFlow) -> dict[str, Any]:
    if flow.discovery_cache is not None:
        return flow.discovery_cache
    flow.discovery_cache = oidc_fetch_discovery(
        str(settings.keycloak_issuer or ""),
        fetcher=_fetch_json,
    )
    return flow.discovery_cache



def _build_keycloak_authorization_url(
    *,
    discovery: dict[str, Any],
    redirect_uri: str,
    state: str,
    code_challenge: str,
) -> str:
    return oidc_build_authorization_url(
        authorization_endpoint=str(discovery["authorization_endpoint"]),
        client_id=settings.keycloak_client_id,
        redirect_uri=redirect_uri,
        scope=settings.keycloak_scope,
        state=state,
        code_challenge=code_challenge,
    )



def _exchange_keycloak_code(
    discovery: dict[str, Any],
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    return oidc_exchange_code(
        token_endpoint=str(discovery["token_endpoint"]),
        client_id=settings.keycloak_client_id,
        client_secret=settings.keycloak_client_secret,
        code=code,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        fetcher=_fetch_json,
    )



def _fetch_keycloak_userinfo(
    discovery: dict[str, Any], access_token: str
) -> dict[str, Any]:
    return oidc_fetch_userinfo(
        str(discovery["userinfo_endpoint"]),
        access_token,
        fetcher=_fetch_json,
    )



def _upsert_keycloak_user(db: Session, userinfo: dict[str, Any]) -> User:
    return auth_service_upsert_keycloak_user(db, userinfo)



def _set_session_cookie(response: Response, request: Request, session_id: str) -> None:
    security_set_session_cookie(
        response=response,
        request=request,
        cookie_name=SESSION_COOKIE_NAME,
        session_id=session_id,
    )
