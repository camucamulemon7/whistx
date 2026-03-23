from __future__ import annotations

from . import legacy_app as legacy
from .config import settings
from .core.application import create_app
from .schemas import BootstrapAdminRequest, HistorySaveRequest, LoginRequest, RegisterRequest

app = create_app()

LOGIN_ATTEMPTS = legacy.LOGIN_ATTEMPTS
SummarizeRequest = legacy.SummarizeRequest
ProofreadRequest = legacy.ProofreadRequest
get_user_by_email = legacy.get_user_by_email


async def auth_login(payload: LoginRequest, request, db):
    legacy.settings = settings
    legacy.get_user_by_email = get_user_by_email
    return await legacy.auth_login(payload, request, db)


def _map_keycloak_auth_error(exc: Exception) -> str:
    legacy.settings = settings
    return legacy._map_keycloak_auth_error(exc)


def _upsert_keycloak_user(db, userinfo):
    legacy.settings = settings
    return legacy._upsert_keycloak_user(db, userinfo)
