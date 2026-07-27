from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import auth
from ..core.config import settings
from ..core.security import client_ip, serialize_user
from ..models import User
from ..repositories import quota_repository, user_repository

logger = logging.getLogger(__name__)

KEYCLOAK_PROVIDER = 'keycloak'
LOGIN_RATE_LIMIT_WINDOW_SECONDS = 300
LOGIN_RATE_LIMIT_ATTEMPTS = 5


class AuthServiceError(Exception):
    def __init__(self, code: str, status_code: int, *, retry_after_sec: int | None = None):
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.retry_after_sec = retry_after_sec


@dataclass(slots=True)
class AuthResult:
    user: User
    session_id: str | None = None
    pending: bool = False


def map_keycloak_auth_error(exc: Exception) -> str:
    mapping = {
        'keycloak_email_not_verified': 'keycloak_email_not_verified',
        'keycloak_account_link_required': 'keycloak_account_link_required',
        'keycloak_identity_conflict': 'keycloak_identity_conflict',
    }
    return mapping.get(str(exc), 'keycloak_failed')


def _client_ip(request: Request) -> str:
    return client_ip(request)


def _login_rate_limit_keys(request: Request, email: str) -> tuple[str, str]:
    normalized = email.strip().lower()
    return f'{_client_ip(request)}::{normalized}', f'email::{normalized}'


def _record_failed_login(key: str) -> None:
    quota_repository.consume_rate_limit(
        bucket='login',
        subject=key,
        limit=LOGIN_RATE_LIMIT_ATTEMPTS,
        window_seconds=LOGIN_RATE_LIMIT_WINDOW_SECONDS,
    )


def _clear_failed_login(key: str) -> None:
    quota_repository.clear_rate_limit(bucket='login', subject=key)


def _login_retry_after_seconds(key: str) -> int | None:
    return quota_repository.rate_limit_retry_after(
        bucket='login',
        subject=key,
        limit=LOGIN_RATE_LIMIT_ATTEMPTS,
    )


def _login_retry_after_seconds_for_keys(keys: tuple[str, str]) -> int | None:
    retry_after_values = [value for value in (_login_retry_after_seconds(key) for key in keys) if value is not None]
    if not retry_after_values:
        return None
    return max(retry_after_values)


def get_optional_user_from_request(request: Request, db: Session) -> User | None:
    session_id = request.cookies.get(auth.SESSION_COOKIE_NAME)
    return auth.get_user_by_session_id(db, session_id)


def build_auth_me_payload(request: Request, db: Session) -> dict[str, Any]:
    session_cookie_present = bool(request.cookies.get(auth.SESSION_COOKIE_NAME))
    user = get_optional_user_from_request(request, db)
    admin_exists = auth.has_admin_account(db)
    http_bootstrap_allowed = getattr(settings, "app_env", "development") != "production"
    return {
        'authenticated': user is not None,
        'sessionInvalid': session_cookie_present and user is None,
        'user': serialize_user(user) if user is not None else None,
        'selfSignupEnabled': settings.enable_self_signup,
        'guestTranscriptionAllowed': bool(getattr(settings, 'allow_guest_transcription', False)),
        'historyRetentionDays': settings.history_retention_days,
        'bootstrapAdminRequired': not admin_exists and http_bootstrap_allowed,
        'bootstrapAdminCliRequired': not admin_exists and not http_bootstrap_allowed,
        'pendingApprovalCount': user_repository.count_pending_users(db) if user is not None and user.is_admin else 0,
        'keycloakEnabled': bool(settings.keycloak_enabled and settings.keycloak_issuer and settings.keycloak_client_id),
        'keycloakButtonLabel': settings.keycloak_button_label,
    }


def login_user(payload, request: Request, db: Session) -> AuthResult:
    rate_limit_keys = _login_rate_limit_keys(request, payload.email)
    retry_after = _login_retry_after_seconds_for_keys(rate_limit_keys)
    if retry_after is not None:
        raise AuthServiceError('too_many_login_attempts', 429, retry_after_sec=retry_after)

    user = auth.get_user_by_email(db, payload.email)
    if user is None or not auth.verify_password(user.password_hash, payload.password):
        for key in rate_limit_keys:
            _record_failed_login(key)
        db.rollback()
        raise AuthServiceError('invalid_credentials', 401)
    if not user.is_active:
        db.rollback()
        raise AuthServiceError('approval_required', 403)

    for key in rate_limit_keys:
        _clear_failed_login(key)
    user.last_login_at = datetime.now(timezone.utc)
    session_id = auth.create_user_session(
        db,
        user=user,
        user_agent=request.headers.get('user-agent'),
        ip_address=client_ip(request),
    )
    db.commit()
    return AuthResult(user=user, session_id=session_id)


def bootstrap_admin(payload, request: Request, db: Session) -> AuthResult:
    if getattr(settings, "app_env", "development") == "production":
        logger.warning("initial administrator HTTP bootstrap rejected in production: ip=%s", client_ip(request))
        raise AuthServiceError('bootstrap_admin_cli_required', 403)
    if auth.has_admin_account(db):
        logger.warning("duplicate administrator bootstrap rejected: ip=%s", client_ip(request))
        raise AuthServiceError('admin_already_exists', 409)
    existing = auth.get_user_by_email(db, payload.email)
    if existing is not None:
        raise AuthServiceError('email_already_exists', 409)
    try:
        user_repository.claim_initial_admin_bootstrap(db)
        user = auth.create_user(
            db,
            email=payload.email.strip().lower(),
            password=payload.password,
            display_name=payload.display_name,
            is_admin=True,
            is_active=True,
        )
        user.approved_by_user_id = user.id
        user.last_login_at = datetime.now(timezone.utc)
        session_id = auth.create_user_session(
            db,
            user=user,
            user_agent=request.headers.get('user-agent'),
            ip_address=client_ip(request),
        )
        db.commit()
        return AuthResult(user=user, session_id=session_id)
    except ValueError:
        db.rollback()
        raise AuthServiceError('password_too_short', 400) from None
    except IntegrityError:
        db.rollback()
        logger.warning("concurrent administrator bootstrap rejected: ip=%s", client_ip(request))
        raise AuthServiceError('admin_already_exists', 409) from None


def register_user(payload, db: Session) -> AuthResult:
    if not auth.has_admin_account(db):
        raise AuthServiceError('bootstrap_admin_required', 409)
    if not settings.enable_self_signup:
        raise AuthServiceError('self_signup_disabled', 403)
    if len(payload.password) < 8:
        raise AuthServiceError('password_too_short', 400)
    existing = auth.get_user_by_email(db, payload.email)
    if existing is not None:
        raise AuthServiceError('email_already_exists', 409)
    try:
        user = auth.create_user(
            db,
            email=payload.email.strip().lower(),
            password=payload.password,
            display_name=payload.display_name,
            is_admin=False,
            is_active=False,
        )
        db.commit()
        return AuthResult(user=user, pending=True)
    except ValueError:
        db.rollback()
        raise AuthServiceError('password_too_short', 400) from None
    except IntegrityError:
        db.rollback()
        raise AuthServiceError('email_already_exists', 409) from None


def logout_user(request: Request, db: Session) -> None:
    auth.delete_user_session(db, request.cookies.get(auth.SESSION_COOKIE_NAME))
    db.commit()


def update_display_name(*, user: User, display_name: str | None, db: Session) -> User:
    user.display_name = (display_name or '').strip() or None
    db.commit()
    return user


def revoke_all_sessions(*, user: User, db: Session) -> None:
    auth.delete_all_user_sessions(db, user.id)
    db.commit()


def change_password(*, user: User, current_password: str, new_password: str, request: Request, db: Session) -> str:
    if not auth.verify_password(user.password_hash, current_password):
        raise AuthServiceError('invalid_current_password', 403)
    if len(new_password) < 8:
        raise AuthServiceError('password_too_short', 400)
    user.password_hash = auth.hash_password(new_password)
    auth.delete_all_user_sessions(db, user.id)
    session_id = auth.create_user_session(
        db,
        user=user,
        user_agent=request.headers.get('user-agent'),
        ip_address=client_ip(request),
    )
    db.commit()
    return session_id


def upsert_keycloak_user(db: Session, userinfo: dict[str, Any]) -> User:
    subject = str(userinfo.get('sub') or '').strip()
    email = str(userinfo.get('email') or '').strip().lower()
    email_verified = userinfo.get('email_verified')
    display_name = (
        str(userinfo.get('name') or '').strip()
        or str(userinfo.get('preferred_username') or '').strip()
        or email
    )
    if not subject or not email:
        raise RuntimeError('keycloak_userinfo_missing_identity')
    if email_verified is False or (email_verified is None and settings.keycloak_require_email_verified):
        raise RuntimeError('keycloak_email_not_verified')

    user = auth.get_user_by_identity(db, provider=KEYCLOAK_PROVIDER, subject=subject)
    if user is None:
        user = auth.get_user_by_email(db, email)
        if user is not None:
            if user.auth_provider and user.auth_subject and (user.auth_provider != KEYCLOAK_PROVIDER or user.auth_subject != subject):
                raise RuntimeError('keycloak_identity_conflict')
            raise RuntimeError('keycloak_account_link_required')
        user = auth.create_user(
            db,
            email=email,
            password=secrets.token_urlsafe(32),
            display_name=display_name,
            is_admin=False,
            is_active=True,
            auth_provider=KEYCLOAK_PROVIDER,
            auth_subject=subject,
        )
    else:
        if not user.is_active:
            user.is_active = True
        if user.approved_at is None:
            user.approved_at = datetime.now(timezone.utc)
        if display_name and not user.display_name:
            user.display_name = display_name
    return user
