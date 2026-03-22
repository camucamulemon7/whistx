from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy.orm import Session

from .config import settings
from .models import User, UserSession
from .repositories import session_repository, user_repository

SESSION_COOKIE_NAME = 'whistx_session'
_password_hasher = PasswordHasher()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def create_user(
    db: Session,
    *,
    email: str,
    password: str,
    display_name: str | None = None,
    is_admin: bool = False,
    is_active: bool = True,
    approved_by_user_id: int | None = None,
    auth_provider: str | None = None,
    auth_subject: str | None = None,
) -> User:
    if len(password) < 8:
        raise ValueError('password_too_short')
    user = User(
        email=email.strip().lower(),
        password_hash=hash_password(password),
        display_name=(display_name or '').strip() or None,
        auth_provider=(auth_provider or '').strip() or None,
        auth_subject=(auth_subject or '').strip() or None,
        is_active=is_active,
        is_admin=is_admin,
        approved_at=utcnow() if is_active else None,
        approved_by_user_id=approved_by_user_id,
    )
    db.add(user)
    db.flush()
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    return user_repository.get_user_by_email(db, email)


def get_user_by_identity(db: Session, *, provider: str, subject: str) -> User | None:
    return user_repository.get_user_by_identity(db, provider=provider, subject=subject)


def authenticate_user(db: Session, *, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None or not user.is_active:
        return None
    if not verify_password(user.password_hash, password):
        return None
    user.last_login_at = utcnow()
    return user


def has_admin_account(db: Session) -> bool:
    return user_repository.has_admin_account(db)


def count_admin_users(db: Session) -> int:
    return user_repository.count_admin_users(db)


def count_pending_users(db: Session) -> int:
    return user_repository.count_pending_users(db)


def list_pending_users(db: Session) -> list[User]:
    return user_repository.list_pending_users(db)


def list_all_users(db: Session) -> list[User]:
    return user_repository.list_all_users(db)


def approve_user(db: Session, *, user: User, admin: User) -> User:
    user.is_active = True
    user.approved_at = utcnow()
    user.approved_by_user_id = admin.id
    return user


def create_user_session(
    db: Session,
    *,
    user: User,
    user_agent: str | None,
    ip_address: str | None,
) -> UserSession:
    session_repository.prune_expired_sessions(db, now=utcnow())
    session = UserSession(
        id=secrets.token_urlsafe(32),
        user_id=user.id,
        created_at=utcnow(),
        expires_at=utcnow() + timedelta(days=settings.app_session_days),
        user_agent=(user_agent or '').strip()[:512] or None,
        ip_address=(ip_address or '').strip()[:128] or None,
    )
    db.add(session)
    db.flush()
    return session


def prune_expired_sessions(db: Session) -> None:
    session_repository.prune_expired_sessions(db, now=utcnow())


def get_user_by_session_id(db: Session, session_id: str | None) -> User | None:
    return session_repository.get_user_by_session_id(db, session_id, now=utcnow())


def delete_user_session(db: Session, session_id: str | None) -> None:
    session_repository.delete_session(db, session_id)
