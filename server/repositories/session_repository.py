from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..models import User, UserSession


def prune_expired_sessions(db: Session, *, now: datetime) -> None:
    db.execute(delete(UserSession).where(UserSession.expires_at < now))


def get_user_by_session_id(db: Session, session_id: str | None, *, now: datetime) -> User | None:
    if not session_id:
        return None
    prune_expired_sessions(db, now=now)
    session = db.scalar(select(UserSession).where(UserSession.id == session_id))
    expires_at = session.expires_at if session is not None else None
    if expires_at is not None and expires_at.tzinfo is None and now.tzinfo is not None:
        expires_at = expires_at.replace(tzinfo=now.tzinfo)
    if session is None or expires_at is None or expires_at < now:
        return None
    return session.user


def delete_session(db: Session, session_id: str | None) -> None:
    if not session_id:
        return
    session = db.get(UserSession, session_id)
    if session is not None:
        db.delete(session)
