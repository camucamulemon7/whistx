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
    if session is None or session.expires_at < now:
        return None
    return session.user


def delete_session(db: Session, session_id: str | None) -> None:
    if not session_id:
        return
    session = db.get(UserSession, session_id)
    if session is not None:
        db.delete(session)
