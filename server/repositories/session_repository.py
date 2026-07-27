from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..models import User, UserSession


@dataclass(frozen=True)
class SessionCleanupResult:
    deleted_count: int
    oldest_expired_at: datetime | None


def prune_expired_sessions(
    db: Session,
    *,
    now: datetime,
    batch_size: int = 1_000,
) -> SessionCleanupResult:
    oldest_expired_at = db.scalar(
        select(func.min(UserSession.expires_at)).where(UserSession.expires_at < now)
    )
    expired_ids = (
        select(UserSession.id)
        .where(UserSession.expires_at < now)
        .order_by(UserSession.expires_at, UserSession.id)
        .limit(max(1, batch_size))
    )
    result = db.execute(delete(UserSession).where(UserSession.id.in_(expired_ids)))
    return SessionCleanupResult(
        deleted_count=max(0, int(result.rowcount or 0)),
        oldest_expired_at=oldest_expired_at,
    )


def get_user_by_session_id(db: Session, session_id: str | None, *, now: datetime) -> User | None:
    if not session_id:
        return None
    return db.scalar(
        select(User)
        .join(UserSession, UserSession.user_id == User.id)
        .where(
            UserSession.id == session_id,
            UserSession.expires_at >= now,
        )
    )


def delete_session(db: Session, session_id: str | None) -> None:
    if not session_id:
        return
    session = db.get(UserSession, session_id)
    if session is not None:
        db.delete(session)


def delete_sessions_for_user(db: Session, user_id: int) -> None:
    db.execute(delete(UserSession).where(UserSession.user_id == user_id))
