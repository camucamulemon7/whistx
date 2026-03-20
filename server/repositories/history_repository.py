from __future__ import annotations

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload

from ..models import TranscriptHistory


def get_history_by_runtime_session(db: Session, *, user_id: int, runtime_session_id: str) -> TranscriptHistory | None:
    stmt = select(TranscriptHistory).where(
        TranscriptHistory.user_id == user_id,
        TranscriptHistory.runtime_session_id == runtime_session_id,
    )
    return db.scalar(stmt)


def history_query_for_user(*, user_id: int, query: str | None = None) -> Select[tuple[TranscriptHistory]]:
    stmt = (
        select(TranscriptHistory)
        .where(TranscriptHistory.user_id == user_id)
        .order_by(TranscriptHistory.saved_at.desc())
    )
    clean_query = (query or '').strip()
    if clean_query:
        like = f'%{clean_query}%'
        stmt = stmt.where(TranscriptHistory.title.ilike(like) | TranscriptHistory.plain_text.ilike(like))
    return stmt


def count_histories_for_user(db: Session, *, user_id: int, query: str | None = None) -> int:
    stmt = select(func.count()).select_from(history_query_for_user(user_id=user_id, query=query).subquery())
    return int(db.scalar(stmt) or 0)


def list_histories_for_user(
    db: Session,
    *,
    user_id: int,
    limit: int,
    offset: int,
    query: str | None = None,
) -> list[TranscriptHistory]:
    stmt = history_query_for_user(user_id=user_id, query=query).limit(limit).offset(offset)
    return list(db.scalars(stmt).all())


def get_history_for_user(
    db: Session,
    *,
    user_id: int,
    history_id: str,
    with_segments: bool = True,
) -> TranscriptHistory | None:
    stmt = select(TranscriptHistory).where(
        TranscriptHistory.id == history_id,
        TranscriptHistory.user_id == user_id,
    )
    if with_segments:
        stmt = stmt.options(selectinload(TranscriptHistory.segments))
    return db.scalar(stmt)
