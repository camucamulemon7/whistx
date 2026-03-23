from __future__ import annotations

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, selectinload

from ..models import TranscriptHistory


def get_history_by_runtime_session(db: Session, *, user_id: int, runtime_session_id: str) -> TranscriptHistory | None:
    stmt = select(TranscriptHistory).where(
        TranscriptHistory.user_id == user_id,
        TranscriptHistory.runtime_session_id == runtime_session_id,
    )
    return db.scalar(stmt)


def history_query_for_user(*, user_id: int, query: str | None = None) -> Select[tuple[TranscriptHistory]]:
    return select(TranscriptHistory).where(TranscriptHistory.user_id == user_id).order_by(TranscriptHistory.saved_at.desc())


def build_history_search_clause(*, query: str | None = None):
    clean_query = (query or "").strip()
    if not clean_query:
        return None
    like = f"%{clean_query}%"
    return or_(TranscriptHistory.title.ilike(like), TranscriptHistory.plain_text.ilike(like))


def build_history_list_stmt(*, user_id: int, query: str | None = None) -> Select[tuple[TranscriptHistory]]:
    stmt = history_query_for_user(user_id=user_id, query=None)
    search_clause = build_history_search_clause(query=query)
    if search_clause is not None:
        stmt = stmt.where(search_clause)
    return stmt


def build_history_count_stmt(*, user_id: int, query: str | None = None):
    return select(func.count()).select_from(build_history_list_stmt(user_id=user_id, query=query).subquery())


def count_histories_for_user(db: Session, *, user_id: int, query: str | None = None) -> int:
    return int(db.scalar(build_history_count_stmt(user_id=user_id, query=query)) or 0)


def list_histories_for_user(
    db: Session,
    *,
    user_id: int,
    limit: int,
    offset: int,
    query: str | None = None,
) -> list[TranscriptHistory]:
    stmt = build_history_list_stmt(user_id=user_id, query=query).limit(limit).offset(offset)
    return list(db.scalars(stmt).all())


def list_runtime_session_ids(db: Session) -> list[str]:
    stmt = select(TranscriptHistory.runtime_session_id)
    return [value for value in db.scalars(stmt).all() if isinstance(value, str) and value.strip()]


def list_histories_saved_before(db: Session, *, cutoff) -> list[TranscriptHistory]:
    stmt = select(TranscriptHistory).where(TranscriptHistory.saved_at < cutoff)
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


def delete_history(db: Session, history: TranscriptHistory) -> None:
    db.delete(history)
