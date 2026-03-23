from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import User


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.strip().lower()))


def get_user_by_identity(db: Session, *, provider: str, subject: str) -> User | None:
    stmt = select(User).where(User.auth_provider == provider.strip(), User.auth_subject == subject.strip())
    return db.scalar(stmt)


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def has_admin_account(db: Session) -> bool:
    return db.scalar(select(User.id).where(User.is_admin.is_(True)).limit(1)) is not None


def count_admin_users(db: Session) -> int:
    stmt = select(func.count()).select_from(User).where(User.is_admin.is_(True))
    return int(db.scalar(stmt) or 0)


def count_pending_users(db: Session) -> int:
    stmt = (
        select(func.count())
        .select_from(User)
        .where(User.is_active.is_(False))
        .where(User.approved_at.is_(None))
    )
    return int(db.scalar(stmt) or 0)


def list_pending_users(db: Session) -> list[User]:
    stmt = (
        select(User)
        .where(User.is_active.is_(False))
        .where(User.approved_at.is_(None))
        .order_by(User.created_at.asc(), User.id.asc())
    )
    return list(db.scalars(stmt).all())


def list_all_users(db: Session) -> list[User]:
    stmt = select(User).order_by(User.created_at.desc(), User.id.desc())
    return list(db.scalars(stmt).all())
