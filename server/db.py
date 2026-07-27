from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .core.config import settings


class Base(DeclarativeBase):
    pass


def _build_engine():
    connect_args: dict[str, object] = {}
    engine_options: dict[str, object] = {
        "future": True,
        "pool_pre_ping": settings.db_pool_pre_ping,
    }
    if settings.app_db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        engine_options.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
            pool_recycle=settings.db_pool_recycle_seconds,
        )
    return create_engine(
        settings.app_db_url,
        connect_args=connect_args,
        **engine_options,
    )


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@dataclass(frozen=True)
class SchemaRevisionStatus:
    ready: bool
    current_revisions: tuple[str, ...]
    expected_revisions: tuple[str, ...]
    error: str | None = None


def _alembic_config() -> Config:
    alembic_ini = Path(__file__).resolve().parent.parent / "alembic.ini"
    config = Config(str(alembic_ini))
    config.set_main_option("sqlalchemy.url", settings.app_db_url)
    return config


def schema_revision_status() -> SchemaRevisionStatus:
    try:
        expected = tuple(sorted(ScriptDirectory.from_config(_alembic_config()).get_heads()))
        with engine.connect() as connection:
            current = tuple(sorted(MigrationContext.configure(connection).get_current_heads()))
        return SchemaRevisionStatus(
            ready=current == expected,
            current_revisions=current,
            expected_revisions=expected,
        )
    except Exception as exc:  # noqa: BLE001
        return SchemaRevisionStatus(
            ready=False,
            current_revisions=(),
            expected_revisions=(),
            error=type(exc).__name__,
        )


def require_schema_current() -> None:
    status = schema_revision_status()
    if status.ready:
        return
    raise RuntimeError(
        "database schema is not current; run `alembic upgrade head` "
        f"(current={status.current_revisions}, expected={status.expected_revisions}, error={status.error})"
    )


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
