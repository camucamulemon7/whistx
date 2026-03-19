from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def _build_engine():
    connect_args: dict[str, object] = {}
    if settings.app_db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(settings.app_db_url, future=True, connect_args=connect_args)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    from . import models  # noqa: F401

    alembic_ini = Path(__file__).resolve().parent.parent / "alembic.ini"
    if alembic_ini.exists():
        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", settings.app_db_url)
        command.upgrade(config, "head")
        return
    Base.metadata.create_all(bind=engine)


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
