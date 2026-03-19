from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("auth_provider", "auth_subject", name="uq_users_auth_identity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    auth_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    auth_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    histories: Mapped[list["TranscriptHistory"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    approved_by: Mapped["User | None"] = relationship(remote_side=[id], foreign_keys=[approved_by_user_id])


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(128), nullable=True)

    user: Mapped["User"] = relationship(back_populates="sessions")


class TranscriptHistory(Base):
    __tablename__ = "transcript_histories"
    __table_args__ = (UniqueConstraint("user_id", "runtime_session_id", name="uq_history_user_runtime_session"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    runtime_session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    audio_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    segment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    plain_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    proofread_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_diarization: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    artifact_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    txt_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    jsonl_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    zip_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    user: Mapped["User"] = relationship(back_populates="histories")
    segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="history",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.seq",
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    history_id: Mapped[str] = mapped_column(
        ForeignKey("transcript_histories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ts_start: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_end: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_offset_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    history: Mapped["TranscriptHistory"] = relationship(back_populates="segments")
