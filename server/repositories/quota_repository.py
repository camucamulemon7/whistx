from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from ..db import db_session
from ..models import ConnectionLease, ConnectionQuotaLock, RateLimitBucket


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def consume_rate_limit(*, bucket: str, subject: str, limit: int, window_seconds: int) -> bool:
    key = f"{bucket}:{subject}"[:512]
    for attempt in range(2):
        try:
            with db_session() as db:
                now = utcnow()
                row = db.scalar(
                    select(RateLimitBucket).where(RateLimitBucket.key == key).with_for_update()
                )
                if row is None:
                    db.add(
                        RateLimitBucket(
                            key=key,
                            count=1,
                            window_ends_at=now + timedelta(seconds=window_seconds),
                        )
                    )
                    db.flush()
                    return True
                if _as_aware(row.window_ends_at) <= now:
                    row.count = 1
                    row.window_ends_at = now + timedelta(seconds=window_seconds)
                    return True
                if row.count >= limit:
                    return False
                row.count += 1
                return True
        except IntegrityError:
            if attempt:
                raise
    return False


def rate_limit_retry_after(*, bucket: str, subject: str, limit: int) -> int | None:
    key = f"{bucket}:{subject}"[:512]
    with db_session() as db:
        row = db.get(RateLimitBucket, key)
        if row is None or row.count < limit:
            return None
        remaining = int((_as_aware(row.window_ends_at) - utcnow()).total_seconds())
        return max(1, remaining) if remaining > 0 else None


def clear_rate_limit(*, bucket: str | None = None, subject: str | None = None) -> None:
    with db_session() as db:
        stmt = delete(RateLimitBucket)
        if bucket is not None and subject is not None:
            stmt = stmt.where(RateLimitBucket.key == f"{bucket}:{subject}"[:512])
        elif bucket is not None:
            stmt = stmt.where(RateLimitBucket.key.like(f"{bucket}:%"))
        db.execute(stmt)


def acquire_connection_lease(
    *,
    subject: str,
    is_guest: bool,
    ttl_seconds: int,
    guest_total_limit: int | None = None,
    guest_subject_limit: int | None = None,
) -> str | None:
    with db_session() as db:
        lock = db.scalar(
            select(ConnectionQuotaLock)
            .where(ConnectionQuotaLock.key == "connections")
            .with_for_update()
        )
        if lock is None:
            raise RuntimeError("connection_quota_lock_missing")

        now = utcnow()
        db.execute(delete(ConnectionLease).where(ConnectionLease.expires_at <= now))
        if is_guest:
            total = int(
                db.scalar(
                    select(func.count())
                    .select_from(ConnectionLease)
                    .where(ConnectionLease.is_guest.is_(True))
                )
                or 0
            )
            subject_count = int(
                db.scalar(
                    select(func.count())
                    .select_from(ConnectionLease)
                    .where(ConnectionLease.is_guest.is_(True))
                    .where(ConnectionLease.subject == subject)
                )
                or 0
            )
            if (
                (guest_total_limit is not None and total >= guest_total_limit)
                or (guest_subject_limit is not None and subject_count >= guest_subject_limit)
            ):
                return None

        lease_id = secrets.token_hex(24)
        db.add(
            ConnectionLease(
                id=lease_id,
                subject=subject[:255],
                is_guest=is_guest,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
        )
        db.flush()
        return lease_id


def release_connection_lease(lease_id: str | None) -> None:
    if not lease_id:
        return
    with db_session() as db:
        db.execute(delete(ConnectionLease).where(ConnectionLease.id == lease_id))


def count_active_connections() -> int:
    with db_session() as db:
        now = utcnow()
        db.execute(delete(ConnectionLease).where(ConnectionLease.expires_at <= now))
        return int(db.scalar(select(func.count()).select_from(ConnectionLease)) or 0)


def _as_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
