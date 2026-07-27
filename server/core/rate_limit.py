from __future__ import annotations

from ..repositories import quota_repository


def consume(*, bucket: str, subject: str, limit: int, window_seconds: int) -> bool:
    return quota_repository.consume_rate_limit(
        bucket=bucket,
        subject=subject,
        limit=limit,
        window_seconds=window_seconds,
    )


def clear() -> None:
    quota_repository.clear_rate_limit()
