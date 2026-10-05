"""Post-commit storage and session cleanup, separate from resource lifecycle."""
from datetime import datetime, timezone
import logging
from ..core.config import settings
from ..db import db_session
from ..repositories import session_repository
from .history_service import cleanup_expired_runtime_data

logger = logging.getLogger(__name__)


def run_cleanup_once(reason: str) -> None:
    try:
        with db_session() as db:
            cleanup_expired_runtime_data(db)
            deleted_sessions = 0
            oldest_expired_at = None
            for _ in range(10):
                result = session_repository.prune_expired_sessions(
                    db,
                    now=datetime.now(timezone.utc),
                    batch_size=1_000,
                )
                deleted_sessions += result.deleted_count
                oldest_expired_at = oldest_expired_at or result.oldest_expired_at
                if result.deleted_count < 1_000:
                    break
            if deleted_sessions:
                logger.info(
                    "expired session cleanup completed: reason=%s deleted=%d oldest_expired_at=%s",
                    reason,
                    deleted_sessions,
                    oldest_expired_at.isoformat() if oldest_expired_at else None,
                )
        from .services.artifact_deletion import process_deletions
        from .services.artifact_reconciliation import scan
        with db_session() as db:
            process_deletions(db, settings.history_dir)
            scan(db, settings.history_dir)
    except Exception:
        logger.warning(
            "scheduled cleanup failed during %s", reason, exc_info=True
        )
