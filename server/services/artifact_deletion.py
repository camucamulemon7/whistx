"""Commit history removal and its cleanup request together; retry after commit."""
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import shutil

from sqlalchemy import func, select

from ..models import ArtifactDeletion, TranscriptHistory

logger = logging.getLogger(__name__)


def enqueue(db, history):
    key = history.artifact_dir or (str(Path(history.txt_path).parent) if history.txt_path else None)
    if db.get(ArtifactDeletion, history.id) is None:
        db.add(ArtifactDeletion(history_id=history.id, user_id=history.user_id, artifact_key=key))


def owned_paths(root, request):
    if not request.artifact_key:
        return []
    parts = Path(request.artifact_key).parts
    if (Path(request.artifact_key).is_absolute() or len(parts) not in {2, 4}
            or parts[0] != str(request.user_id) or parts[-1] != request.history_id
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request.history_id)
            or (len(parts) == 4 and not (re.fullmatch(r'\d{4}', parts[1]) and re.fullmatch(r'\d{2}', parts[2])))):
        raise ValueError('invalid_artifact_owner_path')
    paths = {root.joinpath(*parts), root / str(request.user_id) / request.history_id}
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()) or any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
            raise ValueError('unsafe_artifact_symlink')
    return sorted(paths)


def process_deletions(db, root, *, batch_size=100):
    """Only invoke after enqueue transaction commits. Each row is retryable."""
    identifiers = list(db.scalars(select(ArtifactDeletion.history_id).order_by(ArtifactDeletion.attempts, ArtifactDeletion.requested_at).limit(batch_size)))
    completed = 0
    for identifier in identifiers:
        request = db.scalar(select(ArtifactDeletion).where(ArtifactDeletion.history_id == identifier).with_for_update(skip_locked=True))
        if request is None:
            continue
        try:
            # Never remove a path if a live record still claims this history.
            if db.get(TranscriptHistory, identifier) is not None:
                raise ValueError('history_still_exists')
            for path in owned_paths(root, request):
                try:
                    shutil.rmtree(path)
                except FileNotFoundError:
                    pass
        except (OSError, ValueError) as exc:
            request.attempts += 1
            request.last_error = type(exc).__name__
            logger.warning('artifact deletion retry: history_id=%s error=%s', identifier, request.last_error)
        else:
            db.delete(request)
            completed += 1
            logger.info('artifact deletion completed: history_id=%s', identifier)
        db.commit()
    pending = db.scalar(select(func.count()).select_from(ArtifactDeletion)) or 0
    oldest = db.scalar(select(func.min(ArtifactDeletion.requested_at)))
    age = max(0, (datetime.now(timezone.utc) - oldest.replace(tzinfo=timezone.utc)).total_seconds()) if oldest else 0
    logger.info('artifact cleanup metrics: completed=%d pending=%d oldest_pending_seconds=%.0f', completed, pending, age)
    return {'completed': completed, 'pending': pending, 'oldestPendingSeconds': age}
