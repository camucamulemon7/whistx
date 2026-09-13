"""Read-only online audit; offline quarantine is an explicit operator action."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time

from sqlalchemy import select
from ..models import ArtifactDeletion, TranscriptHistory

logger = logging.getLogger(__name__)


def scan(db, root: Path, *, grace_seconds=86400):
    root = root.resolve()
    referenced = set()
    findings = []
    now = time.time()
    for history in db.scalars(select(TranscriptHistory)):
        key = history.artifact_dir or (str(Path(history.txt_path).parent) if history.txt_path else '')
        directory = root / key
        if not key or not directory.resolve().is_relative_to(root):
            findings.append({'kind': 'invalid_record_path', 'historyId': history.id})
            continue
        referenced.add(directory.resolve())
        referenced.add((root / str(history.user_id) / history.id).resolve())
        if not all((directory / name).is_file() for name in ['transcript.txt', 'transcript.jsonl']):
            findings.append({'kind': 'missing_artifact', 'historyId': history.id, 'path': key})
    for request in db.scalars(select(ArtifactDeletion)):
        if request.artifact_key:
            referenced.add((root / request.artifact_key).resolve())
        referenced.add((root / str(request.user_id) / request.history_id).resolve())
    candidates = set(root.glob('*/*/*/*')) | set(root.glob('*/*'))
    candidates.update((root / '_exports').glob('*.zip'))
    for path in sorted(candidates):
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve() in referenced:
            continue
        parts = path.relative_to(root).parts
        kind = None
        if len(parts) == 2 and parts[0] == '_exports' and path.is_file():
            kind = 'temporary_zip'
        elif len(parts) in {2, 4} and parts[0].isdigit() and path.is_dir():
            if len(parts) == 4 and not (re.fullmatch(r'\d{4}', parts[1]) and re.fullmatch(r'\d{2}', parts[2])):
                continue
            if re.fullmatch(r'hist[-_][A-Za-z0-9_-]+', path.name):
                kind = 'orphan_artifact'
            elif re.fullmatch(r'\.hist[-_][A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', path.name):
                kind = 'abandoned_staging'
        if kind and now - path.stat().st_mtime >= grace_seconds:
            findings.append({'kind': kind, 'path': path.relative_to(root).as_posix()})
    counts = {kind: sum(f['kind'] == kind for f in findings) for kind in sorted({f['kind'] for f in findings})}
    logger.info('artifact reconciliation metrics: %s', json.dumps(counts, sort_keys=True))
    return {'counts': counts, 'findings': findings}


def quarantine(db, root, *, grace_seconds=86400):
    # The caller must stop all writers. Re-scan immediately before every move.
    root = root.resolve()
    report = scan(db, root, grace_seconds=grace_seconds)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = root / '_quarantine' / stamp
    moved = []
    for finding in report['findings']:
        if finding['kind'] not in {'orphan_artifact', 'abandoned_staging', 'temporary_zip'}:
            continue
        source = root / finding['path']
        target = destination / finding['path']
        if (source.is_symlink() or not source.resolve().is_relative_to(root)
                or any(parent.is_symlink() for parent in source.parents if parent != root.parent)):
            raise ValueError('unsafe_quarantine_path')
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        moved.append(finding)
        (destination / 'manifest.json').write_text(json.dumps(moved, indent=2), encoding='utf-8')
        logger.info('artifact quarantined: kind=%s path=%s', finding['kind'], finding['path'])
    return {**report, 'quarantine': str(destination) if moved else None, 'moved': len(moved)}
