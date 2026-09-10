"""Sample-anchored replacement of complete intervals; no guessed word times."""
from __future__ import annotations


def apply_revision(records: list[dict], event: dict) -> list[dict]:
    replacement = event.get('record')
    if not isinstance(replacement, dict) or replacement.get('type') != 'final':
        raise ValueError('invalid_transcript_revision')
    existing = next((row for row in records if row.get('segmentId') == replacement.get('segmentId')), None)
    if existing == replacement:
        return records
    track, start, end = event.get('track'), event.get('startSample'), event.get('endSample')
    if not isinstance(track, str) or type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('invalid_revision_range')
    if (replacement.get('track'), replacement.get('startSample'), replacement.get('endSample')) != (track, start, end):
        raise ValueError('revision_timestamp_mismatch')
    if replacement.get('tsStart') != start * 1000 // 16000 or replacement.get('tsEnd') != end * 1000 // 16000:
        raise ValueError('revision_timestamp_mismatch')
    targets = []
    for row in records:
        if row.get('type') != 'final' or row.get('track') != track:
            continue
        left, right = row['startSample'], row['endSample']
        if right <= start or left >= end:
            continue
        if left < start or right > end or row.get('quality') == 'high_accuracy':
            raise ValueError('revision_conflicts_with_interval')
        targets.append(row)
    ids = [row['segmentId'] for row in targets]
    expected = event.get('replacesSegmentIds')
    if not isinstance(expected, list) or not all(isinstance(value, str) for value in expected) or len(set(expected)) != len(expected) or set(ids) != set(expected):
        raise ValueError('stale_transcript_revision')
    if not targets:
        raise ValueError('revision_has_no_sources')
    kept = [row for row in records if row.get('segmentId') not in set(ids)]
    kept.append(replacement)
    return sorted(kept, key=lambda row: (row.get('tsStart', 0), row.get('seq', 0)))
