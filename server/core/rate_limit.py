from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

_LOCK = threading.Lock()
_EVENTS: dict[str, deque[float]] = defaultdict(deque)


def consume(*, bucket: str, subject: str, limit: int, window_seconds: int) -> bool:
    key = f"{bucket}:{subject}"
    now = time.monotonic()
    cutoff = now - window_seconds
    with _LOCK:
        events = _EVENTS[key]
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            return False
        events.append(now)
        return True


def clear() -> None:
    with _LOCK:
        _EVENTS.clear()
