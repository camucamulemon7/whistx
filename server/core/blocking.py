from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from time import monotonic
from typing import Any, TypeVar

from .config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")


class BlockingWorkPool:
    def __init__(self) -> None:
        self._semaphores = {
            "artifact": asyncio.Semaphore(settings.artifact_worker_concurrency),
            "asr": asyncio.Semaphore(settings.asr_worker_concurrency),
            "diarization": asyncio.Semaphore(settings.diarization_worker_concurrency),
            "llm": asyncio.Semaphore(settings.llm_worker_concurrency),
            "media": asyncio.Semaphore(settings.media_worker_concurrency),
        }

    async def run(
        self,
        kind: str,
        func: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        semaphore = self._semaphores[kind]
        queued_at = monotonic()
        try:
            async with asyncio.timeout(settings.blocking_worker_queue_timeout_seconds):
                await semaphore.acquire()
        except TimeoutError:
            logger.warning("blocking worker queue timeout: kind=%s", kind)
            raise

        wait_seconds = monotonic() - queued_at
        if wait_seconds >= 0.1:
            logger.info(
                "blocking worker queue wait: kind=%s wait_ms=%d",
                kind,
                round(wait_seconds * 1000),
            )
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        finally:
            semaphore.release()


blocking_work_pool = BlockingWorkPool()
