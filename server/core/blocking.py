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
        self._workers: set[asyncio.Task] = set()
        self._operations: set[asyncio.Future[None]] = set()
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
        completion = asyncio.get_running_loop().create_future()
        self._operations.add(completion)
        try:
            return await self._run(kind, func, *args, **kwargs)
        finally:
            self._operations.discard(completion)
            completion.set_result(None)

    async def _run(self, kind: str, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
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
        worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
        self._workers.add(worker)
        worker.add_done_callback(self._workers.discard)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # A synchronous call cannot be cancelled once dispatched. Keep
            # both its permit and the caller's file/client lifetime until it
            # returns, even if shutdown cancels the caller more than once.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not worker.cancelled():
                worker.exception()
            raise
        finally:
            self._workers.discard(worker)
            semaphore.release()

    async def drain(self) -> None:
        """Finish queued and dispatched work before owning clients are closed."""
        while self._operations:
            await asyncio.gather(*(asyncio.shield(operation) for operation in tuple(self._operations)))


blocking_work_pool = BlockingWorkPool()
