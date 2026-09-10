"""Bounded bridge from the synchronous model stream to cancellable SSE."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading

from ..core.blocking import blocking_work_pool
from .meeting_intelligence import answer_events
from .meeting_source import MeetingError

logger = logging.getLogger(__name__)


async def stream_answer(snapshot, model, question):
    async for event in stream_events(lambda cancelled: answer_events(snapshot, model, question=question, cancelled=cancelled)):
        yield event


async def stream_events(factory, *, pool="llm"):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=32)
    cancelled = threading.Event()

    def emit(item):
        future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while not cancelled.is_set():
            try:
                future.result(timeout=0.2)
                return
            except concurrent.futures.TimeoutError:
                continue
        future.cancel()

    def produce():
        try:
            for event in factory(cancelled):
                if cancelled.is_set():
                    break
                emit(event)
        except Exception as exc:
            logger.warning("meeting answer failed: %s", type(exc).__name__)
            emit({"type": "error", "error": exc.code if isinstance(exc, MeetingError) else "meeting_model_unavailable"})
        finally:
            emit(None)

    async def run():
        try:
            await blocking_work_pool.run(pool, produce)
        except Exception:
            await queue.put({"type": "error", "error": "meeting_model_busy"})
            await queue.put(None)

    task = asyncio.create_task(run())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), 10)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        await task
    finally:
        cancelled.set()
        # Do not cancel the to_thread task: the model timeout bounds its life
        # and the worker slot stays held until the underlying call has closed.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
