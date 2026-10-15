"""Cancellation must not outlive the synchronous work it protects."""
import asyncio
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'blocking-lifetime-tests-only-session-secret')

from server.core.blocking import BlockingWorkPool


class BlockingLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def until(self, condition):
        async with asyncio.timeout(3):
            while not condition():
                await asyncio.sleep(.001)

    async def test_repeated_cancel_holds_permit_and_caller_resources_until_thread_returns(self):
        pool = BlockingWorkPool()
        pool._semaphores['asr'] = asyncio.Semaphore(1)
        started, release, second_started, caller_closed = (threading.Event() for _ in range(4))
        def first_call():
            started.set()
            if not release.wait(3):
                raise TimeoutError('test release deadline')
            self.assertFalse(caller_closed.is_set())

        async def owner():
            try:
                await pool.run('asr', first_call)
            finally:
                caller_closed.set()

        first = asyncio.create_task(owner())
        second = None
        try:
            await self.until(started.is_set)
            first.cancel()
            await asyncio.sleep(0)
            second = asyncio.create_task(pool.run('asr', second_started.set))
            await self.until(lambda: bool(pool._semaphores['asr']._waiters))
            first.cancel()
            await asyncio.sleep(0)
            self.assertFalse(first.done())
            self.assertFalse(caller_closed.is_set())
            self.assertFalse(second_started.is_set())
            self.assertEqual(pool._semaphores['asr']._value, 0)
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        self.assertTrue(first.cancelled())
        self.assertTrue(second_started.is_set())
        self.assertTrue(caller_closed.is_set())
        self.assertEqual(pool._semaphores['asr']._value, 1)
        await pool.drain()
        self.assertFalse(pool._workers)

    async def test_queued_cancel_never_dispatches_and_thread_exception_is_collected(self):
        pool = BlockingWorkPool()
        pool._semaphores['artifact'] = asyncio.Semaphore(0)
        called = threading.Event()
        queued = asyncio.create_task(pool.run('artifact', called.set))
        await self.until(lambda: bool(pool._semaphores['artifact']._waiters))
        queued.cancel()
        await asyncio.gather(queued, return_exceptions=True)
        pool._semaphores['artifact'].release()
        self.assertFalse(called.is_set())
        started, release = threading.Event(), threading.Event()
        def failure():
            started.set()
            if not release.wait(3):
                raise TimeoutError('test release deadline')
            raise ValueError('synthetic worker error after cancellation')
        task = asyncio.create_task(pool.run('artifact', failure))
        try:
            await self.until(started.is_set)
            task.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
            result = await asyncio.gather(task, return_exceptions=True)
        self.assertIsInstance(result[0], asyncio.CancelledError)
        self.assertEqual(pool._semaphores['artifact']._value, 1)
        await pool.drain()

    async def test_timeout_waits_for_dispatched_work_and_queue_timeout_never_dispatches(self):
        pool = BlockingWorkPool()
        pool._semaphores['media'] = asyncio.Semaphore(1)
        started, finished, release = (threading.Event() for _ in range(3))
        def work():
            started.set()
            if not release.wait(3):
                raise TimeoutError('test release deadline')
            finished.set()
        async def timed():
            async with asyncio.timeout(.02):
                await pool.run('media', work)
        task = asyncio.create_task(timed())
        try:
            await self.until(started.is_set)
            await self.until(lambda: task.cancelling() > 0)
            self.assertFalse(task.done())
            self.assertEqual(pool._semaphores['media']._value, 0)
        finally:
            release.set()
            result = await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(finished.is_set())
        self.assertIsInstance(result[0], TimeoutError)
        called = threading.Event()
        pool._semaphores['media'] = asyncio.Semaphore(0)
        with patch('server.core.blocking.settings', SimpleNamespace(blocking_worker_queue_timeout_seconds=.02)):
            with self.assertRaises(TimeoutError):
                await pool.run('media', called.set)
        self.assertFalse(called.is_set())
        await pool.drain()
        self.assertFalse(pool._operations)

    async def test_drain_includes_queued_work_before_returning(self):
        pool = BlockingWorkPool()
        pool._semaphores['llm'] = asyncio.Semaphore(0)
        started, release = threading.Event(), threading.Event()
        def work():
            started.set()
            if not release.wait(3):
                raise TimeoutError('test release deadline')
        queued = asyncio.create_task(pool.run('llm', work))
        drain = None
        try:
            await self.until(lambda: bool(pool._semaphores['llm']._waiters))
            drain = asyncio.create_task(pool.drain())
            await asyncio.sleep(0)
            self.assertFalse(drain.done())
            pool._semaphores['llm'].release()
            await self.until(started.is_set)
            self.assertFalse(drain.done())
        finally:
            release.set()
            await asyncio.gather(queued, *([drain] if drain else []))
        self.assertFalse(pool._operations)
