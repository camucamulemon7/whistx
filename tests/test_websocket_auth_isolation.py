"""Keep the complete synchronous authentication transaction off the loop."""
import asyncio
from contextlib import contextmanager
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'websocket-auth-isolation-session-secret')

from server.api.ws import transcribe


class WebSocketAuthenticationTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_lookup_does_not_block_another_connection_and_session_stays_on_one_thread(self):
        started, release = threading.Event(), threading.Event()
        thread_ids = []
        main_thread = threading.get_ident()
        @contextmanager
        def session():
            thread_ids.append(threading.get_ident())
            try:
                yield object()
            finally:
                thread_ids.append(threading.get_ident())
        def lookup(ws, db):
            thread_ids.append(threading.get_ident())
            if ws.slow:
                started.set()
                if not release.wait(3):
                    raise TimeoutError('test release deadline')
            return SimpleNamespace(id=1, is_active=False)
        class Socket:
            def __init__(self, slow):
                self.slow = slow
                self.closed = asyncio.Event()
            async def close(self, **kwargs):
                self.code = kwargs['code']
                self.closed.set()
        first, second = Socket(True), Socket(False)
        with patch.object(transcribe, 'origin_is_allowed', return_value=True), \
                patch.object(transcribe, 'db_session', session), \
                patch.object(transcribe, 'get_optional_user_from_request', lookup):
            operation = asyncio.create_task(transcribe.ws_transcribe(first))
            try:
                async with asyncio.timeout(2):
                    while not started.is_set():
                        await asyncio.sleep(.001)
                    await transcribe.ws_transcribe(second)
                self.assertTrue(second.closed.is_set())
                self.assertFalse(first.closed.is_set())
            finally:
                release.set()
                await operation
        self.assertEqual((first.code, second.code), (4403, 4403))
        self.assertNotIn(main_thread, thread_ids)
        self.assertEqual(thread_ids[0], thread_ids[1])
        self.assertEqual(thread_ids[2], thread_ids[3])
        self.assertEqual(thread_ids[3], thread_ids[4])
        self.assertEqual(thread_ids[0], thread_ids[5])
