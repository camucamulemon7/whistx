"""Optional real loopback disconnect checks; isolated DB, synthetic audio, no provider."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'media-disconnect-tests-only-long-session-secret')

import asyncio
import httpx
import uvicorn

from server.api.routes import media
from server.core.blocking import BlockingWorkPool
from server.services import media_import, meeting_stream_response
from tests import test_media_security as security_cases


@unittest.skipUnless(os.environ.get('WHISTX_MEDIA_SOCKET_TESTS') == '1' and
    (os.environ.get('FFMPEG_BIN') or shutil.which('ffmpeg')), 'enable isolated media socket checks with FFmpeg explicitly')
class MediaDisconnectTests(unittest.TestCase):
    def setUp(self):
        self.fixture = security_cases.MediaSecurityTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.scratch = self.fixture.root / 'scratch'
        self.scratch.mkdir()
        self.uploads = []
        make_upload = media.tempfile.NamedTemporaryFile
        make_decode = media_import.tempfile.TemporaryDirectory
        def upload(*args, **kwargs):
            result = make_upload(*args, **kwargs, dir=self.scratch)
            self.uploads.append(Path(result.name))
            return result
        self.fixture.stack.enter_context(patch.object(media.tempfile, 'NamedTemporaryFile', upload))
        self.fixture.stack.enter_context(patch.object(media_import.tempfile, 'TemporaryDirectory',
            lambda *args, **kwargs: make_decode(*args, **kwargs, dir=self.scratch)))
        self.pool = BlockingWorkPool()
        self.fixture.stack.enter_context(patch.object(media, 'blocking_work_pool', self.pool))
        self.fixture.stack.enter_context(patch.object(meeting_stream_response, 'blocking_work_pool', self.pool))
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(64)
        self.addCleanup(listener.close)
        self.port = listener.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(self.fixture.app, lifespan='off', log_level='error', access_log=False))
        self.thread = threading.Thread(target=self.server.run, kwargs={'sockets': [listener]}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.wait(lambda: self.server.started)
        self.client = httpx.Client(base_url=f'http://127.0.0.1:{self.port}',
            cookies={'whistx_session': self.fixture.tokens[71]}, timeout=5)
        self.addCleanup(self.client.close)

    def stop_server(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def wait(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            self.assertLess(time.monotonic(), deadline, 'isolated media check timed out')
            time.sleep(.02)

    def clean(self):
        self.wait(lambda: bool(self.uploads) and all(not path.exists() for path in self.uploads))
        self.wait(lambda: not list(self.scratch.glob('whistx-decode-*')))
        self.assertEqual(list(self.fixture.root.rglob('*.jsonl')), [])
        self.assertEqual(list(self.fixture.root.rglob('*.meta.json')), [])
        self.assertEqual(list(self.fixture.config.debug_chunks_dir.rglob('*.wav')), [])

    def test_disconnect_while_uploading_deletes_partial_upload_without_asr(self):
        factory = self.fixture.app.state.runtime_resources.transcriber_factory
        with patch.object(self.fixture.app.state.runtime_resources, 'transcriber_factory', side_effect=factory) as calls:
            connection = socket.create_connection(('127.0.0.1', self.port), timeout=5)
            try:
                headers = ('POST /api/media/transcribe?filename=partial.wav HTTP/1.1\r\n'
                    f'Host: 127.0.0.1:{self.port}\r\nContent-Length: 1048576\r\n'
                    f'Cookie: whistx_session={self.fixture.tokens[71]}\r\n\r\n').encode()
                connection.sendall(headers + b'\0' * 65536)
                self.wait(lambda: bool(self.uploads) and self.uploads[0].exists() and self.uploads[0].stat().st_size > 0)
            finally:
                connection.close()
            self.clean()
            self.assertEqual(calls.call_count, 0)

    def test_disconnect_during_asr_retains_worker_until_return_and_cleans_up(self):
        started, release, closed = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def recognize(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise RuntimeError('synthetic provider timed out')
            return SimpleNamespace(text='中断後の合成結果')
        self.fixture.app.state.runtime_resources.transcriber_factory = lambda: SimpleNamespace(
            transcribe_chunk=recognize, close=closed.set)
        initial = self.pool._semaphores['asr']._value
        with self.client.stream('POST', '/api/media/transcribe?filename=synthetic.wav', content=self.fixture.audio) as response:
            self.assertEqual(response.status_code, 200)
            lines = response.iter_lines()
            self.assertTrue(next(lines).startswith('data:'))
            self.assertTrue(started.wait(3))
        self.wait(lambda: all(not path.exists() for path in self.uploads))
        self.assertEqual(self.pool._semaphores['asr']._value, initial - 1)
        self.assertFalse(closed.is_set())
        release.set()
        self.assertTrue(closed.wait(3))
        self.clean()
        self.wait(lambda: self.pool._semaphores['asr']._value == initial)

    def test_disconnect_during_decoder_kills_the_process_and_removes_temporary_files(self):
        binary = self.fixture.root / 'synthetic-decoder'
        binary.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(30)\n')
        binary.chmod(0o700)
        self.fixture.config.ffmpeg_bin = str(binary)
        processes = []
        start = subprocess.Popen
        def spawn(*args, **kwargs):
            process = start(*args, **kwargs)
            processes.append(process)
            self.addCleanup(lambda: process.kill() if process.poll() is None else None)
            return process
        with patch.object(media_import.subprocess, 'Popen', spawn):
            with self.client.stream('POST', '/api/media/transcribe?filename=synthetic.wav', content=self.fixture.audio) as response:
                self.assertEqual(response.status_code, 200)
                lines = response.iter_lines()
                self.assertTrue(next(lines).startswith('data:'))
                self.wait(lambda: bool(processes))
            self.wait(lambda: processes[0].poll() is not None)
            self.clean()

    def test_disconnect_before_worker_start_never_calls_the_provider(self):
        semaphore = asyncio.Semaphore(0)
        self.pool._semaphores['asr'] = semaphore
        factory = self.fixture.app.state.runtime_resources.transcriber_factory
        with patch.object(self.fixture.app.state.runtime_resources, 'transcriber_factory', side_effect=factory) as calls:
            with self.client.stream('POST', '/api/media/transcribe?filename=queued.wav', content=self.fixture.audio) as response:
                self.assertEqual(response.status_code, 200)
            self.wait(lambda: bool(self.uploads) and all(not path.exists() for path in self.uploads))
            self.assertEqual(calls.call_count, 0)
            self.wait(lambda: semaphore._loop is not None)
            semaphore._loop.call_soon_threadsafe(semaphore.release)
            self.wait(lambda: semaphore._value == 1)
            self.clean()
            self.assertEqual(calls.call_count, 0)


if __name__ == '__main__':
    unittest.main()
