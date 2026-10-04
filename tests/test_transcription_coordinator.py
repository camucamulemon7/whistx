import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'coordinator-test-secret-abcdefghijklmnopqrstuvwxyz')
from starlette.websockets import WebSocketDisconnect
from server.core.runtime_resources import RuntimeResources
from server.transcription import coordinator as service


class SyntheticSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.state = SimpleNamespace(authenticated_user_id=1, is_guest=False)
        self.events = []
        self.closed = None
        self.accepted = False
    async def accept(self):
        self.accepted = True
    async def receive_text(self):
        try:
            item = next(self.messages)
        except StopIteration:
            raise WebSocketDisconnect()
        return item if isinstance(item, str) else json.dumps(item)
    async def send_json(self, event):
        self.events.append(event)
    async def close(self, code=1000, reason=''):
        self.closed = (code, reason)


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_and_stop_finalize_artifacts_and_release_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            original = service.settings
            class Config:
                transcripts_dir = Path(directory) / 'transcripts'
                debug_chunks_dir = Path(directory) / 'audio'
                def __getattr__(self, key):
                    return getattr(original, key)
            resources = RuntimeResources(transcriber_factory=lambda: SimpleNamespace())
            socket = SyntheticSocket([{'type': 'start', 'language': 'ja'}, {'type': 'stop'}])
            sessions = []
            create = service.create_live_session
            def capture(*args, **kwargs):
                session = create(*args, **kwargs)
                sessions.append(session)
                return session
            with patch.object(service, 'settings', Config()), \
                 patch.object(service, 'create_live_session', side_effect=capture), \
                 patch.object(service.quota_repository, 'count_active_connections', return_value=1):
                await service.ws_transcribe(socket, resources=resources)
            self.assertTrue(socket.accepted)
            self.assertEqual(resources.active_sockets, set())
            self.assertTrue(sessions[0].store.read_metadata()['finalized'])
            messages = [event.get('message') for event in socket.events if event['type'] == 'info']
            self.assertEqual(messages, ['ready', 'stopping', 'finalized'])

    async def test_invalid_message_budget_closes_and_releases_socket(self):
        original = service.settings
        class Config:
            ws_max_invalid_messages = 2
            def __getattr__(self, key):
                return getattr(original, key)
        resources = RuntimeResources()
        socket = SyntheticSocket(['invalid-json', '[]'])
        with patch.object(service, 'settings', Config()), \
             patch.object(service.quota_repository, 'count_active_connections', return_value=1):
            await service.ws_transcribe(socket, resources=resources)
        self.assertEqual(socket.closed, (4400, 'too_many_invalid_messages'))
        self.assertEqual(resources.active_sockets, set())
        self.assertEqual([event['message'] for event in socket.events if event['type'] == 'error'], ['invalid_json', 'invalid_payload'])

    async def test_missing_provider_error_is_public_and_ping_still_works(self):
        resources = RuntimeResources()
        socket = SyntheticSocket([{'type': 'start'}, {'type': 'ping', 'ts': 123}, {'type': 'stop'}])
        with patch.object(service.quota_repository, 'count_active_connections', return_value=1), \
             self.assertLogs('server', level='ERROR'):
            await service.ws_transcribe(socket, resources=resources)
        error = next(event for event in socket.events if event['type'] == 'error')
        self.assertEqual(error['message'], 'session_create_failed')
        self.assertEqual(len(error['correlationId']), 32)
        self.assertNotIn('transcriber_not_ready', str(error))
        self.assertIn({'type': 'pong', 'ts': 123}, socket.events)
        self.assertEqual(resources.active_sockets, set())
