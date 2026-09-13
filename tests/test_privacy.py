from __future__ import annotations

import asyncio
import json
import logging
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'privacy-tests-only-abcdefghijklmnopqrstuvwxyz')

from server import runtime
from server.langfuse_observer import LangfuseObserver, _safe_serialize
from server.runtime import SummarizeRequest, ProofreadRequest
from server.services.meeting_stream_response import stream_events
from server.transcription.qwen_live import QwenLiveMeeting


class PrivacyTests(unittest.TestCase):
    def test_websocket_provider_error_has_a_traceable_public_envelope(self):
        meeting = object.__new__(QwenLiveMeeting)
        meeting.disconnected = False
        meeting.stopping = True
        meeting.session_id = 'privacy-fixture'
        meeting.data = {'tracks': {'mic': {'received': 100, 'windowStart': 0}}}
        meeting._rt_window = AsyncMock(side_effect=RuntimeError('private-provider-url-and-key'))
        meeting._recover_commits = AsyncMock()
        meeting.send = AsyncMock()
        with self.assertLogs('server', level=logging.ERROR) as logs:
            asyncio.run(meeting._rt_lane('mic'))
        event = meeting.send.call_args.args[0]
        self.assertEqual(event['error'], 'transcription_failed')
        self.assertEqual(len(event['correlationId']), 32)
        self.assertNotIn('private-provider', json.dumps(event))
        self.assertTrue(any(event['correlationId'] in line for line in logs.output))

    def test_sdk_update_failure_does_not_interrupt_application_work(self):
        def fail(**kwargs):
            raise RuntimeError('telemetry unavailable')
        class Context:
            def __enter__(self):
                return SimpleNamespace(update=fail, end=fail)
            def __exit__(self, *args):
                pass
        observer = LangfuseObserver(public_key='', secret_key='')
        observer.enabled = True
        observer._client = SimpleNamespace(start_as_current_span=lambda **kwargs: Context())
        with observer.span(name='test') as span:
            span.update(output='private')
            span.end()

    def test_metadata_default_and_explicit_content_redaction(self):
        value = {'chars': 42, 'language': 'ja', 'transcript': 'confidential discussion',
                 'prompt': 'private instructions', 'glossary': ['secret product'],
                 'nested': {'email': 'alice@example.test', 'api_key': 'secret-value'}}
        safe = _safe_serialize(value)
        self.assertEqual((safe['chars'], safe['language']), (42, 'ja'))
        self.assertNotIn('confidential', json.dumps(safe))
        self.assertNotIn('private', json.dumps(safe))
        self.assertNotIn('secret', json.dumps(safe))
        opted = _safe_serialize({'text': 'discussion alice@example.test https://private.test/key sk-abcdefghijk', 'password': 'secret'}, capture_content=True)
        self.assertIn('discussion', opted['text'])
        for private in ['alice@', 'private.test', 'sk-abcdefghijk', 'secret']:
            self.assertNotIn(private, json.dumps(opted))

    def test_observation_updates_are_filtered_and_body_errors_do_not_reach_sdk(self):
        calls, exits = [], []
        class Context:
            def __enter__(self):
                return SimpleNamespace(update=lambda **values: calls.append(values), end=lambda **values: calls.append(values))
            def __exit__(self, *args):
                exits.append(args)
        observer = LangfuseObserver(public_key='', secret_key='')
        observer.enabled = True
        observer._client = SimpleNamespace(start_as_current_observation=lambda **values: (calls.append(values) or Context()))
        with self.assertRaisesRegex(RuntimeError, 'provider-secret'):
            with observer.generation(name='asr.test', input='private-transcript') as generation:
                generation.update(output='private-transcript', metadata={'email': 'a@example.test'})
                raise RuntimeError('provider-secret')
        self.assertNotIn('private-transcript', json.dumps(calls))
        self.assertNotIn('a@example.test', json.dumps(calls))
        self.assertEqual(exits, [(None, None, None)])

    def test_http_and_sse_provider_errors_have_codes_and_correlations(self):
        def fail(**kwargs):
            raise RuntimeError('provider-secret https://private.test/token')
        async def run():
            model = SimpleNamespace(summarize_long=fail, proofread_long=fail)
            with patch.object(runtime, 'SUMMARIZER', model), patch.object(runtime, 'PROOFREADER', model):
                responses = [await runtime.summarize(SummarizeRequest(text='sample')),
                             await runtime.proofread(ProofreadRequest(text='sample'))]
            for response in responses:
                body = json.loads(response.body)
                self.assertEqual(response.status_code, 502)
                self.assertEqual(len(body['correlationId']), 32)
                self.assertNotIn('provider-secret', str(body))
                self.assertNotIn('private.test', str(body))
            events = [item async for item in stream_events(lambda cancelled: fail())]
            payload = json.loads(next(item for item in events if item.startswith('data:'))[5:])
            self.assertEqual(payload['error'], 'meeting_model_unavailable')
            self.assertEqual(len(payload['correlationId']), 32)
            self.assertNotIn('provider-secret', str(payload))
        with self.assertLogs('server', level=logging.ERROR) as logs:
            asyncio.run(run())
        self.assertTrue(any('correlation_id=' in line for line in logs.output))
