from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'summary-service-test-secret-abcdefghijklmnopqrstuvwxyz')

from server.schemas import ProofreadRequest, SummarizeRequest
from server.services import summary_service as service


class SummaryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_uses_explicit_model_and_observer(self):
        model = Mock()
        model.summarize_long.return_value = SimpleNamespace(text='要約', model='synthetic', chunk_count=2, reduced=True)
        observer = Mock()
        observer.create_trace_context.return_value = {'trace': 'synthetic'}
        response = await service.summarize(
            SummarizeRequest(text='  議事録  ', language='ja', prompt='決定事項'),
            summarizer=model, observer=observer,
        )
        self.assertEqual(json.loads(response.body), {
            'summary': '要約', 'model': 'synthetic', 'inputChars': 3, 'chunkCount': 2, 'reduced': True,
        })
        model.summarize_long.assert_called_once_with(
            text='議事録', language='ja', max_chars=service.settings.summary_input_max_chars,
            custom_template='決定事項', trace_context={'trace': 'synthetic'},
        )

    async def test_invalid_inputs_do_not_call_provider(self):
        model = Mock()
        cases = [
            (SummarizeRequest(text=' '), 'empty_text', 400),
            (SummarizeRequest(text='x' * (service.settings.summary_input_max_chars + 1)), 'summary_input_too_large', 413),
            (SummarizeRequest(text='x', prompt='x' * (service.settings.ws_prompt_max_chars + 1)), 'summary_prompt_too_large', 413),
        ]
        for payload, code, status in cases:
            response = await service.summarize(payload, summarizer=model)
            self.assertEqual(response.status_code, status)
            self.assertEqual(json.loads(response.body)['error'], code)
        model.summarize_long.assert_not_called()
        for function in [service.proofread, service.proofread_stream]:
            for text, code, status in [
                (' ', 'empty_text', 400),
                ('x' * (service.settings.proofread_input_max_chars + 1), 'proofread_input_too_large', 413),
            ]:
                response = await function(ProofreadRequest(text=text), proofreader=model)
                self.assertEqual(response.status_code, status)
                self.assertEqual(json.loads(response.body)['error'], code)
        model.proofread_long.assert_not_called()
        model.proofread_stream_long.assert_not_called()

    async def test_missing_models_preserve_unavailable_errors(self):
        response = await service.summarize(SummarizeRequest(text='sample'), summarizer=None)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)['error'], 'summary_not_configured')
        for function in [service.proofread, service.proofread_stream]:
            response = await function(ProofreadRequest(text='sample'), proofreader=None)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(json.loads(response.body)['error'], 'proofread_not_configured')

    async def test_proofread_applies_glossary_and_normalizes_mode(self):
        model = Mock()
        model.proofread_long.return_value = SimpleNamespace(text='draft', model='synthetic', chunk_count=1, reduced=False)
        with patch.object(service, 'load_shared_glossary', return_value={'text': 'terminology'}), \
             patch.object(service, 'apply_shared_glossary_replacements', return_value='corrected') as replace:
            response = await service.proofread(ProofreadRequest(text='sample', mode='UNKNOWN'), proofreader=model)
        self.assertEqual(json.loads(response.body)['corrected'], 'corrected')
        self.assertEqual(json.loads(response.body)['mode'], 'proofread')
        self.assertEqual(model.proofread_long.call_args.kwargs['glossary_text'], 'terminology')
        replace.assert_called_once_with('draft', 'terminology')

    async def test_stream_keeps_model_captured_at_request_time(self):
        model = Mock()
        model.proofread_stream_long.return_value = iter([
            {'type': 'delta', 'delta': 'draft'}, {'type': 'done'},
        ])
        with patch.object(service, 'load_shared_glossary', return_value={'text': 'terminology'}), \
             patch.object(service, 'apply_shared_glossary_replacements', return_value='corrected'):
            response = await service.proofread_stream(
                ProofreadRequest(text='sample', mode=' TRANSLATE_EN '), proofreader=model,
            )
            events = [json.loads(event[5:]) async for event in response.body_iterator]
        self.assertEqual(events, [
            {'type': 'delta', 'delta': 'draft'}, {'type': 'done'},
            {'type': 'final_text', 'text': 'corrected'},
        ])
        self.assertEqual(model.proofread_stream_long.call_args.kwargs['mode'], 'translate_en')
        self.assertEqual(response.headers['x-accel-buffering'], 'no')

    async def test_stream_provider_failure_is_redacted(self):
        model = Mock()
        model.proofread_stream_long.side_effect = RuntimeError('synthetic-provider-secret')
        with patch.object(service, 'load_shared_glossary', return_value={'text': ''}), \
             self.assertLogs('server', level='ERROR'):
            response = await service.proofread_stream(ProofreadRequest(text='sample'), proofreader=model)
            events = [json.loads(event[5:]) async for event in response.body_iterator]
        self.assertEqual(events[0]['type'], 'error')
        self.assertEqual(events[0]['error'], 'proofread_failed')
        self.assertEqual(len(events[0]['correlationId']), 32)
        self.assertNotIn('synthetic-provider-secret', str(events))
