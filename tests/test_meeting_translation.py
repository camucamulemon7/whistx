import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from server.services.meeting_source import MeetingSnapshot, MeetingError
from server.services.meeting_translation import translation_events


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / 'meeting.jsonl'
        self.rows = [dict(id='hq-0', text='公開日は未定。', quality='high_accuracy', startMs=0, endMs=30000, speaker=''),
                     dict(id='rt-30', text='金曜日に確認。', quality='realtime', startMs=30000, endMs=35000, speaker='')]
        self.snapshot = MeetingSnapshot('runtime:test', path, path.with_suffix('.meeting.json'), self.rows, [], 'r1', 35000, False)
        self.calls = []
        def complete(messages, **kwargs):
            self.calls.append(messages)
            rows = json.loads(messages[-1]['content'])
            return json.dumps({'translations': [dict(id=row['id'], text='Translated '+row['text']) for row in rows]})
        self.model = SimpleNamespace(complete_meeting=complete)

    def run_translation(self, snapshot=None, language='en', cancelled=None):
        return list(translation_events(snapshot or self.snapshot, self.model, language, cancelled=cancelled or threading.Event()))

    def test_only_hq_is_translated_during_recording_and_cached(self):
        events = self.run_translation()
        self.assertEqual([e['id'] for e in events if e['type'] == 'translation'], ['hq-0'])
        self.run_translation()
        self.assertEqual(len(self.calls), 1)
        self.run_translation(language='ja')
        self.assertEqual(len(self.calls), 2)

    def test_changed_text_invalidates_translation_and_stop_covers_remaining(self):
        self.run_translation()
        changed = replace(self.snapshot, segments=[dict(self.rows[0], text='公開日は来月。'), self.rows[1]], finalized=True)
        events = self.run_translation(changed)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([e['id'] for e in events if e['type']=='translation'], ['hq-0', 'rt-30'])
        self.assertIn('来月', events[1]['text'])

    def test_invalid_translation_is_not_saved(self):
        self.model.complete_meeting = lambda *a, **kw: '{"translations":[{"id":"invented","text":"wrong"}]}'
        with self.assertRaises(MeetingError):
            self.run_translation()
        self.assertFalse(self.snapshot.insight_path.exists())

    def test_cancel_and_invalid_language_make_no_model_calls(self):
        stop = threading.Event()
        stop.set()
        self.run_translation(cancelled=stop)
        with self.assertRaises(MeetingError):
            self.run_translation(language='unknown')
        self.assertEqual(self.calls, [])


    def test_api_streams_only_selected_authorized_snapshot(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from unittest.mock import patch, AsyncMock
        from server.api.routes.summary import router
        from server.deps import get_current_user
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=123)
        with TestClient(app) as client, patch('server.api.routes.summary._meeting_source', AsyncMock(return_value=self.snapshot)) as source, patch('server.api.routes.summary._allow_costly_request', return_value=True), patch('server.api.routes.summary.runtime.SUMMARIZER', self.model):
            response = client.post('/api/meeting/translate', json={'runtimeSessionId':'test','language':'en'})
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual(events[-1]['type'], 'done')
            self.assertEqual([e['id'] for e in events if e['type']=='translation'], ['hq-0'])
            self.assertEqual(source.call_args.args[0].runtimeSessionId, 'test')
            self.assertEqual(source.call_args.args[1].id, 123)
            self.assertEqual(client.post('/api/meeting/translate', json={'runtimeSessionId':'test','language':'unknown'}).status_code, 400)
