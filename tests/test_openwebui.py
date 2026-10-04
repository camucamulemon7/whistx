"""Synthetic HTTP contracts for installed OpenWebUI 0.11.4; no credentials or live data."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'synthetic-openwebui-tests-secret-abcdefghijklmnopqrstuvwxyz')

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.api.routes import notes as notes_routes
from server.core.config import load_settings
from server.core.config.openwebui import DEFAULT_GENERATION_MODEL, normalize_openwebui_url
from server.deps import get_current_user
from server.models import User
from server.services import openwebui_notes as notes
from server.services.meeting_intelligence import generate_recap
from server.services.meeting_source import MeetingError, MeetingSnapshot
from server.summarizer import OpenAISummarizer


class OpenWebUIConfigTests(unittest.TestCase):
    def test_generation_uses_openwebui_and_qwopus_without_reusing_asr_credentials(self):
        with patch.dict(os.environ, {'OPENWEBUI_BASE_URL': 'http://localhost:3000/api/', 'OPENWEBUI_API_KEY': 'synthetic-webui',
                                      'SUMMARY_MODEL': '', 'SUMMARY_BASE_URL': 'http://wrong.test/v1',
                                      'SUMMARY_API_KEY': 'wrong', 'ASR_API_KEY': 'asr-only', 'ASR_BASE_URL': 'http://asr.test/v1'}):
            settings = load_settings()
            self.assertEqual(settings.summary_base_url, 'http://localhost:3000/api')
            self.assertEqual(settings.summary_model, DEFAULT_GENERATION_MODEL)
            self.assertEqual(settings.summary_api_key, 'synthetic-webui')
            self.assertEqual(settings.openai_base_url, 'http://asr.test/v1')
            with patch.dict(os.environ, {'OPENWEBUI_API_KEY': ''}):
                self.assertEqual(load_settings().summary_api_key, '')

    def test_endpoint_rejects_credentials_and_queries_without_echoing_them(self):
        for url in ['ftp://example.test', 'https://secret@example.test', 'https://example.test?token=secret', 'http:///missing']:
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, '^invalid_openwebui_url$'):
                normalize_openwebui_url(url)
        self.assertEqual(normalize_openwebui_url('https://example.test/prefix/api/v1/'), 'https://example.test/prefix')

    def test_real_openai_client_posts_generation_to_openwebui_api_with_qwopus(self):
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(200, json={'id': 'synthetic', 'object': 'chat.completion', 'created': 0,
                'model': DEFAULT_GENERATION_MODEL, 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'synthetic summary'}}]})
        model = OpenAISummarizer(api_key='synthetic-generation', base_url='https://openwebui.test/api',
                                 model=DEFAULT_GENERATION_MODEL, temperature=0.2)
        model.client.close()
        from openai import OpenAI
        model.client = OpenAI(api_key='synthetic-generation', base_url='https://openwebui.test/api',
                              http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        self.addCleanup(model.client.close)
        self.assertEqual(model.complete_meeting([{'role': 'user', 'content': 'synthetic meeting'}]), 'synthetic summary')
        self.assertEqual(str(calls[0].url), 'https://openwebui.test/api/chat/completions')
        self.assertEqual(json.loads(calls[0].content)['model'], DEFAULT_GENERATION_MODEL)
        self.assertEqual(calls[0].headers['authorization'], 'Bearer synthetic-generation')


class NotesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        path = self.root / 'meeting.jsonl'
        rows = [dict(id='1', seq=1, text='合成会議では担当を決めた。', startMs=0, endMs=1000, speaker='', audioUrl=None, imageId=None)]
        self.snapshot = MeetingSnapshot('runtime:synthetic', path, path.with_suffix('.meeting.json'), rows, [], 'rev1', 1000, True)
        chapter = {'title': '合成会議', 'start_id': '0', 'end_id': '0', 'summary': [{'text': '担当を決めた。', 'source_ids': ['0']}]}
        from types import SimpleNamespace
        generate_recap(self.snapshot, SimpleNamespace(model=DEFAULT_GENERATION_MODEL,
            complete_meeting=lambda *a, **kw: json.dumps({'chapters': [chapter]}, ensure_ascii=False)))
        settings_patch = patch.object(notes, 'settings', replace(notes.settings, app=replace(notes.settings.app, app_data_dir=self.root),
                                                                openwebui_base_url='https://openwebui.test'))
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.saved = []
        self.posts = 0
        self.mode = 'ok'
        self.identity = {'id': 'own-webui', 'email': 'user@example.test'}
        self.client = httpx.Client(base_url='https://openwebui.test/api/v1/',
                                  headers={'Authorization': 'Bearer synthetic-own-token'}, transport=httpx.MockTransport(self.handle))
        self.addCleanup(self.client.close)

    def handle(self, request):
        if request.url.path.endswith('/auths/'):
            return httpx.Response(401 if self.mode == 'identity401' else 200, json=self.identity)
        if request.method == 'GET' and request.url.path.endswith('/notes/search'):
            return httpx.Response(200, json={'items': self.saved, 'total': len(self.saved)})
        if request.url.path.endswith('/notes/create'):
            self.posts += 1
            body = json.loads(request.content)
            self.assertEqual(body['access_grants'], [])
            if self.mode == 'reject':
                return httpx.Response(422, json={})
            if self.mode == 'create401':
                return httpx.Response(401, json={})
            if self.mode == 'connect':
                raise httpx.ConnectError('synthetic', request=request)
            note = {**body, 'id': 'synthetic-note', 'user_id': 'own-webui'}
            if self.mode not in {'server500', 'server400'}:
                self.saved.append(note)
            if self.mode in {'timeout', 'unknown'}:
                if self.mode == 'unknown':
                    self.saved.clear()
                raise httpx.ReadTimeout('synthetic', request=request)
            if self.mode in {'server500', 'server400'}:
                return httpx.Response(500 if self.mode == 'server500' else 400, json={})
            return httpx.Response(200, json=note)
        self.fail(f'unexpected method/path {request.method} {request.url.path}')

    def save(self, **kwargs):
        return notes.save_recap(kwargs.pop('snapshot', self.snapshot), user_id=1, email='user@example.test',
                               token='synthetic-own-token', title=kwargs.pop('title', '合成会議'), client=self.client, **kwargs)

    def error(self, expected):
        with self.assertRaises(notes.NotesError) as caught:
            self.save()
        self.assertEqual(caught.exception.code, expected)

    def test_explicit_create_then_duplicate_never_overwrites_existing_note(self):
        self.assertEqual(self.posts, 0)
        self.assertFalse(self.save()['alreadySaved'])
        self.assertIn('担当を決めた', self.saved[0]['data']['content']['md'])
        self.saved[0]['data']['content']['md'] = 'User edited this existing note'
        self.assertTrue(self.save(title='Changed title')['alreadySaved'])
        self.assertEqual(self.posts, 1)
        self.assertEqual(self.saved[0]['data']['content']['md'], 'User edited this existing note')
        self.assertNotIn('synthetic-own-token', ''.join(p.read_text() for p in self.root.rglob('*.json')))

    def test_lost_create_response_reconciles_on_explicit_retry(self):
        self.mode = 'timeout'
        self.error('notes_save_uncertain')
        self.mode = 'ok'
        self.assertTrue(self.save()['alreadySaved'])
        self.assertEqual(self.posts, 1)

    def test_unknown_result_persists_across_client_restart_and_does_not_duplicate(self):
        self.mode = 'unknown'
        self.error('notes_save_uncertain')
        self.mode = 'ok'
        self.error('notes_save_uncertain')
        self.assertEqual(self.posts, 1)

    def test_ambiguous_400_and_500_never_blindly_retry(self):
        for mode in ['server400', 'server500']:
            with self.subTest(mode=mode):
                for file in (self.root / 'notes_exports').rglob('*.json'):
                    file.unlink()
                self.mode = mode
                before = self.posts
                self.error('notes_save_uncertain')
                self.mode = 'ok'
                self.error('notes_save_uncertain')
                self.assertEqual(self.posts, before + 1)

    def test_known_rejection_and_auth_failure_can_be_retried(self):
        for mode, code in [('reject', 'notes_save_rejected'), ('create401', 'notes_auth_required'), ('connect', 'notes_connection_failed')]:
            with self.subTest(mode=mode):
                for file in (self.root / 'notes_exports').rglob('*.json'):
                    file.unlink()
                self.mode = mode
                self.error(code)
                self.mode = 'ok'
                self.assertTrue(self.save()['ok'])

    def test_identity_and_destination_mismatch_do_not_post(self):
        self.mode = 'identity401'
        self.error('notes_auth_required')
        self.mode = 'ok'
        self.identity['email'] = 'other@example.test'
        self.error('notes_account_mismatch')
        self.assertEqual(self.posts, 0)

    def test_notes_uses_only_per_operation_credential_at_configured_origin(self):
        with patch.object(notes.httpx, 'Client', return_value=self.client) as factory:
            result = notes.save_recap(self.snapshot, user_id=1, email='user@example.test',
                                      token='synthetic-operation-token', title='Synthetic')
        self.assertTrue(result['ok'])
        self.assertEqual(factory.call_args.kwargs['base_url'], 'https://openwebui.test/api/v1/')
        self.assertEqual(factory.call_args.kwargs['headers'], {'Authorization': 'Bearer synthetic-operation-token'})
        self.assertFalse(factory.call_args.kwargs['follow_redirects'])
        self.assertFalse(factory.call_args.kwargs['trust_env'])

    def test_shared_note_from_another_owner_cannot_reconcile(self):
        self.mode = 'timeout'
        self.error('notes_save_uncertain')
        self.saved[0]['user_id'] = 'another-owner'
        self.mode = 'ok'
        self.error('notes_save_uncertain')
        self.assertEqual(self.posts, 1)

    def test_stale_recap_is_not_sent(self):
        with self.assertRaisesRegex(notes.NotesError, 'notes_current_recap_required'):
            self.save(snapshot=replace(self.snapshot, revision='rev2'))
        self.assertEqual(self.posts, 0)

    def test_two_simultaneous_saves_create_one_note(self):
        outcomes, errors = [], []
        def run():
            try:
                outcomes.append(self.save())
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(self.posts, 1)

    def test_route_requires_whistx_login_and_owner_authorized_source(self):
        app = FastAPI()
        app.include_router(notes_routes.router)
        with TestClient(app) as client:
            # Avoid any real database or user lookup in this synthetic route test.
            from fastapi import HTTPException
            def unauthenticated():
                raise HTTPException(401, 'login_required')
            app.dependency_overrides[get_current_user] = unauthenticated
            body = {'runtimeSessionId': 'other-runtime', 'token': 'synthetic-own-token'}
            self.assertEqual(client.post('/api/meeting/notes', json=body).status_code, 401)
            app.dependency_overrides[get_current_user] = lambda: User(id=1, email='user@example.test')
            with patch.object(notes_routes, 'consume', return_value=True), patch.object(notes_routes, 'load_meeting', side_effect=MeetingError('meeting_not_found', 404)) as load, patch.object(notes_routes, 'save_recap') as send:
                self.assertEqual(client.post('/api/meeting/notes', json=body).status_code, 404)
                self.assertEqual(load.call_args.kwargs['user_id'], 1)
                send.assert_not_called()
            invalid = client.post('/api/meeting/notes', json={**body, 'historyId': 'also-set'})
            self.assertEqual(invalid.status_code, 422)
            self.assertNotIn('synthetic-own-token', invalid.text)
