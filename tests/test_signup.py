"""HTTP registration/login and authorization against isolated synthetic SQLite."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'synthetic-signup-tests-secret-abcdefghijklmnopqrstuvwxyz')

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server import auth
from server.api.routes import admin as admin_routes, auth as auth_routes, history as history_routes
from server.db import Base, get_db
from server.models import TranscriptHistory
from server.services import auth_service


class SignupTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.addCleanup(self.db.close)
        admin = auth.create_user(self.db, email='admin@example.test', password='SyntheticPassword1!', is_admin=True)
        old = auth.create_user(self.db, email='inactive@example.test', password='SyntheticPassword1!', is_active=False)
        self.old_id = old.id
        self.db.add(TranscriptHistory(id='private-history', user_id=admin.id, runtime_session_id='private-runtime',
                                      title='synthetic private', plain_text='synthetic only'))
        self.db.commit()
        app = FastAPI()
        for router in [auth_routes.router, admin_routes.router, history_routes.router]:
            app.include_router(router)
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        settings_patch = patch.object(auth_service, 'settings', replace(auth_service.settings, app=replace(auth_service.settings.app, enable_self_signup=True)))
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        for name, result in [('_login_retry_after_seconds_for_keys', None), ('_clear_failed_login', None), ('_record_failed_login', None)]:
            mocked = patch.object(auth_service, name, return_value=result)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.payload = {'email': 'new@example.test', 'password': 'SyntheticPassword1!', 'display_name': 'New user',
                        'is_admin': True, 'is_active': False}

    def test_signup_immediately_logs_in_as_ordinary_user_and_respects_ownership(self):
        response = self.client.post('/api/auth/register', json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['pending'])
        self.assertFalse(response.json()['user']['isAdmin'])
        created = auth.get_user_by_email(self.db, self.payload['email'])
        self.assertTrue(created.is_active)
        self.assertIsNone(created.approved_by_user_id)
        self.assertEqual(self.client.get('/api/history').status_code, 401)
        self.assertEqual(self.client.post('/api/auth/login', json=self.payload).status_code, 200)
        self.assertTrue(self.client.get('/api/auth/me').json()['authenticated'])
        self.assertEqual(self.client.get('/api/admin/users').status_code, 403)
        self.assertEqual(self.client.get('/api/history/private-history').status_code, 404)
        self.assertEqual(self.client.delete('/api/history/private-history').status_code, 404)
        self.assertFalse(self.db.get(type(created), self.old_id).is_active)

    def test_duplicate_signup_and_bad_login_do_not_enable_or_escalate_accounts(self):
        self.assertEqual(self.client.post('/api/auth/register', json=self.payload).status_code, 200)
        self.assertEqual(self.client.post('/api/auth/register', json=self.payload).status_code, 409)
        wrong = {**self.payload, 'password': 'WrongPassword123'}
        self.assertEqual(self.client.post('/api/auth/login', json=wrong).status_code, 401)
        self.assertEqual(self.client.get('/api/history').status_code, 401)
        inactive = {**self.payload, 'email': 'inactive@example.test'}
        self.assertEqual(self.client.post('/api/auth/register', json=inactive).status_code, 409)
        self.assertEqual(self.client.post('/api/auth/login', json=inactive).status_code, 403)

    def test_registration_still_requires_opt_in_and_initial_admin(self):
        disabled = replace(auth_service.settings, app=replace(auth_service.settings.app, enable_self_signup=False))
        with patch.object(auth_service, 'settings', disabled):
            self.assertEqual(self.client.post('/api/auth/register', json=self.payload).status_code, 403)
        with patch.object(auth_service.auth, 'has_admin_account', return_value=False):
            self.assertEqual(self.client.post('/api/auth/register', json=self.payload).status_code, 409)
