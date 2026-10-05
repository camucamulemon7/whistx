from datetime import datetime, timezone
import os
import time
import unittest

os.environ.setdefault('APP_SESSION_SECRET', 'search-tests-secret-abcdefghijklmnopqrstuvwxyz')

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.api.routes import admin, history
from server.db import Base
from server.models import TranscriptHistory, User
from server.repositories import history_repository, user_repository
from server.repositories.search import normalize_query
from server.services import admin_service, history_service


class SearchPagingTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.addCleanup(self.db.close)
        now = datetime.now(timezone.utc)
        self.db.add_all([User(email=f'u{i}@example.test', display_name=f'User {i}', password_hash='unused',
                              is_active=i % 2 == 0, created_at=now) for i in range(2100)])
        self.db.commit()
        self.owner = self.db.get(User, 1)
        self.owner.is_admin = True
        self.owner.display_name = r'100%_literal\name'
        self.db.add_all([TranscriptHistory(id=f'hist_{i:04}', user_id=self.owner.id,
            runtime_session_id=f'synthetic-{i}', title=r'100%_literal\name' if i == 0 else 'ordinary',
            plain_text='synthetic', saved_at=now) for i in range(1100)])
        self.db.add(TranscriptHistory(id='hist_other', user_id=2, runtime_session_id='other', title=r'100%_literal\name', plain_text='synthetic'))
        self.db.commit()

    def test_large_lists_are_bounded_have_totals_and_no_page_overlap(self):
        started = time.monotonic()
        first = admin_service.list_users_payload(self.db, limit=50)
        second = admin_service.list_users_payload(self.db, limit=50, offset=50)
        self.assertEqual(first['total'], 2100)
        self.assertEqual(len(first['items']), 50)
        self.assertFalse({x['id'] for x in first['items']} & {x['id'] for x in second['items']})
        pending = admin_service.list_pending_users_payload(self.db, limit=30, offset=30)
        self.assertEqual(pending['total'], 1050)
        self.assertEqual(len(pending['items']), 30)
        self.assertEqual((pending['limit'], pending['offset']), (30, 30))
        self.assertEqual(admin_service.list_users_payload(self.db, query='no-match')['total'], 0)
        self.assertEqual(admin_service.list_users_payload(self.db, offset=3000)['items'], [])
        self.assertLess(time.monotonic() - started, 5, 'bounded SQLite fixture regression budget')

    def test_exact_last_pages_empty_pages_and_search_length_boundary(self):
        users = admin_service.list_users_payload(self.db, limit=100, offset=2000)
        self.assertEqual(len(users['items']), 100)
        self.assertEqual(admin_service.list_users_payload(self.db, offset=2100)['items'], [])
        pending = admin_service.list_pending_users_payload(self.db, limit=100, offset=1000)
        self.assertEqual(len(pending['items']), 50)
        self.assertEqual(admin_service.list_pending_users_payload(self.db, offset=1050)['items'], [])
        histories = history_repository.list_histories_for_user(self.db, user_id=1, limit=100, offset=1099)
        self.assertEqual([row.id for row in histories], ['hist_0000'])
        self.assertEqual(history_repository.list_histories_for_user(self.db, user_id=1, limit=1, offset=1100), [])
        self.assertEqual(normalize_query('x' * 200), 'x' * 200)
        self.assertEqual(admin_service.list_users_payload(self.db, query='   ')['total'], 2100)

    def test_search_treats_wildcards_and_backslashes_literally(self):
        for query in ['%', '_literal', '\\name', ' 100%_literal\\name ']:
            users = user_repository.search_users(self.db, query=query)
            self.assertEqual([user.id for user in users], [self.owner.id])
            histories = history_repository.list_histories_for_user(self.db, user_id=self.owner.id,
                                                                  query=query, limit=20, offset=0)
            self.assertEqual([item.id for item in histories], ['hist_0000'])
            self.assertEqual(history_repository.count_histories_for_user(self.db, user_id=self.owner.id, query=query), 1)
        with self.assertRaisesRegex(ValueError, 'search_query_too_long'):
            normalize_query('x' * 201)

    def test_history_order_is_stable_at_equal_timestamps(self):
        first = history_repository.list_histories_for_user(self.db, user_id=1, limit=20, offset=0)
        second = history_repository.list_histories_for_user(self.db, user_id=1, limit=20, offset=20)
        self.assertEqual([item.id for item in first], sorted((item.id for item in first), reverse=True))
        self.assertFalse({item.id for item in first} & {item.id for item in second})

    def test_routes_validate_limits_and_query_length(self):
        app = FastAPI()
        app.include_router(admin.router)
        app.include_router(history.router)
        app.dependency_overrides[admin.get_current_admin] = lambda: self.owner
        app.dependency_overrides[history.get_current_user] = lambda: self.owner
        app.dependency_overrides[admin.get_db] = lambda: self.db
        client = TestClient(app)
        response = client.get('/api/admin/users?limit=3&offset=4')
        self.assertEqual(response.status_code, 200)
        self.assertEqual((len(response.json()['items']), response.json()['total']), (3, 2100))
        for path in ['/api/admin/users', '/api/admin/pending-users', '/api/history']:
            for suffix in ['?limit=101', '?limit=0', '?offset=-1']:
                self.assertEqual(client.get(path + suffix).status_code, 422)
        for path in ['/api/admin/users', '/api/history']:
            self.assertEqual(client.get(path, params={'q': 'x'*201}).status_code, 422)
        self.assertEqual(client.get('/api/admin/users', params={'q': '%'}).json()['total'], 1)

    def test_history_preview_is_bounded_without_loading_full_text(self):
        item = self.db.get(TranscriptHistory, 'hist_0000')
        item.plain_text = 'synthetic ' * 100000
        item.summary_text = 'summary ' * 100000
        item.proofread_text = 'proofread ' * 100000
        self.db.commit()
        self.db.expunge_all()
        rows = history_repository.list_histories_for_user(self.db, user_id=1, query='100%', limit=20, offset=0)
        self.assertEqual(len(rows), 1)
        self.assertNotIn('plain_text', rows[0].__dict__)
        self.assertNotIn('summary_text', rows[0].__dict__)
        self.assertNotIn('proofread_text', rows[0].__dict__)
        self.assertEqual(len(rows[0].list_preview), 2048)
        self.assertLessEqual(len(history_service.build_history_list_item(rows[0])['preview']), 180)
        self.assertNotIn('plain_text', rows[0].__dict__, 'rendering must not lazy-load the transcript')
