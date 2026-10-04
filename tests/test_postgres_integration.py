"""Synthetic integration tests restricted to the dedicated whistx_test database."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'synthetic-pg-tests-secret-abcdefghijklmnopqrstuvwxyz')

from sqlalchemy import create_engine, delete, insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from server.models import ArtifactDeletion, TranscriptHistory, User
from server.repositories import history_repository, user_repository
from server.services import artifact_deletion


@unittest.skipUnless(os.environ.get('WHISTX_PG_TEST_DB_URL'), 'requires dedicated whistx_test PostgreSQL')
class PostgreSQLIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.url = os.environ['WHISTX_PG_TEST_DB_URL']
        if make_url(self.url).database != 'whistx_test':
            self.fail('integration tests require the dedicated whistx_test database')
        self.engine = create_engine(self.url)
        self.addCleanup(self.engine.dispose)
        self.db = Session(self.engine)
        self.addCleanup(self.db.close)
        self.prefix = 'pg_' + uuid.uuid4().hex[:12]
        self.temp = tempfile.TemporaryDirectory(prefix='whistx-pg-artifacts-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.user = User(email=self.prefix+'@example.test', display_name='100%_literal\\name', password_hash='synthetic')
        self.db.add(self.user)
        self.db.commit()
        self.owner_id = self.user.id
        self.addCleanup(self.cleanup_rows, self.owner_id)

    def cleanup_rows(self, owner):
        self.db.rollback()
        self.db.execute(delete(ArtifactDeletion).where(ArtifactDeletion.user_id == owner))
        self.db.execute(delete(TranscriptHistory).where(TranscriptHistory.user_id == owner))
        self.db.execute(delete(User).where(User.email.like(self.prefix+'%')))
        self.db.commit()

    def test_search_plans_latency_and_page_boundaries(self):
        self.db.execute(insert(User), [{'email':f'{self.prefix}_{i}@example.test', 'display_name':f'synthetic user {i}',
                                      'password_hash':'synthetic'} for i in range(10000)])
        self.db.execute(insert(TranscriptHistory), [{'id':f'{self.prefix}_{i:05}', 'user_id':self.owner_id,
            'runtime_session_id':f'{self.prefix}_runtime_{i}',
            'title':'pgneedle92731' if i == 0 else f'synthetic meeting {i}',
            'plain_text':'synthetic transcript ' * 100} for i in range(20000)])
        self.db.commit()
        with self.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as maintenance:
            maintenance.execute(text('VACUUM ANALYZE users'))
            maintenance.execute(text('VACUUM ANALYZE transcript_histories'))
        durations = []
        for _ in range(20):
            started = time.perf_counter()
            rows = history_repository.list_histories_for_user(self.db, user_id=self.owner_id, query='pgneedle92731', limit=50, offset=0)
            self.assertEqual(len(rows), 1)
            durations.append((time.perf_counter()-started)*1000)
        plan = self.db.execute(text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id FROM transcript_histories WHERE user_id=:owner AND (title ILIKE '%pgneedle92731%' OR plain_text ILIKE '%pgneedle92731%') LIMIT 50"), {'owner':self.owner_id}).scalar_one()
        rendered = json.dumps(plan)
        self.assertIn('trgm', rendered)
        users = user_repository.search_users(self.db, query='100%_literal\\name')
        self.assertEqual([u.id for u in users], [self.owner_id])
        first = history_repository.list_histories_for_user(self.db, user_id=self.owner_id, limit=100, offset=0)
        second = history_repository.list_histories_for_user(self.db, user_id=self.owner_id, limit=100, offset=100)
        self.assertFalse({r.id for r in first} & {r.id for r in second})
        self.assertEqual(len(history_repository.list_histories_for_user(self.db, user_id=self.owner_id, limit=100, offset=19999)), 1)
        self.assertEqual(history_repository.list_histories_for_user(self.db, user_id=self.owner_id, limit=100, offset=20000), [])
        durations.sort()
        self.assertLess(durations[-1], 2000, 'synthetic query regression budget')
        print(json.dumps({'fixtureUsers':10001,'fixtureHistories':20000,'queryP50Ms':round(durations[10],2),
                          'queryP95Ms':round(durations[18],2),'queryPlan':plan}))

    def enqueue(self, number):
        identifier = f'{self.prefix}_{number}'
        key = f'{self.owner_id}/{identifier}'
        folder = self.root/key
        folder.mkdir(parents=True)
        (folder/'synthetic.txt').write_text('synthetic')
        self.db.add(ArtifactDeletion(history_id=identifier, user_id=self.owner_id, artifact_key=key))
        self.db.commit()
        return identifier, folder

    def test_locked_request_is_skipped_and_recovered_after_release(self):
        identifier, folder = self.enqueue('locked')
        with Session(self.engine) as holder:
            holder.scalar(select(ArtifactDeletion).where(ArtifactDeletion.history_id == identifier).with_for_update())
            result = artifact_deletion.process_deletions(self.db, self.root)
            self.assertEqual(result['completed'], 0)
            self.assertTrue(folder.exists())
            holder.rollback()
        self.assertEqual(artifact_deletion.process_deletions(self.db, self.root)['completed'], 1)
        self.assertFalse(folder.exists())

    def test_six_process_workers_delete_each_request_once(self):
        for number in range(40):
            self.enqueue(number)
        program = '''
import json,sys
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from server.services.artifact_deletion import process_deletions
with Session(create_engine(sys.argv[1])) as db:
 print(json.dumps(process_deletions(db,sys.argv[2])))
'''
        processes = [subprocess.Popen([sys.executable,'-c',program,self.url,str(self.root)],stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,text=True) for _ in range(6)]
        try:
            completed = 0
            for process in processes:
                stdout, stderr = process.communicate(timeout=40)
                self.assertEqual(process.returncode, 0, stderr)
                completed += json.loads(stdout)['completed']
            self.assertEqual(completed, 40)
            self.assertEqual(artifact_deletion.process_deletions(self.db,self.root)['completed'], 0)
            self.assertEqual(list(self.db.scalars(select(ArtifactDeletion).where(ArtifactDeletion.user_id == self.owner_id))), [])
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()

    def test_retry_after_restart_and_rollback(self):
        identifier, folder = self.enqueue('retry')
        with patch.object(artifact_deletion.shutil,'rmtree',side_effect=PermissionError('synthetic')):
            self.assertEqual(artifact_deletion.process_deletions(self.db,self.root)['completed'],0)
        self.db.close()
        with Session(self.engine) as restarted:
            self.assertEqual(restarted.get(ArtifactDeletion,identifier).attempts,1)
            self.assertEqual(artifact_deletion.process_deletions(restarted,self.root)['completed'],1)
        self.assertFalse(folder.exists())
        key = f'{self.owner_id}/{self.prefix}_rollback'
        folder = self.root/key
        folder.mkdir(parents=True)
        self.db.add(ArtifactDeletion(history_id=self.prefix+'_rollback', user_id=self.owner_id, artifact_key=key))
        self.db.flush()
        self.db.rollback()
        self.assertEqual(artifact_deletion.process_deletions(self.db,self.root)['completed'],0)
        self.assertTrue(folder.exists())
