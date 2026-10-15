"""Exercise the scheduler through committed retention and artifact retries."""
from contextlib import contextmanager
from pathlib import Path
import os
from unittest.mock import patch
import unittest

from sqlalchemy import select
from sqlalchemy.orm import Session

os.environ.setdefault('APP_SESSION_SECRET', 'scheduled-cleanup-tests-only-session-secret')

from server.core.config import settings
from server.models import ArtifactDeletion, TranscriptHistory
from server.services import artifact_deletion, artifact_storage, history_service, runtime_cleanup
from tests import test_artifact_reconciliation as artifact_cases


class ScheduledCleanupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = artifact_cases.ArtifactConsistencyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        root = Path(self.fixture.directory.name)
        class Config:
            history_dir = self.fixture.root
            transcripts_dir = root / 'transcripts'
            debug_chunks_dir = root / 'audio'
            history_retention_days = 7
            def __getattr__(self, name):
                return getattr(settings, name)
        config = Config()
        for module in (runtime_cleanup, history_service, artifact_storage):
            self.enterContext(patch.object(module, 'settings', config))
        @contextmanager
        def sessions():
            with Session(self.fixture.engine) as db:
                try:
                    yield db
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise
        self.enterContext(patch.object(runtime_cleanup, 'db_session', sessions))

    def test_scheduler_commits_retention_then_retries_files_and_runs_reconciliation(self):
        history, path = self.fixture.history()
        with patch.object(artifact_deletion.shutil, 'rmtree', side_effect=PermissionError('synthetic')), \
                patch.object(runtime_cleanup.artifact_reconciliation, 'scan', wraps=runtime_cleanup.artifact_reconciliation.scan) as scan:
            self.assertTrue(runtime_cleanup.run_cleanup_once('test'))
            scan.assert_called_once()
        with Session(self.fixture.engine) as db:
            self.assertIsNone(db.get(TranscriptHistory, history.id))
            self.assertEqual(db.get(ArtifactDeletion, history.id).attempts, 1)
        self.assertTrue(path.exists())
        self.assertTrue(runtime_cleanup.run_cleanup_once('retry'))
        self.assertTrue(runtime_cleanup.run_cleanup_once('idempotent retry'))
        with Session(self.fixture.engine) as db:
            self.assertEqual(list(db.scalars(select(ArtifactDeletion))), [])
        self.assertFalse(path.exists())

    def test_retention_failure_rolls_back_before_files_are_processed_and_reports_failure(self):
        history, path = self.fixture.history()
        original = runtime_cleanup.cleanup_expired_runtime_data
        def fail_after_retention(db):
            original(db)
            raise RuntimeError('synthetic failure before commit')
        with patch.object(runtime_cleanup, 'cleanup_expired_runtime_data', fail_after_retention), \
                patch.object(runtime_cleanup.artifact_deletion, 'process_deletions') as process, \
                self.assertLogs(runtime_cleanup.logger, level='WARNING'):
            self.assertFalse(runtime_cleanup.run_cleanup_once('failure'))
            process.assert_not_called()
        with Session(self.fixture.engine) as db:
            self.assertIsNotNone(db.get(TranscriptHistory, history.id))
            self.assertEqual(list(db.scalars(select(ArtifactDeletion))), [])
        self.assertTrue(path.exists())
