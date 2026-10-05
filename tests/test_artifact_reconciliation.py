from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'artifact-tests-secret-abcdefghijklmnopqrstuvwxyz')
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from server.db import Base
from server.models import ArtifactDeletion, TranscriptHistory, User
from server.services import artifact_deletion, artifact_reconciliation, history_service


class ArtifactConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'history'
        self.root.mkdir()
        self.engine = create_engine('sqlite:///' + str(Path(self.directory.name) / 'test.db'))
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine, expire_on_commit=False)
        self.addCleanup(self.db.close)
        self.user = User(email='test@example.test', password_hash='unused', is_active=True)
        self.db.add(self.user)
        self.db.commit()
        self.config = SimpleNamespace(history_dir=self.root, history_retention_days=7)

    def history(self, identifier='hist_old'):
        key = f'{self.user.id}/2026/09/{identifier}'
        path = self.root / key
        path.mkdir(parents=True)
        (path / 'transcript.txt').write_text('synthetic')
        (path / 'transcript.jsonl').write_text('{}\n')
        history = TranscriptHistory(id=identifier, user_id=self.user.id, runtime_session_id='session-'+identifier,
            title='synthetic', plain_text='synthetic', artifact_dir=key, txt_path=key+'/transcript.txt',
            jsonl_path=key+'/transcript.jsonl', saved_at=datetime.now(timezone.utc)-timedelta(days=10))
        self.db.add(history)
        self.db.commit()
        return history, path

    def test_failed_filesystem_delete_survives_restart_and_retries_idempotently(self):
        history, path = self.history()
        with patch.object(history_service, 'settings', self.config), patch.object(artifact_deletion.shutil, 'rmtree', side_effect=PermissionError('fixture')):
            self.assertTrue(history_service.delete_history_for_user(self.db, user=self.user, history_id=history.id))
        self.assertIsNone(self.db.get(TranscriptHistory, history.id))
        self.assertTrue(path.exists())
        self.db.close()
        with Session(self.engine) as restarted:
            self.assertEqual(restarted.get(ArtifactDeletion, history.id).attempts, 1)
            self.assertEqual(artifact_deletion.process_deletions(restarted, self.root)['completed'], 1)
            self.assertEqual(artifact_deletion.process_deletions(restarted, self.root)['completed'], 0)
        self.assertFalse(path.exists())

    def test_retention_rollback_does_not_remove_files_or_leave_queue(self):
        history, path = self.history()
        with patch.object(history_service, 'settings', self.config):
            self.assertEqual(history_service.cleanup_expired_histories(self.db), 1)
        self.assertTrue(path.exists())
        self.db.rollback()
        self.assertIsNotNone(self.db.get(TranscriptHistory, history.id))
        self.assertEqual(list(self.db.scalars(select(ArtifactDeletion))), [])
        with patch.object(history_service, 'settings', self.config):
            history_service.cleanup_expired_histories(self.db)
        self.db.commit()
        artifact_deletion.process_deletions(self.db, self.root)
        self.assertFalse(path.exists())

    def test_owner_path_and_symlink_are_never_deleted(self):
        outside = Path(self.directory.name) / 'other'
        outside.mkdir()
        (outside / 'keep').write_text('keep')
        for key in ['../other', '2/2026/09/hist_invalid']:
            request = ArtifactDeletion(history_id='hist_invalid', user_id=1, artifact_key=key)
            with self.assertRaises(ValueError):
                artifact_deletion.owned_paths(self.root, request)
        parent = self.root / '1'
        parent.mkdir()
        (parent / 'hist_invalid').symlink_to(outside, target_is_directory=True)
        request = ArtifactDeletion(history_id='hist_invalid', user_id=1, artifact_key='1/hist_invalid')
        with self.assertRaises(ValueError):
            artifact_deletion.owned_paths(self.root, request)
        self.assertEqual((outside / 'keep').read_text(), 'keep')

    def test_dry_run_and_offline_quarantine_preserve_live_and_recent_files(self):
        history, live = self.history()
        (live / 'transcript.jsonl').unlink()
        orphan = live.parent / 'hist_orphan'
        staging = live.parent / '.hist_stage.abcd'
        fresh = live.parent / 'hist_fresh'
        export = self.root / '_exports' / 'hist_export.zip'
        for path in [orphan, staging, fresh]:
            path.mkdir()
        export.parent.mkdir()
        export.write_bytes(b'zip')
        for path in [orphan, staging, export]:
            os.utime(path, (time.time()-90000, time.time()-90000))
        report = artifact_reconciliation.scan(self.db, self.root)
        self.assertEqual(report['counts'], {'abandoned_staging': 1, 'missing_artifact': 1, 'orphan_artifact': 1, 'temporary_zip': 1})
        self.assertTrue(orphan.exists())
        result = artifact_reconciliation.quarantine(self.db, self.root)
        self.assertEqual(result['moved'], 3)
        self.assertTrue(live.exists())
        self.assertTrue(fresh.exists())
        self.assertFalse(orphan.exists())
        self.assertTrue((Path(result['quarantine']) / 'manifest.json').exists())
        self.assertEqual(artifact_reconciliation.quarantine(self.db, self.root)['moved'], 0)

    def test_configured_root_alias_is_safe_but_child_symlinks_are_rejected(self):
        history, path = self.history()
        alias = Path(self.directory.name) / 'root-alias'
        alias.symlink_to(self.root, target_is_directory=True)
        request = ArtifactDeletion(history_id=history.id, user_id=self.user.id, artifact_key=history.artifact_dir)
        self.assertIn(path.resolve(), artifact_deletion.owned_paths(alias, request))
        self.assertIn(path.resolve(), artifact_deletion.owned_paths(self.root, request))
        history.user_id += 1
        request.user_id = history.user_id
        with self.assertRaisesRegex(ValueError, 'invalid_artifact_owner_path'):
            artifact_deletion.owned_paths(alias, request)

    def test_quarantine_destination_symlink_cannot_move_files_outside_root(self):
        orphan = self.root / '1' / 'hist_orphan'
        orphan.mkdir(parents=True)
        (orphan / 'keep').write_text('synthetic')
        os.utime(orphan, (time.time()-90000, time.time()-90000))
        outside = Path(self.directory.name) / 'external'
        outside.mkdir()
        (self.root / '_quarantine').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unsafe_quarantine_destination'):
            artifact_reconciliation.quarantine(self.db, self.root)
        self.assertTrue((orphan / 'keep').is_file())
        self.assertEqual(list(outside.iterdir()), [])

    def test_commit_failure_keeps_history_and_files_without_outbox(self):
        history, path = self.history()
        with patch.object(history_service, 'settings', self.config), \
             patch.object(self.db, 'commit', side_effect=RuntimeError('synthetic-commit-failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic-commit-failure'):
                history_service.delete_history_for_user(self.db, user=self.user, history_id=history.id)
        self.assertIsNotNone(self.db.get(TranscriptHistory, history.id))
        self.assertTrue(path.is_dir())
        self.assertEqual(list(self.db.scalars(select(ArtifactDeletion))), [])

    def test_live_history_blocks_cleanup_and_preserves_retry(self):
        history, path = self.history()
        artifact_deletion.enqueue(self.db, history)
        self.db.commit()
        result = artifact_deletion.process_deletions(self.db, self.root)
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['pending'], 1)
        self.assertEqual(self.db.get(ArtifactDeletion, history.id).attempts, 1)
        self.assertTrue(path.is_dir())

    def test_manifest_failure_prevents_quarantine_move(self):
        orphan = self.root / '1' / 'hist_orphan'
        orphan.mkdir(parents=True)
        (orphan / 'keep').write_text('synthetic')
        os.utime(orphan, (time.time()-90000, time.time()-90000))
        with patch.object(artifact_reconciliation, '_write_manifest', side_effect=OSError('synthetic-write-failure')):
            with self.assertRaisesRegex(OSError, 'synthetic-write-failure'):
                artifact_reconciliation.quarantine(self.db, self.root)
        self.assertTrue((orphan / 'keep').is_file())
