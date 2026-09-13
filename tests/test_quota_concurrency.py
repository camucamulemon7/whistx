"""Shared-quota integration against a disposable PostgreSQL database."""
import json
import os
import subprocess
import sys
import unittest
import uuid


@unittest.skipUnless(os.environ.get('QUOTA_TEST_DB_URL'), 'requires disposable PostgreSQL')
class SharedQuotaConcurrencyTests(unittest.TestCase):
    def test_six_processes_cannot_exceed_shared_budget(self):
        environment = dict(os.environ, APP_DB_URL=os.environ['QUOTA_TEST_DB_URL'])
        bucket = 'ci-concurrency-' + uuid.uuid4().hex
        program = '''
import json, sys
from server.repositories.quota_repository import consume_rate_limit
print(json.dumps(sum(consume_rate_limit(bucket=sys.argv[1], subject="test", limit=20, window_seconds=300) for _ in range(10))))
'''
        # Seed the row to exercise lock contention rather than insertion timing.
        seed = 'from server.repositories.quota_repository import consume_rate_limit; import sys; consume_rate_limit(bucket=sys.argv[1], subject="test", limit=20, window_seconds=300)'
        subprocess.run([sys.executable, '-c', seed, bucket], env=environment, check=True, timeout=20)
        processes = [subprocess.Popen([sys.executable, '-c', program, bucket], env=environment,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(6)]
        try:
            accepted = 1
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr)
                accepted += json.loads(stdout)
            self.assertEqual(accepted, 20)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
            cleanup = 'from server.repositories.quota_repository import clear_rate_limit; import sys; clear_rate_limit(bucket=sys.argv[1], subject="test")'
            subprocess.run([sys.executable, '-c', cleanup, bucket], env=environment, check=True, timeout=20)
