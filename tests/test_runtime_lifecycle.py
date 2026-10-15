"""Resource ownership across app instances, partial startup and shutdown."""
import asyncio
import json
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'runtime-lifecycle-tests-only-session-secret')

from fastapi.testclient import TestClient
from server import runtime
from server.api.routes import health
from server.core.application import create_app
from server.core.blocking import BlockingWorkPool
from server.core.runtime_resources import RuntimeResources
from server.openai_whisper import OpenAIWhisperTranscriber
from server.summarizer import OpenAISummarizer


class RuntimeOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_builds_fresh_clients_and_tasks_for_the_same_app_owner(self):
        resources = RuntimeResources()
        models, observers = [Mock() for _ in range(4)], [Mock() for _ in range(2)]
        original_settings = runtime.settings
        class Config:
            diarization_enabled = False
            def __getattr__(self, name):
                return getattr(original_settings, name)
        with patch.object(runtime, 'settings', Config()), \
                patch.object(runtime.runtime_cleanup, 'run_cleanup_once'), \
                patch.object(runtime, 'make_langfuse_observer', side_effect=observers), \
                patch.object(runtime, 'AudioPreprocessor'), \
                patch.object(runtime, 'OpenAISummarizer', side_effect=models):
            await runtime.on_startup(resources)
            first_tasks = (resources.cleanup_task, resources.event_loop_monitor_task)
            self.assertIs(resources.summarizer, models[0])
            await runtime.on_shutdown(resources)
            self.assertTrue(all(task.done() for task in first_tasks))
            await runtime.on_startup(resources)
            try:
                self.assertIs(resources.summarizer, models[2])
                self.assertIs(resources.observer, observers[1])
                self.assertIsNot(resources.cleanup_task, first_tasks[0])
                models[2].close.assert_not_called()
            finally:
                await runtime.on_shutdown(resources)
        for model in models:
            model.close.assert_called_once()
        for observer in observers:
            observer.flush.assert_called_once()
            observer.shutdown.assert_called_once()

    async def test_partial_startup_failure_cancels_created_tasks_and_closes_observer(self):
        resources = RuntimeResources()
        observer = Mock()
        with patch.object(runtime.runtime_cleanup, 'run_cleanup_once'), \
                patch.object(runtime, 'make_langfuse_observer', return_value=observer), \
                patch.object(runtime, 'AudioPreprocessor', side_effect=RuntimeError('synthetic startup failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic startup failure'):
                await runtime.on_startup(resources)
        self.assertIsNone(resources.cleanup_task)
        self.assertIsNone(resources.event_loop_monitor_task)
        self.assertIsNone(resources.observer)
        self.assertIsNone(resources.transcriber_factory)
        observer.flush.assert_called_once()
        observer.shutdown.assert_called_once()

    async def test_clients_close_on_owner_shutdown_and_repeated_close_is_safe(self):
        whisper = OpenAIWhisperTranscriber(api_key='synthetic', base_url='http://synthetic.invalid/v1', model='test')
        models = [OpenAISummarizer(api_key='synthetic', base_url='http://synthetic.invalid/v1', model='test', temperature=0) for _ in range(2)]
        try:
            whisper.close()
            whisper.close()
            self.assertTrue(whisper.client.is_closed())
            resources = RuntimeResources(summarizer=models[0], proofreader=models[1])
            await runtime.on_shutdown(resources)
            await runtime.on_shutdown(resources)
            self.assertTrue(all(model.client.is_closed() for model in models))
            self.assertIsNone(resources.summarizer)
            self.assertIsNone(resources.proofreader)
        finally:
            whisper.close()
            for model in models:
                model.close()

    async def test_shutdown_drains_worker_before_closing_its_client(self):
        pool = BlockingWorkPool()
        client = Mock()
        started, release = threading.Event(), threading.Event()
        def work():
            started.set()
            if not release.wait(3):
                raise TimeoutError('test release deadline')
            client.close.assert_not_called()
        operation = asyncio.create_task(pool.run('llm', work))
        shutdown = None
        with patch.object(runtime, 'blocking_work_pool', pool):
            try:
                async with asyncio.timeout(2):
                    while not started.is_set():
                        await asyncio.sleep(.001)
                shutdown = asyncio.create_task(runtime.on_shutdown(RuntimeResources(summarizer=client)))
                await asyncio.sleep(0)
                client.close.assert_not_called()
                self.assertFalse(shutdown.done())
            finally:
                release.set()
                await asyncio.gather(operation, *([shutdown] if shutdown else []))
        client.close.assert_called_once()

    async def test_cleanup_continues_when_one_client_close_fails(self):
        first, second, observer = Mock(), Mock(), Mock()
        first.close.side_effect = RuntimeError('synthetic close failure')
        resources = RuntimeResources(proofreader=first, summarizer=second, observer=observer)
        with self.assertLogs(runtime.logger, level='WARNING'):
            await runtime.on_shutdown(resources)
        second.close.assert_called_once()
        observer.shutdown.assert_called_once()
        self.assertIsNone(resources.proofreader)


class AppResourceIsolationTests(unittest.TestCase):
    def test_app_instances_and_health_routes_use_their_own_resources(self):
        first, second = create_app(), create_app()
        resources = first.state.runtime_resources
        self.assertIsNot(resources, second.state.runtime_resources)
        resources.transcriber_factory = lambda: None
        resources.active_sockets.add('synthetic')
        self.assertEqual(second.state.runtime_resources.active_sockets, set())
        with patch.object(health, 'settings', SimpleNamespace(openai_api_key='', asr_backend='speech_cpp')), \
                patch.object(health, 'schema_revision_status', return_value=SimpleNamespace(ready=True, current_revisions=('test',), expected_revisions=('test',))), \
                patch.object(health.health_service.quota_repository, 'count_active_connections', return_value=0):
            for app, configured in [(first, True), (second, False)]:
                response = TestClient(app).get('/api/health/ready')
                self.assertEqual(response.status_code, 200 if configured else 503)
                self.assertEqual(json.loads(response.content)['asr']['ready'], configured)
