import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'composition-tests-secret-abcdefghijklmnopqrstuvwxyz')
from server import runtime
from server.core.application import create_app
from server.core.runtime_resources import RuntimeResources
from server.services import health_service


class RuntimeCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_constructs_explicit_resources_and_cancels_tasks(self):
        original = runtime.settings
        class Config:
            openai_api_key = 'synthetic'
            diarization_enabled = True
            def __getattr__(self, name):
                return getattr(original, name)
        resources = RuntimeResources()
        observer, audio, model, diarizer = Mock(), Mock(), Mock(), Mock()
        with patch.object(runtime, 'resources', resources), patch.object(runtime, 'settings', Config()), \
             patch.object(runtime.runtime_cleanup, 'run_cleanup_once') as cleanup, \
             patch.object(runtime, 'make_langfuse_observer', return_value=observer), \
             patch.object(runtime, 'AudioPreprocessor', return_value=audio), \
             patch.object(runtime, 'OpenAISummarizer', return_value=model), \
             patch.object(runtime, 'PyannoteSpeakerDiarizer', return_value=diarizer):
            await runtime.on_startup()
            tasks = (resources.cleanup_task, resources.event_loop_monitor_task)
            try:
                self.assertIs(resources.audio_preprocessor, audio)
                self.assertIs(resources.summarizer, model)
                self.assertIs(resources.proofreader, model)
                self.assertIs(resources.diarizer, diarizer)
                self.assertIsNotNone(resources.transcriber_factory)
                self.assertIs(create_app().state.runtime_resources, resources)
                cleanup.assert_called_once_with('startup')
                diarizer.preflight.assert_called_once()
            finally:
                await runtime.on_shutdown()
        self.assertTrue(all(task.done() for task in tasks))
        self.assertIsNone(resources.cleanup_task)
        self.assertIsNone(resources.event_loop_monitor_task)
        observer.flush.assert_called_once()
        observer.shutdown.assert_called_once()

    async def test_health_reads_supplied_resources(self):
        resources = RuntimeResources(transcriber_factory=lambda: None, summarizer=Mock())
        with patch.object(health_service.quota_repository, 'count_active_connections', return_value=2):
            configured = json.loads((await health_service.health(resources=resources)).body)
            empty = json.loads((await health_service.health(resources=RuntimeResources())).body)
        self.assertTrue(configured['asrReady'])
        self.assertEqual(configured['activeConnections'], 2)
        self.assertFalse(empty['asrReady'])
        self.assertIsNone(empty['summaryModel'])

    def test_resource_instances_do_not_share_socket_sets(self):
        first, second = RuntimeResources(), RuntimeResources()
        first.active_sockets.add('synthetic')
        self.assertEqual(second.active_sockets, set())
