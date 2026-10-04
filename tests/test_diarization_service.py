import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'diarization-tests-secret-abcdefghijklmnopqrstuvwxyz')
from server.diarizer import SpeakerTurn
from server.services import diarization_service as service


class DiarizationServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Mock()
        self.session = SimpleNamespace(store=self.store, session_id='synthetic', audio_chunks=['synthetic'],
            collect_audio_for_diarization=True, diarization_num_speakers=0,
            diarization_min_speakers=0, diarization_max_speakers=2)
        self.config = SimpleNamespace(diarization_work_dir='synthetic', diarization_keep_chunks=False)

    async def test_disabled_or_empty_sessions_clean_chunks_without_model_calls(self):
        model = Mock()
        send = AsyncMock()
        await service.run_diarization_for_session(object(), self.session, diarizer=None, send=send, config=self.config)
        self.store.cleanup_chunks.assert_called_once()
        self.store.reset_mock()
        self.session.audio_chunks = []
        await service.run_diarization_for_session(object(), self.session, diarizer=model, send=send, config=self.config)
        model.diarize.assert_not_called()
        send.assert_not_called()

    async def test_final_labels_are_rewritten_and_patch_is_sorted(self):
        model = Mock()
        model.diarize.return_value = [SpeakerTurn(0, 1000, 'A'), SpeakerTurn(1000, 3000, 'B')]
        records = [{'type': 'final', 'seq': 2, 'tsStart': 1000, 'tsEnd': 2000},
                   {'type': 'partial', 'seq': 0, 'tsStart': 0, 'tsEnd': 900},
                   {'type': 'final', 'seq': 1, 'tsStart': 0, 'tsEnd': 900}]
        send = AsyncMock()
        with patch.object(service, 'read_jsonl_records', return_value=records):
            await service.run_diarization_for_session(object(), self.session, diarizer=model, send=send, config=self.config)
        self.store.rewrite_records.assert_called_once_with(records)
        self.assertNotIn('speaker', records[1])
        payloads = [call.args[1] for call in send.call_args_list]
        self.assertEqual(payloads[1]['segments'], [{'seq': 1, 'speaker': 'A'}, {'seq': 2, 'speaker': 'B'}])
        self.assertEqual(payloads[-1]['message'], 'diarization_done')
        self.store.cleanup_chunks.assert_called_once()

    async def test_failure_is_redacted_and_keep_chunks_is_preserved(self):
        model = Mock()
        model.diarize.side_effect = RuntimeError('synthetic-secret')
        self.config.diarization_keep_chunks = True
        send = AsyncMock()
        with self.assertLogs('server', level='ERROR'):
            await service.run_diarization_for_session(object(), self.session, diarizer=model, send=send, config=self.config)
        payloads = [call.args[1] for call in send.call_args_list]
        self.assertEqual(payloads[1]['error'], 'diarization_failed')
        self.assertNotIn('synthetic-secret', str(payloads))
        self.store.cleanup_chunks.assert_not_called()

    def test_speaker_overlap_and_nearest_distance(self):
        turns = [SpeakerTurn(0, 1000, 'A'), SpeakerTurn(1000, 2000, 'B')]
        self.assertEqual(service.pick_speaker(turns, 800, 1900), 'B')
        self.assertEqual(service.pick_speaker(turns, 2200, 2300), 'B')
        self.assertIsNone(service.pick_speaker(turns, 10000, 11000))
