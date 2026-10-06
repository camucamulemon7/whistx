"""Rolling HTTP Whisper plus automatic long-window re-recognition."""
from __future__ import annotations

import asyncio

from ..core.blocking import blocking_work_pool
from ..core.config import settings
from ..core.public_errors import public_error
from ..openai_whisper import OpenAIWhisperTranscriber
from ..services.meeting_refinement import _atomic_text
from ..services.meeting_source import MeetingError, write_json_atomic
from ..transcript_store import _render_txt_line
from .high_accuracy import HighAccuracyMeetingMixin
from .live import LiveMeeting, UPDATE_SAMPLES, logger
from .local_agreement import SAMPLE_RATE, pcm_wav
from .text_processing import _sanitize_transcript_text


class WhisperLiveMeeting(HighAccuracyMeetingMixin, LiveMeeting):
    hq_lock_name = '.whisper-hq.lock'
    retain_failed_hq = True

    @property
    def lane_settings(self):
        return settings

    def __init__(self, ws, payload, *, resources=None):
        self.hq_transcriber = None
        super().__init__(ws, payload, resources=resources)
        try:
            if self.data.get('asrModel') and self.data['asrModel'] != settings.asr_model:
                raise MeetingError('meeting_model_mismatch')
            self.data.update(asrBackend='whisper', asrModel=settings.asr_model)
            self.data.setdefault('hqWindowSamples', settings.asr_high_accuracy_window_seconds * SAMPLE_RATE)
            self.data.setdefault('rtWindowSamples', settings.asr_realtime_window_seconds * SAMPLE_RATE)
            self.data.setdefault('highAccuracyEnabled', settings.asr_high_accuracy_enabled)
            upgrading = not self.data.get('whisperDualLane')
            for track, state in self.data['tracks'].items():
                state.setdefault('hqCommitted', 0)
                for row in self.records:
                    if row.get('type') != 'final' or row.get('track'):
                        continue
                    if str(row.get('segmentId', '')).startswith(track + '-'):
                        start = row['chunkOffsetMs'] * 16
                        end = start + row['chunkDurationMs'] * 16
                        row.update(track=track, startSample=start, endSample=end, quality='realtime', revision=0)
                        # Old windows may cross the new HQ boundaries. Preserve
                        # these records; automatic HQ starts at their final end.
                        if upgrading:
                            state['hqCommitted'] = max(state['hqCommitted'], end)
                own = [r for r in self.records if r.get('type') == 'final' and r.get('track') == track]
                state['windowStart'] = max([state['windowStart']] + [r['endSample'] for r in own])
                state['lastDecode'] = max(state['lastDecode'], state['windowStart'])
                state['hqCommitted'] = max([state['hqCommitted']] + [r['endSample'] for r in own if r.get('quality') == 'high_accuracy'])
            if upgrading and self.records:
                self.store.rewrite_records(self.records)
            self.data['whisperDualLane'] = True
            _atomic_text(self.store.txt_path, ''.join(_render_txt_line(r) + '\n' for r in self.records if r.get('type') == 'final'))
            write_json_atomic(self.state_path, self.data)
            metadata = self.store.read_metadata()
            metadata.update(asrBackend='whisper', asrModel=settings.asr_model, highAccuracyEnabled=self.data['highAccuracyEnabled'])
            self.store.write_metadata(metadata)
            if self.data['highAccuracyEnabled']:
                # Separate clients prevent RT/HQ response-format and timeout
                # state from racing. Both use the same configured Whisper model.
                self.hq_transcriber = OpenAIWhisperTranscriber(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.asr_model)
                self.hq_transcriber.multi_pass_enabled = False
                self.hq_transcriber.retry_max_attempts = 1
                self.hq_transcriber.client = self.hq_transcriber.client.with_options(timeout=settings.asr_high_accuracy_timeout_seconds, max_retries=0)
        except BaseException:
            self.close()
            raise
        self.rt_done = False
        self.rt_failed = False
        # Mic/display share the RT adapter, which remembers the provider's
        # response-format fallback. Keep those requests serialized; HQ has
        # its own adapter and can run concurrently.
        self.rt_decode_lock = asyncio.Lock()

    def close(self):
        if self.hq_transcriber is not None:
            self.hq_transcriber.client.close()
        super().close()

    def _window_end(self, start):
        size = self.data['hqWindowSamples']
        return min(start + self.data['rtWindowSamples'], (start // size + 1) * size)

    def _final_record(self, record, track, start, end):
        row = super()._final_record(record, track, start, end)
        row.update(track=track, startSample=start, endSample=end, quality='realtime', revision=0)
        return row

    def _silence_samples(self):
        return settings.asr_high_accuracy_silence_ms * SAMPLE_RATE // 1000

    def _note_pause(self, track, end, quiet_samples):
        if quiet_samples >= self._silence_samples():
            self.data['tracks'][track]['hqPauseEnd'] = end

    def _rt_busy(self):
        threshold = self.lane_settings.asr_high_accuracy_max_rt_lag_seconds * SAMPLE_RATE
        return any(s['received'] - s['lastDecode'] > UPDATE_SAMPLES + threshold
                   for s in self.data['tracks'].values())

    async def _batch_text(self, client, pcm):
        await self._request_budget()
        result = await blocking_work_pool.run('asr', self.hq_transcriber.transcribe_chunk, pcm_wav(pcm),
            mime_type='audio/wav', language=self.data['language'] or None,
            prompt=(self.data['vocabulary'] + ' ' + self.data['prompt']).strip() or None, temperature=0.0)
        return _sanitize_transcript_text(result.text, language=self.data['language']).strip()

    async def _rt_lane(self, track):
        failures = 0
        while not self.disconnected:
            if self.stopping and self.data['tracks'][track]['received'] <= self.data['tracks'][track]['windowStart']:
                return
            try:
                async with self.rt_decode_lock:
                    inferred = await self._infer(track, flush=self.stopping)
                if inferred:
                    failures = 0
                else:
                    await asyncio.sleep(.025)
            except Exception as exc:
                failures += 1
                await self._recover_commits()
                logger.warning('Whisper realtime failed: session=%s error=%s', self.session_id, type(exc).__name__)
                await self.send(dict(type='error', message='transcription_failed', buffered=True,
                    **public_error('transcription_failed', exc, logger),
                    detail='音声は保存されています。Whisperの接続と停止処理を再試行してください。'))
                if self.stopping:
                    self.rt_failed = True
                    return
                await asyncio.sleep(min(10, failures * 2))
