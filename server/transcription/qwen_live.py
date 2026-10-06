"""Two remote inference lanes sharing one vLLM model and durable PCM journal."""
from __future__ import annotations

import asyncio
import copy

import httpx

from ..core.blocking import blocking_work_pool
from ..core.config import settings
from ..core.rate_limit import consume
from ..qwen_asr import QwenRealtime, batch_request, clean_qwen_text, response_text
from ..core.public_errors import public_error
from ..services.meeting_source import MeetingError, write_json_atomic
from ..services.meeting_refinement import _atomic_text
from ..transcript_store import _render_txt_line
from .high_accuracy import HighAccuracyMeetingMixin
from .live import LiveMeeting, logger
from .local_agreement import SAMPLE_RATE, pcm_wav, speech_bounds


class QwenLiveMeeting(HighAccuracyMeetingMixin, LiveMeeting):
    hq_lock_name = '.qwen-hq.lock'

    @property
    def lane_settings(self):
        return settings

    def create_transcriber(self):
        return None

    def __init__(self, ws, payload, *, resources=None):
        super().__init__(ws, payload, resources=resources)
        try:
            backend = self.data.get('asrBackend')
            if payload.get('resumeSessionId') and backend != 'qwen3_vllm':
                raise MeetingError('meeting_backend_mismatch')
            if backend and self.data.get('asrModel') != settings.asr_model:
                raise MeetingError('meeting_model_mismatch')
            if not payload.get('resumeSessionId'):
                self.data['language'] = payload.get('language') if payload.get('language') in {'ja', 'en'} else 'auto'
            self.data.update(asrBackend='qwen3_vllm', asrModel=settings.asr_model)
            self.data.setdefault('hqWindowSamples', settings.asr_high_accuracy_window_seconds * SAMPLE_RATE)
            self.data.setdefault('rtWindowSamples', settings.asr_realtime_window_seconds * SAMPLE_RATE)
            self.data.setdefault('highAccuracyEnabled', settings.asr_high_accuracy_enabled)
            for track, state in self.data['tracks'].items():
                state.setdefault('hqCommitted', 0)
                # Recover the append-before-checkpoint crash gap without re-emitting RT.
                rows = [r for r in self.records if r.get('track') == track and r.get('type') == 'final']
                state['windowStart'] = max([state['windowStart']] + [r['endSample'] for r in rows])
                state['lastDecode'] = state['windowStart']
                state['hqCommitted'] = max([state['hqCommitted']] + [r['endSample'] for r in rows if r.get('quality') == 'high_accuracy'])
            _atomic_text(self.store.txt_path, ''.join(_render_txt_line(r) + '\n' for r in self.records if r.get('type') == 'final'))
            write_json_atomic(self.state_path, self.data)
            metadata = self.store.read_metadata()
            metadata.update(asrBackend='qwen3_vllm', asrModel=settings.asr_model, language=self.data['language'])
            self.store.write_metadata(metadata)
        except BaseException:
            self.close()
            raise
        self.rt_done = False
        self.rt_failed = False
        self.pending_rt = set()

    async def _request_budget(self):
        async with self.state_lock:
            if self.is_guest and self.data['asrRequests'] >= settings.guest_ws_max_asr_requests:
                raise MeetingError('guest_asr_request_limit', 429)
            allowed = await asyncio.to_thread(consume, bucket='asr', subject=self.rate_subject,
                limit=settings.costly_api_rate_limit_requests, window_seconds=settings.costly_api_rate_limit_window_seconds)
            if not allowed:
                raise MeetingError('rate_limit_exceeded', 429)
            self.data['asrRequests'] += 1
            await blocking_work_pool.run('artifact', write_json_atomic, self.state_path, copy.deepcopy(self.data))

    async def _rt_window(self, track):
        state = self.data['tracks'][track]
        start = state['windowStart']
        hq_size = self.data['hqWindowSamples']
        limit = min(start + self.data['rtWindowSamples'], (start // hq_size + 1) * hq_size)
        sent = start
        text = ''
        await self._request_budget()
        self.pending_rt.add(track)
        try:
            if self.data['language'] in {'ja', 'en'}:
                # vLLM Realtime has no language parameter. Use the same model's
                # transcription endpoint for each short capture window.
                while state['received'] < limit and not self.stopping and not self.disconnected:
                    await asyncio.sleep(0.025)
                if self.disconnected:
                    return
                sent = min(state['received'], limit)
                pcm = await blocking_work_pool.run('artifact', self._read_audio, track, start, sent)
                if await blocking_work_pool.run('media', speech_bounds, pcm) is not None:
                    async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + settings.openai_api_key},
                                                 timeout=settings.asr_api_timeout_seconds) as client:
                        endpoint, options = batch_request(pcm_wav(pcm), mime_type='audio/wav', model=settings.asr_model,
                            context=(self.data['vocabulary'] + ' ' + self.data['prompt']).strip(), priority=0, language=self.data['language'])
                        response = await client.post(str(settings.openai_base_url).rstrip('/') + endpoint, **options)
                        response.raise_for_status()
                        text = response_text(response.json())
            else:
                async with QwenRealtime(base_url=settings.openai_base_url, api_key=settings.openai_api_key,
                                        model=settings.asr_model, timeout=settings.asr_api_timeout_seconds) as stream:
                    async def receive():
                        nonlocal text
                        while not stream.done:
                            event = await stream.receive()
                            if event.get('type') == 'transcription.delta':
                                text += event.get('delta', '')
                                await self.send(dict(type='partial', track=track, segmentId=f'{track}-rt-{start}',
                                    text=clean_qwen_text(text, final=False), stableText='', quality='realtime',
                                    tsStart=start * 1000 // SAMPLE_RATE, tsEnd=sent * 1000 // SAMPLE_RATE))
                            elif event.get('type') == 'transcription.done':
                                text = event.get('text', text)
                    receiver = asyncio.create_task(receive())
                    try:
                        while sent < limit and not self.disconnected:
                            if receiver.done():
                                await receiver
                                raise RuntimeError('qwen_realtime_ended_early')
                            end = min(state['received'], limit)
                            if end > sent:
                                pcm = await blocking_work_pool.run('artifact', self._read_audio, track, sent, end)
                                await stream.append(pcm)
                                sent = end
                            elif self.stopping:
                                break
                            else:
                                await asyncio.sleep(0.025)
                        if self.disconnected:
                            return
                        await stream.finish()
                        await receiver
                    finally:
                        if not receiver.done():
                            receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)
            pcm = await blocking_work_pool.run('artifact', self._read_audio, track, start, sent)
            text = clean_qwen_text(text)
            silence_samples = settings.asr_high_accuracy_silence_ms * SAMPLE_RATE // 1000
            quiet_tail = len(pcm) >= silence_samples * 2 and await blocking_work_pool.run(
                'media', speech_bounds, pcm[-silence_samples * 2:]) is None
            if await blocking_work_pool.run('media', speech_bounds, pcm) is None:
                text = ''
            async with self.state_lock:
                record = None
                if text:
                    record = self._record(track, start, sent, text, quality='realtime', seq=self.next_seq)
                    await self._save_clip(record, pcm)
                    await blocking_work_pool.run('artifact', self.store.append_record, record)
                    self.records.append(record)
                    self.next_seq += 1
                if quiet_tail:
                    state['hqPauseEnd'] = sent
                state.update(windowStart=sent, lastDecode=sent)
                await blocking_work_pool.run('artifact', write_json_atomic, self.state_path, copy.deepcopy(self.data))
                if record:
                    await self.send(record)
            await self.send(dict(type='partial', track=track, text='', stableText=''))
        finally:
            self.pending_rt.discard(track)

    async def _rt_lane(self, track):
        failures = 0
        while not self.disconnected:
            state = self.data['tracks'][track]
            if state['received'] <= state['windowStart']:
                if self.stopping:
                    return
                await asyncio.sleep(0.05)
                continue
            try:
                await self._rt_window(track)
                failures = 0
            except Exception as exc:
                failures += 1
                await self._recover_commits()
                logger.warning('Qwen RT failed: session=%s error=%s', self.session_id, type(exc).__name__)
                await self.send(dict(type='error', message='transcription_failed', buffered=True,
                    **public_error('transcription_failed', exc, logger),
                    detail='音声は保存されています。vLLMの音声対応・Realtime設定と接続を確認してください。'))
                if self.stopping:
                    self.rt_failed = True
                    return
                await asyncio.sleep(min(10, failures * 2))

    async def _batch_text(self, client, pcm):
        await self._request_budget()
        endpoint, options = batch_request(pcm_wav(pcm), mime_type='audio/wav', model=settings.asr_model,
            context=(self.data['vocabulary'] + ' ' + self.data['prompt']).strip(), priority=settings.asr_high_accuracy_priority, language=self.data['language'])
        response = await client.post(str(settings.openai_base_url).rstrip('/') + endpoint, **options)
        response.raise_for_status()
        return response_text(response.json(), allow_empty=True)
