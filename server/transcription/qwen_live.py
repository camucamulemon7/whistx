"""Two remote inference lanes sharing one vLLM model and durable PCM journal."""
from __future__ import annotations

import asyncio
import copy
import fcntl
from datetime import datetime, timezone

import httpx

from ..core.blocking import blocking_work_pool
from ..core.config import settings
from ..core.rate_limit import consume
from ..qwen_asr import QwenRealtime, batch_request, clean_qwen_text, response_text, language_coverage_lost, script_groups
from ..services.meeting_source import MeetingError, write_json_atomic
from ..services.meeting_refinement import _atomic_text
from ..transcript_store import _render_txt_line, read_jsonl_records
from .live import LiveMeeting, logger
from .local_agreement import SAMPLE_RATE, pcm_wav, speech_bounds


class QwenLiveMeeting(LiveMeeting):
    def create_transcriber(self):
        return None

    def __init__(self, ws, payload):
        super().__init__(ws, payload)
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

    async def _recover_commits(self):
        # A durable journal append can succeed even if TXT/checkpoint writing
        # fails. Reload it before retrying so the same interval is not appended
        # again with a different revision body.
        async with self.state_lock:
            rows = await blocking_work_pool.run('artifact', read_jsonl_records, self.store.jsonl_path)
            changed = rows != self.records
            self.records = rows
            self.next_seq = max(self.next_seq, max((r.get('seq', -1) for r in rows if r.get('type') == 'final'), default=-1) + 1)
            for track, state in self.data['tracks'].items():
                own = [r for r in rows if r.get('type') == 'final' and r.get('track') == track]
                state['windowStart'] = max([state['windowStart']] + [r['endSample'] for r in own])
                state['lastDecode'] = state['windowStart']
                state['hqCommitted'] = max([state['hqCommitted']] + [r['endSample'] for r in own if r.get('quality') == 'high_accuracy'])
            await blocking_work_pool.run('artifact', _atomic_text, self.store.txt_path,
                ''.join(_render_txt_line(r) + '\n' for r in rows if r.get('type') == 'final'))
            await blocking_work_pool.run('artifact', write_json_atomic, self.state_path, copy.deepcopy(self.data))
        if changed:
            await self.send(dict(type='transcript_snapshot', records=[r for r in rows if r.get('type') == 'final']))

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

    def _record(self, track, start, end, text, *, quality, seq):
        stamp = datetime.now(timezone.utc).isoformat()
        return dict(type='final', segmentId=f'{track}-{quality}-{start}-{end}', seq=seq, text=text,
            track=track, startSample=start, endSample=end, quality=quality, revision=int(quality == 'high_accuracy'),
            tsStart=start * 1000 // SAMPLE_RATE, tsEnd=end * 1000 // SAMPLE_RATE,
            chunkOffsetMs=start * 1000 // SAMPLE_RATE, chunkDurationMs=(end-start) * 1000 // SAMPLE_RATE,
            language=self.data['language'], createdAt=stamp,
            speaker='自分（マイク）' if track == 'mic' and len(self.data['tracks']) > 1 else '共有音声' if track == 'display' else None)

    async def _save_clip(self, record, pcm):
        name = f"{record['segmentId']}.wav"
        await blocking_work_pool.run('artifact', (self.audio_dir / name).write_bytes, pcm_wav(pcm))
        record['rawAudioPath'] = f'/api/transcripts/{self.session_id}/audio/{name}'
        images = [r for r in self.records if r.get('type') == 'screen' and r.get('tsStart', 0) <= record['tsEnd']]
        record['screenshotPath'] = images[-1].get('screenshotPath') if images else None

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
                    detail='音声は保存されています。vLLMの音声対応・Realtime設定と接続を確認してください。'))
                if self.stopping:
                    self.rt_failed = True
                    return
                await asyncio.sleep(min(10, failures * 2))

    def _next_hq(self):
        size = self.data['hqWindowSamples']
        for track, state in self.data['tracks'].items():
            start = state['hqCommitted']
            end = min(start + size, state['windowStart'])
            if end > start and (end - start >= size or self.rt_done):
                return track, start, end
        return None

    def _rt_busy(self):
        # An outstanding completed window is inference lag; a partially filled
        # active window is normal capture and must not starve the HQ lane.
        threshold = settings.asr_high_accuracy_max_rt_lag_seconds * SAMPLE_RATE
        return any(s['received'] - s['windowStart'] > self.data['rtWindowSamples'] + threshold
                   for s in self.data['tracks'].values())

    async def _batch_text(self, client, pcm):
        await self._request_budget()
        endpoint, options = batch_request(pcm_wav(pcm), mime_type='audio/wav', model=settings.asr_model,
            context=(self.data['vocabulary'] + ' ' + self.data['prompt']).strip(), priority=settings.asr_high_accuracy_priority, language=self.data['language'])
        response = await client.post(str(settings.openai_base_url).rstrip('/') + endpoint, **options)
        response.raise_for_status()
        return response_text(response.json())

    async def _hq_interval(self, client, track, start, end):
        targets = [r for r in self.records if r.get('type') == 'final' and r.get('track') == track
                   and r['startSample'] >= start and r['endSample'] <= end and r.get('quality') == 'realtime']
        if not targets:
            self.data['tracks'][track]['hqCommitted'] = end
            await self.checkpoint()
            return
        await self.send(dict(type='hq_status', state='running', track=track, tsStart=start//16, tsEnd=end//16))
        pcm = await blocking_work_pool.run('artifact', self._read_audio, track, start, end)
        text = await self._batch_text(client, pcm)
        original = ' '.join(r['text'] for r in targets)
        retained = []
        fallback = []
        if language_coverage_lost(original, text):
            # Qwen may choose the majority language and omit/translate the
            # minority. Re-read contiguous script groups with the SAME model;
            # never splice untranslated source text into an unanchored result.
            groups = script_groups(targets)
            if len(groups) == 1 or len(groups) > 8:
                groups = [targets]
                pieces = [original]
                retained = [r['segmentId'] for r in targets]
            else:
                await self.send(dict(type='hq_status', state='running', track=track, tsStart=start//16, tsEnd=end//16,
                    message='日英の脱落を検出したため、同じモデルで区間を分けて再認識しています。'))
                pieces = []
                left = start
                for index, group in enumerate(groups):
                    right = groups[index + 1][0]['startSample'] if index + 1 < len(groups) else end
                    while self._rt_busy() and not self.rt_done and not self.disconnected:
                        await asyncio.sleep(.1)
                    if self.disconnected:
                        return
                    group_pcm = pcm[(left-start)*2:(right-start)*2]
                    candidate = await self._batch_text(client, group_pcm)
                    source = ' '.join(r['text'] for r in group)
                    if language_coverage_lost(source, candidate):
                        candidate = source
                        retained.extend(r['segmentId'] for r in group)
                    pieces.append(candidate)
                    fallback.append(dict(startSample=left, endSample=right, text=candidate))
                    left = right
            text = '\n'.join(pieces)
        if len(retained) == len(targets):
            # A result that demonstrably drops a language cannot supersede RT.
            self.data['tracks'][track]['hqCommitted'] = end
            self.data.setdefault('hqRetainedIntervals', []).append(dict(track=track, startSample=start, endSample=end))
            await self.checkpoint()
            await self.send(dict(type='hq_status', state='retained', track=track, tsStart=start//16, tsEnd=end//16,
                message='言語の脱落を検出した区間は、速報を維持しました。'))
            return
        record = self._record(track, start, end, text, quality='high_accuracy', seq=min(r['seq'] for r in targets))
        record.update(originalText=original, realtimeSegments=targets)
        if fallback:
            record['highAccuracySegments'] = fallback
        if retained:
            record['retainedRealtimeSegmentIds'] = retained
        await self._save_clip(record, pcm)
        event = dict(type='revision', track=track, startSample=start, endSample=end,
                     replacesSegmentIds=[r['segmentId'] for r in targets], record=record)
        async with self.state_lock:
            self.records = await blocking_work_pool.run('artifact', self.store.append_revision, event, self.records)
            self.data['tracks'][track]['hqCommitted'] = end
            await blocking_work_pool.run('artifact', write_json_atomic, self.state_path, copy.deepcopy(self.data))
            await self.send({**event, 'type': 'transcript_revision'})
        await self.send(dict(type='hq_status', state='completed', track=track, tsStart=start//16, tsEnd=end//16))

    async def _hq_lane(self):
        if not self.data['highAccuracyEnabled']:
            return
        # File lock bounds HQ concurrency across meetings and app workers.
        lock_path = settings.transcripts_dir / '.qwen-hq.lock'
        failures = 0
        async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + settings.openai_api_key},
                                     timeout=settings.asr_high_accuracy_timeout_seconds) as client:
            while not self.disconnected:
                job = self._next_hq()
                if self.rt_failed:
                    return
                if not job:
                    if self.rt_done:
                        return
                    await asyncio.sleep(0.1)
                    continue
                if self._rt_busy() and not self.rt_done:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    with lock_path.open('a') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            await asyncio.sleep(0.1)
                            continue
                        await self._hq_interval(client, *job)
                    failures = 0
                except Exception as exc:
                    failures += 1
                    await self._recover_commits()
                    logger.warning('Qwen HQ failed: session=%s error=%s', self.session_id, type(exc).__name__)
                    await self.send(dict(type='hq_status', state='error', message='再認識を再試行します。速報と音声は保存されています。'))
                    if self.rt_done and failures >= 3:
                        raise
                    await asyncio.sleep(min(10, failures * 2))

    async def work(self):
        rt = [asyncio.create_task(self._rt_lane(track)) for track in self.data['tracks']]
        hq = asyncio.create_task(self._hq_lane())
        try:
            await asyncio.gather(*rt)
            self.rt_done = True
            await hq
            if self.disconnected:
                return
            if self.rt_failed:
                raise RuntimeError('qwen_rt_incomplete')
            # Preserve the revision journal before legacy speaker materialization.
            journal = await blocking_work_pool.run('artifact', self.store.jsonl_path.read_text, encoding='utf-8')
            await blocking_work_pool.run('artifact', _atomic_text, self.store.jsonl_path.with_suffix('.revisions.jsonl'), journal)
            await self.diarize()
            self.data['finalized'] = True
            await self.checkpoint()
            metadata = self.store.read_metadata()
            metadata.update(finalized=True, finalizedAt=datetime.now(timezone.utc).isoformat())
            await blocking_work_pool.run('artifact', write_json_atomic, self.store.metadata_path, metadata)
            await self.send(dict(type='info', message='finalized', state='completed'))
        except Exception as exc:
            logger.warning('Qwen finalization failed: session=%s error=%s', self.session_id, type(exc).__name__)
            await self.send(dict(type='error', message='finalize_failed', buffered=True,
                detail='認識が未完了です。音声は保存されています。接続を確認して停止処理を再試行してください。'))
        finally:
            for task in [*rt, hq]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*rt, hq, return_exceptions=True)
