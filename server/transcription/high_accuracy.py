"""Shared sample-anchored high-accuracy scheduling and durable revisions."""
from __future__ import annotations

import asyncio
import copy
import fcntl
import logging
from datetime import datetime, timezone

import httpx

from ..core.blocking import blocking_work_pool
from ..core.public_errors import public_error
from ..qwen_asr import language_coverage_lost, script_groups
from ..services.meeting_refinement import _atomic_text
from ..services.meeting_source import write_json_atomic
from ..transcript_store import _render_txt_line, read_jsonl_records
from .local_agreement import SAMPLE_RATE, pcm_wav

logger = logging.getLogger(__name__)


class HighAccuracyMeetingMixin:
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

    def _next_hq(self):
        size = self.data['hqWindowSamples']
        for track, state in self.data['tracks'].items():
            start = state['hqCommitted']
            boundary = (start // size + 1) * size
            end = min(boundary, state['windowStart'])
            # Only revise complete RT windows: a pause must never split a record.
            pause = min(state.get('hqPauseEnd', 0), end)
            if pause - start >= self.lane_settings.asr_high_accuracy_min_seconds * SAMPLE_RATE:
                end = pause
                return track, start, end
            if end > start and (end == boundary or self.rt_done):
                return track, start, end
        return None

    def _rt_busy(self):
        # An outstanding completed window is inference lag; a partially filled
        # active window is normal capture and must not starve the HQ lane.
        threshold = self.lane_settings.asr_high_accuracy_max_rt_lag_seconds * SAMPLE_RATE
        return any(s['received'] - s['windowStart'] > self.data['rtWindowSamples'] + threshold
                   for s in self.data['tracks'].values())

    async def _retain_hq(self, track, start, end, *, reason, message):
        self.data['tracks'][track]['hqCommitted'] = end
        self.data.setdefault('hqRetainedIntervals', []).append(
            dict(track=track, startSample=start, endSample=end, reason=reason))
        await self.checkpoint()
        await self.send(dict(type='hq_status', state='retained', track=track,
                             tsStart=start//16, tsEnd=end//16, message=message))

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
        if not text.strip():
            await self._retain_hq(track, start, end, reason='empty_result',
                message='高精度認識が空の区間は速報を保持し、次の区間へ進みました。')
            return
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
                    if not candidate.strip() or language_coverage_lost(source, candidate):
                        candidate = source
                        retained.extend(r['segmentId'] for r in group)
                    pieces.append(candidate)
                    fallback.append(dict(startSample=left, endSample=right, text=candidate))
                    left = right
            text = '\n'.join(pieces)
        if len(retained) == len(targets):
            # A result that demonstrably drops a language cannot supersede RT.
            await self._retain_hq(track, start, end, reason='language_coverage',
                message='言語の脱落を検出した区間は、速報を維持しました。')
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
        lock_path = self.lane_settings.transcripts_dir / self.hq_lock_name
        failures = 0
        async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + self.lane_settings.openai_api_key},
                                     timeout=self.lane_settings.asr_high_accuracy_timeout_seconds) as client:
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
                    logger.warning('ASR HQ failed: session=%s error=%s', self.session_id, type(exc).__name__)
                    if failures >= 3 and getattr(self, 'retain_failed_hq', False):
                        await self._retain_hq(*job, reason='recognition_failed',
                            message='高精度再認識に失敗した区間は速報と音声を保持し、次の区間へ進みました。')
                        failures = 0
                        continue
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
                raise RuntimeError('asr_rt_incomplete')
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
            logger.warning('ASR finalization failed: session=%s error=%s', self.session_id, type(exc).__name__)
            await self.send(dict(type='error', message='finalize_failed', buffered=True,
                **public_error('finalize_failed', exc, logger),
                detail='認識が未完了です。音声は保存されています。接続を確認して停止処理を再試行してください。'))
        finally:
            for task in [*rt, hq]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*rt, hq, return_exceptions=True)
