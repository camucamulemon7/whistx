"""Bounded local-media decoding and cancellable, sample-anchored ASR import."""
from __future__ import annotations

import shutil
import concurrent.futures
import fcntl
import os
import subprocess
import tempfile
import time
import uuid
import wave
import httpx
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import settings
from ..langfuse_observer import high_accuracy_observation
from ..pcm_gate import pcm_speech_bounds
from ..transcript_store import TranscriptStore, build_debug_chunks_dir
from ..transcription.local_agreement import SAMPLE_RATE, pcm_wav
from ..transcription.text_processing import _sanitize_transcript_text
from .meeting_source import MeetingError
from .media_segments import file_policy, iter_file_segments, split_limited_segment, trim_overlap_text

MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_DURATION_SECONDS = 2 * 60 * 60
# Force a local demuxer: playlists/manifests must never fetch external media.
MEDIA_FORMATS = {
    '.wav': 'wav', '.mp3': 'mp3', '.m4a': 'mov', '.mp4': 'mov', '.mov': 'mov',
    '.webm': 'matroska', '.mkv': 'matroska', '.ogg': 'ogg', '.opus': 'ogg',
    '.flac': 'flac', '.aac': 'aac', '.aiff': 'aiff', '.aif': 'aiff',
}


def media_format(filename: str) -> str:
    value = MEDIA_FORMATS.get(Path(filename).suffix.lower())
    if value is None:
        raise MeetingError('media_unsupported_format', 415)
    return value


def decode_media(source: Path, destination: Path, *, filename: str, cancelled, ffmpeg_bin=None, progress=None):
    if cancelled.is_set():
        raise MeetingError('media_cancelled', 409)
    executable = ffmpeg_bin or settings.ffmpeg_bin
    if shutil.which(executable) is None:
        raise MeetingError('media_decoder_unavailable', 503)
    command = [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-xerror',
               '-progress', 'pipe:1', '-stats_period', '0.25',
               '-protocol_whitelist', 'file,pipe', '-threads', '1', '-f', media_format(filename), '-i', str(source),
               '-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', '1', '-ar', str(SAMPLE_RATE),
               '-t', str(MAX_DURATION_SECONDS + 1), '-c:a', 'pcm_s16le', '-threads', '1', str(destination)]
    # Discard decoder logs: embedded metadata/file paths are not public errors.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    buffer = b''
    deadline = time.monotonic() + max(120, settings.ffmpeg_timeout_seconds)
    try:
        while process.poll() is None:
            try:
                buffer += os.read(process.stdout.fileno(), 4096)
            except BlockingIOError:
                pass
            lines = buffer.split(b'\n')
            buffer = lines.pop()
            if progress:
                for line in lines:
                    if line.startswith(b'out_time_us='):
                        try:
                            progress['convertedMs'] = max(0, int(line.partition(b'=')[2]) // 1000)
                        except ValueError:
                            pass
            if cancelled.wait(.1):
                raise MeetingError('media_cancelled', 409)
            if time.monotonic() >= deadline:
                raise MeetingError('media_decode_timeout', 422)
        if process.returncode:
            raise MeetingError('media_decode_failed', 422)
        try:
            with wave.open(str(destination)) as audio:
                if not audio.getnframes():
                    raise MeetingError('media_no_audio', 422)
                if audio.getnframes() > MAX_DURATION_SECONDS * SAMPLE_RATE:
                    raise MeetingError('media_too_long', 413)
        except (wave.Error, EOFError) as exc:
            raise MeetingError('media_decode_failed', 422) from exc
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def _step(executor, operation, *, cancelled, state, status):
    """Poll a single bounded operation without dropping its running worker."""
    began = time.monotonic()
    future = executor.submit(operation)
    last_phase, last_report = None, 0.0
    while not future.done():
        if cancelled.is_set():
            return None
        now = time.monotonic()
        phase = state.get('phase')
        if phase != last_phase or now - last_report >= 1:
            yield status(elapsedSeconds=round(now - began, 1), **state.copy())
            last_phase, last_report = phase, now
        try:
            future.result(timeout=.1)
        except concurrent.futures.TimeoutError:
            pass
    return future.result()


def _recognize(model, segment, *, cancelled, state, run_inference, bounds):
    # One file request globally across app workers. Decode/quota waiting never
    # occupy the shared ASR worker pool; recording remains an independent lane.
    lock_path = settings.transcripts_dir / '.file-import-asr.lock'
    with lock_path.open('a+b') as lock:
        deadline = time.monotonic() + settings.blocking_worker_queue_timeout_seconds
        state.update(phase='waiting', waitReason='file_slot')
        while True:
            if cancelled.is_set():
                raise MeetingError('media_cancelled', 409)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise MeetingError('meeting_model_busy', 503)
                cancelled.wait(.1)
        try:
            def call():
                if cancelled.is_set():
                    raise MeetingError('media_cancelled', 409)
                state.update(phase='recognizing', waitReason=None)
                return model.transcribe_chunk(pcm_wav(segment.pcm), mime_type='audio/wav',
                    language=state['language'] or None, prompt=state['prompt'] or None, temperature=0.0,
                    **({'pcm_bounds': bounds} if settings.asr_backend == 'speech_cpp' else {}))
            return run_inference(call, state) if run_inference else call()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def import_events(source: Path, *, filename: str, language: str, prompt: str, user_id: int,
                  cancelled, transcriber_factory, allow_request, observer=None, decoder=decode_media,
                  run_inference=None, policy=None):
    """Preview completed intervals, but commit only the completed owned runtime."""
    store = None
    audio_dir = None
    model = None
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix='whistx-decode-') as directory, \
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            decoded = Path(directory) / 'audio.wav'
            yield dict(type='status', phase='decoding', message='音声を取り出しています', progress=0)
            state = dict(phase='decoding')
            def decoding_status(**values):
                converted = values.get('convertedMs', 0) // 1000
                return dict(type='status', message=f'音声を取り出しています · {converted}秒を変換 · {values["elapsedSeconds"]}秒経過', **values)
            yield from _step(executor, lambda: decoder(source, decoded, filename=filename,
                cancelled=cancelled, **({'progress': state} if decoder is decode_media else {})),
                cancelled=cancelled, state=state, status=decoding_status)
            if cancelled.is_set():
                return
            session_id = 'file-' + uuid.uuid4().hex
            store = TranscriptStore(settings.transcripts_dir, session_id)
            audio_dir = build_debug_chunks_dir(settings.debug_chunks_dir, session_id)
            audio_dir.mkdir(parents=True, exist_ok=False)
            model = transcriber_factory()
            client = getattr(model, 'client', None)
            if client is not None:
                client.timeout = settings.media_asr_timeout_seconds
            if hasattr(model, 'multi_pass_enabled'):
                model.multi_pass_enabled = False
            records = []
            policy = policy or file_policy(settings.asr_backend, language=language)
            with wave.open(str(decoded)) as audio:
                total = audio.getnframes()
                duration_ms = total * 1000 // SAMPLE_RATE
                metadata = dict(sessionId=session_id, ownerUserId=user_id, language=language,
                    audioSource='file', sourceName=Path(filename).name[:200], mode='file',
                    asrBackend=settings.asr_backend, asrModel=settings.asr_model,
                    finalized=False, durationMs=duration_ms, prompt=prompt,
                    recognitionMode='file_import', fileTargetSeconds=policy.target_seconds,
                    fileMaxSeconds=policy.max_seconds, fileConcurrency=1,
                    fileStrongPauseMs=policy.strong_pause_ms,
                    createdAt=datetime.now(timezone.utc).isoformat())
                store.write_metadata(metadata)
                through = 0
                previous = ''
                def recognition_status(**values):
                    reason = values.get('waitReason')
                    label = {'rate_limit': '利用上限の回復を待っています', 'file_slot': '別のファイル認識の完了を待っています',
                             'asr_worker': '認識処理の空きを待っています'}.get(reason, 'ファイルを認識しています')
                    percentage = round(through * 100 / total)
                    # Prompts/language are internal request context, not status.
                    public = {key: value for key, value in values.items() if key not in ('language', 'prompt')}
                    return dict(type='status', progress=percentage, throughMs=through * 1000 // SAMPLE_RATE,
                        durationMs=duration_ms, completedSegments=len(records),
                        message=f'{label} · {percentage}% · {values.get("elapsedSeconds", 0)}秒経過', **public)
                segments = iter_file_segments(audio, policy)
                pending = []
                while True:
                    try:
                        segment, depth = pending.pop() if pending else (next(segments), 0)
                    except StopIteration:
                        break
                    if cancelled.is_set():
                        return
                    bounds = pcm_speech_bounds(segment.pcm)
                    if bounds is not None:
                        waiting = False
                        waited_at = time.monotonic()
                        while not allow_request():
                            yield recognition_status(phase='waiting', waitReason='rate_limit',
                                elapsedSeconds=round(time.monotonic() - waited_at, 1))
                            waiting = True
                            if cancelled.wait(1):
                                return
                        if waiting:
                            yield recognition_status(phase='recognizing', waitReason=None)
                        state = dict(phase='recognizing', language=language, prompt=prompt)
                        with high_accuracy_observation(observer, session_id=session_id, model=settings.asr_model,
                                start_ms=segment.start * 1000 // SAMPLE_RATE, end_ms=segment.end * 1000 // SAMPLE_RATE,
                                audio_bytes=len(segment.pcm), language=language, track='file', kind='file_import') as observation:
                            try:
                                result = yield from _step(executor, lambda: _recognize(model, segment,
                                    cancelled=cancelled, state=state, run_inference=run_inference, bounds=bounds),
                                    cancelled=cancelled, state=state, status=recognition_status)
                            except ValueError as exc:
                                if str(exc) != 'speech_cpp_batch_model_limit':
                                    raise
                                if depth >= 3 or segment.end - segment.start <= 10 * SAMPLE_RATE:
                                    raise MeetingError('media_model_limit', 422) from exc
                                left, right = split_limited_segment(segment)
                                observation.update(output=dict(succeeded=False, stop='model_limit', retrying=True))
                                pending.extend([(right, depth + 1), (left, depth + 1)])
                                yield dict(type='status', phase='splitting', progress=round(through * 100 / total),
                                    message='モデルの出力上限に達したため、区間を短くして認識し直しています')
                                continue
                            except (TimeoutError, httpx.TimeoutException) as exc:
                                code = 'meeting_model_busy' if state.get('phase') == 'waiting' else 'media_recognition_timeout'
                                raise MeetingError(code, 504) from exc
                            if cancelled.is_set():
                                observation.update(output=dict(succeeded=False, cancelled=True))
                                return
                            text = _sanitize_transcript_text(result.text, language=language).strip()
                            boundary_original = text
                            text = trim_overlap_text(previous, text, overlapping=bool(segment.overlap_samples))
                            observation.update(output=dict(text=text, chars=len(text), succeeded=True))
                        if cancelled.is_set():
                            return
                        if text:
                            seq = len(records)
                            name = f'file-{seq:06d}.wav'
                            (audio_dir / name).write_bytes(pcm_wav(segment.pcm))
                            # Interval timing stays anchored to the imported media.
                            row = dict(type='final', segmentId=f'file-{segment.start}-{segment.end}', seq=seq, text=text,
                                track='file', startSample=segment.start, endSample=segment.end, quality='file_import',
                                tsStart=segment.start * 1000 // SAMPLE_RATE, tsEnd=segment.end * 1000 // SAMPLE_RATE,
                                chunkOffsetMs=segment.start * 1000 // SAMPLE_RATE,
                                chunkDurationMs=(segment.end-segment.start) * 1000 // SAMPLE_RATE, language=language,
                                boundary=segment.boundary, overlapMs=segment.overlap_samples * 1000 // SAMPLE_RATE,
                                createdAt=datetime.now(timezone.utc).isoformat(),
                                rawAudioPath=f'/api/transcripts/{session_id}/audio/{name}')
                            if text != boundary_original:
                                row['boundaryOriginalText'] = boundary_original
                            store.append_record(row)
                            records.append(row)
                            previous = text
                            yield dict(type='segment', seq=seq, text=text, tsStart=row['tsStart'], tsEnd=row['tsEnd'])
                    through = max(through, segment.end)
                    yield recognition_status(phase='recognizing')
                if cancelled.is_set():
                    return
                if not records:
                    raise MeetingError('media_no_speech', 422)
                # A close failure must precede the commit/done event, not turn
                # a published success into a later stream error.
                closed_model, model = model, None
                closed_model.close()
                if cancelled.is_set():
                    return
                store.write_metadata({**metadata, 'finalized': True})
                if cancelled.is_set():
                    return
                completed = True
                yield dict(type='done', sessionId=session_id, records=records, durationMs=duration_ms,
                           title=Path(filename).stem[:200], message='文字起こしが完了しました')
    finally:
        try:
            if model is not None:
                model.close()
        finally:
            source.unlink(missing_ok=True)
            if store is not None and not completed:
                for path in (store.jsonl_path, store.txt_path, store.metadata_path):
                    path.unlink(missing_ok=True)
                shutil.rmtree(store.chunks_dir, ignore_errors=True)
                shutil.rmtree(store.screenshots_dir, ignore_errors=True)
                if audio_dir is not None:
                    shutil.rmtree(audio_dir, ignore_errors=True)
