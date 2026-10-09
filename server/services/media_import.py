"""Bounded local-media decoding and cancellable, sample-anchored ASR import."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import settings
from ..langfuse_observer import high_accuracy_observation
from ..pcm_gate import pcm_speech_bounds
from ..transcript_store import TranscriptStore, build_debug_chunks_dir
from ..transcription.local_agreement import SAMPLE_RATE, pcm_wav
from ..transcription.text_processing import _sanitize_transcript_text
from .meeting_source import MeetingError

MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_DURATION_SECONDS = 2 * 60 * 60
WINDOW_SECONDS = 30
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


def decode_media(source: Path, destination: Path, *, filename: str, cancelled, ffmpeg_bin=None):
    if cancelled.is_set():
        raise MeetingError('media_cancelled', 409)
    executable = ffmpeg_bin or settings.ffmpeg_bin
    if shutil.which(executable) is None:
        raise MeetingError('media_decoder_unavailable', 503)
    command = [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-protocol_whitelist', 'file,pipe', '-f', media_format(filename), '-i', str(source),
               '-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', '1', '-ar', str(SAMPLE_RATE),
               '-t', str(MAX_DURATION_SECONDS + 1), '-c:a', 'pcm_s16le', str(destination)]
    # Discard decoder logs: embedded metadata/file paths are not public errors.
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + max(120, settings.ffmpeg_timeout_seconds)
    try:
        while process.poll() is None:
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


def import_events(source: Path, *, filename: str, language: str, prompt: str, user_id: int,
                  cancelled, transcriber_factory, allow_request, observer=None, decoder=decode_media):
    """Publish a completed runtime only; errors/cancellation remove partial artifacts."""
    store = None
    audio_dir = None
    model = None
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix='whistx-decode-') as directory:
            decoded = Path(directory) / 'audio.wav'
            yield dict(type='status', phase='decoding', message='音声を取り出しています', progress=0)
            decoder(source, decoded, filename=filename, cancelled=cancelled)
            if cancelled.is_set():
                return
            session_id = 'file-' + uuid.uuid4().hex
            store = TranscriptStore(settings.transcripts_dir, session_id)
            audio_dir = build_debug_chunks_dir(settings.debug_chunks_dir, session_id)
            audio_dir.mkdir(parents=True, exist_ok=False)
            model = transcriber_factory()
            if hasattr(model, 'multi_pass_enabled'):
                model.multi_pass_enabled = False
            records = []
            with wave.open(str(decoded)) as audio:
                total = audio.getnframes()
                duration_ms = total * 1000 // SAMPLE_RATE
                metadata = dict(sessionId=session_id, ownerUserId=user_id, language=language,
                    audioSource='file', sourceName=Path(filename).name[:200], mode='file',
                    asrBackend=settings.asr_backend, asrModel=settings.asr_model,
                    finalized=False, durationMs=duration_ms, prompt=prompt,
                    createdAt=datetime.now(timezone.utc).isoformat())
                store.write_metadata(metadata)
                start = 0
                while start < total:
                    if cancelled.is_set():
                        return
                    pcm = audio.readframes(WINDOW_SECONDS * SAMPLE_RATE)
                    end = start + len(pcm) // 2
                    if pcm_speech_bounds(pcm) is not None:
                        waiting = False
                        while not allow_request():
                            if not waiting:
                                yield dict(type='status', phase='waiting', progress=round(start * 100 / total),
                                           message='順番を待っています。まもなく再開します')
                                waiting = True
                            if cancelled.wait(1):
                                return
                        with high_accuracy_observation(observer, session_id=session_id, model=settings.asr_model,
                                start_ms=start * 1000 // SAMPLE_RATE, end_ms=end * 1000 // SAMPLE_RATE,
                                audio_bytes=len(pcm), language=language, track='file') as observation:
                            result = model.transcribe_chunk(pcm_wav(pcm), mime_type='audio/wav',
                                language=language or None, prompt=prompt or None, temperature=0.0)
                            text = _sanitize_transcript_text(result.text, language=language).strip()
                            observation.update(output=dict(text=text, chars=len(text), succeeded=True))
                        if cancelled.is_set():
                            return
                        if text:
                            seq = len(records)
                            name = f'file-{seq:06d}.wav'
                            (audio_dir / name).write_bytes(pcm_wav(pcm))
                            # Interval timing stays anchored to the imported media.
                            row = dict(type='final', segmentId=f'file-{start}-{end}', seq=seq, text=text,
                                track='file', startSample=start, endSample=end, quality='high_accuracy',
                                tsStart=start * 1000 // SAMPLE_RATE, tsEnd=end * 1000 // SAMPLE_RATE,
                                chunkOffsetMs=start * 1000 // SAMPLE_RATE,
                                chunkDurationMs=(end-start) * 1000 // SAMPLE_RATE, language=language,
                                createdAt=datetime.now(timezone.utc).isoformat(),
                                rawAudioPath=f'/api/transcripts/{session_id}/audio/{name}')
                            store.append_record(row)
                            records.append(row)
                    start = end
                    yield dict(type='status', phase='recognizing', progress=round(start * 100 / total),
                               throughMs=start * 1000 // SAMPLE_RATE, durationMs=duration_ms,
                               message=f'文字起こし中 · {round(start * 100 / total)}%')
                if cancelled.is_set():
                    return
                if not records:
                    raise MeetingError('media_no_speech', 422)
                # A close failure must precede the commit/done event, not turn
                # a published success into a later stream error.
                closed_model, model = model, None
                closed_model.close()
                store.write_metadata({**metadata, 'finalized': True})
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
