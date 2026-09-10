"""Explicit, cancellable audio re-recognition before a runtime is saved.

Original text is retained per record. The JSONL rename is the commit point;
all expensive inference completes before the transcript becomes visible.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import settings
from ..openai_whisper import OpenAIWhisperTranscriber
from ..transcript_store import _render_txt_line, read_jsonl_records, resolve_debug_audio_path
from ..transcription.text_processing import _sanitize_transcript_text
from .meeting_source import MeetingError, read_json, write_json_atomic


def _atomic_text(path: Path, text: str):
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def refine_events(snapshot, *, cancelled, transcriber_factory=None, allow_request=None):
    if transcriber_factory is None:
        from ..qwen_asr import QwenBatchTranscriber
        transcriber_factory = QwenBatchTranscriber if settings.asr_backend == "qwen3_vllm" else OpenAIWhisperTranscriber
    if not snapshot.key.startswith('runtime:') or not snapshot.finalized:
        raise MeetingError('refinement_requires_completed_runtime', 409)
    session_id = snapshot.key.split(':', 1)[1]
    lock_path = snapshot.transcript_path.with_suffix('.live.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MeetingError('meeting_already_connected', 409) from exc
        metadata_path = snapshot.transcript_path.with_suffix('.meta.json')
        metadata = read_json(metadata_path)
        if not metadata.get('finalized'):
            raise MeetingError('refinement_requires_completed_runtime', 409)
        records = read_jsonl_records(snapshot.transcript_path, strict=True)
        finals = [row for row in records if row.get('type') == 'final']
        if not finals:
            raise MeetingError('empty_transcript')
        audio = []
        for row in finals:
            filename = Path(str(row.get('rawAudioPath') or '')).name
            path = resolve_debug_audio_path(settings.debug_chunks_dir, session_id, filename) if filename else None
            if path is None or not path.is_file():
                raise MeetingError('refinement_audio_missing', 409)
            if path.stat().st_size > 8_000_000:
                raise MeetingError('refinement_audio_too_large', 413)
            audio.append(path)
        model = transcriber_factory(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.asr_model)
        try:
            for i, (row, path) in enumerate(zip(finals, audio)):
                if cancelled.is_set():
                    return
                if allow_request is not None and not allow_request():
                    raise MeetingError('rate_limit_exceeded', 429)
                yield {'type': 'status', 'message': f'音声を再認識しています {i+1}/{len(finals)}', 'completed': i, 'total': len(finals)}
                result = model.transcribe_chunk(path.read_bytes(), mime_type='audio/wav' if path.suffix == '.wav' else 'audio/webm',
                                                language=row.get('language') or None,
                                                prompt=str(metadata.get('prompt') or '')[:500] or None, temperature=0.0)
                text = _sanitize_transcript_text(result.text, language=row.get('language')).strip()
                if not text:
                    raise MeetingError('refinement_empty_result', 502)
                if settings.asr_backend == 'qwen3_vllm':
                    from ..qwen_asr import language_coverage_lost
                    if language_coverage_lost(row['text'], text):
                        raise MeetingError('refinement_language_loss', 502)
                row.setdefault('originalText', row['text'])
                row['text'] = text
                row['refinedAt'] = datetime.now(timezone.utc).isoformat()
            if cancelled.is_set():
                return
            # Keep the complete pre-refinement source, including screenshot rows.
            original = snapshot.transcript_path.with_suffix('.original.jsonl')
            if not original.exists():
                _atomic_text(original, snapshot.transcript_path.read_text(encoding='utf-8'))
            _atomic_text(snapshot.transcript_path, ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records))
            _atomic_text(snapshot.transcript_path.with_suffix('.txt'), ''.join(_render_txt_line(row)+'\n' for row in finals))
            metadata['refinedAt'] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(metadata_path, metadata)
            yield {'type': 'done', 'records': finals, 'model': settings.asr_model, 'message': '音声から再認識しました。原文は保存されています。'}
        finally:
            model.client.close()
