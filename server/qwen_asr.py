"""Remote-only Qwen3-ASR clients. Both lanes use one vLLM URL and model.

Realtime follows vLLM's model/commit/append/final-commit protocol, rather
than the different OpenAI Realtime conversation protocol. No model is loaded.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import connect

from .asr import ASRChunkResult
from .core.config import settings


def realtime_url(base_url: str | None) -> str:
    value = urlsplit(base_url or '')
    if value.scheme not in {'http', 'https'} or not value.netloc or value.username or value.query or value.fragment:
        raise ValueError('Qwen requires an HTTP(S) ASR_BASE_URL without credentials or query')
    return urlunsplit(('wss' if value.scheme == 'https' else 'ws', value.netloc, value.path.rstrip('/') + '/realtime', '', ''))


def clean_qwen_text(text: str, *, final: bool = True) -> str:
    # Qwen emits a leading language header even for mixed speech. Suppress
    # its fragmented prefix without deleting the ordinary English word later.
    value = text.strip()
    if not final and ('language'.startswith(value) or (value.startswith('language ') and '<asr_text>' not in value)):
        return ''
    value = re.sub(r'^language\s+[^<\n]{1,80}<asr_text>', '', value)
    value = value.replace('<asr_text>', '')
    value = re.sub(r'<\|[^|]*\|>', '', value)
    if not final:
        value = re.sub(r'<[^>]*$', '', value)
    return value.strip()


def batch_payload(audio: bytes, *, mime_type: str, model: str, context: str, priority: int) -> dict:
    context = re.sub(r'<\|.*?\|>|<asr_text>', '', context)[:4000]
    messages = []
    if context:
        messages.append({'role': 'system', 'content': context})
    messages.append({'role': 'user', 'content': [{'type': 'audio_url', 'audio_url': {
        'url': 'data:' + mime_type + ';base64,' + base64.b64encode(audio).decode('ascii'),
    }}]})
    return {'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': 4096,
            'priority': priority, 'stream': False}


def batch_request(audio: bytes, *, mime_type: str, model: str, context: str, priority: int, language: str = 'auto') -> tuple[str, dict]:
    if language in {'ja', 'en'}:
        # The original Qwen chat template discards assistant messages, so a
        # chat prefill is not portable. The transcription endpoint accepts language.
        return '/audio/transcriptions', {
            'data': {'model': model, 'language': language, 'prompt': re.sub(r'<\|.*?\|>|<asr_text>', '', context)[:4000],
                     'temperature': '0', 'priority': str(priority), 'response_format': 'json'},
            'files': {'file': ('audio.wav', audio, mime_type)},
        }
    return '/chat/completions', {'json': batch_payload(audio, mime_type=mime_type, model=model, context=context, priority=priority)}


def response_text(payload: dict) -> str:
    if isinstance(payload.get('text'), str):
        text = clean_qwen_text(payload['text'])
        if not text:
            raise ValueError('qwen_empty_response')
        return text
    choices = payload.get('choices') or []
    if not choices or choices[0].get('finish_reason') != 'stop':
        raise ValueError('qwen_incomplete_response')
    text = choices[0].get('message', {}).get('content')
    if not isinstance(text, str):
        raise ValueError('qwen_invalid_response')
    text = clean_qwen_text(text)
    if not text:
        raise ValueError('qwen_empty_response')
    return text


class QwenBatchTranscriber:
    """Sync adapter for the existing manual refinement/chunk interface."""
    def __init__(self, *, api_key: str, base_url: str | None, model: str, **kwargs):
        realtime_url(base_url)
        self.base_url, self.model = str(base_url).rstrip('/'), model
        self.client = httpx.Client(headers={'Authorization': 'Bearer ' + api_key}, timeout=settings.asr_high_accuracy_timeout_seconds)

    def transcribe_chunk(self, audio_bytes: bytes, *, mime_type: str, language: str | None, prompt: str | None,
                         temperature: float, **kwargs) -> ASRChunkResult:
        endpoint, options = batch_request(audio_bytes, mime_type=mime_type, model=self.model, context=prompt or '',
                                          priority=settings.asr_high_accuracy_priority, language=language or 'auto')
        result = self.client.post(self.base_url + endpoint, **options)
        result.raise_for_status()
        return ASRChunkResult(text=response_text(result.json()), start_ms=None, end_ms=None)

    def close(self):
        self.client.close()


class QwenRealtime:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float):
        self.url, self.api_key, self.model, self.timeout = realtime_url(base_url), api_key, model, timeout
        self.socket = None
        self.done = False

    async def __aenter__(self):
        self.socket = await connect(self.url, additional_headers={'Authorization': 'Bearer ' + self.api_key},
                                    open_timeout=min(15, self.timeout), close_timeout=3, max_size=2_000_000, max_queue=16)
        try:
            created = json.loads(await asyncio.wait_for(self.socket.recv(), self.timeout))
            if created.get('type') != 'session.created':
                raise ValueError('qwen_realtime_handshake_failed')
            await self.socket.send(json.dumps({'type': 'session.update', 'model': self.model}))
            await self.socket.send(json.dumps({'type': 'input_audio_buffer.commit', 'final': False}))
            return self
        except BaseException:
            await self.socket.close()
            raise

    async def append(self, pcm: bytes):
        await self.socket.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm).decode('ascii')}))

    async def finish(self):
        await self.socket.send(json.dumps({'type': 'input_audio_buffer.commit', 'final': True}))

    async def receive(self):
        event = json.loads(await asyncio.wait_for(self.socket.recv(), self.timeout))
        if event.get('type') == 'error':
            # Do not surface upstream error bodies (they can contain request text).
            raise RuntimeError('qwen_realtime_error')
        if event.get('type') == 'transcription.done':
            self.done = True
        return event

    async def __aexit__(self, *exc):
        if self.socket:
            await self.socket.close()


def script_counts(text: str) -> tuple[int, int]:
    """A conservative omission signal, not a language/accuracy classifier."""
    japanese = len(re.findall(r'[\u3040-\u30ff\u3400-\u9fff]', text))
    latin = len(re.findall(r'[A-Za-z]', text))
    return japanese, latin


def language_coverage_lost(original: str, candidate: str) -> bool:
    before_ja, before_en = script_counts(original)
    after_ja, after_en = script_counts(candidate)
    # Do not mistake a lone acronym or punctuation normalization for omission.
    return (before_ja >= 8 and after_ja < 2) or (before_en >= 24 and after_en < 5)


def script_groups(records: list[dict]) -> list[list[dict]]:
    """Keep adjacent source windows together where the visible script agrees."""
    groups = []
    previous = None
    for row in sorted(records, key=lambda item: item['startSample']):
        ja, en = script_counts(row['text'])
        script = 'ja' if ja >= 2 else 'en' if en else 'other'
        if not groups or script != previous:
            groups.append([])
        groups[-1].append(row)
        previous = script
    return groups
