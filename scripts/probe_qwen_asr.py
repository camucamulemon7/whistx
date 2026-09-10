#!/usr/bin/env python3
"""Check the configured single vLLM server's batch and realtime audio support.

Uses a generated silence fixture unless a 16 kHz mono PCM16 WAV is supplied.
Silence probes protocol support only, not recognition accuracy.
"""
import argparse
import asyncio
import json
import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

os.environ.setdefault("APP_SESSION_SECRET", "isolated-audio-probe-only-long-session-secret")
from server.qwen_asr import QwenRealtime, batch_payload
from server.transcription.local_agreement import pcm_wav


async def probe(args):
    pcm = b'\0\0' * 16000
    if args.audio:
        with wave.open(str(args.audio), 'rb') as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
                raise ValueError('Audio must be 16 kHz mono PCM16 WAV')
            pcm = source.readframes(source.getnframes())
    async def batch():
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            response = await client.post(args.url.rstrip('/') + '/chat/completions',
                json=batch_payload(pcm_wav(pcm), mime_type='audio/wav', model=args.model, context='', priority=10))
            print(json.dumps(dict(lane='batch', status=response.status_code, response=response.json()), ensure_ascii=False), flush=True)
            return response.is_success
    async def realtime():
        async with QwenRealtime(base_url=args.url, api_key='', model=args.model, timeout=args.timeout) as stream:
            await stream.append(pcm)
            await stream.finish()
            while not stream.done:
                event = await stream.receive()
                print(json.dumps(dict(lane='realtime', event=event), ensure_ascii=False), flush=True)
        return True
    results = await asyncio.gather(batch(), realtime(), return_exceptions=True)
    for lane, result in zip(['batch', 'realtime'], results):
        if isinstance(result, BaseException):
            print(json.dumps(dict(lane=lane, error=type(result).__name__, detail=str(result))), flush=True)
    return all(result is True for result in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:8004/v1')
    parser.add_argument('--model', default='Qwen3-ASR-1.7B')
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--timeout', type=float, default=30)
    sys.exit(0 if asyncio.run(probe(parser.parse_args())) else 1)
