#!/usr/bin/env python3
"""Reproducible remote-ASR comparison. Input: local 16 kHz mono PCM16 WAV + reference.

Windows are submitted sequentially without real-time pacing: reported API time
is inference/transport wall time, not microphone-to-screen latency. Silence offsets
probe cut alignment; they are not independent speech samples. No audio is uploaded
anywhere except the configured ASR endpoint. Credentials are never written to reports.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import unicodedata
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def edits(reference, hypothesis):
    """Return substitutions, deletions, insertions for a minimum edit alignment."""
    previous = [(0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, left in enumerate(reference, 1):
        current = [(0, i, 0)]
        for j, right in enumerate(hypothesis, 1):
            s, d, ins = previous[j - 1]
            diagonal = (s + (left != right), d, ins)
            s, d, ins = previous[j]
            deletion = (s, d + 1, ins)
            s, d, ins = current[-1]
            insertion = (s, d, ins + 1)
            current.append(min((diagonal, deletion, insertion), key=sum))
        previous = current
    return previous[-1]


async def run(args):
    import httpx
    from server.qwen_asr import QwenRealtime, batch_request, clean_qwen_text, response_text
    from server.transcription.local_agreement import pcm_wav, speech_bounds
    with wave.open(str(args.audio), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Expected 16 kHz mono PCM16 WAV')
        original = wav.readframes(wav.getnframes())
    reference = args.reference.read_text().strip()
    if not normalized(reference):
        raise ValueError('Reference must contain speech')
    base = os.environ['ASR_BASE_URL'].rstrip('/')
    key = os.environ['ASR_API_KEY']
    output = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + key}, timeout=120) as client:
        for repeat in range(args.repeats):
            for offset in args.offsets:
                pcm = bytes(int(offset * 32000)) + original
                for profile in args.profiles:
                    begin = time.monotonic()
                    row = {'profile': profile, 'offsetSeconds': offset, 'repeat': repeat,
                           'audioSeconds': len(pcm) / 32000, 'requests': 0}
                    try:
                        chunks = []
                        if profile.startswith('qwen-rt-'):
                            size = int(profile.rsplit('-', 1)[1]) * 32000
                            for start in range(0, len(pcm), size):
                                audio = pcm[start:start + size]
                                if speech_bounds(audio) is None:
                                    continue
                                row['requests'] += 1
                                async with QwenRealtime(base_url=base, api_key=key, model=args.qwen_model, timeout=120) as stream:
                                    await stream.append(audio)
                                    await stream.finish()
                                    text = ''
                                    while not stream.done:
                                        event = await stream.receive()
                                        if event.get('type') == 'transcription.delta':
                                            text += event.get('delta', '')
                                        elif event.get('type') == 'transcription.done':
                                            text = event.get('text', text)
                                    chunks.append(clean_qwen_text(text))
                        else:
                            row['requests'] += 1
                            if profile == 'qwen-hq':
                                endpoint, options = batch_request(pcm_wav(pcm), mime_type='audio/wav', model=args.qwen_model,
                                    context='', priority=10, language='auto')
                            else:
                                endpoint, options = '/audio/transcriptions', {'data': {'model': args.whisper_model, 'response_format': 'json'},
                                    'files': {'file': ('sample.wav', pcm_wav(pcm), 'audio/wav')}}
                            response = await client.post(base + endpoint, **options)
                            response.raise_for_status()
                            chunks.append(response_text(response.json()))
                        hypothesis = ''.join(chunks)
                        s, d, ins = edits(normalized(reference), normalized(hypothesis))
                        row.update(hypothesis=hypothesis, substitutions=s, deletions=d, insertions=ins,
                                   cer=(s+d+ins)/len(normalized(reference)))
                    except Exception as exc:
                        row['error'] = type(exc).__name__
                    row['apiWallSeconds'] = round(time.monotonic() - begin, 3)
                    output.append(row)
                    args.output.write_text(json.dumps({'reference': reference, 'audio': args.audio.name,
                        'note': 'Unpaced API timing; offsets reuse the same speech sample.', 'results': output}, ensure_ascii=False, indent=2)+'\n')
                    print(json.dumps({k: v for k, v in row.items() if k != 'hypothesis'}, ensure_ascii=False), flush=True)
    return int(any('error' in row for row in output))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offsets', nargs='+', type=float, default=[0, 1, 2])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--profiles', nargs='+', choices=['qwen-rt-5', 'qwen-rt-10', 'qwen-rt-15', 'qwen-hq', 'whisper-hq'],
                        default=['qwen-rt-5', 'qwen-rt-10', 'qwen-rt-15', 'qwen-hq', 'whisper-hq'])
    parser.add_argument('--qwen-model', default='Qwen3-ASR-1.7B')
    parser.add_argument('--whisper-model', default='whisper-large-v3-turbo')
    args = parser.parse_args()
    if args.repeats < 1 or any(x < 0 or x > 10 for x in args.offsets):
        parser.error('repeats must be positive and offsets between 0 and 10 seconds')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
