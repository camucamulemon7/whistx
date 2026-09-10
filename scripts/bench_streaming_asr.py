#!/usr/bin/env python3
"""Replay a 16 kHz mono PCM WAV through the real live adapter and .env ASR.

This is an offline harness: isolated temporary artifacts, no browser/network
transport measurement, no application quota writes. --realtime paces capture.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = row
    return previous[-1]


async def replay(args, pcm):
    from server.core.config import settings
    from server.transcription.live import LiveMeeting
    from server.transcription.qwen_live import QwenLiveMeeting

    events = []
    begin = time.monotonic()
    class Socket:
        state = SimpleNamespace(authenticated_user_id=1, is_guest=False, rate_limit_subject='offline-benchmark')
        async def send_json(self, data):
            events.append({**data, 'observedMs': round((time.monotonic()-begin)*1000)})

    class IsolatedSettings:
        def __init__(self, root):
            self.transcripts_dir = root / 'transcripts'
            self.debug_chunks_dir = root / 'audio'
        def __getattr__(self, name):
            return getattr(settings, name)

    with tempfile.TemporaryDirectory(prefix='whistx-asr-bench-') as directory:
        with ExitStack() as stack:
            for module in ['server.transcription.live', 'server.transcription.qwen_live']:
                stack.enter_context(patch(module + '.settings', IsolatedSettings(Path(directory))))
                stack.enter_context(patch(module + '.consume', return_value=True))
            meeting_class = QwenLiveMeeting if settings.asr_backend == 'qwen3_vllm' else LiveMeeting
            meeting = meeting_class(Socket(), {'tracks': ['mic'], 'language': args.language, 'prompt': args.prompt})
            worker = asyncio.create_task(meeting.work())
            try:
                for seq, start in enumerate(range(0, len(pcm), 32000)):
                    packet = pcm[start:start+32000]
                    if args.realtime:
                        target = begin + (start + len(packet)) / 32000
                        await asyncio.sleep(max(0, target - time.monotonic()))
                    await meeting.accept_audio(dict(track='mic', seq=seq, sampleStart=start//2, pcm=base64.b64encode(packet).decode()))
                    # Backpressure needs replay from the durable offset.
                    while meeting.data['tracks']['mic']['seq'] < seq:
                        await asyncio.sleep(.1)
                        await meeting.accept_audio(dict(track='mic', seq=seq, sampleStart=start//2, pcm=base64.b64encode(packet).decode()))
                capture_done = time.monotonic()
                meeting.stopping = True
                meeting.wake.set()
                await asyncio.wait_for(worker, args.timeout)
                if not meeting.data['finalized']:
                    raise RuntimeError('ASR did not finalize; inspect endpoint health')
                finals = [row for row in meeting.records if row['type'] == 'final']
                partials = [event for event in events if event['type'] == 'partial' and event.get('text')]
                text = ''.join(event['text'] for event in finals)
                report = dict(model=settings.asr_model, backend=settings.asr_backend, realtime=args.realtime, durationSeconds=len(pcm)/32000,
                              elapsedSeconds=round(time.monotonic()-begin,3), finalizeSeconds=round(time.monotonic()-capture_done,3),
                              firstPartialMs=partials[0]['observedMs'] if partials else None,
                              apiRequests=meeting.data['asrRequests'], text=text, events=events,
                              limitations='Adapter inference only; excludes microphone, browser and WebSocket transport. CER is raw Unicode character edit rate.')
                if args.reference:
                    reference = args.reference.read_text(encoding='utf-8').strip()
                    report['cer'] = distance(reference, text)/max(1,len(reference))
                return report
            finally:
                meeting.disconnected = True
                meeting.wake.set()
                if not worker.done():
                    await asyncio.shield(worker)
                meeting.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--language', default='ja')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    with wave.open(str(args.audio), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, 'NONE'):
            parser.error('audio must be an uncompressed 16 kHz mono signed 16-bit PCM WAV')
        pcm = audio.readframes(audio.getnframes())
    if not pcm or len(pcm) > 32000*3600:
        parser.error('provide between one sample and one hour of audio')
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault('APP_SESSION_SECRET', 'offline-benchmark-only-abcdefghijklmnopqrstuvwxyz')
    report = asyncio.run(replay(args, pcm))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'events','text'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
