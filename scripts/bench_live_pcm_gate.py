"""Compare duplicate/reused PCM gates through the speech.cpp HTTP live adapter.

Use supplied synthetic 16 kHz mono PCM16 WAV and ASR_BASE_URL/ASR_MODEL/ASR_API_KEY.
The first five seconds are decoded at the normal single-track sample boundaries.
Artifacts are temporary; application quotas are stubbed, gateway limits remain.
This measures adapter overhead, not microphone/browser latency or ASR accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def compare(args, pcm, root):
    import httpx
    from server.core.config import settings
    from server.pcm_gate import pcm_speech_bounds
    from server.transcription.speech_cpp_live import SpeechCppLiveMeeting

    report = dict(kind='real-speech-cpp-pcm-gate', model=settings.asr_model, runs=[],
        limitations='Fixed sample-boundary adapter replay; excludes capture, browser, network WebSocket, HQ and quota DB latency.')
    # Alternate order to reduce warmup/order bias. The first pair is a warmup.
    for repeat in range(args.repeats + 1):
        for profile in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
            run = dict(profile=profile, repeat=repeat, warmup=repeat == 0, decodes=[], upstream=[])
            gates = []

            def gate(audio):
                started = time.perf_counter()
                result = pcm_speech_bounds(audio)
                gates.append((time.perf_counter() - started) * 1000)
                return result

            original_post = httpx.Client.post

            def post(client, url, **options):
                started = time.perf_counter()
                response = original_post(client, url, **options)
                run['upstream'].append(dict(status=response.status_code,
                    durationMs=(time.perf_counter() - started) * 1000,
                    providerBase=response.headers.get('x-litellm-model-api-base'),
                    stop=response.headers.get('X-Speech-Stop') or response.headers.get('llm_provider-X-Speech-Stop'),
                    uploadSha256=hashlib.sha256(options['files']['file'][1]).hexdigest()))
                return response

            class Config:
                transcripts_dir = root / f'{repeat}-{profile}' / 'transcripts'
                debug_chunks_dir = root / f'{repeat}-{profile}' / 'audio'
                asr_high_accuracy_enabled = False
                asr_realtime_window_seconds = 5

                def __getattr__(self, name):
                    return getattr(settings, name)

            class Socket:
                state = SimpleNamespace(authenticated_user_id=1, is_guest=False, rate_limit_subject='isolated-benchmark')

                async def send_json(self, event):
                    pass

            with ExitStack() as stack:
                for module in ('live', 'whisper_live', 'speech_cpp_live'):
                    stack.enter_context(patch('server.transcription.' + module + '.settings', Config()))
                stack.enter_context(patch('server.transcription.live.consume', return_value=True))
                stack.enter_context(patch('server.transcription.whisper_live.whisper_speech_bounds', gate))
                stack.enter_context(patch('server.pcm_gate.pcm_speech_bounds', gate))
                stack.enter_context(patch('httpx.Client.post', post))
                live = SpeechCppLiveMeeting(Socket(), dict(tracks=['mic'], language=args.language))
                live.reuse_pcm_bounds = profile == 'after'
                try:
                    # Fixed boundaries keep inference input independent of provider latency.
                    for seq, start in enumerate(range(0, len(pcm), 40000)):
                        packet = pcm[start:start + 40000]
                        await live.accept_audio(dict(track='mic', seq=seq, sampleStart=start // 2,
                            pcm=base64.b64encode(packet).decode()))
                        gates.clear()
                        started = time.perf_counter()
                        await live._infer('mic', flush=False)
                        run['decodes'].append(dict(inferMs=(time.perf_counter() - started) * 1000,
                            gateCalls=len(gates), gateMs=sum(gates)))
                    run['text'] = '\n'.join(r['text'] for r in live.records)
                    run['rawPcmPreserved'] = (live.store.chunks_dir / 'mic.pcm').read_bytes() == pcm
                finally:
                    live.close()
            report['runs'].append(run)
            print(json.dumps(dict(profile=profile, repeat=repeat, warmup=run['warmup'],
                inferMs=round(sum(d['inferMs'] for d in run['decodes']), 2),
                gateMs=round(sum(d['gateMs'] for d in run['decodes']), 2),
                gateCalls=sum(d['gateCalls'] for d in run['decodes']),
                requests=len(run['upstream'])), ensure_ascii=False), flush=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    report['summary'] = {}
    for profile in ('before', 'after'):
        runs = [r for r in report['runs'] if r['profile'] == profile and not r['warmup']]
        report['summary'][profile] = {
            name: statistics.median(sum(d[name] for d in r['decodes']) for r in runs)
            for name in ('inferMs', 'gateCalls', 'gateMs')}
    report['sameUploads'] = all(
        [r['uploadSha256'] for r in run['upstream']] ==
        [r['uploadSha256'] for r in report['runs'][0]['upstream']] for run in report['runs'])
    report['sameFinalText'] = len({r['text'] for r in report['runs']}) == 1
    report['allComplete'] = all(r['upstream'] and all(u['status'] == 200 and u['stop'] == 'complete'
        for u in r['upstream']) for r in report['runs'])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, ensure_ascii=False), flush=True)
    if not report['sameUploads'] or not report['allComplete'] or not all(r['rawPcmPreserved'] for r in report['runs']):
        raise RuntimeError('Upload, provider completion, or source preservation check failed; inspect the report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--language', choices=['auto', 'ja', 'en'], default='ja')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('--repeats must be between 1 and 10')
    with wave.open(str(args.audio)) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, 'NONE'):
            parser.error('audio must be uncompressed 16 kHz mono PCM16 WAV')
        pcm = audio.readframes(5 * 16000)
    if len(pcm) != 5 * 32000:
        parser.error('provide at least five seconds of synthetic speech')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='whistx-gate-benchmark-') as directory:
        os.environ.update(APP_ENV='development', APP_DATA_DIR=directory, ASR_BACKEND='speech_cpp',
            APP_SESSION_SECRET='isolated-gate-benchmark-session-secret', LANGFUSE_ENABLED='0')
        asyncio.run(compare(args, pcm, Path(directory)))


if __name__ == '__main__':
    main()
