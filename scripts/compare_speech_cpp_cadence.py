"""Compare old/new HTTP live cadence using ONLY supplied synthetic PCM WAVs.

Uses an isolated journal and the configured LiteLLM alias. Does not start an
application server, capture a screen/microphone, or touch production app data.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import wave


async def compare(args, root):
    from server.core.config import settings
    from server.speech_cpp_asr import SpeechCppBatchTranscriber
    from server.transcription.speech_cpp_live import SpeechCppLiveMeeting
    import httpx

    report = {'kind': 'real-model-synthetic-pcm', 'browserCaptureTested': False,
              'productionDataTouched': False, 'profiles': []}
    for audio_path in args.audio:
        with wave.open(str(audio_path)) as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
            pcm = audio.readframes(audio.getnframes()) + b'\0\0'*16000
        for profile in ('before', 'after'):
            began = time.monotonic()
            events, requests, http_requests = [], [], []
            active = {'rt': 0, 'hq': 0}
            maximum = {'rt': 0, 'hq': 0}
            guard = threading.Lock()
            config_root = root / (audio_path.stem + '-' + profile)

            class Config:
                transcripts_dir = config_root / 'transcripts'
                debug_chunks_dir = config_root / 'audio'

                def __getattr__(self, name):
                    return getattr(settings, name)

            class Socket:
                state = SimpleNamespace(authenticated_user_id=991001, is_guest=False,
                                        rate_limit_subject='isolated-synthetic-cadence')

                async def send_json(self, event):
                    if event['type'] != 'capture_ack':
                        events.append({**event, 'elapsedSeconds': round(time.monotonic()-began, 3)})

            def factory(**kwargs):
                lane = 'rt' if not requests_factory else 'hq'
                model = SpeechCppBatchTranscriber(**kwargs)
                transcribe = model.transcribe_chunk

                def timed(audio, **options):
                    with wave.open(io.BytesIO(audio)) as wav:
                        duration = wav.getnframes()/16000
                    started = time.monotonic()
                    with guard:
                        active[lane] += 1
                        maximum[lane] = max(maximum[lane], active[lane])
                    try:
                        return transcribe(audio, **options)
                    finally:
                        with guard:
                            active[lane] -= 1
                            requests.append({'lane': lane, 'audioSeconds': duration,
                                'elapsedStart': round(started-began, 3),
                                'durationMs': round((time.monotonic()-started)*1000)})
                model.transcribe_chunk = timed
                requests_factory.append(model)
                return model

            original_send = httpx.Client.send

            def inspect_send(client, request, *values, **options):
                response = original_send(client, request, *values, **options)
                if request.url.path.endswith('/audio/transcriptions'):
                    http_requests.append({'url': str(request.url), 'status': response.status_code,
                        'providerBase': response.headers.get('x-litellm-model-api-base'),
                        'stop': response.headers.get('X-Speech-Stop') or response.headers.get('llm_provider-X-Speech-Stop')})
                return response

            requests_factory = []
            with ExitStack() as stack:
                for module in ('live', 'whisper_live', 'speech_cpp_live'):
                    stack.enter_context(patch('server.transcription.' + module + '.settings', Config()))
                # The fixture has no application database. Request counts are
                # still recorded, and the real gateway retains its own limits.
                stack.enter_context(patch('server.transcription.live.consume', return_value=True))
                stack.enter_context(patch('server.transcription.speech_cpp_live.SpeechCppBatchTranscriber', side_effect=factory))
                stack.enter_context(patch('httpx.Client.send', inspect_send))
                if profile == 'before':
                    stack.enter_context(patch.object(SpeechCppLiveMeeting, 'update_samples', 32000))
                live = SpeechCppLiveMeeting(Socket(), dict(tracks=['display'], language='auto'))
                worker = asyncio.create_task(live.work())
                capture_start = time.monotonic()
                try:
                    for seq, offset in enumerate(range(0, len(pcm), 8000)):
                        packet = pcm[offset:offset+8000]
                        deadline = capture_start + (offset+len(packet))/32000
                        await asyncio.sleep(max(0, deadline-time.monotonic()))
                        await live.accept_audio(dict(track='display', seq=seq, sampleStart=offset//2,
                            pcm=base64.b64encode(packet).decode()))
                    live.stopping = True
                    await asyncio.wait_for(worker, 180)
                    assert live.data['finalized']
                    assert (live.store.chunks_dir/'display.pcm').read_bytes() == pcm
                    assert not [e for e in events if e['type']=='error'], events
                    assert maximum['rt'] == 1
                    assert len({r['segmentId'] for r in live.records}) == len(live.records)
                    nonempty = [e for e in events if e['type']=='partial' and e.get('text')]
                    finals = [e for e in events if e['type']=='final']
                    rt = [r for r in requests if r['lane']=='rt']
                    report['profiles'].append({'audio': audio_path.name, 'profile': profile,
                        'captureSeconds': len(pcm)/32000, 'packetMs': 250,
                        'updateMs': live.update_samples/16, 'rtWindowSeconds': live.data['rtWindowSamples']/16000,
                        'firstPartialSeconds': nonempty[0]['elapsedSeconds'] if nonempty else None,
                        'partialCount': len(nonempty), 'finalCount': len(finals),
                        'changedPartialCount': sum(i==0 or e.get('text')!=nonempty[i-1].get('text') for i,e in enumerate(nonempty)),
                        'medianPartialGapSeconds': statistics.median([b['elapsedSeconds']-a['elapsedSeconds']
                            for a,b in zip(nonempty, nonempty[1:])]) if len(nonempty)>1 else None,
                        'rtRequestCount': len(rt), 'hqRequestCount': len(requests)-len(rt),
                        'rtMedianInferenceMs': statistics.median(r['durationMs'] for r in rt) if rt else None,
                        'maxBacklogMs': max([e.get('backlogMs',0) for e in events]+[0]),
                        'maxConcurrentRequests': maximum, 'rawPcmPreserved': True,
                        'finalText': '\n'.join(r['text'] for r in live.records),
                        'events': events, 'requests': requests, 'upstreamRequests': http_requests})
                    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                    print(json.dumps({k:v for k,v in report['profiles'][-1].items()
                        if k not in {'events','requests','upstreamRequests','finalText'}}, ensure_ascii=False), flush=True)
                finally:
                    live.disconnected = True
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    live.close()
    assert all(r['status']==200 and r['stop']=='complete' for p in report['profiles'] for r in p['upstreamRequests'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='whistx-cadence-') as directory:
        os.environ.update(APP_ENV='development', APP_DATA_DIR=directory,
            APP_SESSION_SECRET='isolated-synthetic-cadence-secret-only', LANGFUSE_ENABLED='0')
        asyncio.run(compare(args, Path(directory)))


if __name__ == '__main__':
    main()
