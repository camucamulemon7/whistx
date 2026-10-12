"""Bounded file strategies, using explicitly supplied synthetic audio only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import time
import wave


def normalized(text):
    return ''.join(re.findall(r'\w', text.casefold()))


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = next_row
    return row[-1]


def main():
    os.environ.setdefault('APP_SESSION_SECRET', 'synthetic-file-comparison-only-secret-value')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--language', choices=['ja', 'en', 'auto'], default='ja')
    parser.add_argument('--profiles', nargs='+', default=['fixed30', 'fixed60', 'fixed90', 'pause60', 'pause75', 'fixed30-parallel2'])
    args = parser.parse_args()
    from server.core.config import settings
    from server.services.media_segments import FileSegment, FilePolicy, iter_file_segments, trim_overlap_text
    from server.speech_cpp_asr import SpeechCppBatchTranscriber
    from server.transcription.local_agreement import pcm_wav
    import httpx

    reference = normalized(args.reference.read_text())
    decoded = args.report.parent / 'synthetic.wav'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(args.audio),
        '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(decoded)], check=True)
    with wave.open(str(decoded)) as wav:
        duration = wav.getnframes() / 16000
        assert duration <= 180, 'Keep real-model comparisons short and synthetic'
        pcm = wav.readframes(wav.getnframes())
    report = dict(syntheticOnly=True, productionDataTouched=False, inputSeconds=duration,
                  referenceChars=len(reference), profiles=[])
    for profile in args.profiles:
        health = httpx.get('http://127.0.0.1:8005/api/health', timeout=5).json()
        assert health.get('activeConnections') == 0, 'Stop when recording is active'
        began = time.monotonic()
        if profile.startswith(('pause', 'forced')):
            target = int(re.search(r'\d+', profile).group())
            policy = (FilePolicy(target_seconds=target, pause_ms=100000, strong_pause_ms=0)
                      if profile.startswith('forced') else FilePolicy(target_seconds=target,
                          strong_pause_ms=800 if args.language == 'auto' else 0))
            with wave.open(str(decoded)) as wav:
                segments = list(iter_file_segments(wav, policy))
        else:
            size = int(re.search(r'\d+', profile).group()) * 16000
            overlap = 12800 if 'overlap' in profile else 0
            segments = [FileSegment(max(0,start-overlap), min(start + size,len(pcm)//2),
                        pcm[max(0,start-overlap)*2:(start+size)*2], min(start,overlap), 'limit')
                        for start in range(0,len(pcm)//2,size)]
        planning = time.monotonic() - began
        def recognize(segment):
            model = SpeechCppBatchTranscriber(api_key=settings.openai_api_key,
                base_url=settings.openai_base_url, model=settings.asr_model)
            observed = []
            original = model.client.send
            def inspect(request, *a, **kw):
                response = original(request, *a, **kw)
                observed.append(dict(url=str(request.url), status=response.status_code,
                    provider=response.headers.get('x-litellm-model-api-base'),
                    stop=response.headers.get('X-Speech-Stop') or response.headers.get('llm_provider-X-Speech-Stop')))
                return response
            model.client.send = inspect
            started = time.monotonic()
            try:
                result = model.transcribe_chunk(pcm_wav(segment.pcm), mime_type='audio/wav',
                    language=args.language, prompt=None)
                return dict(startSeconds=segment.start/16000, endSeconds=segment.end/16000,
                    boundary=segment.boundary, overlapSeconds=segment.overlap_samples/16000,
                    seconds=time.monotonic()-started, completedAt=time.monotonic()-began,
                    text=result.text, http=observed)
            finally:
                model.close()
        workers = 2 if 'parallel2' in profile else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(recognize, segments))
        previous = ''
        for row in rows:
            row['text'] = trim_overlap_text(previous, row['text'], overlapping=bool(row['overlapSeconds']))
            previous = row['text']
        text = '\n'.join(row['text'] for row in rows)
        value = dict(profile=profile, concurrency=workers, planningSeconds=planning,
            totalSeconds=time.monotonic()-began, firstCompletionSeconds=min(row['completedAt'] for row in rows),
            asrRequests=sum(len(row['http']) for row in rows),
            charErrorRate=distance(reference, normalized(text))/max(1, len(reference)),
            text=text, segments=rows)
        report['profiles'].append(value)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({key: value[key] for key in ('profile', 'concurrency', 'totalSeconds', 'firstCompletionSeconds', 'asrRequests', 'charErrorRate')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
