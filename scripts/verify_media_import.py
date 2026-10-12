"""Authenticated synthetic file import before/after; isolated SQLite/artifacts.

Requires the test fixtures in this checkout. Set ASR_* to the existing gateway.
The reference and all audio inputs MUST be synthetic, each at most 180 seconds.
No application server or production account is created; quota mocks affect only
the isolated fixture. The real gateway keeps its own protection.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, nargs='+', required=True)
    parser.add_argument('--reference', type=Path, nargs='+', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert len(args.audio) == len(args.reference)
    os.environ.setdefault('APP_SESSION_SECRET', 'synthetic-file-verification-only-secret-value')
    from tests.test_media_security import MediaSecurityTests
    from server.api.routes import media
    from server.services import media_import
    from server.speech_cpp_asr import SpeechCppBatchTranscriber
    from server.core.config import settings
    from scripts.compare_media_import import normalized, distance
    import httpx

    specification = importlib.util.spec_from_file_location('server.services._file_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(baseline)
    report = dict(syntheticOnly=True, productionDataTouched=False, productionWorkerQueueMeasured=False, profiles=[])
    for path, reference_path in zip(args.audio, args.reference):
        with wave.open(str(path)) as wav:
            duration = wav.getnframes() / wav.getframerate()
        assert duration <= 180
        reference = normalized(reference_path.read_text())
        for profile, service in [('before', baseline), ('after', media_import)]:
            assert httpx.get('http://127.0.0.1:8005/api/health', timeout=5).json().get('activeConnections') == 0
            fixture = MediaSecurityTests()
            fixture.setUp()
            requests, events, stages = [], [], []
            began = time.monotonic()
            def factory():
                model = SpeechCppBatchTranscriber(api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url, model=settings.asr_model)
                send = model.client.send
                def inspected(request, *args, **kwargs):
                    started = time.monotonic()
                    response = send(request, *args, **kwargs)
                    requests.append(dict(url=str(request.url), status=response.status_code,
                        seconds=time.monotonic()-started, provider=response.headers.get('x-litellm-model-api-base'),
                        stop=response.headers.get('X-Speech-Stop') or response.headers.get('llm_provider-X-Speech-Stop')))
                    return response
                model.client.send = inspected
                return model
            fixture.app.state.runtime_resources.transcriber_factory = factory
            original = service.import_events
            def observed(*args, **kwargs):
                if profile == 'before':
                    kwargs.pop('run_inference', None)
                for value in original(*args, **kwargs):
                    events.append(dict(type=value['type'], elapsedSeconds=time.monotonic()-began,
                        phase=value.get('phase'), seq=value.get('seq')))
                    yield value
            decoder = service.decode_media
            def timed_decoder(*args, **kwargs):
                start = time.monotonic()
                try:
                    return decoder(*args, **kwargs)
                finally:
                    stages.append(dict(stage='decode', seconds=time.monotonic()-start))
            # import_events binds its default decoder at definition time.
            def with_decoder(*args, **kwargs):
                kwargs['decoder'] = timed_decoder
                return observed(*args, **kwargs)
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(baseline, 'settings', fixture.config))
                    stack.enter_context(patch.object(media, 'import_events', with_decoder))
                    response = fixture.upload(path.name, data=path.read_bytes(), language='auto' if 'mixed' in path.name else 'ja')
                    done = fixture.done(response)
                total = time.monotonic()-began
                text = '\n'.join(row['text'] for row in done['records'])
                previews = [e for e in events if e['type'] == 'segment']
                result = dict(profile=profile, input=path.name, inputSeconds=duration,
                    totalSeconds=total, firstVisibleSeconds=previews[0]['elapsedSeconds'] if previews else total,
                    asrRequests=len(requests), httpSeconds=sum(r['seconds'] for r in requests),
                    charErrorRate=distance(reference, normalized(text))/max(1,len(reference)),
                    text=text, intervals=[dict(startMs=r['tsStart'],endMs=r['tsEnd'],quality=r['quality']) for r in done['records']],
                    events=events, requests=requests, stages=stages)
                assert requests and all(r['status'] == 200 and r['stop'] == 'complete' for r in requests)
                assert all(r['url'] == settings.openai_base_url + '/audio/transcriptions' for r in requests)
                fixture.login(72)
                assert fixture.client.get(done['records'][0]['rawAudioPath']).status_code == 404
                assert fixture.client.post('/api/history', json={'runtimeSessionId': done['sessionId'], 'title': 'synthetic'}).status_code == 404
                fixture.login(71)
                saved = fixture.client.post('/api/history', json={'runtimeSessionId': done['sessionId'], 'title': 'synthetic'})
                assert saved.status_code == 200, saved.status_code
                result['ownershipAndHistoryPassed'] = True
                report['profiles'].append(result)
                args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps({k:result[k] for k in ('profile','input','totalSeconds','firstVisibleSeconds','asrRequests','charErrorRate')},ensure_ascii=False),flush=True)
            finally:
                fixture.doCleanups()


if __name__ == '__main__':
    main()
