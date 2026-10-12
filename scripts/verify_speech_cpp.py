"""Exercise real ASR through isolated Whistx routes using supplied synthetic WAVs.

Run from the repository root with --audio (16 kHz mono PCM16, >=10 s) and
--video-audio (synthetic WAV for MP4 import). Existing SUMMARY/OPENWEBUI env
can optionally exercise translation with --translate. Never uses live app data.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import wave
from unittest.mock import patch


def sse(response):
    response.raise_for_status()
    events = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith('data:')]
    assert events and events[-1]['type'] == 'done', events
    assert not any(event['type'] == 'error' for event in events), events
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--video-audio', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='Qwen3-ASR-1.7B')
    parser.add_argument('--translate', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='whistx-real-asr-') as scratch:
        root = Path(scratch)
        os.environ.update(APP_ENV='development', APP_DATA_DIR=scratch,
            APP_DB_URL='sqlite:///' + str(root/'app.db'), APP_TRANSCRIPTS_DIR=str(root/'transcripts'),
            APP_HISTORY_DIR=str(root/'history'), APP_DEBUG_CHUNKS_DIR=str(root/'audio'),
            APP_SESSION_SECRET='isolated-synthetic-asr-session-secret-only',
            APP_PUBLIC_URL='http://testserver', APP_ALLOWED_HOSTS='testserver',
            APP_CONTAINER_HOST_GATEWAY='', ASR_BACKEND='speech_cpp', ASR_BASE_URL=args.base_url,
            ASR_MODEL=args.model, LANGFUSE_ENABLED='0', DIARIZATION_ENABLED='0',
            ASR_HIGH_ACCURACY_ENABLED='1', COSTLY_API_RATE_LIMIT_REQUESTS='1000')
        subprocess.run([sys.executable, '-m', 'alembic', 'upgrade', 'head'], check=True)
        sys.path.insert(0, str(Path.cwd()))
        from fastapi.testclient import TestClient
        from server import auth
        from server.core.application import create_app
        from server.db import db_session
        from server.models import User
        import httpx
        with db_session() as db:
            user = User(email='synthetic-asr@example.invalid', password_hash='synthetic', is_active=True)
            db.add(user)
            db.flush()
            token = auth.create_user_session(db, user=user, user_agent='synthetic-asr', ip_address='127.0.0.1')
        report = {'kind':'real-model', 'transport':'rolling-http', 'model':args.model,
                  'liveEvents':[], 'upstreamRequests':[]}
        original_send = httpx.Client.send
        def inspect_send(client, request, *values, **options):
            response = original_send(client, request, *values, **options)
            if request.url.path.endswith('/audio/transcriptions'):
                report['upstreamRequests'].append({'url':str(request.url),
                    'status':response.status_code,
                    'providerBase':response.headers.get('x-litellm-model-api-base'),
                    'stop':response.headers.get('X-Speech-Stop') or response.headers.get('llm_provider-X-Speech-Stop')})
            return response
        with patch('httpx.Client.send', inspect_send), TestClient(create_app()) as client:
            client.cookies.set('whistx_session', token)
            health = client.get('/api/health')
            assert health.status_code == 200, health.text
            assert health.json()['asrReady'], health.text
            report['health'] = health.json()
            with wave.open(str(args.audio)) as audio:
                assert (audio.getframerate(),audio.getnchannels(),audio.getsampwidth()) == (16000,1,2)
                pcm = audio.readframes(audio.getnframes())
            began = time.monotonic()
            with client.websocket_connect('/ws/transcribe/live', headers={'Origin':'http://testserver'}) as ws:
                ws.send_json({'type':'start','protocolVersion':2,'tracks':['mic'],'language':'auto',
                              'sessionId':'synthetic-real-asr','diarizationEnabled':False})
                ready = ws.receive_json()
                assert ready['message'] == 'ready', ready
                assert client.get('/api/health').json()['activeConnections'] == 1
                report['liveSessionId'] = ready['sessionId']
                failures = []
                def receive():
                    try:
                        while True:
                            event = ws.receive_json()
                            if event['type'] != 'capture_ack':
                                report['liveEvents'].append({**event,'elapsedSeconds':round(time.monotonic()-began,3)})
                            if event.get('message') == 'finalized':
                                return
                    except Exception as exc:
                        failures.append(type(exc).__name__)
                receiver = threading.Thread(target=receive, daemon=True)
                receiver.start()
                capture_start = time.monotonic()
                for seq, offset in enumerate(range(0,len(pcm),8000)):
                    delay = capture_start + offset/32000 - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    ws.send_json({'type':'audio','track':'mic','seq':seq,'sampleStart':offset//2,
                                  'pcm':base64.b64encode(pcm[offset:offset+8000]).decode()})
                ws.send_json({'type':'stop'})
                receiver.join(timeout=180)
                assert not receiver.is_alive(), 'live finalization timed out'
                assert not failures, failures
                # Keep TestClient's scope open until server cleanup finishes;
                # leaving it early cancels the handler's pending finally block.
                close = ws.receive()
                assert close['type'] == 'websocket.close', close
                cleanup_deadline = time.monotonic() + 3
                while client.get('/api/health').json()['activeConnections']:
                    assert time.monotonic() < cleanup_deadline, 'live connection lease was not released'
                    time.sleep(.1)
            assert client.get('/api/health').json()['activeConnections'] == 0
            events = report['liveEvents']
            assert not any(event['type']=='error' for event in events), events
            assert any(event['type']=='partial' and event.get('text') for event in events), 'missing partial'
            assert any(event['type']=='final' and event.get('text') for event in events), 'missing final'
            assert any(event['type']=='transcript_revision' for event in events), 'missing HQ'
            print('Real rolling HTTP recording + automatic high accuracy passed', flush=True)
            video = root/'synthetic.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=160x90:r=5',
                '-i',str(args.video_audio),'-shortest','-c:v','libx264','-c:a','aac',str(video)], check=True)
            events = sse(client.post('/api/media/transcribe',params={'filename':'synthetic.mp4','language':'ja'},
                                     content=video.read_bytes()))
            report['videoImport'] = events[-1]
            session_id = events[-1]['sessionId']
            assert events[-1]['records'], 'empty video transcript'
            report['manualRefinement'] = sse(client.post('/api/meeting/refine',json={'runtimeSessionId':session_id}))[-1]
            print('Real MP4 import + manual audio re-recognition passed', flush=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if args.translate:
                report['translations'] = {}
                for language in ('en','ja'):
                    translated = sse(client.post('/api/meeting/translate',json={
                        'runtimeSessionId':report['liveSessionId'],'language':language}))
                    assert any(event['type']=='translation' for event in translated), translated
                    report['translations'][language] = translated
                print('Real OpenWebUI LLM translation passed', flush=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
