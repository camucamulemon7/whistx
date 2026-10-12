import test from 'node:test';
import assert from 'node:assert/strict';
import { createTranscriptionController } from '../../web/src/controllers/transcription.js';

test('legacy Whisper forwards finalization and ignores events from replaced sockets', async () => {
  const originalSocket = globalThis.WebSocket, originalLocation = globalThis.location;
  class Socket extends EventTarget {
    static OPEN = 1; static CONNECTING = 0; readyState = 1;
    emit(data) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) })); }
  }
  const state = { wsPath: '/ws/transcribe', runtimeSessionId: 'current' };
  const events = [], records = [], progressEvents = [];
  try {
    globalThis.WebSocket = Socket;
    globalThis.location = { protocol: 'http:', host: 'synthetic.test' };
    const controller = createTranscriptionController({ state, logWsEvent(){}, setStatus(){}, updateDownloadLinks(){},
      addLogLine: text => records.push(text), applyTranscriptProgress: event => progressEvents.push(event),
      meetingWorkspace: { liveEvent: event => events.push(event) } });
    const socket = await controller.ensureSocket();
    socket.emit({type:'partial',sessionId:'current',segmentId:'mixed-0',text:'source tail',stableText:'source'});
    socket.emit({type:'final',sessionId:'other',text:'stale'});
    socket.emit({type:'final',sessionId:'current',text:'source'});
    socket.emit({type:'info',sessionId:'current',message:'finalized'});
    assert.deepEqual(records, ['source']);
    assert.equal(state.runtimeSessionFinalized, true);
    assert.deepEqual(events.map(event => event.type), ['final', 'info']);
    assert.deepEqual(progressEvents.map(event => event.type), ['partial', 'final', 'info'], 'Whisper finals replace progress before saving');
    state.ws = new Socket();
    socket.emit({type:'info',sessionId:'old',message:'ready'});
    socket.emit({type:'final',sessionId:'current',text:'late source'});
    assert.equal(state.runtimeSessionId, 'current');
    assert.deepEqual(records, ['source']);
    assert.equal(events.length, 2);
    assert.equal(progressEvents.length, 3, 'replaced sockets cannot change inline progress');
  } finally {
    if (originalSocket === undefined) delete globalThis.WebSocket; else globalThis.WebSocket = originalSocket;
    if (originalLocation === undefined) delete globalThis.location; else globalThis.location = originalLocation;
  }
});
