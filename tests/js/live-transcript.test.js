import test from 'node:test';
import assert from 'node:assert/strict';
import { createLiveTranscript } from '../../web/src/meeting/live-transcript.js';
import { createTranscriptionController } from '../../web/src/controllers/transcription.js';

class Element {
  children = []; dataset = {}; hidden = true; textContent = '';
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
}
const document = { createElement: () => new Element() };
const textOf = element => element.textContent + element.children.map(textOf).join('');

test('live hypotheses replace by track, stay separate from records, and clear on finalization', () => {
  const root = new Element();
  const preview = createLiveTranscript({ root, document });
  preview.event({ type: 'partial', track: 'mic', segmentId: 'mic-0', text: '公開日は未定', stableText: '公開日は' });
  assert.equal(root.hidden, false);
  assert.equal(root.children[0].children[1].children[0].textContent, '公開日は');
  preview.event({ type: 'partial', track: 'mic', segmentId: 'mic-0', text: '公開日は来月', stableText: '公開日は' });
  preview.event({ type: 'partial', track: 'display', text: '<script>synthetic</script>', stableText: 'wrong prefix' });
  assert.equal(root.children.length, 2);
  assert.match(textOf(root), /公開日は来月/);
  assert.doesNotMatch(textOf(root), /未定|wrong prefix/);
  assert.equal(root.children[1].children[1].children[1].textContent, '<script>synthetic</script>');
  preview.event({ type: 'final', track: 'mic', segmentId: 'older-segment' });
  assert.equal(root.children.length, 2, 'an older final must not erase the newer hypothesis');
  preview.event({ type: 'final', track: 'mic', segmentId: 'mic-0' });
  assert.equal(root.children.length, 1);
  preview.event({ type: 'partial', track: 'display', text: '' });
  assert.equal(root.hidden, true);
  preview.event({ type: 'partial', track: 'mic', text: 'temporary' });
  preview.event({ type: 'info', message: 'ready' });
  assert.equal(root.hidden, true);
  preview.event({ type: 'partial', track: 'mic', text: 'temporary' });
  preview.event({ type: 'info', message: 'finalized' });
  assert.equal(root.hidden, true);
  preview.event({ type: 'partial', track: 'mic', text: 'temporary' });
  preview.reset();
  assert.equal(root.hidden, true);
});

test('legacy Whisper forwards finalization and ignores events from replaced sockets', async () => {
  const originalSocket = globalThis.WebSocket, originalLocation = globalThis.location;
  class Socket extends EventTarget {
    static OPEN = 1; static CONNECTING = 0; readyState = 1;
    emit(data) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) })); }
  }
  const state = { wsPath: '/ws/transcribe', runtimeSessionId: 'current' };
  const events = [], records = [];
  try {
    globalThis.WebSocket = Socket;
    globalThis.location = { protocol: 'http:', host: 'synthetic.test' };
    const controller = createTranscriptionController({ state, logWsEvent(){}, setStatus(){}, updateDownloadLinks(){},
      addLogLine: text => records.push(text), meetingWorkspace: { liveEvent: event => events.push(event) } });
    const socket = await controller.ensureSocket();
    socket.emit({type:'final',sessionId:'other',text:'stale'});
    socket.emit({type:'final',sessionId:'current',text:'source'});
    socket.emit({type:'info',sessionId:'current',message:'finalized'});
    assert.deepEqual(records, ['source']);
    assert.equal(state.runtimeSessionFinalized, true);
    assert.deepEqual(events.map(event => event.type), ['final', 'info']);
    state.ws = new Socket();
    socket.emit({type:'info',sessionId:'old',message:'ready'});
    socket.emit({type:'final',sessionId:'current',text:'late source'});
    assert.equal(state.runtimeSessionId, 'current');
    assert.deepEqual(records, ['source']);
    assert.equal(events.length, 2);
  } finally {
    if (originalSocket === undefined) delete globalThis.WebSocket; else globalThis.WebSocket = originalSocket;
    if (originalLocation === undefined) delete globalThis.location; else globalThis.location = originalLocation;
  }
});
