import test from 'node:test';
import assert from 'node:assert/strict';
import { LiveCapture } from '../../web/src/meeting/live-capture.js';
import { safeMediaUrl } from '../../web/src/meeting/workspace.js';

globalThis.location = { origin: 'http://localhost' };
globalThis.WebSocket = { OPEN: 1 };

test('meeting media only links to authorized relative artifact routes', () => {
  assert.equal(safeMediaUrl('/api/history/123/screenshots/slide.png'), '/api/history/123/screenshots/slide.png');
  for (const url of ['javascript:alert(1)', '//evil.test/api/history/123/audio/a.wav', '/api/history/123/audio/../../secret', '/api/admin/users']) {
    assert.equal(safeMediaUrl(url), '');
  }
});

test('capture resends unacknowledged PCM and finalizes only after durable ACK', () => {
  const events = [];
  const sent = [];
  const capture = new LiveCapture({ path: '/ws/live', streams: { mic: {} }, start: {}, onEvent: e => events.push(e) });
  capture.ws = { readyState: 1, bufferedAmount: 0, send: raw => sent.push(JSON.parse(raw)) };
  capture.enqueue('mic', { pcm: new Int16Array(16000).buffer, sampleStart: 0 });
  capture.enqueue('mic', { pcm: new Int16Array(16000).buffer, sampleStart: 16000 });
  assert.equal(capture.pending.size, 2);
  assert.equal(sent.length, 2);
  capture.wantFinalize = true;
  capture.ack('mic', 0, 16000);
  assert.equal(capture.bufferBytes, 32000);
  assert.equal(sent.filter(e => e.type === 'stop').length, 0);
  capture.sent.clear();
  capture.pump();
  assert.equal(sent.at(-1).seq, 1);
  capture.ack('mic', 1, 32000);
  assert.equal(capture.pending.size, 0);
  assert.equal(sent.at(-1).type, 'stop');
  capture.pump();
  assert.equal(sent.filter(e => e.type === 'stop').length, 1);
});
