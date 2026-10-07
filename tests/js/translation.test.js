import test from 'node:test';
import assert from 'node:assert/strict';
import { createTranslation } from '../../web/src/meeting/translation.js';

test('committed revisions start promptly and merge pending work without concurrent requests', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const elements = new Map();
  for (const id of ['translationEnabled', 'translationLanguage', 'meetingTranslationTab', 'translationRows', 'translationStatus', 'translationRetry']) {
    elements.set(id, { value: 'en', scrollTop: 0, scrollHeight: 0, clientHeight: 0,
      addEventListener() {}, replaceChildren() {} });
  }
  t.mock.method(globalThis, 'fetch', () => new Promise(resolve => pending.push(resolve)));
  const previousDocument = globalThis.document, previousStorage = globalThis.localStorage;
  globalThis.document = { querySelector: selector => elements.get(selector.slice(1)) };
  globalThis.localStorage = { getItem: key => key === 'whistx_translation_enabled' ? '1' : 'en' };
  const pending = [];
  const finish = () => pending.shift()(new Response('data: {"type":"done"}\n\n'));
  const flush = () => new Promise(resolve => setImmediate(resolve));
  try {
    const translation = createTranslation({ getSource: () => ({ runtimeSessionId: 'synthetic' }), getAccess: () => 'ready', setView() {} });
    translation.event({ type: 'transcript_revision' });
    translation.event({ type: 'transcript_snapshot' });
    t.mock.timers.tick(1);
    assert.equal(pending.length, 1, 'start committed translation on the next event-loop turn');
    for (let i = 0; i < 10; i++) {
      translation.event({ type: 'transcript_revision' });
      t.mock.timers.tick(100);
      assert.equal(pending.length, 1, 'new revisions must not create concurrent translations');
    }
    finish();
    await flush();
    t.mock.timers.tick(1);
    assert.equal(pending.length, 1, 'process merged revisions as soon as the previous translation ends');
    finish();
    await flush();
    t.mock.timers.tick(1000);
    assert.equal(pending.length, 0, 'merged revisions must not add duplicate requests');
  } finally {
    while (pending.length) finish();
    await flush();
    globalThis.document = previousDocument;
    globalThis.localStorage = previousStorage;
  }
});
