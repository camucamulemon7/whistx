import assert from 'node:assert/strict';
import test from 'node:test';
import { MEDIA_MAX_BYTES, validateMediaFile, createMediaImportController } from '../../web/src/controllers/media-import.js';
import { createWorkspaceController } from '../../web/src/controllers/workspace.js';
import { createAppState } from '../../web/src/state/store.js';

test('file selection accepts audio/video and rejects empty, oversized or unsupported input', () => {
  for (const name of ['会議.WAV', '会議.mp4', 'test.mov', 'test.webm', 'test.flac']) {
    assert.equal(validateMediaFile({ name, size: 1024 }), '');
  }
  assert.match(validateMediaFile({ name: 'empty.wav', size: 0 }), /空/);
  assert.match(validateMediaFile({ name: 'large.mp4', size: MEDIA_MAX_BYTES + 1 }), /256 MB/);
  assert.match(validateMediaFile({ name: 'remote.m3u8', size: 100 }), /対応/);
  assert.match(validateMediaFile({ name: 'a'.repeat(240) + '.wav', size: 100 }), /240/);
});

test('an active media import locks destructive actions and protects page navigation', () => {
  const state = createAppState({ chunkDefaultSeconds: 30, speakerMax: 8, defaultPromptTemplate: '' });
  const workspace = createWorkspaceController({ state, refinementController: null });
  state.mediaImportController = new AbortController();
  assert.equal(workspace.isRecordingInteractionLocked(), true);
  assert.equal(workspace.shouldProtectWorkspaceFromUnload(), true);
  state.mediaImportController = null;
  assert.equal(workspace.isRecordingInteractionLocked(), false);
});

test('file preview does not replace prior results; cancellation, retry and success commit safely', async () => {
  class Element {
    children = []; textContent = ''; hidden = true; value = 'ja'; disabled = false; dataset = {};
    listeners = {}; classList = { add() {}, remove() {} }; files = [];
    addEventListener(type, callback) { this.listeners[type] = callback; }
    setAttribute() {} removeAttribute() {} append(...children) { this.children.push(...children); }
    replaceChildren() { this.children = []; }
    click() { return this.listeners.click?.(); }
  }
  const nodes = new Map();
  const node = selector => { if (!nodes.has(selector)) nodes.set(selector, new Element()); return nodes.get(selector); };
  const oldDocument = globalThis.document, oldFetch = globalThis.fetch;
  globalThis.document = { querySelector: node, querySelectorAll: () => [], addEventListener() {}, createElement: () => new Element() };
  let stream, requests = 0, commits = 0;
  globalThis.fetch = async (_, options) => {
    requests++;
    return new Response(new ReadableStream({ start(controller) {
      stream = controller;
      options.signal.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
    } }), { headers: { 'Content-Type': 'text/event-stream' } });
  };
  const emit = value => stream.enqueue(new TextEncoder().encode('data: ' + JSON.stringify(value) + '\n\n'));
  const settle = () => new Promise(resolve => setImmediate(resolve));
  const state = { auth: { authenticated: true }, asrAvailable: true, segments: [{ text: '以前の結果' }], runtimeSessionId: 'old-runtime' };
  const noop = () => {};
  const deps = { state, startBtn: new Element(), languageEl: new Element(), sharedVocabularyEl: { value: '' }, promptEl: { value: '' }, saveTitleInputEl: new Element(),
    runtimeUi: { appLocked: false }, isRecordingInteractionLocked: () => Boolean(state.mediaImportController), confirmWorkspaceDiscard: () => true,
    updateSaveControls: noop, setSessionSettingsLocked: noop, updateDownloadLinks: noop, renderHistoryList: noop, syncUnloadProtection: noop,
    commitNewRecordingWorkspace() { commits++; state.segments = []; }, addLogLine(text) { state.segments.push({ text }); }, markWorkspaceDirty: noop, showToast: noop,
    meetingWorkspace: { busy: false, setView: noop, refresh: noop } };
  try {
    createMediaImportController(deps);
    node('#mediaFileInput').files = [{ name: 'synthetic.wav', size: 128 }]; node('#mediaFileInput').listeners.change();
    let operation = node('#mediaImportStart').click(); await settle();
    emit({ type: 'segment', seq: 0, text: '途中の結果', tsStart: 0, tsEnd: 1000 }); await settle();
    assert.equal(node('#mediaImportPreview').hidden, false);
    assert.deepEqual(state.segments, [{ text: '以前の結果' }]); assert.equal(state.runtimeSessionId, 'old-runtime'); assert.equal(commits, 0);
    await node('#mediaImportStart').click(); assert.equal(requests, 1);
    node('#mediaImportCancel').click(); await operation;
    assert.equal(node('#mediaImportPreview').hidden, true); assert.deepEqual(state.segments, [{ text: '以前の結果' }]);
    operation = node('#mediaImportStart').click(); await settle();
    emit({ type: 'segment', seq: 0, text: '後半失敗前の結果', tsStart: 0, tsEnd: 1000 }); await settle();
    emit({ type: 'error', error: 'media_recognition_timeout' }); await operation;
    assert.equal(commits, 0); assert.equal(node('#mediaImportPreview').hidden, true); assert.match(node('#mediaImportStatus').textContent, /時間切れ/);
    operation = node('#mediaImportStart').click(); await settle();
    emit({ type: 'segment', seq: 0, text: '再試行の結果', tsStart: 0, tsEnd: 1000 }); await settle();
    assert.equal(node('#mediaImportPreview').children[1].children[1].textContent, '再試行の結果');
    emit({ type: 'done', sessionId: 'new-runtime', title: '合成テスト', records: [{ text: '完成した結果', tsStart: 0, tsEnd: 1000, seq: 0 }] });
    stream.close(); await operation;
    assert.equal(commits, 1); assert.equal(state.runtimeSessionId, 'new-runtime'); assert.deepEqual(state.segments, [{ text: '完成した結果' }]);
    assert.equal(node('#mediaImportPreview').hidden, true); assert.match(node('#mediaImportStatus').textContent, /完了/);
  } finally {
    state.mediaImportController?.abort(); globalThis.document = oldDocument; globalThis.fetch = oldFetch;
  }
});
