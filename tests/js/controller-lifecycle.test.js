import assert from 'node:assert/strict';
import test from 'node:test';
import { createFeatureContext } from '../../web/src/controllers/dependencies.js';
import { createMediaImportController } from '../../web/src/controllers/media-import.js';
import { applyHistoryDetailPayload } from '../../web/src/history/state.js';

test('feature interfaces keep live callbacks and limit state mutation', () => {
  const app = { state: { recordingPhase: 'idle' }, setStatus: () => 'first' };
  const recording = createFeatureContext(app, 'recording');
  assert.equal(recording.setStatus(), 'first');
  app.setStatus = () => 'updated';
  assert.equal(recording.setStatus(), 'updated');
  recording.state.recordingPhase = 'recording';
  assert.equal(app.state.recordingPhase, 'recording');
  app.state = { recordingPhase: 'idle' };
  assert.equal(recording.state.recordingPhase, 'idle');
  assert.throws(() => { recording.state.unownedField = true; }, TypeError);
  assert.throws(() => { recording.unownedCallback = () => {}; }, TypeError);
  assert.equal(recording.unownedCallback, undefined);
  assert.throws(() => createFeatureContext(app, 'unknown'), /Unknown/);
});

test('all feature interfaces initialize and history helpers can update their owned state', () => {
  const app = { state: { history: {}, log: ['old'], segments: ['old'] } };
  for (const feature of ['recording', 'media-capture', 'history', 'transcript', 'media-import']) {
    assert.ok(createFeatureContext(app, feature));
  }
  applyHistoryDetailPayload(createFeatureContext(app, 'history').state, { id: 'selected' });
  assert.equal(app.state.history.selectedId, 'selected');
  assert.equal(app.state.viewingHistoryId, 'selected');
  assert.equal(app.state.savedHistoryId, 'selected');
  assert.deepEqual(app.state.log, []);
  assert.deepEqual(app.state.segments, []);
});

function fixture() {
  let listeners = 0;
  class Element {
    children = []; textContent = ''; hidden = true; value = 'ja'; disabled = false; dataset = {}; files = [];
    handlers = new Map(); classList = { add() {}, remove() {} };
    addEventListener(type, callback) {
      if (!this.handlers.has(type)) this.handlers.set(type, new Set());
      this.handlers.get(type).add(callback); listeners++;
    }
    removeEventListener(type, callback) { if (this.handlers.get(type)?.delete(callback)) listeners--; }
    setAttribute() {} removeAttribute() {} append(...children) { this.children.push(...children); }
    replaceChildren() { this.children = []; }
    fire(type) { return Promise.all([...this.handlers.get(type) || []].map(callback => callback({}))); }
  }
  const nodes = new Map();
  const node = selector => { if (!nodes.has(selector)) nodes.set(selector, new Element()); return nodes.get(selector); };
  const documentEvents = new Element();
  const original = globalThis.document;
  globalThis.document = { querySelector: node, querySelectorAll: () => [], createElement: () => new Element(),
    addEventListener: documentEvents.addEventListener.bind(documentEvents), removeEventListener: documentEvents.removeEventListener.bind(documentEvents) };
  let commits = 0;
  const noop = () => {};
  const state = { auth: { authenticated: true }, asrAvailable: true };
  const deps = { state, startBtn: new Element(), languageEl: new Element(), sharedVocabularyEl: { value: '' }, promptEl: { value: '' }, saveTitleInputEl: new Element(),
    runtimeUi: { appLocked: false }, isRecordingInteractionLocked: () => Boolean(state.mediaImportController), confirmWorkspaceDiscard: () => true,
    updateSaveControls: noop, setSessionSettingsLocked: noop, updateDownloadLinks: noop, renderHistoryList: noop, syncUnloadProtection: noop,
    commitNewRecordingWorkspace() { commits++; }, addLogLine: noop, markWorkspaceDirty: noop, showToast: noop,
    meetingWorkspace: { busy: false, setView: noop, refresh: noop } };
  return { deps, node, listeners: () => listeners, commits: () => commits,
    restore() { if (original === undefined) delete globalThis.document; else globalThis.document = original; } };
}

test('media import mount is idempotent and dispose removes listeners before remount', () => {
  const ui = fixture();
  const controller = createMediaImportController(ui.deps);
  try {
    assert.equal(ui.listeners(), 0);
    controller.mount();
    const mounted = ui.listeners();
    assert.ok(mounted > 0);
    controller.mount();
    assert.equal(ui.listeners(), mounted);
    controller.dispose();
    controller.dispose();
    assert.equal(ui.listeners(), 0);
    controller.mount();
    assert.equal(ui.listeners(), mounted);
  } finally { controller.dispose(); ui.restore(); }
});

test('dispose clears an upload timer, aborts its request and ignores late completion after remount', async () => {
  const ui = fixture(), pending = [], timers = new Set();
  const original = { fetch: globalThis.fetch, setInterval: globalThis.setInterval, clearInterval: globalThis.clearInterval };
  globalThis.setInterval = () => { const timer = {}; timers.add(timer); return timer; };
  globalThis.clearInterval = timer => timers.delete(timer);
  globalThis.fetch = (_, options) => new Promise(resolve => pending.push({ resolve, signal: options.signal }));
  const controller = createMediaImportController(ui.deps);
  const select = async () => {
    ui.node('#mediaFileInput').files = [{ name: 'synthetic.wav', size: 128 }];
    await ui.node('#mediaFileInput').fire('change');
  };
  const complete = index => pending[index].resolve(new Response('data: '+JSON.stringify({ type: 'done', sessionId: 'new', title: 'test', records: [] })+'\n\n'));
  let first, second;
  try {
    controller.mount(); await select();
    first = ui.node('#mediaImportStart').fire('click');
    assert.equal(timers.size, 1);
    controller.dispose();
    assert.equal(timers.size, 0);
    assert.equal(pending[0].signal.aborted, true);
    assert.equal(ui.deps.state.mediaImportController, null);
    controller.mount(); await select();
    second = ui.node('#mediaImportStart').fire('click');
    const current = ui.deps.state.mediaImportController;
    complete(0); await first;
    assert.equal(ui.deps.state.mediaImportController, current);
    assert.equal(ui.commits(), 0);
    complete(1); await second;
    assert.equal(ui.commits(), 1);
    assert.equal(ui.deps.state.mediaImportController, null);
    assert.equal(timers.size, 0);
  } finally {
    controller.dispose();
    pending.forEach((_, index) => complete(index));
    await Promise.allSettled([first, second]);
    Object.assign(globalThis, original); ui.restore();
  }
});
