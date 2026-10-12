import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createAppState } from '../../web/src/state/store.js';
import { createRecordingController } from '../../web/src/controllers/recording.js';
import { createTranscriptController } from '../../web/src/controllers/transcript.js';
import { createWorkspaceController } from '../../web/src/controllers/workspace.js';
import { createHistoryController } from '../../web/src/controllers/history.js';
import { createMeetingWorkspace } from '../../web/src/meeting/workspace.js';
import { LiveCapture } from '../../web/src/meeting/live-capture.js';

class Element extends EventTarget {
  children = []; dataset = {}; className = ''; textContent = ''; hidden = false; value = '';
  clientWidth = 1200; clientHeight = 100; scrollHeight = 100; scrollTop = 0;
  attributes = new Map(); style = { setProperty() {} };
  classList = {
    add: (...names) => { this.className += ' ' + names.join(' '); },
    remove: (...names) => { this.className = this.className.split(' ').filter(name => !names.includes(name)).join(' '); },
    contains: name => this.className.split(' ').includes(name),
    toggle: (name, value) => { if (value) this.classList.add(name); else this.classList.remove(name); },
  };
  append(...children) { for (const child of children) { child.remove(); child.parent = this; this.children.push(child); } }
  appendChild(child) { this.append(child); }
  prepend(child) { child.remove(); child.parent = this; this.children.unshift(child); }
  insertBefore(child, reference) { if (!reference) { this.append(child); return; } child.remove(); child.parent = this; this.children.splice(this.children.indexOf(reference), 0, child); }
  get parentNode() { return this.parent || null; }
  get lastChild() { return this.children.at(-1) || null; }
  appendData(value) { this.textContent += value; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); this.parent = null; }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  set innerHTML(value) {
    this.replaceChildren();
    if (value.includes('class="empty-state"')) { const empty = new Element(); empty.className = 'empty-state'; this.append(empty); }
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
  getAttribute(name) { return this.attributes.get(name); }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
  closest() { return null; }
  querySelectorAll(selector) {
    const name = selector.match(/^\.([\w-]+)$/)?.[1];
    return this.children.flatMap(child => [...(name && child.classList.contains(name) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
const textOf = node => node.textContent + node.children.map(textOf).join('');

for (const source of ['display', 'both']) {
  test(`${source} capture uses only the main transcript through empty state, HQ, reconnect and save`, async t => {
    const html = await readFile(new URL('../../web/index.html', import.meta.url), 'utf8');
    const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => ['#' + match[1], new Element()]));
    assert.equal(elements.has('#liveTranscript'), false);
    const node = selector => elements.get(selector) || null;
    const original = Object.fromEntries(['document', 'window', 'localStorage', 'fetch', 'requestAnimationFrame'].map(key => [key, globalThis[key]]));
    const originalStart = LiveCapture.prototype.start, originalStop = LiveCapture.prototype.stop;
    t.after(() => {
      for (const [key, value] of Object.entries(original)) { if (value === undefined) delete globalThis[key]; else globalThis[key] = value; }
      LiveCapture.prototype.start = originalStart; LiveCapture.prototype.stop = originalStop;
    });
    globalThis.document = { querySelector: node, querySelectorAll: () => [], createElement: () => new Element(), createTextNode: text => { const node = new Element(); node.textContent = text; return node; }, documentElement: new Element(), body: new Element() };
    globalThis.window = { addEventListener() {}, removeEventListener() {} };
    globalThis.localStorage = { getItem: () => null, setItem() {} };
    globalThis.requestAnimationFrame = callback => callback();
    const saved = [];
    globalThis.fetch = async (url, options = {}) => {
      if (url === '/api/history' && options.method === 'POST') {
        saved.push(JSON.parse(options.body));
        return new Response(JSON.stringify({ history: { id: 'synthetic-history' } }));
      }
      return new Response(JSON.stringify(String(url).startsWith('/api/history') ? { items: [], total: 0 } : { turns: [], images: [] }));
    };
    LiveCapture.prototype.start = async function () {
      this.running = true; this.sessionId = 'synthetic-' + source;
      this.onEvent({ type: 'info', message: 'ready', sessionId: this.sessionId, tracks: Object.fromEntries(Object.keys(this.streams).map(track => [track, {}])) });
      this.onEvent({ type: 'transcript_snapshot', records: [] });
    };
    LiveCapture.prototype.stop = async function () { this.running = false; this.onEvent({ type: 'info', message: 'finalized', sessionId: this.sessionId }); };
    const state = createAppState({ chunkDefaultSeconds: 30, speakerMax: 8, defaultPromptTemplate: '' });
    state.auth.authenticated = true; state.meetingInsights = true;
    const label = new Element(); label.className = 'record-label'; node('#startBtn').append(label);
    const notices = [];
    const dependencies = {
      state, runtimeUi: { appLocked: false }, logEl: node('#log'), startBtn: node('#startBtn'), saveBtn: node('#saveBtn'), clearBtn: node('#clearBtn'),
      dlTxt: node('#dlTxt'), dlJsonl: node('#dlJsonl'), dlZip: node('#dlZip'),
      segmentCountEl: node('#segmentCount'), saveStateBadgeEl: node('#saveStateBadge'), presetButtons: [], promptEl: node('#prompt'),
      prepareInputStream: async () => { state.recordingAudioSource = source; state.micStream = {}; state.displayStream = {}; return {}; },
      hasAudioTrack: stream => Boolean(stream), setupVad: async () => {}, selectedLanguage: () => null, resolveDiarizationStartOptions: () => ({}),
      markWorkspaceClean: () => { state.workspaceDirty = false; }, markWorkspaceDirty: () => { state.workspaceDirty = true; },
      markProofreadStale() {}, renderHistoryList() {}, renderAudioLevel() {}, setSummary() {}, setProofread() {}, cleanupMedia() {},
      updateRecordingTelemetry() {}, setStatus() {}, captureDisplayScreenshot() {}, logClientEvent() {}, buildAutoSaveTitle: () => '合成テスト',
      updateDiarizationSpeakerUi() {}, updateSharedVocabularyMeta() {},
      showToast: message => notices.push(message),
    };
    const transcript = createTranscriptController(dependencies);
    const workspace = createWorkspaceController(dependencies);
    Object.assign(dependencies, transcript, workspace);
    dependencies.meetingWorkspace = createMeetingWorkspace({ getSource: () => state.runtimeSessionId ? { runtimeSessionId: state.runtimeSessionId } : null });
    const recording = createRecordingController(dependencies);
    await recording.startLiveRecording({ liveWsPath: '/ws/live', capturePacketMs: 250 }, source);
    const emit = event => state.liveCapture.onEvent(event);
    assert.ok(node('#log').querySelector('.empty-state'));
    for (const track of Object.keys(state.liveCapture.streams)) emit({ type: 'partial', track, text: '暫定本文だけの合成テスト' });
    assert.equal(state.segments.length, 0);
    assert.equal(node('#log').querySelectorAll('.log-row').length, 0);
    assert.ok(node('#log').querySelector('.empty-state'), 'hypotheses do not replace the empty main transcript');
    assert.equal(node('#liveTranscript'), null);
    emit({ type: 'partial', track: 'display', segmentId: 'display-0', text: '小刻みに表示する本文', stableText: '小刻みに', tsStart: 0, tsEnd: 2500, speaker: '共有音声' });
    const progress = node('#log').querySelector('.transcript-progress');
    assert.equal(textOf(progress), '小刻みに');
    emit({ type: 'partial', track: 'display', segmentId: 'display-0', text: '小刻みに表示する本文です', stableText: '小刻みに表示する', tsStart: 0, tsEnd: 3750, speaker: '共有音声' });
    assert.equal(node('#log').querySelector('.transcript-progress'), progress, 'agreed characters extend the same DOM node');
    assert.equal(textOf(progress), '小刻みに表示する');
    assert.equal(state.segments.length, 0, 'visible progress is never a finalized record');
    assert.equal(transcript.extractTranscriptText(), '', 'copy/export extraction omits agreed but provisional text');
    assert.equal(node('#saveBtn').disabled, true);
    emit({ type: 'connection', message: '再接続中' });
    assert.equal(node('#liveConnection').textContent, '再接続中');
    emit({ type: 'error', message: 'transcription_failed' });
    assert.match(node('#liveConnection').textContent, /受信済みの音声は保存/);

    const rows = [0, 1].map(index => ({ type: 'final', track: 'display', segmentId: `display-${index}`, seq: index,
      text: `確定した発話${index}`, startSample: index * 80000, endSample: (index + 1) * 80000,
      tsStart: index * 5000, tsEnd: (index + 1) * 5000, speaker: '共有音声', quality: 'realtime' }));
    for (const row of rows) emit(row);
    assert.equal(node('#log').querySelector('.transcript-progress'), null);
    assert.doesNotMatch(textOf(node('#log')), /小刻みに表示する/, 'the authoritative final replaces the whole provisional prefix');
    if (source === 'both') {
      const mic = { ...rows[0], track: 'mic', segmentId: 'mic-0', seq: 2, text: '別入力の確定発話', speaker: '自分（マイク）' };
      emit(mic);
    }
    assert.equal(node('#log').querySelector('.empty-state'), null);
    assert.match(textOf(node('#log')), /確定した発話0.*確定した発話1/s);
    emit({ type: 'partial', track: 'display', segmentId: 'display-2', text: '次の発話が続きます', stableText: '次の発話', tsStart: 10000, tsEnd: 12500, speaker: '共有音声' });
    const nextProgress = node('#log').querySelector('.transcript-progress');
    const hq = { ...rows[0], segmentId: 'display-hq', text: '高精度に置換した発話', endSample: 160000, tsEnd: 10000, quality: 'high_accuracy' };
    emit({ type: 'transcript_revision', track: 'display', startSample: 0, endSample: 160000, replacesSegmentIds: rows.map(row => row.segmentId), record: hq });
    emit(rows[0]); // A late realtime result cannot resurrect a replaced record.
    emit({ type: 'partial', track: 'display', text: '次の暫定本文' });
    emit({ type: 'hq_status', state: 'completed', tsStart: 0, tsEnd: 10000 });
    assert.match(textOf(node('#log')), /高精度に置換した発話/);
    assert.doesNotMatch(textOf(node('#log')), /確定した発話[01]|暫定本文/);
    assert.equal(node('#hqStatus').hidden, false);
    assert.equal(node('#log').querySelector('.transcript-progress'), nextProgress, 'HQ for earlier audio retains current progress');
    assert.match(node('#hqStatus').textContent, /高精度認識を反映/);
    if (source === 'both') assert.match(textOf(node('#log')), /別入力の確定発話/);
    const snapshot = structuredClone(state.segments);
    emit({ type: 'info', message: 'ready', sessionId: state.runtimeSessionId, tracks: Object.fromEntries(Object.keys(state.liveCapture.streams).map(track => [track, {}])) });
    emit({ type: 'transcript_snapshot', records: structuredClone(snapshot) });
    emit({ type: 'transcript_snapshot', records: structuredClone(snapshot) });
    assert.deepEqual(state.segments, snapshot);
    assert.equal(node('#log').querySelector('.transcript-progress'), null, 'ready resets provisional text before restoring the snapshot');
    assert.equal(node('#log').querySelectorAll('.log-row').length, snapshot.length);
    assert.equal(node('#liveTranscript'), null, 'reconnection cannot recreate the removed card');
    const tail = { ...rows[0], segmentId: 'display-tail', seq: 3, startSample: 160000, endSample: 176000,
      tsStart: 10000, tsEnd: 11000, text: '停止直前の文字も残す', screenshotPath: '', rawAudioPath: '', audioPath: '' };
    emit({ type: 'partial', track: 'display', segmentId: tail.segmentId, tsStart: 10000, tsEnd: 10500,
      text: tail.text, stableText: '停止直前の文字', speaker: '共有音声' });
    state.liveCapture.stop = async function () {
      this.onEvent(tail); this.running = false;
      this.onEvent({ type: 'info', message: 'finalized', sessionId: this.sessionId });
    };
    await recording.finalizeLiveRecording();
    const finalized = [...snapshot, tail];
    assert.deepEqual(state.segments, finalized);
    assert.equal(node('#log').querySelector('.transcript-progress'), null);
    assert.match(textOf(node('#log')), /停止直前の文字も残す/);
    assert.equal(state.runtimeSessionFinalized, true);
    assert.equal(state.recordingPhase, 'idle');
    assert.match(node('#liveConnection').textContent, /録音完了/);
    assert.equal(node('#saveBtn').disabled, false);
    await createHistoryController(dependencies).saveCurrentHistory();
    assert.equal(saved.length, 1);
    assert.equal(saved[0].runtimeSessionId, 'synthetic-' + source);
    assert.equal(state.savedHistoryId, 'synthetic-history');
    assert.deepEqual(state.segments, finalized, 'finalization and saving preserve all committed text, including the stop tail');
    assert.ok(notices.includes('保存しました'));
  });
}
