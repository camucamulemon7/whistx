import test from 'node:test';
import assert from 'node:assert/strict';
import { createTranscriptController } from '../../web/src/controllers/transcript.js';

class Element {
  children = []; dataset = {}; className = ''; textContent = '';
  append(...children) {
    for (const child of children) { child.remove(); child.parent = this; this.children.push(child); }
  }
  prepend(child) { child.remove(); child.parent = this; this.children.unshift(child); }
  insertBefore(child, reference) { if (!reference) { this.append(child); return; } child.remove(); child.parent = this; this.children.splice(this.children.indexOf(reference), 0, child); }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
    this.parent = null;
  }
  querySelectorAll(selector) {
    return this.children.flatMap(child => [
      ...(child.className.split(' ').includes(selector.slice(1)) ? [child] : []),
      ...child.querySelectorAll(selector),
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

test('rendered live and history headings omit redundant sources without changing transcript records', () => {
  const originalDocument = globalThis.document;
  const root = new Element();
  const shared = { segmentId: 'display-0', track: 'display', speaker: '共有音声', text: '表示する本文', tsStart: 0, tsEnd: 5000 };
  const state = { segments: [shared], liveCapture: null };
  const controller = createTranscriptController({ state, logEl: root });
  const addRow = segment => {
    const row = new Element(), text = new Element();
    row.className = 'log-row'; row.dataset.segmentId = segment.segmentId;
    text.className = 'text'; row.append(text); root.append(row);
    return row;
  };
  const sharedRow = addRow(shared);
  try {
    globalThis.document = { createElement: () => new Element() };
    const saved = structuredClone(state.segments);
    controller.renderTranscriptParagraphs();
    const paragraph = root.querySelector('.transcript-paragraph');
    assert.equal(paragraph.querySelector('.transcript-paragraph-heading').textContent, '00:00');
    assert.equal(sharedRow.querySelector('.text').textContent, shared.text);
    assert.deepEqual(state.segments, saved);
    assert.equal(controller.extractTranscriptText(), '[共有音声] 表示する本文', 'existing copy/export metadata is preserved');
    const content = sharedRow.querySelector('.text');
    let text = content.textContent, writes = 0;
    Object.defineProperty(content, 'textContent', { get: () => text, set: value => { text = value; writes++; } });
    const rootAppend = root.append, paragraphAppend = paragraph.append;
    root.append = paragraph.append = () => { throw new Error('unchanged paragraphs must not be detached and appended again'); };
    controller.renderTranscriptParagraphs();
    assert.equal(writes, 0, 'unchanged committed text is not rewritten');
    root.append = rootAppend; paragraph.append = paragraphAppend;

    state.liveCapture = { streams: { mic: {}, display: {} } };
    controller.renderTranscriptParagraphs();
    assert.equal(paragraph.querySelector('.transcript-paragraph-heading').textContent, '00:00 · 共有音声', 'dual capture is identified before the mic speaks');
    assert.equal(root.querySelector('.transcript-paragraph'), paragraph);
    assert.equal(paragraph.querySelector('.log-row'), sharedRow);

    state.liveCapture = null;
    const mic = { segmentId: 'mic-0', track: 'mic', speaker: '自分（マイク）', text: 'マイクの本文', tsStart: 5000, tsEnd: 10000 };
    state.segments.push(mic); addRow(mic);
    const history = structuredClone(state.segments);
    controller.renderTranscriptParagraphs();
    assert.deepEqual(root.querySelectorAll('.transcript-paragraph-heading').map(node => node.textContent), ['00:00 · 共有音声', '00:05 · 自分（マイク）']);
    assert.deepEqual(state.segments, history, 'rendering stored dual-input history preserves its metadata');

    state.segments = [{ ...shared, speaker: '共有音声 · SPEAKER_00' }];
    controller.renderTranscriptParagraphs();
    assert.equal(root.querySelector('.transcript-paragraph-heading').textContent, '00:00 · SPEAKER_00');
    assert.equal(sharedRow.querySelector('.text').textContent, shared.text);
  } finally { globalThis.document = originalDocument; }
});
