import test from 'node:test';
import assert from 'node:assert/strict';
import { createTranscriptProgress } from '../../web/src/meeting/transcript-progress.js';

class Element {
  children = []; dataset = {}; className = ''; textContent = ''; writes = 0;
  append(...nodes) { for (const node of nodes) { node.remove(); node.parentNode = this; this.children.push(node); } }
  remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(node => node !== this); this.parentNode = null; }
  get lastChild() { return this.children.at(-1) || null; }
  setAttribute() {}
  appendData(text) { this.textContent += text; this.writes++; }
  querySelector(selector) { return this.children.find(node => node.className.split(' ').includes(selector.slice(1))) || null; }
}
const document = { createElement: () => new Element(), createTextNode: text => Object.assign(new Element(), { textContent: text }) };
const textOf = node => node.textContent + node.children.map(textOf).join('');
const partial = (text, stableText, extra = {}) => ({ type: 'partial', track: 'display', segmentId: 'display-0', tsStart: 0, tsEnd: 3000, text, stableText, ...extra });

test('agreed prefixes append immediately to one text node, without volatile tails or retyping', () => {
  const root = new Element(), empty = new Element(); empty.className = 'empty-state'; root.append(empty);
  const records = [], changes = [];
  const progress = createTranscriptProgress({ root, document, getRecords: () => records, onChange: value => changes.push(value) });
  progress.event(partial('公開日は来月です', ''));
  assert.equal(root.children.length, 1);
  progress.event(partial('公開日は来月です', '公開日は'));
  const paragraph = root.children[1], node = paragraph.lastChild, content = node.lastChild;
  assert.equal(textOf(node), '公開日は');
  assert.equal(empty.hidden, true);
  progress.event(partial('公開日は来月です', '公開日は来月'));
  assert.equal(paragraph.lastChild, node);
  assert.equal(node.lastChild, content);
  assert.equal(textOf(node), '公開日は来月');
  const writes = content.writes;
  for (const event of [partial('公開日は来月です', '公開日は来月'), partial('公開日時は変更されます', '公開'), partial('違う本文', '不正な接頭辞')]) progress.event(event);
  assert.equal(content.writes, writes);
  assert.equal(changes.length, 2, 'only new agreed text triggers scrolling work');
  assert.deepEqual(records, [], 'preview text never enters committed storage');
  progress.event({ type: 'info', message: 'finalized' });
  assert.equal(root.children.length, 1);
  assert.equal(empty.hidden, false);
});

test('progress joins the last paragraph, preserves reading position and survives earlier HQ changes', () => {
  const root = new Element(), paragraph = new Element(), row = new Element();
  paragraph.dataset.firstSegment = 'display-final'; paragraph.append(row); root.append(paragraph);
  let records = [{ segmentId: 'display-final', track: 'display', speaker: '共有音声', tsStart: 0, tsEnd: 5000, text: 'the next' }];
  const changes = [];
  const progress = createTranscriptProgress({ root, document, getRecords: () => records, isFollowing: () => false, onChange: value => changes.push(value) });
  progress.event(partial('topic is ready', 'topic', { segmentId: 'display-next', tsStart: 5000, tsEnd: 7500, speaker: '共有音声' }));
  assert.equal(root.children.length, 1, 'the progress text shares the existing lower paragraph');
  const node = paragraph.lastChild;
  assert.equal(textOf(node), ' topic');
  assert.deepEqual(changes, [{ follow: false }]);
  const newParagraph = new Element(); newParagraph.dataset.firstSegment = 'display-hq'; root.append(newParagraph); paragraph.remove();
  records = [{ ...records[0], segmentId: 'display-hq', text: 'the revised' }];
  progress.reconcile();
  assert.equal(newParagraph.lastChild, node);
  assert.equal(textOf(node), ' topic');
  progress.event(partial('stale old text', 'stale', { segmentId: 'display-final', tsStart: 0, tsEnd: 4000 }));
  assert.equal(newParagraph.lastChild, node, 'HQ coverage rejects old partial intervals');
  progress.event({ type: 'final', track: 'display', segmentId: 'older' });
  assert.equal(newParagraph.lastChild, node);
  progress.event({ type: 'final', track: 'display', segmentId: 'display-next' });
  assert.equal(newParagraph.children.length, 0);
});

test('two source drafts clear independently, and reconnect cannot restore stale progress', () => {
  const root = new Element();
  const progress = createTranscriptProgress({ root, document, getRecords: () => [], getSourceTracks: () => ['mic', 'display'] });
  progress.event(partial('<script>文字</script>', '<script>文字', { speaker: '共有音声' }));
  progress.event(partial('自分の発話です', '自分の発話', { track: 'mic', segmentId: 'mic-0', speaker: '自分（マイク）' }));
  assert.equal(root.children.length, 2);
  assert.match(textOf(root), /共有音声|自分（マイク）/);
  progress.event({ type: 'final', track: 'display', segmentId: 'display-0' });
  assert.equal(root.children.length, 1);
  assert.match(textOf(root), /自分の発話/);
  progress.event({ type: 'info', message: 'ready' });
  assert.equal(root.children.length, 0);
  progress.event(partial('次の発話', '次の', { track: 'mic', segmentId: 'mic-1', tsStart: 5000, tsEnd: 7500 }));
  progress.reset();
  assert.equal(root.children.length, 0);
});

test('legacy mixed finals retire their covered prefix even when the final has no track or segment ID', () => {
  const root = new Element(), records = [];
  const progress = createTranscriptProgress({ root, document, getRecords: () => records });
  progress.event(partial('missing clock', 'missing', { tsStart: undefined }));
  assert.equal(root.children.length, 0, 'a preview requires a source clock');
  progress.event(partial('legacy complete text', 'legacy', { track: 'mixed', segmentId: 'mixed-0' }));
  assert.equal(root.children.length, 1);
  records.push({ segmentId: '0', tsStart: 0, tsEnd: 5000, text: 'legacy complete text' });
  progress.reconcile();
  assert.equal(root.children.length, 0);
  assert.equal(records[0].text, 'legacy complete text');
});
