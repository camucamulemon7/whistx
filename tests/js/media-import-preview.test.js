import assert from 'node:assert/strict';
import test from 'node:test';
import { createMediaImportPreview } from '../../web/src/meeting/media-import-preview.js';
import { transcriptQualityLabel } from '../../web/src/controllers/transcript.js';

class Element {
  children = []; hidden = true; textContent = ''; scrollHeight = 100; scrollTop = 0; clientHeight = 100;
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
}
const document = { createElement: () => new Element() };

test('completed file intervals preview safely, without modifying saved transcript state', () => {
  const root = new Element(), saved = [{ text: '既存の結果' }];
  const preview = createMediaImportPreview({ root, document });
  preview.event({ type: 'segment', seq: 0, text: '<script>合成テスト</script>', tsStart: 0, tsEnd: 75000 });
  assert.equal(root.hidden, false);
  assert.equal(root.children[1].children[1].textContent, '<script>合成テスト</script>');
  assert.equal(root.children[1].children[0].textContent, '00:00–01:15');
  preview.event({ type: 'segment', seq: 0, text: '重複イベント', tsStart: 0, tsEnd: 75000 });
  assert.equal(root.children.length, 2);
  assert.deepEqual(saved, [{ text: '既存の結果' }]);
  preview.reset();
  assert.equal(root.hidden, true);
  assert.equal(root.children.length, 0);
  preview.event({ type: 'segment', seq: 0, text: '再試行の結果', tsStart: 0, tsEnd: 1000 });
  assert.equal(root.children[1].children[1].textContent, '再試行の結果');
});

test('reading an older preview interval prevents forced scrolling', () => {
  const root = new Element();
  const preview = createMediaImportPreview({ root, document });
  preview.event({ type: 'segment', seq: 0, text: '前半', tsStart: 0, tsEnd: 1000 });
  root.scrollHeight = 1000; root.scrollTop = 0; root.clientHeight = 100;
  preview.event({ type: 'segment', seq: 1, text: '後半', tsStart: 1000, tsEnd: 2000 });
  assert.equal(root.scrollTop, 0);
});

test('file labels include legacy imports while recording quality retains its meaning', () => {
  assert.equal(transcriptQualityLabel({ quality: 'file_import' }), 'ファイル認識');
  assert.equal(transcriptQualityLabel({ quality: 'high_accuracy', track: 'file' }), 'ファイル認識');
  assert.equal(transcriptQualityLabel({ quality: 'high_accuracy', track: 'mic' }), '高精度');
  assert.equal(transcriptQualityLabel({ quality: 'realtime', track: 'display' }), 'リアルタイム');
  assert.equal(transcriptQualityLabel({ quality: 'high_accuracy', retainedRealtimeSegmentIds: ['r0'] }), '高精度・一部速報を保持');
});
