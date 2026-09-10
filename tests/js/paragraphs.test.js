import test from 'node:test';
import assert from 'node:assert/strict';
import { transcriptParagraphs, transcriptJoiner } from '../../web/src/meeting/paragraphs.js';

test('capture chunks form continuous paragraphs, with breaks for speakers and pauses', () => {
  const rows = [
    { text: '次の', tsStart: 0, tsEnd: 5000, track: 'mic' },
    { text: '議題です。', tsStart: 5000, tsEnd: 10000, track: 'mic' },
    { text: '了解です。', tsStart: 10000, tsEnd: 15000, track: 'display' },
    { text: '再開します。', tsStart: 25000, tsEnd: 30000, track: 'display' },
  ];
  assert.deepEqual(transcriptParagraphs(rows).map(group => group.length), [2, 1, 1]);
  assert.equal(transcriptJoiner('次の', '議題'), '');
  assert.equal(transcriptJoiner('the next', 'topic'), ' ');
});

test('long paragraphs break at sentence endings, never an arbitrary packet boundary', () => {
  const rows = [{ text: 'あ'.repeat(460), tsStart: 0, tsEnd: 5000 },
    { text: 'です。', tsStart: 5000, tsEnd: 10000 }, { text: '次です。', tsStart: 10000, tsEnd: 15000 }];
  assert.deepEqual(transcriptParagraphs(rows).map(group => group.length), [2, 1]);
});
