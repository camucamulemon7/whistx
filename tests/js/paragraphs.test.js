import test from 'node:test';
import assert from 'node:assert/strict';
import { transcriptParagraphs, transcriptJoiner, transcriptSpeakerLabel } from '../../web/src/meeting/paragraphs.js';

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

test('single-source headings omit capture labels while dual input and diarized speakers remain distinct', () => {
  const shared = { track: 'display', speaker: '共有音声', text: '元の本文' };
  const original = structuredClone(shared);
  assert.equal(transcriptSpeakerLabel(shared, false), '');
  assert.equal(transcriptSpeakerLabel({ track: 'display' }, false), '');
  assert.equal(transcriptSpeakerLabel(shared, true), '共有音声');
  assert.equal(transcriptSpeakerLabel({ track: 'mic', speaker: '自分（マイク）' }, false), '');
  assert.equal(transcriptSpeakerLabel({ track: 'mic', speaker: '自分（マイク）' }, true), '自分（マイク）');
  assert.equal(transcriptSpeakerLabel({ track: 'mic' }, true), 'マイク');
  assert.equal(transcriptSpeakerLabel({ track: 'display', speaker: '共有音声 · SPEAKER_00' }, false), 'SPEAKER_00');
  assert.equal(transcriptSpeakerLabel({ track: 'display', speaker: '共有音声 · SPEAKER_00' }, true), '共有音声 · SPEAKER_00');
  assert.equal(transcriptSpeakerLabel({ speaker: '田中' }, false), '田中');
  assert.equal(transcriptSpeakerLabel({ track: 'file' }, false), '');
  assert.deepEqual(shared, original, 'presentation cannot change source metadata or transcript content');
});
