import test from 'node:test';
import assert from 'node:assert/strict';
import { applyTranscriptRevision } from '../../web/src/transcription/revisions.js';

const records = ['mic', 'display'].flatMap(track => [0, 1].map(i => ({
  type: 'final', track, segmentId: `${track}-${i}`, seq: i, text: '速報',
  startSample: i * 16000, endSample: (i+1) * 16000, tsStart: i*1000, tsEnd: (i+1)*1000, quality: 'realtime',
})));
const event = { track: 'mic', startSample: 0, endSample: 32000, replacesSegmentIds: ['mic-0', 'mic-1'],
  record: { type: 'final', track: 'mic', segmentId: 'mic-hq', seq: 0, text: 'APIをreviewします。',
    startSample: 0, endSample: 32000, tsStart: 0, tsEnd: 2000, quality: 'high_accuracy' } };

test('revision preserves the other audio track and the sample clock, and is idempotent', () => {
  const updated = applyTranscriptRevision(records, event);
  assert.equal(updated.length, 3);
  assert.deepEqual(updated.filter(row => row.track === 'display'), records.filter(row => row.track === 'display'));
  assert.deepEqual(applyTranscriptRevision(updated, event), updated);
  assert.equal(updated.find(row => row.segmentId === 'mic-hq').text, 'APIをreviewします。');
  assert.equal(records.length, 4);
});

test('stale IDs, partial intervals and timestamp changes cannot overwrite live text', () => {
  for (const changed of [
    { replacesSegmentIds: ['mic-0'] },
    { replacesSegmentIds: ['mic-0', 'display-1'] },
    { record: { ...event.record, tsEnd: 1900 } },
    { endSample: 16000 },
  ]) assert.throws(() => applyTranscriptRevision(records, { ...event, ...changed }));
});
