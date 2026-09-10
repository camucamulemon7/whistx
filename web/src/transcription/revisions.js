// Only complete, sample-anchored source intervals can be superseded.
export function applyTranscriptRevision(records, event) {
  const row = event.record;
  if (!row || row.type !== "final" || !Array.isArray(event.replacesSegmentIds)) throw new Error("invalid_revision");
  if (records.some((item) => item.segmentId === row.segmentId)) return records;
  const { track, startSample: start, endSample: end } = event;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start ||
      row.track !== track || row.startSample !== start || row.endSample !== end ||
      row.tsStart !== Math.floor(start / 16) || row.tsEnd !== Math.floor(end / 16)) throw new Error("invalid_revision_range");
  const ids = new Set(event.replacesSegmentIds);
  const targets = records.filter((item) => item.track === track && item.startSample < end && item.endSample > start);
  if (!targets.length || ids.size !== targets.length || targets.some((item) => !ids.has(item.segmentId) ||
      item.startSample < start || item.endSample > end || item.quality === "high_accuracy")) throw new Error("stale_revision");
  return [...records.filter((item) => !ids.has(item.segmentId)), row].sort((a, b) => a.tsStart - b.tsStart || a.seq - b.seq);
}
