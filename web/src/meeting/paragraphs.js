// Capture packets remain individually addressable; reading paragraphs follow speech.
export function transcriptParagraphs(segments) {
  const groups = [];
  let previous = null;
  let length = 0;
  for (const segment of segments) {
    const speaker = segment.speaker || segment.track || "";
    const boundary = !previous || speaker !== (previous.speaker || previous.track || "")
      || Number(segment.tsStart) - Number(previous.tsEnd) > 8000
      || (length >= 450 && /[。！？.!?][」”"']?\s*$/.test(previous.text || ""));
    if (boundary) { groups.push([]); length = 0; }
    groups.at(-1).push(segment);
    length += (segment.text || "").length;
    previous = segment;
  }
  return groups;
}

export function transcriptJoiner(previous, next) {
  return /[\p{Script=Latin}\d][,;:.!?]?$/u.test(previous || "") && /^[\p{Script=Latin}\d]/u.test(next || "") ? " " : "";
}
