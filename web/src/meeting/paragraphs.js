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

// Default capture-source names are useful only when the transcript contains both inputs.
// Keep speaker identity and the stored metadata intact for history, copies and exports.
export function transcriptSpeakerLabel(segment, multipleSources) {
  const speaker = String(segment.speaker || "").trim();
  if (!multipleSources) {
    if (segment.track === "display") {
      if (speaker === "共有音声") return "";
      if (speaker.startsWith("共有音声 · ")) return speaker.slice("共有音声 · ".length);
    }
    if (segment.track === "mic" && speaker === "自分（マイク）") return "";
    return speaker;
  }
  return speaker || (segment.track === "display" ? "共有音声" : segment.track === "mic" ? "マイク" : "");
}
