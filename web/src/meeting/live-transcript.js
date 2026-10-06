// Preview hypotheses separately from saved/copyable transcript records.
export function createLiveTranscript({ root, document = globalThis.document }) {
  const tracks = new Map();
  function render() {
    root.replaceChildren();
    for (const [track, value] of tracks) {
      const row = document.createElement('div');
      row.className = 'live-transcript-row';
      row.dataset.track = track;
      const label = document.createElement('small');
      label.textContent = `認識中（内容は更新されます）${track === 'display' ? ' · 共有音声' : track === 'mic' ? ' · マイク' : ''}`;
      const text = document.createElement('p');
      const stable = document.createElement('span');
      stable.className = 'live-transcript-stable';
      stable.textContent = value.stable;
      const hypothesis = document.createElement('span');
      hypothesis.className = 'live-transcript-hypothesis';
      hypothesis.textContent = value.text.slice(value.stable.length);
      text.append(stable, hypothesis);
      row.append(label, text);
      root.append(row);
    }
    root.hidden = tracks.size === 0;
  }
  function reset() { tracks.clear(); render(); }
  function event(value) {
    const track = value.track || 'mixed';
    if (value.type === 'partial') {
      if (value.text) {
        const text = String(value.text);
        const stable = text.startsWith(value.stableText || '') ? String(value.stableText || '') : '';
        tracks.set(track, { text, stable, segmentId: value.segmentId });
      } else tracks.delete(track);
      render();
    } else if (value.type === 'final') {
      const previous = tracks.get(track);
      if (previous && (!previous.segmentId || previous.segmentId === value.segmentId)) {
        tracks.delete(track); render();
      }
    } else if (value.type === 'info' && ['ready', 'finalized'].includes(value.message)) reset();
  }
  return { event, reset };
}
