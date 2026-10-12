import { formatTimestamp } from '../ui/format.js';

// A temporary view only: no runtime ID, downloads, audio, or saved segments.
export function createMediaImportPreview({ root, document = globalThis.document }) {
  const seen = new Set();
  function reset() {
    seen.clear(); root.replaceChildren(); root.hidden = true;
  }
  function event(value) {
    if (value.type !== 'segment' || !value.text || seen.has(value.seq)) return;
    seen.add(value.seq);
    const follow = root.hidden || root.scrollHeight - root.scrollTop - root.clientHeight < 48;
    if (root.hidden) {
      const label = document.createElement('p'); label.className = 'media-preview-label';
      label.textContent = '取込中のプレビュー · 完了後に結果を置き換えます'; root.append(label);
    }
    const row = document.createElement('p'); row.className = 'media-preview-row';
    const range = document.createElement('span'); range.className = 'media-preview-time';
    range.textContent = `${formatTimestamp(value.tsStart)}–${formatTimestamp(value.tsEnd)}`;
    const text = document.createElement('span'); text.textContent = value.text;
    row.append(range, text); root.append(row); root.hidden = false;
    if (follow) root.scrollTop = root.scrollHeight;
  }
  return { reset, event };
}
