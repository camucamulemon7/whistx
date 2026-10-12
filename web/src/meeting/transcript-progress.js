import { transcriptParagraphs, transcriptJoiner, transcriptSpeakerLabel } from './paragraphs.js';
import { formatTimestamp } from '../ui/format.js';

// An agreed prefix is still provisional. It never becomes a .log-row or a saved record.
export function createTranscriptProgress({ root, getRecords, getSourceTracks = () => [],
  isFollowing = () => false, onChange = () => {}, document = globalThis.document }) {
  const drafts = new Map();
  const committed = value => getRecords().some(record => record.segmentId === value.segmentId ||
    ((record.track || 'mixed') === value.track && Number.isFinite(value.tsStart) && Number.isFinite(value.tsEnd) &&
      value.tsStart < record.tsEnd && value.tsEnd <= record.tsEnd));

  function remove(value) {
    value.node?.remove();
    value.owned?.remove();
    drafts.delete(value.track);
  }
  function reconcile() {
    const records = getRecords();
    const groups = transcriptParagraphs(records);
    const last = groups.at(-1);
    const multiple = new Set([...getSourceTracks(), ...records.map(row => row.track), ...drafts.keys()].filter(track => track === 'mic' || track === 'display')).size > 1;
    for (const value of drafts.values()) {
      if (committed(value)) { remove(value); continue; }
      let paragraph;
      const previous = last?.at(-1);
      if (previous && previous.track === value.track && value.tsStart - previous.tsEnd <= 8000 &&
          (previous.speaker || '') === (value.speaker || '')) {
        paragraph = [...root.children].find(node => node.dataset.firstSegment === String(last[0].segmentId || last[0].seq));
      }
      if (!paragraph) {
        if (!value.owned) {
          value.owned = document.createElement('section');
          value.owned.className = 'transcript-paragraph transcript-progress-paragraph';
          const heading = document.createElement('div');
          heading.className = 'transcript-paragraph-heading';
          value.owned.append(heading);
        }
        paragraph = value.owned;
        const label = transcriptSpeakerLabel(value, multiple);
        const heading = `${formatTimestamp(value.tsStart)}${label ? ` · ${label}` : ''}`;
        if (paragraph.children[0].textContent !== heading) paragraph.children[0].textContent = heading;
        if (paragraph.parentNode !== root) root.append(paragraph);
      }
      if (value.node.parentNode !== paragraph || paragraph.lastChild !== value.node) paragraph.append(value.node);
      if (value.owned && paragraph !== value.owned) { value.owned.remove(); value.owned = null; }
      const joiner = paragraph === value.owned ? '' : transcriptJoiner(previous.text, value.text);
      if (value.joiner.textContent !== joiner) value.joiner.textContent = joiner;
    }
    const empty = root.querySelector('.empty-state');
    if (empty) empty.hidden = drafts.size > 0;
  }
  function reset() { for (const value of [...drafts.values()]) remove(value); reconcile(); }
  function event(value) {
    const track = value.track || 'mixed';
    if (value.type === 'info' && ['ready', 'finalized'].includes(value.message)) { reset(); return; }
    if (value.type === 'final') {
      const previous = drafts.get(track);
      if (previous?.segmentId === value.segmentId) { remove(previous); reconcile(); }
      return;
    }
    if (value.type !== 'partial') return;
    const previous = drafts.get(track);
    if (!value.text) { if (previous) { remove(previous); reconcile(); } return; }
    const text = String(value.stableText || '');
    if (!value.segmentId || !text || !Number.isFinite(value.tsStart) || !Number.isFinite(value.tsEnd) ||
        value.tsEnd < value.tsStart || !String(value.text).startsWith(text) || committed({ ...value, track })) return;
    // Ignore unstable regressions rather than repeatedly erasing and retyping a prefix.
    if (previous?.segmentId === value.segmentId && (!text.startsWith(previous.text) || text === previous.text)) return;
    const follow = isFollowing();
    if (previous && previous.segmentId !== value.segmentId) remove(previous);
    let draft = drafts.get(track);
    if (!draft) {
      const node = document.createElement('span');
      node.className = 'transcript-progress';
      node.setAttribute('aria-label', '確定前の認識結果');
      node.title = '確定前の認識結果（保存・コピーには含まれません）';
      const joiner = document.createTextNode('');
      const content = document.createTextNode('');
      node.append(joiner, content);
      draft = { ...value, track, text: '', node, joiner, content, owned: null };
      drafts.set(track, draft);
    }
    draft.content.appendData(text.slice(draft.text.length));
    draft.text = text;
    draft.tsEnd = value.tsEnd;
    reconcile();
    onChange({ follow });
  }
  return { event, reconcile, reset };
}
