import { readSseJsonStream } from '../api/sse.js';
import { formatTimestamp } from '../ui/format.js';

export function createTranslation({ getSource, getAccess, setView }) {
  const enabled = document.querySelector('#translationEnabled');
  const language = document.querySelector('#translationLanguage');
  const tab = document.querySelector('#meetingTranslationTab');
  const root = document.querySelector('#translationRows');
  const status = document.querySelector('#translationStatus');
  const retry = document.querySelector('#translationRetry');
  let controller = null, timer = null, version = 0, queued = false;
  let sources = [], translations = new Map();
  try {
    enabled.checked = localStorage.getItem('whistx_translation_enabled') === '1';
    language.value = localStorage.getItem('whistx_translation_language') || 'en';
    if (!language.value) language.value = 'en';
  } catch {}
  tab.hidden = !enabled.checked;

  function render() {
    const top = root.scrollTop;
    const follow = root.scrollHeight - root.clientHeight - top < 80;
    root.replaceChildren();
    for (const source of sources) {
      const row = document.createElement('article');
      row.className = 'translation-row';
      const original = document.createElement('div');
      const time = document.createElement('small');
      time.textContent = formatTimestamp(source.startMs);
      const text = document.createElement('p');
      text.textContent = source.text;
      original.append(time, text);
      const translated = document.createElement('p');
      const result = translations.get(source.id);
      translated.textContent = result?.sourceText === source.text ? result.text : '高精度認識の確定後に翻訳します';
      translated.lang = language.value;
      translated.className = result?.sourceText === source.text ? '' : 'translation-pending';
      row.append(original, translated);
      root.append(row);
    }
    root.scrollTop = follow ? root.scrollHeight : top;
  }
  function reset() {
    version++;
    clearTimeout(timer);
    controller?.abort();
    controller = null;
    queued = false;
    retry.hidden = true;
    sources = [];
    translations.clear();
    render();
    status.textContent = enabled.checked ? '高精度認識が確定すると翻訳します（録音終了時は残りも翻訳）。' : '';
  }
  async function run() {
    if (!enabled.checked || !getSource()) return;
    if (getAccess() !== 'ready') { status.textContent = '翻訳にはログインと要約モデルの設定が必要です。'; return; }
    if (controller) { queued = true; return; }
    const current = new AbortController();
    controller = current;
    const stamp = version;
    const timeout = setTimeout(() => current.abort('timeout'), 300_000);
    status.textContent = '高精度の文字起こしを翻訳しています…';
    retry.hidden = true;
    let done = false;
    try {
      const response = await fetch('/api/meeting/translate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...getSource(), language: language.value }), signal: current.signal });
      if (!response.ok) throw new Error((await response.json()).error || 'translation_failed');
      await readSseJsonStream(response, event => {
        if (stamp !== version || current.signal.aborted) return;
        if (event.type === 'error') throw new Error(event.error);
        if (event.type === 'sources') { sources = event.sources; render(); }
        if (event.type === 'translation') { translations.set(event.id, event); render(); }
        if (event.type === 'done') {
          done = true;
          const count = sources.filter(row => translations.get(row.id)?.sourceText === row.text).length;
          status.textContent = count ? `${count}区間を翻訳済み · 新しい高精度確定分は自動で追加します` : '高精度認識が確定すると翻訳します';
        }
      });
      if (!done && !current.signal.aborted) throw new Error('incomplete_translation');
    } catch (error) {
      if (stamp === version && (current.signal.reason === 'timeout' || !current.signal.aborted)) {
        status.textContent = error.message === 'rate_limit_exceeded' ? '翻訳が混み合っています。少し待って再試行してください。' : '翻訳に失敗しました。原文と翻訳済みの内容は保持しています。';
        retry.hidden = false;
      }
    } finally {
      clearTimeout(timeout);
      if (controller === current) {
        controller = null;
        if (queued) { queued = false; schedule(); }
      }
    }
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(run, 400); }
  enabled.addEventListener('change', () => {
    reset();
    tab.hidden = !enabled.checked;
    try { localStorage.setItem('whistx_translation_enabled', enabled.checked ? '1' : '0'); } catch {}
    setView(enabled.checked ? 'translation' : 'transcript');
    if (enabled.checked) schedule();
  });
  language.addEventListener('change', () => {
    reset();
    try { localStorage.setItem('whistx_translation_language', language.value); } catch {}
    schedule();
  });
  retry.addEventListener('click', run);
  return {
    sync() { reset(); schedule(); },
    event(event) {
      if (event.type === 'transcript_revision' || event.type === 'transcript_snapshot' || (event.type === 'info' && event.message === 'finalized')) schedule();
    },
  };
}
