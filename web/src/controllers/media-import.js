import { readSseJsonStream } from '../api/sse.js';

export const MEDIA_MAX_BYTES = 256 * 1024 * 1024;
export const MEDIA_EXTENSIONS = ['wav', 'mp3', 'm4a', 'mp4', 'mov', 'webm', 'mkv', 'ogg', 'opus', 'flac', 'aac', 'aiff', 'aif'];
export function validateMediaFile(file) {
  if (!file?.size) return '空のファイルは取り込めません。';
  if (file.size > MEDIA_MAX_BYTES) return '256 MB以下のファイルを選んでください。';
  if (!MEDIA_EXTENSIONS.includes(file.name.split('.').pop().toLowerCase())) return '対応する動画・音声ファイルを選んでください。';
  if (file.name.length > 240) return 'ファイル名を240文字以内にしてください。';
  return '';
}
const messages = {
  media_too_large: '256 MB以下のファイルを選んでください。',
  media_too_long: '2時間以内の動画・音声を選んでください。',
  media_unsupported_format: 'このファイル形式は取り込めません。',
  media_decode_failed: '音声を読み取れませんでした。音声を含むファイルか確認してください。',
  media_no_audio: 'このファイルには音声がありません。',
  media_no_speech: '発話を検出できませんでした。音声を確認してください。',
  media_decoder_unavailable: 'ファイル取り込みの準備ができていません。管理者に確認してください。',
  media_decode_timeout: '音声の読み取りに時間がかかりました。短いファイルで再試行してください。',
  media_upload_timeout: '送信に時間がかかりました。接続を確認して再試行してください。',
  asr_not_ready: '文字起こしに接続できません。少し待って再試行してください。',
  rate_limit_exceeded: '処理の上限に達しました。少し待って再試行してください。',
  login_required: 'ファイルを取り込むにはログインしてください。',
  meeting_model_busy: '文字起こしが混み合っています。少し待って再試行してください。',
};
export function createMediaImportController(deps) {
  const $ = selector => document.querySelector(selector);
  const panel = $('#mediaImportPanel'), recordingPanel = $('#recordingInputPanel');
  const input = $('#mediaFileInput'), drop = $('#mediaDropzone');
  const status = $('#mediaImportStatus'), progress = $('#mediaImportProgress');
  const button = $('#mediaImportStart'), cancel = $('#mediaImportCancel'), language = $('#mediaLanguage');
  const tabs = [...document.querySelectorAll('[data-capture-mode]')];
  let file = null, mode = 'record';
  function resetSelection() {
    file = null; $('#mediaFileName').textContent = '動画・音声ファイルを選ぶ';
    $('#mediaFileDetail').textContent = 'ここにドロップ、またはクリックして選択';
    drop.classList.remove('has-file'); progress.hidden = true;
  }
  function sync() {
    const locked = deps.isRecordingInteractionLocked(), active = Boolean(deps.state.mediaImportController);
    if (!deps.state.auth.authenticated) {
      if (file) resetSelection();
      if (active) deps.state.mediaImportController.abort('auth_changed');
    }
    tabs.forEach(tab => {
      tab.disabled = locked;
      const selected = tab.dataset.captureMode === mode;
      tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
    });
    input.disabled = locked || !deps.state.auth.authenticated;
    drop.disabled = input.disabled;
    language.disabled = locked;
    button.disabled = locked || !file || !deps.state.auth.authenticated || !deps.state.asrAvailable || Boolean(deps.meetingWorkspace?.busy) || deps.state.saveInFlight || deps.state.summaryInFlight || deps.state.proofreadInFlight;
    cancel.hidden = !active; panel.setAttribute('aria-busy', String(active));
    if (active) deps.startBtn.disabled = true;
    $('#mediaLoginHint').hidden = Boolean(deps.state.auth.authenticated);
  }
  function setMode(next) {
    if (deps.isRecordingInteractionLocked()) return;
    mode = next; recordingPanel.hidden = mode !== 'record'; panel.hidden = mode !== 'file';
    if (mode === 'file') language.value = deps.languageEl.value;
    sync();
  }
  tabs.forEach(tab => {
    tab.addEventListener('click', () => setMode(tab.dataset.captureMode));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const target = tabs[event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : 1 - tabs.indexOf(tab)];
      target.click(); target.focus();
    });
  });
  function choose(next) {
    if (deps.isRecordingInteractionLocked()) return;
    const error = validateMediaFile(next);
    if (error) { status.textContent = error; resetSelection(); sync(); return; }
    file = next; $('#mediaFileName').textContent = file.name;
    $('#mediaFileDetail').textContent = (file.size / 1024 / 1024).toFixed(1) + ' MB · 文字起こしの準備ができました';
    status.textContent = '音声の言語を確認して、文字起こしを開始してください。';
    progress.hidden = true; drop.classList.add('has-file'); sync();
  }
  input.addEventListener('change', () => { if (input.files[0]) choose(input.files[0]); input.value = ''; });
  drop.addEventListener('click', () => { if (!input.disabled) input.click(); });
  drop.addEventListener('dragover', event => { event.preventDefault(); if (!input.disabled) drop.classList.add('is-dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-dragover'));
  drop.addEventListener('drop', event => {
    event.preventDefault(); drop.classList.remove('is-dragover');
    if (!input.disabled && event.dataTransfer?.files[0]) choose(event.dataTransfer.files[0]);
  });
  // Dropping outside the target must not navigate away from a recording.
  document.addEventListener('dragover', event => {
    if (event.dataTransfer?.types.includes('Files')) event.preventDefault();
  });
  document.addEventListener('drop', event => {
    if (!event.dataTransfer?.files.length || event.defaultPrevented) return;
    event.preventDefault();
    if (!input.disabled) { setMode('file'); choose(event.dataTransfer.files[0]); }
  });
  cancel.addEventListener('click', () => deps.state.mediaImportController?.abort());
  button.addEventListener('click', async () => {
    if (button.disabled || !file || deps.isRecordingInteractionLocked() || deps.state.summaryInFlight || deps.state.proofreadInFlight) return;
    if (!deps.confirmWorkspaceDiscard('ファイルの文字起こし結果に置き換え')) return;
    const controller = new AbortController(); deps.state.mediaImportController = controller;
    deps.updateSaveControls(); deps.setSessionSettingsLocked(); deps.updateDownloadLinks(); deps.renderHistoryList(); deps.syncUnloadProtection();
    status.textContent = 'ファイルを送信しています…'; progress.hidden = false; progress.removeAttribute('value');
    let result = null;
    try {
      const params = new URLSearchParams({ filename: file.name, language: language.value,
        prompt: (deps.sharedVocabularyEl.value.slice(0, 250) + ' ' + deps.promptEl.value.slice(0, 500)).trim().slice(0, 750) });
      const response = await fetch('/api/media/transcribe?' + params, { method: 'POST', body: file,
        headers: { 'Content-Type': 'application/octet-stream' }, signal: controller.signal });
      if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(error.error || error.detail || 'import_failed'); }
      await readSseJsonStream(response, event => {
        if (event.type === 'error') throw new Error(event.error);
        if (event.type === 'status') {
          status.textContent = event.message;
          if (event.phase === 'decoding') progress.removeAttribute('value');
          else progress.value = Math.min(100, Math.max(0, Number(event.progress) || 0));
        }
        if (event.type === 'done') result = event;
      });
      if (!result || controller.signal.aborted) throw new Error('import_interrupted');
      deps.commitNewRecordingWorkspace(); deps.state.liveCapture?.dispose(); deps.state.liveCapture = null;
      deps.state.runtimeSessionId = result.sessionId; deps.state.runtimeSessionToken = ''; deps.state.runtimeSessionFinalized = true;
      deps.state.recordingAudioSource = 'file'; deps.saveTitleInputEl.value = result.title;
      for (const row of result.records) deps.addLogLine(row.text, row.tsStart, row.tsEnd, row.seq, row.speaker || '', '', row.rawAudioPath || '', '', row.segmentId, row);
      deps.markWorkspaceDirty(); deps.meetingWorkspace.setView('transcript');
      $('#liveConnection').textContent = 'ファイルから文字起こし · ' + file.name;
      progress.value = 100; status.textContent = '完了しました。履歴への保存・議事録の作成・書き出しができます。';
      deps.showToast('ファイルの文字起こしが完了しました', 'success');
    } catch (error) {
      status.textContent = controller.signal.aborted ? 'キャンセルしました。表示中の結果はそのまま残ります。'
        : messages[error.message] || '取り込みに失敗しました。表示中の結果はそのまま残ります。再試行してください。';
      result = null; controller.abort();
      progress.hidden = true;
    } finally {
      deps.state.mediaImportController = null; deps.startBtn.disabled = deps.runtimeUi.appLocked || !deps.state.asrAvailable;
      deps.updateSaveControls(); deps.setSessionSettingsLocked(); deps.updateDownloadLinks(); deps.renderHistoryList(); deps.syncUnloadProtection();
      if (result && !controller.signal.aborted) deps.meetingWorkspace.refresh({ transcriptChanged: true });
    }
  });
  sync(); return { sync };
}
