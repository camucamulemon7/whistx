const $ = (q, scope = document) => scope.querySelector(q);
const $$ = (q, scope = document) => Array.from(scope.querySelectorAll(q));

const recordBtn = $('#recordButton');
const timerEl = $('#recordTimer');
const sessionIdEl = $('#sessionId');
const statusEl = $('#status');
const vadDot = $('#vadDot');
const connEl = $('#connCount');
const partialEl = $('#partial');
const dlTxt = $('#dlTxt');
const dlJsonl = $('#dlJsonl');
const dlSrt = $('#dlSrt');
const finalListEl = $('#finalList');
const autoscrollEl = $('#autoscroll');
const copyAllBtn = $('#copyAll');
const clearAllBtn = $('#clearAll');
const languageEl = $('#language');
const vadBackendEl = $('#vadBackend');
const sileroThEl = $('#sileroThreshold');
const autoVadEnableEl = $('#autoVadEnable');
const autoVadWindowMsEl = $('#autoVadWindowMs');
const calibBtn = $('#calibBtn');
const vadStatusEl = $('#vadStatus');
const asrBackendEl = $('#asrBackend');
const utterSilEl = $('#utteranceSilenceMs');
const forceUttEl = $('#forceUtteranceMs');
const forceOvEl = $('#forceOverlapMs');
const partialIntEl = $('#partialIntervalMs');
const windowSecEl = $('#windowSeconds');
const diarEl = $('#enableDiarization');
const beamSizeEl = $('#beamSize');
const patienceEl = $('#patience');
const minFinalMsEl = $('#minFinalMs');
const vacEnableEl = $('#vacEnable');
const vacMinSpeechMsEl = $('#vacMinSpeechMs');
const vacHangoverMsEl = $('#vacHangoverMs');
const vacMinFinalMsEl = $('#vacMinFinalMs');
const agcEl = $('#agc');
const nsEl = $('#ns');
const ecEl = $('#ec');
const gainEl = $('#gain');
const gainValEl = $('#gainVal');
const levelNowEl = $('#levelNow');
const hotwordsEl = $('#hotwords');
const hotStatusEl = $('#hotwordsStatus');
const saveHotBtn = $('#saveHotwords');
const themeRadios = $$('input[name="theme"]');
const profileRadios = $$('input[name="profile"]');
const chunkModeEl = $('#chunkMode');
const refineStatusEl = $('#refineStatus');
const longformStatusEl = $('#longformStatus');
const longformPlaceholderEl = $('#longformPlaceholder');

const waveCanvas = $('#wave');

const state = {
  recordState: 'idle',
  timerStart: 0,
  timerHandle: null,
  pendingStart: null,
};

let ws = null;
let ac = null;
let workletNode = null;
let mediaStream = null;
let mediaStream2 = null;
let seq = 0;
let waveBuf = new Float32Array(16000 * 5);
let wavePos = 0;
let drawReq = null;
let awaitingReady = false;
let captureActive = false;
let gainListenerAttached = false;
let audioStreaming = false;
let pendingStart = null;
let recordLock = false;
const finalNodes = new Map();
const pendingRefineSegments = new Set();
let refineStatusTimer = null;
function generateSessionId() {
  return 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 6);
}

function ensureSessionId() {
  const value = (sessionIdEl?.value || '').trim();
  if (!value || value === 'sess-') {
    const next = generateSessionId();
    if (sessionIdEl) sessionIdEl.value = next;
  }
}
ensureSessionId();


const PREF_KEY = 'whistx-profile';

function getSelectedProfile() {
  const active = profileRadios.find((input) => input.checked);
  return active ? active.value : 'realtime';
}

function applyPreferences() {
  try {
    const raw = localStorage.getItem(PREF_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed.profile) {
        profileRadios.forEach((input) => {
          input.checked = input.value === parsed.profile;
        });
      }
      if (parsed.chunkMode && chunkModeEl) {
        chunkModeEl.value = parsed.chunkMode;
      }
    }
  } catch {}
}

function savePreferences() {
  const payload = {
    profile: getSelectedProfile(),
    chunkMode: chunkModeEl?.value || 'utterance',
  };
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify(payload));
  } catch {}
}

applyPreferences();

function resetStatuses() {
  pendingRefineSegments.clear();
  if (refineStatusTimer) {
    clearTimeout(refineStatusTimer);
    refineStatusTimer = null;
  }
  if (refineStatusEl) {
    refineStatusEl.textContent = '';
    refineStatusEl.hidden = true;
  }
  if (longformStatusEl) {
    longformStatusEl.textContent = '';
    longformStatusEl.hidden = true;
  }
  if (longformPlaceholderEl) {
    longformPlaceholderEl.hidden = true;
    longformPlaceholderEl.textContent = '';
  }
}

function resetTranscripts() {
  if (finalListEl) finalListEl.innerHTML = '';
  finalNodes.clear();
  if (partialEl) partialEl.textContent = '';
  resetStatuses();
}

function updatePlaceholderVisibility() {
  if (pendingRefineSegments.size === 0) {
    if (longformPlaceholderEl) {
      longformPlaceholderEl.hidden = true;
      longformPlaceholderEl.textContent = '';
    }
    if (longformStatusEl) {
      longformStatusEl.hidden = true;
      longformStatusEl.textContent = '';
    }
  }
}


function setRecordState(nextState) {
  state.recordState = nextState;
  if (!recordBtn) return;
  recordBtn.dataset.state = nextState;
  const recordingLike = nextState === 'ready' || nextState === 'recording';
  recordBtn.classList.toggle('recording', recordingLike);
  recordBtn.classList.toggle('loading', nextState === 'preparing');
  recordBtn.disabled = nextState === 'preparing';
  recordBtn.setAttribute('aria-pressed', recordingLike ? 'true' : 'false');
  if (!recordingLike) {
    stopTimer();
    if (timerEl) timerEl.textContent = '00:00';
  }
}

function startTimer() {
  if (!timerEl) return;
  state.timerStart = Date.now();
  timerEl.textContent = '00:00';
  if (state.timerHandle) clearInterval(state.timerHandle);
  state.timerHandle = setInterval(() => {
    const diff = Date.now() - state.timerStart;
    const totalSeconds = Math.max(0, Math.round(diff / 1000));
    const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
    const seconds = (totalSeconds % 60).toString().padStart(2, '0');
    timerEl.textContent = `${minutes}:${seconds}`;
  }, 1000);
}

function stopTimer() {
  if (state.timerHandle) {
    clearInterval(state.timerHandle);
    state.timerHandle = null;
  }
}

function setStatus(message) {
  if (statusEl) statusEl.textContent = message;
}

function updateDownloadLinks() {
  const id = sessionIdEl?.value?.trim();
  if (!id) return;
  const url = (ext) => `/api/transcript/${id}.${ext}`;
  if (dlTxt) dlTxt.href = url('txt');
  if (dlJsonl) dlJsonl.href = url('jsonl');
  if (dlSrt) dlSrt.href = url('srt');
}
updateDownloadLinks();


if (gainEl && gainValEl) {
  const setGainLabel = () => { gainValEl.textContent = 'x' + Number(gainEl.value).toFixed(2); };
  setGainLabel();
  gainEl.addEventListener('input', setGainLabel);
}

function readOpts() {
  const profile = getSelectedProfile();
  const chunkSelection = chunkModeEl?.value || 'utterance';
  let chunkMode = 'utterance';
  let chunkSeconds = undefined;
  if (chunkSelection.startsWith('longform')) {
    chunkMode = 'longform';
    chunkSeconds = chunkSelection.endsWith('120') ? 120 : 60;
  }
  return {
    language: languageEl?.value || 'ja',
    asrBackend: asrBackendEl?.value || 'parakeet',
    vadBackend: vadBackendEl?.value || 'silero',
    sileroThreshold: Number(sileroThEl?.value || 0.5),
    autoVadEnable: autoVadEnableEl?.checked ? 1 : 0,
    autoVadWindowMs: Number(autoVadWindowMsEl?.value || 3000),
    vacEnable: vacEnableEl?.checked ? 1 : 0,
    vacMinSpeechMs: Number(vacMinSpeechMsEl?.value || 220),
    vacHangoverMs: Number(vacHangoverMsEl?.value || 360),
    vacMinFinalMs: Number(vacMinFinalMsEl?.value || 700),
    utteranceSilenceMs: Number(utterSilEl?.value || 600),
    forceUtteranceMs: Number(forceUttEl?.value || 9000),
    forceOverlapMs: Number(forceOvEl?.value || 1200),
    partialIntervalMs: Number(partialIntEl?.value || 650),
    windowSeconds: Number(windowSecEl?.value || 8),
    enableDiarization: diarEl?.checked ? 1 : 0,
    beamSize: Number(beamSizeEl?.value || 12),
    patience: Number(patienceEl?.value || 1.2),
    minFinalMs: Number(minFinalMsEl?.value || 800),
    transcribeProfile: profile,
    highAccuracy: profile === 'high_accuracy' ? 1 : 0,
    chunkMode,
    chunkSeconds,
    chunkOverlapSeconds: chunkMode === 'longform' ? 5 : undefined,
  };
}

function formatTimestamp(tsStart = 0) {
  const ts = Math.max(0, Math.round(tsStart / 1000));
  const mm = Math.floor(ts / 60).toString().padStart(2, '0');
  const ss = (ts % 60).toString().padStart(2, '0');
  return `${mm}:${ss}`;
}

function ensureFinalNode(segmentId) {
  let node = finalNodes.get(segmentId);
  if (!node && finalListEl) {
    node = document.createElement('div');
    node.className = 'item';
    node.dataset.segmentId = segmentId;
    const timeEl = document.createElement('div');
    timeEl.className = 'time';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    node.append(timeEl, bubble);
    finalListEl.appendChild(node);
    finalNodes.set(segmentId, node);
  }
  return node;
}

function renderFinalNode(node, msg) {
  if (!node) return;
  const timeEl = node.querySelector('.time');
  const bubble = node.querySelector('.bubble');
  if (timeEl) timeEl.textContent = formatTimestamp(msg.tsStart || 0);
  if (bubble) {
    bubble.innerHTML = '';
    if (msg.speaker) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = msg.speaker;
      bubble.appendChild(chip);
    }
    const span = document.createElement('span');
    span.textContent = msg.text || '';
    bubble.appendChild(span);
  }
}

function removeFinalNode(segmentId) {
  const node = finalNodes.get(segmentId);
  if (node && node.parentElement) {
    node.parentElement.removeChild(node);
  }
  finalNodes.delete(segmentId);
  updatePlaceholderVisibility();
}

function appendFinal(msg) {
  if (!finalListEl || !msg.segmentId) return;
  const text = (msg.text || '').trim();
  if (!text) {
    removeFinalNode(msg.segmentId);
    return;
  }
  const node = ensureFinalNode(msg.segmentId);
  renderFinalNode(node, msg);
  if (autoscrollEl?.checked) {
    finalListEl.scrollTop = finalListEl.scrollHeight;
  }
  updatePlaceholderVisibility();
}

function handleStatusMessage(msg) {
  const stage = msg.stage;
  if (!stage) return;
  if (stage === 'refining') {
    if (msg.segmentId) pendingRefineSegments.add(msg.segmentId);
    if (refineStatusTimer) {
      clearTimeout(refineStatusTimer);
      refineStatusTimer = null;
    }
    if (refineStatusEl) {
      const count = pendingRefineSegments.size;
      refineStatusEl.hidden = false;
      refineStatusEl.textContent = count > 1 ? `精度向上処理中… (${count})` : '精度向上処理中…';
    }
    if (msg.segmentId && msg.segmentId.startsWith('chunk_') && longformPlaceholderEl) {
      longformPlaceholderEl.hidden = false;
      longformPlaceholderEl.textContent = '長尺チャンクを処理中…';
    }
  } else if (stage === 'refined') {
    if (msg.segmentId) pendingRefineSegments.delete(msg.segmentId);
    if (refineStatusEl) {
      const latencySec = msg.latencyMs ? (msg.latencyMs / 1000).toFixed(1) : null;
      refineStatusEl.hidden = false;
      refineStatusEl.textContent = latencySec ? `精度向上完了 (${latencySec}s)` : '精度向上完了';
    }
    if (refineStatusTimer) clearTimeout(refineStatusTimer);
    refineStatusTimer = setTimeout(() => {
      if (pendingRefineSegments.size === 0 && refineStatusEl) {
        refineStatusEl.hidden = true;
        refineStatusEl.textContent = '';
      }
    }, 2000);
    updatePlaceholderVisibility();
  } else if (stage === 'longform_pending') {
    const remainingMs = Number(msg.remainingMs ?? 0);
    const seconds = (remainingMs / 1000).toFixed(1);
    if (longformStatusEl) {
      longformStatusEl.hidden = false;
      longformStatusEl.textContent = `長尺チャンク処理中… 残り${seconds}s`;
    }
    if (longformPlaceholderEl) {
      longformPlaceholderEl.hidden = false;
      longformPlaceholderEl.textContent = `長尺チャンク処理中… (残り${seconds}s)`;
    }
  }
}

function ampToDb(a) {
  return 20 * Math.log10(Math.max(1e-9, a));
}

async function startCapture() {
  if (recordLock || state.recordState === 'preparing') {
    return;
  }
  ensureSessionId();
  const sourceInput = document.querySelector('input[name="source"]:checked');
  if (!sourceInput) {
    setStatus('入力ソースを選択してください');
    return;
  }
  const source = sourceInput.value;

  setRecordState('preparing');
  setStatus('デバイス確認中...');
  recordLock = true;

  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  if (mediaStream2) { mediaStream2.getTracks().forEach((t) => t.stop()); mediaStream2 = null; }
  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }

  try {
    if (source === 'mic') {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: !!ecEl?.checked,
          noiseSuppression: !!nsEl?.checked,
          autoGainControl: !!agcEl?.checked,
        },
      });
    } else if (source === 'display') {
      mediaStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      if (mediaStream.getAudioTracks().length === 0) {
        setStatus('共有に音声が含まれていません（「音声を共有」を有効に）');
      }
    } else {
      const mic = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: !!ecEl?.checked,
          noiseSuppression: !!nsEl?.checked,
          autoGainControl: !!agcEl?.checked,
        },
      });
      const disp = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      mediaStream = mic;
      mediaStream2 = disp;
      if (disp.getAudioTracks().length === 0) {
        setStatus('画面共有に音声が含まれていないため、マイクのみで進行します');
        mediaStream2 = null;
      }
    }
  } catch (err) {
    setStatus('音声取得に失敗: ' + err.message);
    setRecordState('idle');
    recordLock = false;
    return;
  }

  const sessionId = sessionIdEl?.value?.trim() || generateSessionId();
  if (sessionIdEl && !sessionIdEl.value) sessionIdEl.value = sessionId;

  pendingStart = { source };
  awaitingReady = true;
  resetStatuses();
  seq = 0;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${proto}://${location.host}/ws/transcribe`;
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    const opts = readOpts();
    pendingStart.opts = opts;
    ws?.send(
      JSON.stringify({
        type: 'start',
        sessionId,
        source,
        lang: opts.language || 'ja',
        format: 'pcm16',
        sampleRate: 16000,
        opts,
      }),
    );
    setStatus('モデル準備中...');
  };
  ws.onmessage = (ev) => handleMessage(ev.data);
  ws.onclose = () => {
    if (captureActive || awaitingReady) {
      setStatus('切断');
    }
    ws = null;
    cleanup();
  };
  ws.onerror = () => setStatus('エラー');
}

async function activateCapture() {
  if (!pendingStart || captureActive) {
    return;
  }
  if (!mediaStream && !mediaStream2) {
    setStatus('入力ストリームが利用できません');
    cleanup();
    return;
  }
  const { source } = pendingStart;
  try {
    if (!ac) {
      ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
      await ac.audioWorklet.addModule('/audio-worklet-processor.js');
    } else if (ac.state === 'suspended') {
      await ac.resume();
    }
    workletNode = new AudioWorkletNode(ac, 'pcm16-downsampler');
    if (gainEl) workletNode.port.postMessage({ type: 'set', gain: Number(gainEl.value) });
    const sink = ac.createGain();
    sink.gain.value = 0;
    const makeLP = () => {
      const f = ac.createBiquadFilter();
      f.type = 'lowpass';
      f.frequency.value = 7000;
      f.Q.value = 0.707;
      return f;
    };
    if (source === 'mix' && mediaStream2) {
      const src1 = ac.createMediaStreamSource(mediaStream);
      const src2 = ac.createMediaStreamSource(mediaStream2);
      const g1 = ac.createGain();
      const g2 = ac.createGain();
      g1.gain.value = 0.7;
      g2.gain.value = 0.7;
      src1.connect(g1).connect(makeLP()).connect(workletNode);
      src2.connect(g2).connect(makeLP()).connect(workletNode);
    } else if (mediaStream) {
      const src = ac.createMediaStreamSource(mediaStream);
      src.connect(makeLP()).connect(workletNode);
    } else {
      throw new Error('入力ソースが見つかりません');
    }
    workletNode.connect(sink).connect(ac.destination);
    workletNode.port.onmessage = handleAudioFrame;
    if (gainEl && !gainListenerAttached) {
      gainEl.addEventListener('input', () => {
        if (workletNode) {
          workletNode.port.postMessage({ type: 'set', gain: Number(gainEl.value) });
        }
      });
      gainListenerAttached = true;
    }
    captureActive = true;
    awaitingReady = false;
    pendingStart = null;
    audioStreaming = false;
    setRecordState('ready');
    setStatus('録音準備完了');
    updateDownloadLinks();
    recordLock = false;
  } catch (err) {
    console.error(err);
    setStatus('音声処理初期化に失敗: ' + err.message);
    pendingStart = null;
    awaitingReady = false;
    captureActive = false;
    setRecordState('idle');
    stopCapture();
  }
}

function handleMessage(data) {
  try {
    const msg = JSON.parse(data);
    if (msg.type === 'partial') {
      if (partialEl) partialEl.textContent = msg.text || '';
    } else if (msg.type === 'final') {
      if (partialEl) partialEl.textContent = '';
      appendFinal(msg);
    } else if (msg.type === 'overwrite') {
      appendFinal(msg);
    } else if (msg.type === 'vad') {
      if (msg.state === 'start') vadDot?.classList.add('on');
      if (msg.state === 'end') vadDot?.classList.remove('on');
    } else if (msg.type === 'calib') {
      if (vadStatusEl) {
        const thr = (msg.newThr ?? msg.oldThr ?? 0).toFixed(2);
        const ratio = ((msg.ratio ?? 0) * 100).toFixed(0);
        const rms = (msg.rms ?? 0).toFixed(3);
        vadStatusEl.textContent = `calib: thr=${thr} talk=${ratio}% rms=${rms}`;
        setTimeout(() => {
          if (vadStatusEl.textContent.startsWith('calib:')) vadStatusEl.textContent = '';
        }, 4000);
      }
    } else if (msg.type === 'status') {
      handleStatusMessage(msg);
    } else if (msg.type === 'conn') {
      if (connEl) connEl.textContent = String(msg.count ?? 0);
    } else if (msg.type === 'info') {
      if (msg.sessionId && sessionIdEl) {
        sessionIdEl.value = msg.sessionId;
        updateDownloadLinks();
      }
      if (msg.backend && asrBackendEl) {
        asrBackendEl.value = msg.backend;
      }
      const normalizedMessage = (msg.message || '').toLowerCase();
      if (normalizedMessage === 'backend_loading' || msg.state === 'loading') {
        setStatus('モデル読み込み中...');
        return;
      }
      if (msg.state === 'ready' || normalizedMessage === 'ready') {
        const readyText = msg.backend ? `モデル準備完了 · ${msg.backend}` : 'モデル準備完了';
        setStatus(readyText);
        activateCapture();
        return;
      }
      if (msg.message && msg.message !== 'ready') {
        if ((msg.message === 'stopping' || msg.message === 'closed') && !captureActive && !awaitingReady) {
          // keep current status
        } else {
          const infoText = msg.backend ? `${msg.message} · ${msg.backend}` : msg.message;
          setStatus(infoText);
        }
      }
    }
  } catch (err) {
    console.error(err);
  }
}

function handleAudioFrame(ev) {
  if (!ws || ws.readyState !== 1) return;
  const { type, ptsMs, payload } = ev.data || {};
  if (type !== 'frame' || !payload) return;
  const header = new ArrayBuffer(8);
  const view = new DataView(header);
  view.setUint32(0, seq++, true);
  view.setUint32(4, ptsMs >>> 0, true);
  const body = payload;
  const out = new Uint8Array(header.byteLength + body.byteLength);
  out.set(new Uint8Array(header), 0);
  out.set(new Uint8Array(body), header.byteLength);
  ws.send(out);
  if (!audioStreaming) {
    audioStreaming = true;
    setRecordState('recording');
    startTimer();
    setStatus('録音中');
  }
  const i16 = new Int16Array(body);
  for (let i = 0; i < i16.length; i++) {
    waveBuf[wavePos++] = i16[i] / 32768;
    if (wavePos >= waveBuf.length) wavePos = 0;
  }
  if (!drawReq) drawReq = requestAnimationFrame(drawWave);
}

function drawWave() {
  drawReq = null;
  if (!waveCanvas) return;
  const ctx = waveCanvas.getContext('2d');
  if (!ctx) return;
  const { width: W, height: H } = waveCanvas;
  ctx.clearRect(0, 0, W, H);
  const styles = getComputedStyle(document.documentElement);
  const accent = (styles.getPropertyValue('--accent') || '#2563eb').trim();
  const toRgba = (hex, alpha) => {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!m) return `rgba(37, 99, 235, ${alpha})`;
    const [r, g, b] = m.slice(1).map((v) => parseInt(v, 16));
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };
  const gradient = ctx.createLinearGradient(0, 0, 0, H);
  gradient.addColorStop(0, toRgba(accent, 0.45));
  gradient.addColorStop(0.6, toRgba(accent, 0.1));
  gradient.addColorStop(1, toRgba(accent, 0));

  const areaPath = new Path2D();
  const strokePath = new Path2D();
  const mid = H / 2;
  const scale = H / 2 - 6;
  const N = waveBuf.length;
  areaPath.moveTo(0, mid);
  strokePath.moveTo(0, mid);
  for (let x = 0; x < W; x++) {
    const idx = (wavePos + Math.floor((x / W) * N)) % N;
    const v = waveBuf[idx] || 0;
    const ease = Math.pow(Math.sin((x / W) * Math.PI), 0.65);
    const y = mid - v * scale * ease;
    areaPath.lineTo(x, y);
    strokePath.lineTo(x, y);
  }
  areaPath.lineTo(W, mid);
  areaPath.lineTo(0, mid);
  areaPath.closePath();
  ctx.fillStyle = gradient;
  ctx.fill(areaPath);
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = toRgba(accent, 0.9);
  ctx.shadowColor = toRgba(accent, 0.25);
  ctx.shadowBlur = 12;
  ctx.stroke(strokePath);
  ctx.shadowBlur = 0;

  const win = 3200;
  let sum2 = 0;
  for (let i = 0; i < win; i++) {
    const idx = (wavePos - 1 - i + N) % N;
    const v = waveBuf[idx] || 0;
    sum2 += v * v;
  }
  const rms = Math.sqrt(sum2 / win);
  const db = ampToDb(rms);
  if (levelNowEl) levelNowEl.textContent = `RMS: ${db.toFixed(1)} dBFS`;
  drawReq = requestAnimationFrame(drawWave);
}

async function stopCapture() {
  if (ws && ws.readyState === 1) {
    try { ws.send(JSON.stringify({ type: 'stop' })); } catch {}
  }
  try { ws?.close(); } catch {}
  cleanup();
  setStatus('停止');
}

function cleanup() {
  if (workletNode) {
    try { workletNode.disconnect(); } catch {}
    workletNode = null;
  }
  if (ac) {
    try { ac.close(); } catch {}
    ac = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (mediaStream2) {
    mediaStream2.getTracks().forEach((t) => t.stop());
    mediaStream2 = null;
  }
  if (drawReq) {
    cancelAnimationFrame(drawReq);
    drawReq = null;
  }
  waveBuf.fill(0);
  awaitingReady = false;
  captureActive = false;
  audioStreaming = false;
  seq = 0;
  pendingStart = null;
  recordLock = false;
  resetStatuses();
  setRecordState('idle');
  stopTimer();
}

if (recordBtn) {
  recordBtn.addEventListener('click', () => {
    if (state.recordState === 'idle') {
      startCapture();
    } else if (state.recordState === 'ready' || state.recordState === 'recording') {
      stopCapture();
    }
  });
}

if (calibBtn) {
  calibBtn.addEventListener('click', () => {
    try {
      ws?.send(JSON.stringify({ type: 'calibrate', durationMs: 2000 }));
      if (vadStatusEl) vadStatusEl.textContent = 'calibrating...';
    } catch {}
  });
}

async function loadHotwords() {
  try {
    const r = await fetch('/api/hotwords');
    const j = await r.json();
    if (hotwordsEl) hotwordsEl.value = (j.words || []).join('\n');
  } catch {}
}

async function saveHotwords() {
  const text = hotwordsEl?.value || '';
  try {
    const r = await fetch('/api/hotwords', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const j = await r.json();
    if (hotStatusEl) {
      hotStatusEl.textContent = `保存しました (${j.count || 0}語)`;
      setTimeout(() => (hotStatusEl.textContent = ''), 2000);
    }
  } catch (err) {
    if (hotStatusEl) hotStatusEl.textContent = '保存失敗';
  }
}

loadHotwords();
if (saveHotBtn) saveHotBtn.addEventListener('click', saveHotwords);

if (copyAllBtn) {
  copyAllBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    const lines = [...(finalListEl?.querySelectorAll('.item .bubble') || [])].map((n) => n.textContent?.trim() || '');
    const txt = lines.join('\n').trim();
    let ok = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(txt);
        ok = true;
      }
    } catch {}
    if (!ok) {
      try {
        const ta = document.createElement('textarea');
        ta.value = txt;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        ok = true;
      } catch {}
    }
    if (ok && hotStatusEl) {
      hotStatusEl.textContent = 'コピーしました';
      setTimeout(() => (hotStatusEl.textContent = ''), 2000);
    }
  });
}

if (clearAllBtn) {
  clearAllBtn.addEventListener('click', (e) => {
    e.preventDefault();
    resetTranscripts();
  });
}

profileRadios.forEach((input) => {
  input.addEventListener('change', () => {
    savePreferences();
    resetTranscripts();
  });
});

if (chunkModeEl) {
  chunkModeEl.addEventListener('change', () => {
    savePreferences();
    resetTranscripts();
  });
}

function applyTheme(mode) {
  const normalized = mode === 'dark' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', normalized);
  localStorage.setItem('theme', normalized);
}

(function initTheme() {
  const stored = localStorage.getItem('theme');
  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
  const initialMode = stored === 'dark' ? 'dark' : stored === 'light' ? 'light' : prefersDark ? 'dark' : 'light';
  applyTheme(initialMode);
  themeRadios.forEach((input) => {
    const value = input.value;
    input.checked = value === initialMode;
    input.addEventListener('change', () => {
      if (input.checked) {
        applyTheme(value);
      }
    });
  });
})();
