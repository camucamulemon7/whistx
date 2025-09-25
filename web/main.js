(() => {
  const $ = (q) => document.querySelector(q);
  const sessionIdEl = $('#sessionId');
  if (sessionIdEl) sessionIdEl.value = 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2,6);
  const startBtn = $('#startBtn');
  const stopBtn = $('#stopBtn');
  const statusEl = $('#status');
  const partialEl = $('#partial');
  // final text area replaced by list view
  const dlTxt = $('#dlTxt');
  const dlJsonl = $('#dlJsonl');
  const dlSrt = $('#dlSrt');
  const waveCanvas = $('#wave');
  const vadDot = $('#vadDot');
  const connEl = $('#connCount');
  const finalListEl = document.querySelector('#finalList');
  const autoscrollEl = document.querySelector('#autoscroll');
  const copyAllBtn = document.querySelector('#copyAll');
  const clearAllBtn = document.querySelector('#clearAll');
  const captionToggle = $('#toggleCaption');
  const audioToggle = $('#toggleAudio');
  const audioBody = $('#audioBody');
  // settings
  // Parakeet 用に Whisper 固有のUIは非表示（保持しても送らない）
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

  let ws = null;
  let ac = null;
  let workletNode = null;
  let mediaStream = null;      // mic または display を格納
  let mediaStream2 = null;     // mix 用の 2 本目（display）
  let seq = 0;
  let waveBuf = new Float32Array(16000 * 5); // 5秒表示用リングバッファ
  let wavePos = 0;
  let drawReq = null;
  let pendingStart = null;
  let awaitingReady = false;
  let captureActive = false;
  let gainListenerAttached = false;
  let audioStreaming = false;
  // ユーティリティ
  function dbToAmp(db) { return Math.pow(10, db / 20); }
  function ampToDb(a) { return 20 * Math.log10(Math.max(1e-9, a)); }
  function setGainLabel() { if (gainEl && gainValEl) gainValEl.textContent = 'x' + Number(gainEl.value).toFixed(2); }
  setGainLabel();
  if (gainEl) gainEl.addEventListener('input', setGainLabel);

  function syncToggle(btn, target, opts = {}) {
    if (!btn || !target) return;
    const { show = '表示', hide = '非表示' } = opts;
    const update = () => {
      const hidden = target.classList.contains('hidden');
      btn.textContent = hidden ? show : hide;
    };
    btn.addEventListener('click', () => {
      target.classList.toggle('hidden');
      update();
    });
    update();
  }

  syncToggle(captionToggle, partialEl);
  syncToggle(audioToggle, audioBody);

  function setStatus(s) { statusEl.textContent = s; }
  function urlForTranscript(ext) {
    const id = sessionIdEl.value.trim();
    return `/api/transcript/${id}.${ext}`;
  }
  // Theme toggle
  (function initTheme(){
    const stored = localStorage.getItem('theme');
    if (stored === 'light' || stored === 'dark') document.documentElement.setAttribute('data-theme', stored);
    const btn = document.querySelector('#themeToggle');
    if (btn) btn.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  })();

  function tsFmt(ms) {
    const s = Math.max(0, Math.round(ms/1000));
    const m = Math.floor(s/60); const sec = s%60;
    return `${m}:${sec.toString().padStart(2,'0')}`;
  }

  function appendFinal(msg) {
    if (!finalListEl) return;
    const div = document.createElement('div');
    div.className = 'item';
    const t = document.createElement('div'); t.className='time'; t.textContent = tsFmt(msg.tsStart||0);
    const b = document.createElement('div'); b.className='bubble';
    if (msg.speaker) { const chip = document.createElement('span'); chip.className='chip'; chip.textContent = msg.speaker; b.appendChild(chip); }
    const span = document.createElement('span'); span.textContent = msg.text || ''; b.appendChild(span);
    div.appendChild(t); div.appendChild(b);
    finalListEl.appendChild(div);
    if (autoscrollEl?.checked) finalListEl.scrollTop = finalListEl.scrollHeight;
  }

  function readOpts() {
    const opts = {
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
    };
    return opts;
  }

  async function start() {
    if ((ws && ws.readyState === 1) || pendingStart) {
      return;
    }
    let sessionId = (sessionIdEl?.value || '').trim();
    if (!sessionId) {
      sessionId = 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2,6);
      if (sessionIdEl) sessionIdEl.value = sessionId;
    }
    const sourceInput = document.querySelector('input[name="source"]:checked');
    if (!sourceInput) {
      setStatus('入力ソースを選択してください');
      return;
    }
    const source = sourceInput.value;
    startBtn.disabled = true;
    stopBtn.disabled = true;
    setStatus('デバイス確認中...');
    // 既存ストリームをリセット
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (mediaStream2) { mediaStream2.getTracks().forEach(t => t.stop()); mediaStream2 = null; }
    try {
      if (source === 'mic') {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: !!ecEl?.checked, noiseSuppression: !!nsEl?.checked, autoGainControl: !!agcEl?.checked }
        });
      } else if (source === 'display') {
        mediaStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
        if (mediaStream.getAudioTracks().length === 0) {
          setStatus('共有に音声が含まれていません（「音声を共有」を有効に）');
        }
      } else { // mix
        const mic = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: !!ecEl?.checked, noiseSuppression: !!nsEl?.checked, autoGainControl: !!agcEl?.checked }
        });
        const disp = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
        mediaStream = mic;
        mediaStream2 = disp;
        if (disp.getAudioTracks().length === 0) {
          setStatus('画面共有に音声が含まれていないため、マイクのみで進行します');
          mediaStream2 = null;
        }
      }
    } catch (e) {
      setStatus('音声取得に失敗: ' + e.message);
      startBtn.disabled = false;
      pendingStart = null;
      return;
    }

    pendingStart = { source };
    awaitingReady = true;
    seq = 0;

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${proto}://${location.host}/ws/transcribe`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      const opts = readOpts();
      pendingStart.opts = opts;
      const lang = opts.language || 'ja';
      ws.send(JSON.stringify({
        type: 'start',
        sessionId,
        source,
        lang,
        format: 'pcm16',
        sampleRate: 16000,
        opts,
      }));
      setStatus('モデル準備中...');
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'partial') {
          partialEl.textContent = msg.text || '';
        } else if (msg.type === 'final') {
          partialEl.textContent = '';
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
            setTimeout(()=>{ if (vadStatusEl.textContent.startsWith('calib:')) vadStatusEl.textContent=''; }, 4000);
          }
        } else if (msg.type === 'conn') {
          if (connEl) connEl.textContent = String(msg.count ?? 0);
        } else if (msg.type === 'info') {
          if (msg.sessionId && sessionIdEl) {
            sessionIdEl.value = msg.sessionId;
            dlTxt.href = urlForTranscript('txt');
            dlJsonl.href = urlForTranscript('jsonl');
            if (dlSrt) dlSrt.href = urlForTranscript('srt');
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
      } catch (_) {}
    };
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
    const { source, opts = {} } = pendingStart;
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
      const makeLP = () => { const f = ac.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 7000; f.Q.value = 0.707; return f; };
      if (source === 'mix' && mediaStream2) {
        const src1 = ac.createMediaStreamSource(mediaStream);
        const src2 = ac.createMediaStreamSource(mediaStream2);
        const g1 = ac.createGain(); g1.gain.value = 0.7;
        const g2 = ac.createGain(); g2.gain.value = 0.7;
        const lp1 = makeLP(); const lp2 = makeLP();
        src1.connect(g1).connect(lp1).connect(workletNode);
        src2.connect(g2).connect(lp2).connect(workletNode);
      } else if (mediaStream) {
        const src = ac.createMediaStreamSource(mediaStream);
        const lp = makeLP();
        src.connect(lp).connect(workletNode);
      } else {
        throw new Error('入力ソースが見つかりません');
      }
      workletNode.connect(sink).connect(ac.destination);
      workletNode.port.onmessage = (ev) => {
        if (!ws || ws.readyState !== 1) return;
        const { type, ptsMs, payload } = ev.data || {};
        if (type !== 'frame' || !payload) return;
        const header = new ArrayBuffer(8);
        const view = new DataView(header);
        view.setUint32(0, seq++ , true);
        view.setUint32(4, ptsMs >>> 0, true);
        const body = payload;
        const out = new Uint8Array(header.byteLength + body.byteLength);
        out.set(new Uint8Array(header), 0);
        out.set(new Uint8Array(body), header.byteLength);
        ws.send(out);
        if (!audioStreaming) {
          audioStreaming = true;
          setStatus('録音中');
        }
        const i16 = new Int16Array(body);
        for (let i = 0; i < i16.length; i++) {
          waveBuf[wavePos++] = i16[i] / 32768;
          if (wavePos >= waveBuf.length) wavePos = 0;
        }
        if (!drawReq) drawReq = requestAnimationFrame(drawWave);
      };
      if (gainEl && !gainListenerAttached) {
        gainEl.addEventListener('input', () => {
          if (workletNode) workletNode.port.postMessage({ type: 'set', gain: Number(gainEl.value) });
        });
        gainListenerAttached = true;
      }
      captureActive = true;
      awaitingReady = false;
      pendingStart = null;
      stopBtn.disabled = false;
      audioStreaming = false;
      setStatus('録音準備完了');
      dlTxt.href = urlForTranscript('txt');
      dlJsonl.href = urlForTranscript('jsonl');
      if (dlSrt) dlSrt.href = urlForTranscript('srt');
    } catch (err) {
      console.error(err);
      setStatus('音声処理初期化に失敗: ' + err.message);
      pendingStart = null;
      awaitingReady = false;
      captureActive = false;
      stopBtn.disabled = false;
      stop();
    }
  }

  function cleanup() {
    if (workletNode) { try { workletNode.disconnect(); } catch {} workletNode = null; }
    if (ac) { try { ac.close(); } catch {} ac = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (mediaStream2) { mediaStream2.getTracks().forEach(t => t.stop()); mediaStream2 = null; }
    if (drawReq) { cancelAnimationFrame(drawReq); drawReq = null; }
    waveBuf.fill(0);
    pendingStart = null;
    awaitingReady = false;
    captureActive = false;
    audioStreaming = false;
    seq = 0;
    if (ws) { ws = null; }
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }

  function drawWave() {
    drawReq = null;
    if (!waveCanvas) return;
    const ctx = waveCanvas.getContext('2d');
    const W = waveCanvas.width, H = waveCanvas.height;
    ctx.clearRect(0,0,W,H);
    const styles = getComputedStyle(document.documentElement);
    const accent = (styles.getPropertyValue('--accent') || '#38bdf8').trim();
    const toRgba = (hex, alpha) => {
      const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      if (!m) return `rgba(56,189,248,${alpha})`;
      const [r, g, b] = m.slice(1).map(v => parseInt(v, 16));
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    };
    const gradient = ctx.createLinearGradient(0, 0, 0, H);
    gradient.addColorStop(0, toRgba(accent, 0.45));
    gradient.addColorStop(0.6, toRgba(accent, 0.10));
    gradient.addColorStop(1, toRgba(accent, 0));

    const areaPath = new Path2D();
    const strokePath = new Path2D();
    const mid = H / 2;
    const scale = H / 2 - 6;
    const N = waveBuf.length;
    areaPath.moveTo(0, mid);
    strokePath.moveTo(0, mid);
    for (let x = 0; x < W; x++) {
      const idx = (wavePos + Math.floor(x / W * N)) % N;
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

    const win = 3200; // 200ms @ 16k
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

  async function stop() {
    if (ws) {
      if (ws.readyState === 1) {
        try { ws.send(JSON.stringify({ type: 'stop' })); } catch {}
      }
      try { ws.close(); } catch {}
    }
    cleanup();
    setStatus('停止');
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
  if (calibBtn) calibBtn.addEventListener('click', () => {
    try {
      ws?.send(JSON.stringify({ type: 'calibrate', durationMs: 2000 }));
      if (vadStatusEl) vadStatusEl.textContent = 'calibrating...';
    } catch {}
  });
  // Hotwords load/save
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
      const r = await fetch('/api/hotwords', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
      const j = await r.json();
      if (hotStatusEl) { hotStatusEl.textContent = `保存しました (${j.count||0}語)`; setTimeout(()=> hotStatusEl.textContent='', 2000); }
    } catch (e) {
      if (hotStatusEl) hotStatusEl.textContent = '保存失敗';
    }
  }
  loadHotwords();
  if (saveHotBtn) saveHotBtn.addEventListener('click', saveHotwords);
  if (copyAllBtn) copyAllBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    const lines = [...(finalListEl?.querySelectorAll('.item .bubble')||[])].map(n=>n.textContent?.trim()||'');
    const txt = lines.join('\n').trim();
    let ok = false;
    try{ if (navigator.clipboard && window.isSecureContext){ await navigator.clipboard.writeText(txt); ok=true; } }catch{}
    if (!ok){ try{ const ta=document.createElement('textarea'); ta.value=txt; ta.style.position='fixed'; ta.style.opacity='0'; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta); ok=true; }catch{} }
    if (ok && hotStatusEl){ hotStatusEl.textContent='コピーしました'; setTimeout(()=> hotStatusEl.textContent='', 2000);} 
  });
  if (clearAllBtn) clearAllBtn.addEventListener('click', (e) => { e.preventDefault(); if (finalListEl) finalListEl.innerHTML=''; if (partialEl) partialEl.textContent=''; });
})();
