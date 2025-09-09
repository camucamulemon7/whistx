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
  // settings
  // Parakeet 用に Whisper 固有のUIは非表示（保持しても送らない）
  const languageEl = $('#language');
  const vadBackendEl = $('#vadBackend');
  const sileroThEl = $('#sileroThreshold');
  const autoVadEnableEl = $('#autoVadEnable');
  const autoVadWindowMsEl = $('#autoVadWindowMs');
  const calibBtn = $('#calibBtn');
  const vadStatusEl = $('#vadStatus');
  const utterSilEl = $('#utteranceSilenceMs');
  const forceUttEl = $('#forceUtteranceMs');
  const forceOvEl = $('#forceOverlapMs');
  const partialIntEl = $('#partialIntervalMs');
  const windowSecEl = $('#windowSeconds');
  const diarEl = $('#enableDiarization');
  const beamSizeEl = $('#beamSize');
  const patienceEl = $('#patience');
  const minFinalMsEl = $('#minFinalMs');
  const agcEl = $('#agc');
  const nsEl = $('#ns');
  const ecEl = $('#ec');
  const gainEl = $('#gain');
  const gainValEl = $('#gainVal');
  const levelNowEl = $('#levelNow');

  let ws = null;
  let ac = null;
  let workletNode = null;
  let mediaStream = null;      // mic または display を格納
  let mediaStream2 = null;     // mix 用の 2 本目（display）
  let seq = 0;
  let waveBuf = new Float32Array(16000 * 5); // 5秒表示用リングバッファ
  let wavePos = 0;
  let drawReq = null;
  // ユーティリティ
  function dbToAmp(db) { return Math.pow(10, db / 20); }
  function ampToDb(a) { return 20 * Math.log10(Math.max(1e-9, a)); }
  function setGainLabel() { if (gainEl && gainValEl) gainValEl.textContent = 'x' + Number(gainEl.value).toFixed(2); }
  setGainLabel();
  if (gainEl) gainEl.addEventListener('input', setGainLabel);

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
      vadBackend: vadBackendEl?.value || 'silero',
      sileroThreshold: Number(sileroThEl?.value || 0.5),
      autoVadEnable: autoVadEnableEl?.checked ? 1 : 0,
      autoVadWindowMs: Number(autoVadWindowMsEl?.value || 3000),
      utteranceSilenceMs: Number(utterSilEl?.value || 600),
      forceUtteranceMs: Number(forceUttEl?.value || 9000),
      forceOverlapMs: Number(forceOvEl?.value || 1200),
      partialIntervalMs: Number(partialIntEl?.value || 600),
      windowSeconds: Number(windowSecEl?.value || 8),
      enableDiarization: diarEl?.checked ? 1 : 0,
      beamSize: Number(beamSizeEl?.value || 10),
      patience: Number(patienceEl?.value || 1.1),
      minFinalMs: Number(minFinalMsEl?.value || 400),
    };
    return opts;
  }

  async function start() {
    let sessionId = (sessionIdEl?.value || '').trim();
    if (!sessionId) {
      sessionId = 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2,6);
      if (sessionIdEl) sessionIdEl.value = sessionId;
    }
    const source = document.querySelector('input[name="source"]:checked').value;
    try {
      setStatus('準備中...');
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
        mediaStream = mic;      // 1 本目
        mediaStream2 = disp;    // 2 本目
        if (disp.getAudioTracks().length === 0) {
          setStatus('画面共有に音声が含まれていないため、マイクのみで進行します');
          mediaStream2 = null;
        }
      }
    } catch (e) {
      setStatus('音声取得に失敗: ' + e.message);
      return;
    }

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${proto}://${location.host}/ws/transcribe`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'start',
        sessionId,
        source,
        lang: 'ja',
        format: 'pcm16',
        sampleRate: 16000,
        opts: readOpts(),
      }));
      setStatus('接続中...');
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
          setStatus(msg.message || 'info');
        }
      } catch (_) {}
    };
    ws.onclose = () => {
      setStatus('切断');
      cleanup();
    };
    ws.onerror = () => setStatus('エラー');

    // Audio pipeline
    ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    await ac.audioWorklet.addModule('/audio-worklet-processor.js');
    workletNode = new AudioWorkletNode(ac, 'pcm16-downsampler');
    if (gainEl) workletNode.port.postMessage({ type: 'set', gain: Number(gainEl.value) });
    const sink = ac.createGain();
    sink.gain.value = 0; // 無音でスピーカに流さない

    // 入力を接続（mix の場合は 2 系統をサミング）
    const makeLP = () => { const f = ac.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 7000; f.Q.value = 0.707; return f; };
    if (source === 'mix' && mediaStream2) {
      const src1 = ac.createMediaStreamSource(mediaStream);
      const src2 = ac.createMediaStreamSource(mediaStream2);
      const g1 = ac.createGain(); g1.gain.value = 0.7; // -3dB 程度
      const g2 = ac.createGain(); g2.gain.value = 0.7;
      const lp1 = makeLP(); const lp2 = makeLP();
      src1.connect(g1).connect(lp1).connect(workletNode);
      src2.connect(g2).connect(lp2).connect(workletNode);
    } else {
      const src = ac.createMediaStreamSource(mediaStream);
      const lp = makeLP();
      src.connect(lp).connect(workletNode);
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

      // 波形バッファに追記（縮約: int16 -> float）
      const i16 = new Int16Array(body);
      for (let i = 0; i < i16.length; i++) {
        waveBuf[wavePos++] = i16[i] / 32768;
        if (wavePos >= waveBuf.length) wavePos = 0;
      }
      if (!drawReq) drawReq = requestAnimationFrame(drawWave);
    };

    if (gainEl) gainEl.addEventListener('input', () => {
      if (workletNode) workletNode.port.postMessage({ type: 'set', gain: Number(gainEl.value) });
    });

    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus('録音中');
    dlTxt.href = urlForTranscript('txt');
    dlJsonl.href = urlForTranscript('jsonl');
    if (dlSrt) dlSrt.href = urlForTranscript('srt');
  }

  function cleanup() {
    if (workletNode) { try { workletNode.disconnect(); } catch {} workletNode = null; }
    if (ac) { try { ac.close(); } catch {} ac = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (mediaStream2) { mediaStream2.getTracks().forEach(t => t.stop()); mediaStream2 = null; }
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
    // 波形のみ（シンプル表示）
    const styles = getComputedStyle(document.documentElement);
    ctx.strokeStyle = styles.getPropertyValue('--accent') || '#1aa3ff';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const N = waveBuf.length;
    for (let x = 0; x < W; x++) {
      const idx = (wavePos + Math.floor(x / W * N)) % N;
      const v = waveBuf[idx] || 0;
      const y = H/2 - v * (H/2 - 2);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // 直近 200ms の RMS を表示
    const win = 3200; // 200ms @ 16k
    let sum2 = 0;
    for (let i = 0; i < win; i++) { const idx = (wavePos - 1 - i + N) % N; const v = waveBuf[idx] || 0; sum2 += v*v; }
    const rms = Math.sqrt(sum2 / win);
    const db = ampToDb(rms);
    if (levelNowEl) levelNowEl.textContent = `RMS: ${db.toFixed(1)} dBFS`;
    // 次フレーム
    drawReq = requestAnimationFrame(drawWave);
  }

  async function stop() {
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: 'stop' }));
      ws.close();
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
