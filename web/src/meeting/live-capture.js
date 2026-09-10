import { arrayBufferToBase64, buildWebSocketUrl } from "../transcription/websocket.js";

const MAX_BUFFER_BYTES = 16 * 1024 * 1024;

export class LiveCapture {
  constructor({ path, streams, start, onEvent, onFatal, screenshot, packetMs = 1000 }) {
    this.path = path;
    this.packetSamples = Math.max(400, Math.min(16000, packetMs * 16));
    this.streams = streams;
    this.startPayload = start;
    this.onEvent = onEvent;
    this.onFatal = onFatal;
    this.screenshot = screenshot;
    this.pending = new Map();
    this.sequences = new Map();
    this.samples = new Map();
    this.nodes = [];
    this.sources = [];
    this.sessionId = "";
    this.bufferBytes = 0;
    this.running = false;
    this.stopping = false;
    this.closed = false;
    this.reconnecting = false;
    this.retry = 0;
    this.sent = new Set();
    this.acked = new Map();
  }

  async start() {
    this.context = new AudioContext({ sampleRate: 16000 });
    if (this.context.sampleRate !== 16000) {
      await this.context.close();
      throw new Error("live_sample_rate_not_supported");
    }
    await this.context.audioWorklet.addModule(new URL("../audio/capture-worklet.js", import.meta.url));
    await this.connect();
    if (this.closed) throw new Error("live_capture_closed");
    this.running = true;
    this.mute = this.context.createGain();
    this.mute.gain.value = 0;
    this.mute.connect(this.context.destination);
    const originFrame = Math.floor(this.context.currentTime * this.context.sampleRate);
    for (const [track, stream] of Object.entries(this.streams)) {
      const source = this.context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(this.context, "meeting-capture", { processorOptions: { originFrame, packetSamples: this.packetSamples } });
      node.port.onmessage = ({ data }) => {
        if (data.type === "pcm") this.enqueue(track, data);
        if (data.type === "flushed") node.resolveFlush?.();
      };
      node.onprocessorerror = () => this.fail(new Error("audio_capture_failed"));
      source.connect(node);
      node.connect(this.mute);
      this.sources.push(source);
      this.nodes.push(node);
    }
    await this.context.resume();
    this.pumpTimer = setInterval(() => this.pump(), 500);
    this.screenTimer = setInterval(() => this.captureScreen(), 3000);
  }

  async connect() {
    const ws = new WebSocket(buildWebSocketUrl(location, this.path));
    this.ws = ws;
    this.sent.clear();
    let ready = false;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { reject(new Error("live_connect_timeout")); ws.close(); }, 15_000);
      ws.addEventListener("open", () => {
        ws.send(JSON.stringify({ ...this.startPayload, type: "start", protocolVersion: 2,
          tracks: Object.keys(this.streams), resumeSessionId: this.sessionId || "" }));
      });
      ws.addEventListener("message", ({ data: raw }) => {
        let data;
        try { data = JSON.parse(raw); } catch { return; }
        if (data.type === "info" && data.message === "ready") {
          ready = true;
          clearTimeout(timeout);
          this.sessionId = data.sessionId;
          this.retry = 0;
          for (const [track, position] of Object.entries(data.tracks || {})) this.ack(track, position.seq, position.samples);
          this.onEvent(data);
          if (data.asrBackend === "qwen3_vllm") this.onEvent({ type: "transcript_snapshot", records: data.records || [] });
          else for (const record of data.records || []) this.onEvent({ ...record, sessionId: this.sessionId });
          resolve();
          this.pump();
          return;
        }
        if (data.type === "capture_ack") this.ack(data.track, data.seq, data.samples);
        if (data.type === "resend" || data.type === "backpressure") {
          this.sent.clear();
          this.sendAfter = performance.now() + (data.retryMs || 500);
        }
        if (data.type === "error" && !ready) {
          clearTimeout(timeout);
          reject(new Error(data.message || "live_start_failed"));
          ws.close();
          return;
        }
        if (data.type === "info" && data.message === "finalized") {
          this.finalized = true;
          this.resolveStop?.();
        }
        if (data.type === "error" && data.message === "finalize_failed") {
          this.wantFinalize = false;
          this.rejectStop?.(new Error("finalize_failed"));
        }
        this.onEvent(data);
      });
      ws.addEventListener("close", () => {
        clearTimeout(timeout);
        if (!ready) reject(new Error("live_connection_closed"));
        if (this.ws === ws && this.running && !this.closed && !this.finalized) this.reconnect();
      });
      ws.addEventListener("error", () => {
        if (!ready) { clearTimeout(timeout); reject(new Error("live_connection_failed")); }
      });
    });
  }

  async reconnect() {
    if (this.reconnecting || this.closed) return;
    this.reconnecting = true;
    this.onEvent({ type: "connection", message: "再接続中・音声を端末に保持しています" });
    while (!this.closed && this.running) {
      await new Promise((resolve) => setTimeout(resolve, Math.min(5000, 500 * 2 ** this.retry++)));
      if (this.closed) break;
      try {
        await this.connect();
        this.onEvent({ type: "connection", message: "接続が復旧しました" });
        break;
      } catch {
        if (this.retry >= 24) {
          this.fail(new Error("live_reconnect_failed"));
          break;
        }
      }
    }
    this.reconnecting = false;
  }

  enqueue(track, { pcm, sampleStart }) {
    if (this.closed) return;
    if (this.bufferBytes + pcm.byteLength > MAX_BUFFER_BYTES) {
      this.fail(new Error("live_buffer_full"));
      return;
    }
    const seq = this.sequences.get(track) || 0;
    this.sequences.set(track, seq + 1);
    this.samples.set(track, sampleStart + pcm.byteLength / 2);
    const key = `${track}:${seq}`;
    this.pending.set(key, { track, seq, sampleStart, pcm });
    this.bufferBytes += pcm.byteLength;
    this.pump();
  }

  ack(track, seq, samples) {
    this.acked.set(track, { seq, samples });
    for (const [key, packet] of this.pending) {
      if (packet.track === track && packet.seq <= seq) {
        this.bufferBytes -= packet.pcm.byteLength;
        this.pending.delete(key);
        this.sent.delete(key);
      }
    }
    this.pump();
  }

  pump() {
    if (this.closed || this.ws?.readyState !== WebSocket.OPEN || performance.now() < (this.sendAfter || 0)) return;
    for (const [key, packet] of this.pending) {
      if (this.sent.has(key)) continue;
      if (this.ws.bufferedAmount > 256_000 || this.sent.size >= 6) break;
      this.ws.send(JSON.stringify({ type: "audio", track: packet.track, seq: packet.seq,
        sampleStart: packet.sampleStart, pcm: arrayBufferToBase64(packet.pcm) }));
      this.sent.add(key);
    }
    this.onEvent({ type: "capture_state", pendingBytes: this.bufferBytes, samples: Math.max(0, ...this.samples.values()) });
    if (this.wantFinalize && !this.pending.size && this.stopSentTo !== this.ws) {
      this.stopSentTo = this.ws;
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  async captureScreen() {
    if (this.capturingScreen || this.closed || this.bufferBytes > 128_000 || this.ws?.readyState !== WebSocket.OPEN || this.ws.bufferedAmount > 256_000) return;
    this.capturingScreen = true;
    try {
      const image = await this.screenshot?.();
      if (image && !this.closed && this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "screen", timeMs: Math.round(Math.max(0, ...this.samples.values()) / 16), image: image.data, mimeType: image.mimeType }));
      }
    } catch {
      this.onEvent({ type: "connection", message: "画像の取得に失敗しました。音声の記録は継続しています。" });
    } finally {
      this.capturingScreen = false;
    }
  }

  async stop() {
    if (this.closed) throw new Error("live_capture_closed");
    if (!this.stopping) {
      this.stopping = true;
      clearInterval(this.screenTimer);
      await Promise.all(this.nodes.map((node) => new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("audio_flush_timeout")), 3000);
        node.resolveFlush = () => { clearTimeout(timeout); resolve(); };
        node.port.postMessage({ type: "flush" });
      })));
      await this.context.close();
    }
    const deadline = performance.now() + 120_000;
    while (this.pending.size || this.ws?.readyState !== WebSocket.OPEN) {
      if (this.closed || performance.now() > deadline) throw new Error("audio_upload_timeout");
      this.pump();
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (this.finalized) return;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("session_finalize_timeout")), 180_000);
      this.resolveStop = () => { clearTimeout(timeout); resolve(); };
      this.rejectStop = (error) => { clearTimeout(timeout); reject(error); };
      this.wantFinalize = true;
      this.stopSentTo = null;
      this.pump();
    });
    this.dispose();
  }

  fail(error) {
    if (this.closed) return;
    this.onFatal(error);
    // Keep the bounded, unacknowledged PCM available for a user download.
    this.dispose();
  }

  downloadPending() {
    // Keep the real sample offset in the filename rather than padding hours of
    // silence. Download one contiguous pending range per input track.
    for (const track of Object.keys(this.streams)) {
      const packets = [...this.pending.values()].filter((packet) => packet.track === track).sort((a, b) => a.sampleStart - b.sampleStart);
      if (!packets.length) continue;
      const bytes = packets.reduce((sum, packet) => sum + packet.pcm.byteLength, 0);
      const header = new ArrayBuffer(44);
      const data = new DataView(header);
      const ascii = (offset, text) => [...text].forEach((char, index) => data.setUint8(offset + index, char.charCodeAt(0)));
      ascii(0, "RIFF"); data.setUint32(4, 36 + bytes, true); ascii(8, "WAVE"); ascii(12, "fmt ");
      data.setUint32(16, 16, true); data.setUint16(20, 1, true); data.setUint16(22, 1, true);
      data.setUint32(24, 16000, true); data.setUint32(28, 32000, true); data.setUint16(32, 2, true); data.setUint16(34, 16, true);
      ascii(36, "data"); data.setUint32(40, bytes, true);
      const url = URL.createObjectURL(new Blob([header, ...packets.map((packet) => packet.pcm)], { type: "audio/wav" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = `${track}-from-${Math.round(packets[0].sampleStart / 16)}ms.wav`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }

  dispose() {
    this.closed = true;
    this.running = false;
    clearInterval(this.pumpTimer);
    clearInterval(this.screenTimer);
    for (const source of this.sources) source.disconnect();
    for (const node of this.nodes) node.disconnect();
    if (this.context?.state !== "closed") this.context?.close();
    this.ws?.close();
  }
}
