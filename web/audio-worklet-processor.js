class Pcm16DownsamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sampleRateIn = sampleRate; // AudioContext rate (likely 48000)
    this.sampleRateOut = 16000;
    this.decimFactor = Math.floor(this.sampleRateIn / this.sampleRateOut);
    this.frac = this.sampleRateIn / this.sampleRateOut;
    this.buffer = [];
    this.totalSamples = 0; // at 16k
    this.gain = 1.0;
    this.port.onmessage = (e) => {
      const d = e.data || {};
      if (d.type === 'set' && typeof d.gain === 'number') {
        this.gain = Math.max(0, Math.min(8, d.gain));
      }
    };
  }

  static get parameterDescriptors() { return []; }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch0 = input[0];
    if (!ch0) return true;
    // mono mix
    const frames = ch0.length;
    let mono = ch0;
    if (input.length > 1 && input[1]) {
      mono = new Float32Array(frames);
      const ch1 = input[1];
      for (let i = 0; i < frames; i++) mono[i] = 0.5 * (ch0[i] + ch1[i]);
    }
    // resample to 16k
    if (this.sampleRateIn === this.sampleRateOut) {
      this.buffer.push(Float32Array.from(mono));
    } else if (this.sampleRateIn === 48000 && this.sampleRateOut === 16000) {
      // 48k -> 16k: 簡易アンチエイリアス（3点移動平均）付きの 1/3 間引き
      const out = new Float32Array(Math.floor(mono.length / 3));
      let j = 0;
      for (let i = 0; i + 2 < mono.length; i += 3) {
        const m = (mono[i] + mono[i + 1] + mono[i + 2]) / 3;
        out[j++] = m;
      }
      this.buffer.push(out);
    } else {
      // linear interpolation resampler
      const ratio = this.sampleRateIn / this.sampleRateOut;
      const outLen = Math.floor(mono.length / ratio);
      const out = new Float32Array(outLen);
      let pos = 0;
      for (let i = 0; i < outLen; i++) {
        const idx = pos | 0;
        const frac = pos - idx;
        const s0 = mono[idx] || 0;
        const s1 = mono[idx + 1] || s0;
        out[i] = s0 + (s1 - s0) * frac;
        pos += ratio;
      }
      this.buffer.push(out);
    }
    // pack frames of 200ms => 3200 samples
    let total = 0;
    for (const b of this.buffer) total += b.length;
    const frameSamples = 3200; // 200ms @ 16k
    if (total >= frameSamples) {
      const out = new Int16Array(frameSamples);
      let needed = frameSamples;
      let offset = 0;
      while (needed > 0 && this.buffer.length > 0) {
        const head = this.buffer[0];
        const take = Math.min(needed, head.length);
        for (let i = 0; i < take; i++) {
          let s = head[i] * this.gain;
          s = Math.max(-1, Math.min(1, s));
          out[offset + i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        needed -= take;
        offset += take;
        if (take === head.length) this.buffer.shift();
        else this.buffer[0] = head.subarray(take);
      }
      const ptsMs = Math.round(((this.totalSamples) / this.sampleRateOut) * 1000);
      this.totalSamples += frameSamples;
      this.port.postMessage({ type: 'frame', ptsMs, payload: out.buffer }, [out.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm16-downsampler', Pcm16DownsamplerProcessor);
