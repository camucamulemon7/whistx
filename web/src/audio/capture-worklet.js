// AudioContext requests 16 kHz; the browser resamples the media stream before
// this processor. Samples, rather than timer callbacks, define the audio clock.
class MeetingCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.samples = new Int16Array(options.processorOptions?.packetSamples || 16000);
    this.used = 0;
    this.offset = 0;
    this.active = true;
    this.originFrame = options.processorOptions?.originFrame ?? currentFrame;
    this.started = false;
    this.port.onmessage = ({ data }) => {
      if (data?.type === "flush") {
        this.emit();
        this.active = false;
        this.port.postMessage({ type: "flushed" });
      }
    };
  }

  emit() {
    if (!this.used) return;
    const pcm = this.samples.slice(0, this.used);
    this.port.postMessage({ type: "pcm", sampleStart: this.offset, pcm: pcm.buffer }, [pcm.buffer]);
    this.offset += this.used;
    this.used = 0;
  }

  append(value) {
    const sample = Math.max(-1, Math.min(1, value));
    this.samples[this.used++] = Math.round(sample * (sample < 0 ? 32768 : 32767));
    if (this.used === this.samples.length) this.emit();
  }

  process(inputs, outputs) {
    if (!this.active) return false;
    if (!this.started) {
      // Nodes for mic and screen may be connected in different render quanta.
      // Padding to their shared origin keeps both tracks on one timeline.
      const padding = Math.max(0, currentFrame - this.originFrame);
      for (let i = 0; i < padding; i += 1) this.append(0);
      this.started = true;
    }
    const input = inputs[0] || [];
    const length = input[0]?.length || outputs[0]?.[0]?.length || 128;
    for (let i = 0; i < length; i += 1) {
      let value = 0;
      for (const channel of input) value += channel[i] || 0;
      this.append(input.length ? value / input.length : 0);
    }
    return true;
  }
}

registerProcessor("meeting-capture", MeetingCaptureProcessor);
