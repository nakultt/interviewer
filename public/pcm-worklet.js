class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pending = new Float32Array(512);
    this.offset = 0;
    this.input = [];
    this.position = 0;
    this.ratio = sampleRate / 16000;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    this.input.push(...channel);
    while (this.position + 1 < this.input.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      const sample = this.input[left] * (1 - fraction) + this.input[left + 1] * fraction;
      this.pending[this.offset] = sample;
      this.offset += 1;
      this.position += this.ratio;
      if (this.offset === 512) {
        const frame = this.pending;
        this.port.postMessage(frame, [frame.buffer]);
        this.pending = new Float32Array(512);
        this.offset = 0;
      }
    }
    const consumed = Math.floor(this.position);
    if (consumed > 0) {
      this.input = this.input.slice(consumed);
      this.position -= consumed;
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
