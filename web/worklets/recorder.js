// Captures mic audio as PCM16 on the audio thread, posting ~100 ms chunks while recording.
// Adapted from sb/sdks/convai/client worklets/rawAudioProcessor.js (PCM16 only; push-to-talk
// start/stop instead of mute). The AudioContext runs at 16 kHz, so there's no resampling here.
// ponytail: Chrome resamples the mic to the 16 kHz context; Firefox needs the SDK's libsamplerate path
class PcmRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.buf = [];
    this.port.onmessage = ({ data }) => {
      if (data.type === "start") {
        this.recording = true;
        this.buf = [];
      } else if (data.type === "stop") {
        this.recording = false;
        this.send();
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  send() {
    if (!this.buf.length) return;
    const pcm = new Int16Array(this.buf.length);
    for (let i = 0; i < pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, this.buf[i]));
      pcm[i] = s < 0 ? s * 32768 : s * 32767;
    }
    this.port.postMessage({ type: "chunk", pcm }, [pcm.buffer]);
    this.buf = [];
  }

  process(inputs) {
    const ch = inputs[0][0];
    if (this.recording && ch) {
      this.buf.push(...ch);
      if (this.buf.length >= sampleRate / 10) this.send();
    }
    return true;
  }
}
registerProcessor("pcm-recorder", PcmRecorder);
