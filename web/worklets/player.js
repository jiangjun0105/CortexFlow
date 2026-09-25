// Plays 24 kHz PCM16 reply chunks back to back on the audio thread.
// Adapted from sb/sdks/convai/client worklets/audioConcatProcessor.js (PCM16 only, no ulaw), plus a
// start prebuffer: chunks arrive in bursts (no audio while the model writes text), so each reply waits
// for `prebuffer` samples, or the end of the reply, before it starts. Messages from the page:
//   buffer {buffer}  queue a chunk      newReply  drop the queue, prebuffer again
//   flush            reply fully sent   interrupt drop the queue
// Posts "started" when the first sample plays and "finished" once a flushed reply has fully played.
class PcmPlayer extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    this.prebuffer = processorOptions.prebuffer;
    this.reset();
    this.port.onmessage = ({ data }) => {
      if (data.type === "buffer") {
        this.queue.push(new Int16Array(data.buffer));
        this.queued += data.buffer.byteLength / 2;
      } else if (data.type === "flush") this.flushed = true;
      else if (data.type === "newReply" || data.type === "interrupt") this.reset();
    };
  }

  reset() {
    this.queue = [];
    this.current = null;
    this.cursor = this.queued = 0;
    this.started = this.flushed = this.done = false;
  }

  process(_, outputs) {
    const out = outputs[0][0];
    if (!this.started && this.queued > 0 && (this.queued >= this.prebuffer || this.flushed)) {
      this.started = true;
      this.port.postMessage({ type: "started" });
    }
    let i = 0;
    if (this.started) {
      for (; i < out.length; i++) {
        if (!this.current) {
          if (!this.queue.length) break; // underrun: silence until the next chunk
          this.current = this.queue.shift();
          this.cursor = 0;
        }
        out[i] = this.current[this.cursor++] / 32768;
        this.queued--;
        if (this.cursor >= this.current.length) this.current = null;
      }
    }
    out.fill(0, i);
    if (this.flushed && !this.done && this.queued === 0) {
      this.done = true;
      this.port.postMessage({ type: "finished" });
    }
    return true;
  }
}
registerProcessor("pcm-player", PcmPlayer);
