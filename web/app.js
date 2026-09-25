// Slice 1: push-to-talk voice chat. Protocol: technical design §11, with PCM16 audio both ways.
const $ = (id) => document.getElementById(id);
const talk = $("talk"), caption = $("caption"), status = $("status"), metrics = $("metrics");

// Chunks arrive faster than realtime overall, but in bursts: no audio comes while the model writes
// its next 6 text tokens (gaps up to ~0.4 s). Delaying only the start of a reply absorbs that:
// measured over 4 replies, 30 ms -> 1 stall, 100 ms -> none. 150 ms for margin (calibration knob).
const PREBUFFER_S = 0.15;

let ws, serverBusy = false, releasedAt = 0, firstSoundMs = null, serverMs = null;

function setStatus(s) { status.textContent = s; }

function showMetrics() {
  if (!serverMs) return;
  metrics.textContent = `server: first audio ${serverMs.voice_first_audio ?? "–"} ms · generation ${serverMs.total} ms`
    + (firstSoundMs != null ? ` · you: release → first sound ${firstSoundMs} ms` : "");
}

// ---- playback: worklets/player.js queues 24 kHz PCM16 chunks and plays them back to back ----
const out = new AudioContext({ sampleRate: 24000 });
const player = out.audioWorklet.addModule("worklets/player.js").then(() => {
  const node = new AudioWorkletNode(out, "pcm-player", { processorOptions: { prebuffer: PREBUFFER_S * 24000 } });
  node.connect(out.destination);
  node.port.onmessage = ({ data }) => {
    if (data.type === "started") {
      firstSoundMs = Math.round(performance.now() - releasedAt + (out.outputLatency || 0) * 1000);
      showMetrics();
    } else if (data.type === "finished") setStatus("idle");
  };
  return node;
});
const toPlayer = async (msg, transfer = []) => (await player).port.postMessage(msg, transfer);

// ---- socket ----
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { talk.disabled = false; serverBusy = false; setStatus("idle"); };
  ws.onclose = (e) => {
    talk.disabled = true;
    if (e.code === 4000) return setStatus("opened in another tab");
    setStatus("reconnecting…");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return toPlayer({ type: "buffer", buffer: e.data }, [e.data]);
    const msg = JSON.parse(e.data);
    if (msg.type === "state") {
      serverBusy = msg.value !== "idle";
      if (serverBusy) setStatus(msg.value);
      else toPlayer({ type: "flush" }); // status goes idle when the player says it's finished
    } else if (msg.type === "caption") caption.textContent += msg.text;
    else if (msg.type === "metrics") { serverMs = msg.ms; showMetrics(); }
  };
}

// ---- push-to-talk: worklets/recorder.js captures 16 kHz PCM16 while the button/Space is held ----
let mic = null, pieces = [], holding = false, recording = false;

async function micReady() {
  if (mic) return mic;
  const ctx = new AudioContext({ sampleRate: 16000 }); // browser resamples the mic to 16 kHz
  await ctx.audioWorklet.addModule("worklets/recorder.js");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const node = new AudioWorkletNode(ctx, "pcm-recorder");
  ctx.createMediaStreamSource(stream).connect(node);
  node.port.onmessage = ({ data }) => {
    if (data.type === "chunk") pieces.push(data.pcm);
    else if (data.type === "stopped") sendUtterance();
  };
  return (mic = { ctx, node });
}

async function start() {
  if (holding || talk.disabled || serverBusy) return; // wait for the current reply to finish generating
  holding = true;
  talk.classList.add("on");
  setStatus("listening");
  out.resume();
  toPlayer({ type: "interrupt" }); // talking over the tail of the last reply stops it
  try {
    await micReady();
  } catch (err) {
    holding = false;
    talk.classList.remove("on");
    return setStatus(`mic unavailable: ${err.message}`);
  }
  if (!holding) return; // released while the first mic permission prompt was open
  pieces = [];
  recording = true;
  mic.node.port.postMessage({ type: "start" });
}

function stop() {
  if (!holding) return;
  holding = false;
  talk.classList.remove("on");
  if (!recording) return setStatus("idle");
  recording = false;
  releasedAt = performance.now();
  mic.node.port.postMessage({ type: "stop" }); // -> flushes the last chunk, then "stopped"
}

function sendUtterance() {
  const pcm = new Int16Array(pieces.reduce((n, p) => n + p.length, 0));
  let i = 0;
  for (const p of pieces) { pcm.set(p, i); i += p.length; }
  pieces = [];
  if (pcm.length < 16000 * 0.2) return setStatus("idle (too short)");
  caption.textContent = "";
  firstSoundMs = null;
  toPlayer({ type: "newReply" });
  ws.send(pcm.buffer);
}

talk.addEventListener("pointerdown", start);
talk.addEventListener("pointerup", stop);
talk.addEventListener("pointerleave", stop);
addEventListener("keydown", (e) => { if (e.code === "Space" && !e.repeat) { e.preventDefault(); start(); } });
addEventListener("keyup", (e) => { if (e.code === "Space") { e.preventDefault(); stop(); } });

connect();
