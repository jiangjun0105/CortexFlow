// Slice 1: hands-free voice chat. The mic streams continuously; the server runs the VAD.
// Protocol: see server.py (technical design §11, with VAD instead of push-to-talk and PCM16 audio).
const $ = (id) => document.getElementById(id);
const talk = $("talk"), caption = $("caption"), status = $("status"), metrics = $("metrics");

// Chunks arrive faster than realtime overall, but in bursts: no audio comes while the model writes
// its next 6 text tokens (gaps up to ~0.4 s). Delaying only the start of a reply absorbs that:
// measured over 4 replies, 30 ms -> 1 stall, 100 ms -> none. 150 ms for margin (calibration knob).
const PREBUFFER_S = 0.15;

let ws, micOn = false, speechEndedAt = 0, firstSoundMs = null, serverMs = null;

function setStatus(s) { status.textContent = micOn || !ws || ws.readyState !== 1 ? s : "paused (press Start)"; }

function showMetrics() {
  if (!serverMs) return;
  metrics.textContent = `server: first audio ${serverMs.voice_first_audio ?? "–"} ms · generation ${serverMs.total} ms`
    + (firstSoundMs != null ? ` · end of speech → first sound ${firstSoundMs} ms (+0.5 s VAD wait before that)` : "");
}

// ---- playback: worklets/player.js queues 24 kHz PCM16 chunks and plays them back to back ----
const out = new AudioContext({ sampleRate: 24000 });
const player = out.audioWorklet.addModule("worklets/player.js").then(() => {
  const node = new AudioWorkletNode(out, "pcm-player", { processorOptions: { prebuffer: PREBUFFER_S * 24000 } });
  node.connect(out.destination);
  node.port.onmessage = ({ data }) => {
    if (data.type === "started") {
      firstSoundMs = Math.round(performance.now() - speechEndedAt + (out.outputLatency || 0) * 1000);
      showMetrics();
    } else if (data.type === "finished") {
      ws.send(JSON.stringify({ type: "playback", value: "done" })); // server starts listening again
    }
  };
  return node;
});
const toPlayer = async (msg, transfer = []) => (await player).port.postMessage(msg, transfer);

// ---- socket ----
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { talk.disabled = false; };
  ws.onclose = (e) => {
    talk.disabled = true;
    if (e.code === 4000) return (status.textContent = "opened in another tab");
    status.textContent = "reconnecting…";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return toPlayer({ type: "buffer", buffer: e.data }, [e.data]);
    const msg = JSON.parse(e.data);
    UI.handle(msg); // orb state, stage views, metrics strip
    if (msg.type === "state") {
      if (msg.value === "thinking") { // the VAD decided you finished: a new reply starts
        speechEndedAt = performance.now();
        firstSoundMs = null;
        caption.textContent = "";
        toPlayer({ type: "newReply" });
      }
      setStatus(msg.value);
    } else if (msg.type === "caption") caption.textContent += msg.text;
    else if (msg.type === "metrics") { // generation done: play out whatever is still buffered
      serverMs = msg.ms;
      showMetrics();
      toPlayer({ type: "flush" });
    } else if (msg.type === "control" && msg.name === "interrupt") toPlayer({ type: "interrupt" }); // you talked over it
    else if (msg.type === "error") caption.textContent = msg.text;
  };
}

// ---- mic: worklets/recorder.js streams 16 kHz PCM16 ~100 ms chunks while the mic is on ----
let mic = null;

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
    if (data.type === "chunk" && micOn && ws.readyState === 1) ws.send(data.pcm.buffer);
  };
  return (mic = { ctx, node });
}

async function toggle() {
  if (talk.disabled) return;
  if (micOn) {
    micOn = false;
    mic.node.port.postMessage({ type: "stop" });
    talk.textContent = "Start";
    talk.classList.remove("on");
    UI.handle({ type: "state", value: "idle" });
    return setStatus("");
  }
  out.resume();
  try {
    await micReady();
  } catch (err) {
    return (status.textContent = `mic unavailable: ${err.message}`);
  }
  micOn = true;
  mic.node.port.postMessage({ type: "start" });
  talk.textContent = "Stop";
  talk.classList.add("on");
  setStatus("listening");
}

talk.addEventListener("click", toggle);
UI.onAction = (a) => ws.readyState === 1 && ws.send(JSON.stringify(a)); // clicks on cards/steps/chips
connect();
