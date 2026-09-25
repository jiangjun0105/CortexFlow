"""Breakfast voice agent backend: hands-free voice chat + ASR -> Jev router -> web agent lookups -> stage views.

  .venv/bin/python server.py      then open http://localhost:8000

WebSocket /ws (technical design §11, but VAD instead of push-to-talk, and PCM16 instead of float32):
  ↑ binary: mic audio, int16 LE 16kHz mono, streamed continuously in small chunks (~100 ms)
  ↑ text:   {"type": "playback", "value": "done"}  the reply finished playing in the browser
  ↓ binary: reply audio, int16 LE 24kHz mono, ~80ms chunks, streamed as they're decoded
  ↓ text:   {"type": "state", "value": "listening"|"hearing"|"thinking"|"speaking"},
            {"type": "caption", "text": piece}, {"type": "metrics", "route": "chat", "ms": {...}}

Silero VAD runs here on 512-sample blocks. When an utterance ends, the turn runs as a task while
mic frames keep arriving. The browser's mic is echo-cancelled, so the VAD keeps listening while the
agent speaks: speech during a reply is a barge-in (↓ {"type": "control", "name": "interrupt"} stops
playback, generation stops, and the new utterance becomes the next turn).
"""
import asyncio
import contextlib
import json
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from lfm_audio import BLOCK, MIC_SR, Endpointer

MAX_FRAME_S = 1  # one mic frame; the browser sends ~0.1 s
# True: drop the mic while the agent thinks/speaks (no barge-in). Flip it if the agent's own voice
# leaks past the browser's echo cancellation and it keeps interrupting itself.
HALF_DUPLEX = True  # the browser echo cancellation let the agent hear itself and barge in on its own replies
AGENT = None  # VoiceAgent, loaded at startup; tests set a fake before starting the app
VAD = None  # block (float32, 512 samples) -> speech prob; silero at startup, tests set a fake
ASR = None  # pcm -> text (mlx-whisper), loaded with the real agent; None: every turn is plain chat
ROUTE = None  # async (text, session) -> (route, target, scores): the Jev router
LOOKUP_TIMEOUT_S = 45  # a cold recommend is ~29 s (3 Nimble search+scrape); cached after that
MODEL_LOCK = asyncio.Lock()
# single session (design §1): a new connection replaces the old one
SESSION = {"ws": None, "screen": "welcome", "views": {}, "meals": [], "dish": None, "video": None,
           "steps": [], "step": 0, "transcript": []}
NAV = {"next", "back", "repeat", "goto", "seek", "show_video", "play", "pause"}


@contextlib.asynccontextmanager
async def lifespan(app):
    global AGENT, VAD, ASR, ROUTE
    if VAD is None:
        import warnings

        import torch
        from silero_vad import load_silero_vad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.jit.load deprecation inside silero
            silero = load_silero_vad()

        def VAD(block):
            with torch.no_grad():
                return float(silero(torch.from_numpy(block), MIC_SR))

        VAD.reset = silero.reset_states
    if AGENT is None:
        import asr
        import notes
        import router
        from voice_agent import VoiceAgent

        await asyncio.to_thread(asr.warmup)
        ASR, ROUTE = asr.transcribe, router.route
        await ROUTE("hello", SESSION)  # warm Jev's connection: the first call takes ~900 ms
        AGENT = await asyncio.to_thread(VoiceAgent, notes.SYSTEM)
    yield


app = FastAPI(lifespan=lifespan)


async def send(ws, msg):
    with contextlib.suppress(Exception):  # dead socket mid-turn: finish the turn, drop the sends
        await (ws.send_bytes(msg) if isinstance(msg, bytes) else ws.send_json(msg))


async def speak(ws, cancel, metrics, key, ms, **reply):
    """Stream one voice-agent reply (audio=/note=, see VoiceAgent.reply). Returns samples sent."""
    samples, said = 0, []
    async with MODEL_LOCK:
        gen = AGENT.reply(**reply)
        try:
            # one next() per thread hop, so each chunk goes out as soon as it's decoded
            while not cancel.is_set() and (item := await asyncio.to_thread(next, gen, None)) is not None:
                if isinstance(item, str):
                    said.append(item)
                    await send(ws, {"type": "caption", "text": item})
                    continue
                if key not in metrics:
                    metrics[key] = ms()
                    await send(ws, {"type": "state", "value": "speaking"})
                await send(ws, (np.clip(item, -1, 1) * 32767).astype("<i2").tobytes())
                samples += len(item)
        finally:
            await asyncio.to_thread(gen.close)  # keeps the turn in history even if we stopped early
    SESSION["transcript"].append(("assistant", "".join(said)))
    return samples


async def show(ws, view):
    """Send a stage view and remember it, so routing and a page refresh know what's on screen."""
    kind = view["view"]
    SESSION["views"][kind], SESSION["screen"] = view, kind
    if kind == "dishes":
        SESSION["meals"] = view["meals"]
    elif kind == "video":
        SESSION["video"] = view
    elif kind == "steps":
        SESSION["steps"], SESSION["step"] = view["steps"], 0
    await send(ws, {"type": "view", **view})


async def lookup(route, target, text):
    import recommender
    import search_agent

    if route == "recommend":
        return await recommender.recommend(text)
    if route == "video":
        meals = SESSION["meals"]
        if target is not None and 0 <= target < len(meals):
            SESSION["dish"] = meals[target]
        dish = recommender.label(SESSION["dish"]) if SESSION["dish"] else text
        return await search_agent.video(dish)
    video = SESSION["video"]
    return await search_agent.steps(video["main"]["id"], video["dish"])


async def nav(ws, route, target):
    """Step and video controls: screen changes only, no speech."""
    steps, video = SESSION["steps"], SESSION["views"].get("video")
    if route in ("next", "back", "repeat", "goto"):
        i = {"next": SESSION["step"] + 1, "back": SESSION["step"] - 1, "repeat": SESSION["step"]}.get(route, target)
        SESSION["step"] = max(0, min(len(steps) - 1, i or 0))
        if SESSION["screen"] != "steps":
            await send(ws, {"type": "view", **SESSION["views"]["steps"]})
            SESSION["screen"] = "steps"
        await send(ws, {"type": "control", "name": "goto_step", "index": SESSION["step"]})
    elif route in ("seek", "show_video") and video:
        await send(ws, {"type": "view", **video})
        SESSION["screen"] = "video"
        if route == "seek" and steps:
            t = steps[SESSION["step"]].get("video_start") or 0
            await send(ws, {"type": "control", "name": "video", "cmd": "seek", "t": t})
    elif route in ("play", "pause"):
        await send(ws, {"type": "control", "name": "video", "cmd": route})


async def turn(ws, heard, cancel):
    """One reply to one utterance (float32 pcm) or page request (dict: typed text / dish click),
    stopping early once `cancel` is set (barge-in).

    Returns the seconds of audio sent (0: the browser won't report playback)."""
    import notes
    import router

    t0 = time.perf_counter()  # end of speech, as decided by the VAD
    ms = lambda: round((time.perf_counter() - t0) * 1000)
    metrics, samples = {}, 0
    route, target, scores, audio, text = "chat", None, {}, None, ""
    await send(ws, {"type": "state", "value": "thinking"})
    if isinstance(heard, dict) and heard.get("name") == "select_dish":
        route, target = "video", int(heard["id"])
    else:
        if isinstance(heard, dict):
            text = heard.get("text", "")
        else:
            audio = heard
            if ASR:
                text = await asyncio.to_thread(ASR, heard)
                metrics["asr"] = ms()
        if text:
            SESSION["transcript"].append(("user", text))
            if ROUTE:
                route, target, scores = await ROUTE(text, SESSION)
                metrics["jev"] = ms()
    # the voice agent hears your voice; typed text goes in as the note
    said = {"audio": audio} if audio is not None else {"note": text}

    if route in router.LOOKUPS:
        job = asyncio.create_task(asyncio.wait_for(lookup(route, target, text), LOOKUP_TIMEOUT_S))
        samples += await speak(ws, cancel, metrics, "voice_first_audio", ms,
                               **(said if audio is not None else {}), note=notes.REASSURE)
        try:
            view = await job
        except Exception as e:  # design §14: apologise, keep the current screen
            view = None
            await send(ws, {"type": "error", "text": f"lookup failed: {e}"})
        metrics["module"] = ms()
        if view:
            await show(ws, view)
        if not cancel.is_set():
            samples += await speak(ws, cancel, metrics, "announce_first_audio", ms,
                                   note=notes.announce(view) if view else notes.FAILED)
    elif route in NAV:
        await nav(ws, route, target)
    else:
        samples += await speak(ws, cancel, metrics, "voice_first_audio", ms, **said)
    metrics["total"] = ms()  # generation done; the browser may still be playing
    await send(ws, {"type": "metrics", "route": route, "target": target, "scores": scores, "ms": metrics})
    return samples / 24_000


class Listener:
    """Per-connection VAD: mic frames in, turns out. Speech during a reply interrupts it."""

    def __init__(self, ws):
        self.ws, self.ep, self.residual = ws, Endpointer(), np.zeros(0, np.float32)
        self.busy, self.task, self.played, self.cancel = False, None, asyncio.Event(), asyncio.Event()

    async def listen(self):
        self.busy = False
        if HALF_DUPLEX:  # start fresh: nothing heard during the reply counts
            self.ep.reset()
            self.residual = self.residual[:0]
            if hasattr(VAD, "reset"):
                VAD.reset()
        await send(self.ws, {"type": "state", "value": "listening"})

    async def frame(self, pcm):
        if HALF_DUPLEX and self.busy:  # agent is thinking/speaking: drop the mic so it can't hear itself
            return
        x = np.concatenate([self.residual, pcm])
        n = len(x) // BLOCK * BLOCK
        self.residual = x[n:]
        for block in x[:n].reshape(-1, BLOCK):
            was_speaking = self.ep.speaking
            speech = self.ep.feed(block, VAD(block))
            if self.ep.speaking and not was_speaking:
                if self.busy:  # barge-in: stop the reply now, the new utterance becomes the next turn
                    self.cancel.set()
                    self.played.set()
                    await send(self.ws, {"type": "control", "name": "interrupt"})
                await send(self.ws, {"type": "state", "value": "hearing"})
            if speech is not None:
                self.start(speech)

    def start(self, heard):
        """Queue a turn for an utterance (pcm) or a page request (dict)."""
        self.busy = True
        prev, self.cancel, self.played = self.task, asyncio.Event(), asyncio.Event()
        self.task = asyncio.create_task(self._turn(heard, prev, self.cancel, self.played))

    async def _turn(self, speech, prev, cancel, played):
        if prev:  # an interrupted turn may still be winding down
            await prev
        try:
            secs = await turn(self.ws, speech, cancel)
        except Exception as e:  # design §14: report, then carry on with the next turn
            await send(self.ws, {"type": "error", "text": f"voice agent failed: {e}"})
            secs = 0
        if secs and not cancel.is_set():  # wait for playback; the timeout covers a closed/hidden tab
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(played.wait(), secs + 5)
        if self.task is asyncio.current_task() and not self.ep.speaking:  # only the latest turn, and not mid-speech
            await self.listen()
        elif self.task is asyncio.current_task():
            self.busy = False

    def playback_done(self):
        self.played.set()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    old, SESSION["ws"] = SESSION["ws"], ws
    if old is not None:
        with contextlib.suppress(Exception):
            await old.close(code=4000)  # second tab takes over (design §14)
    listener = Listener(ws)
    if view := SESSION["views"].get(SESSION["screen"]):  # page refresh: restore the stage
        await send(ws, {"type": "view", **view})
    await listener.listen()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                if data and len(data) % 2 == 0 and len(data) <= MAX_FRAME_S * MIC_SR * 2:
                    await listener.frame(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768)
            elif msg.get("text"):
                with contextlib.suppress(ValueError, AttributeError):
                    m = json.loads(msg["text"])
                    if m == {"type": "playback", "value": "done"}:
                        listener.playback_done()
                    elif m.get("type") == "text" or m.get("name") == "select_dish":
                        listener.start(m)  # welcome chip / dish card click: a full turn
                    elif m.get("name") == "goto_step":  # the page already switched; keep SESSION in sync
                        SESSION["step"], SESSION["screen"] = int(m["index"]), "steps"
                    elif m.get("name") in ("show_video", "show_steps"):
                        SESSION["screen"] = m["name"].removeprefix("show_")
    except WebSocketDisconnect:
        pass
    finally:
        if SESSION["ws"] is ws:
            SESSION["ws"] = None
        if listener.task:
            await listener.task  # let the turn finish so its history is saved (design §14)


app.mount("/", StaticFiles(directory="web", html=True), name="web")  # after /ws so it doesn't shadow it

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
