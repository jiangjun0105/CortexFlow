"""Breakfast voice agent backend, slice 1: hands-free voice chat (no ASR / router / search yet).

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
HALF_DUPLEX = False
AGENT = None  # VoiceAgent, loaded at startup; tests set a fake before starting the app
VAD = None  # block (float32, 512 samples) -> speech prob; silero at startup, tests set a fake
MODEL_LOCK = asyncio.Lock()
SESSION = {"ws": None}  # single session (design §1): a new connection replaces the old one


@contextlib.asynccontextmanager
async def lifespan(app):
    global AGENT, VAD
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
        from voice_agent import VoiceAgent

        AGENT = await asyncio.to_thread(VoiceAgent)
    yield


app = FastAPI(lifespan=lifespan)


async def send(ws, msg):
    with contextlib.suppress(Exception):  # dead socket mid-turn: finish the turn, drop the sends
        await (ws.send_bytes(msg) if isinstance(msg, bytes) else ws.send_json(msg))


async def turn(ws, pcm, cancel):
    """One reply to one utterance, stopping early once `cancel` is set (barge-in).

    Returns the seconds of audio sent (0: the browser won't report playback)."""
    t0 = time.perf_counter()  # end of speech, as decided by the VAD
    ms = lambda: round((time.perf_counter() - t0) * 1000)
    metrics, samples = {}, 0
    await send(ws, {"type": "state", "value": "thinking"})
    async with MODEL_LOCK:
        gen = AGENT.reply(audio=pcm)
        try:
            # one next() per thread hop, so each chunk goes out as soon as it's decoded
            while not cancel.is_set() and (item := await asyncio.to_thread(next, gen, None)) is not None:
                if isinstance(item, str):
                    await send(ws, {"type": "caption", "text": item})
                    continue
                if "voice_first_audio" not in metrics:
                    metrics["voice_first_audio"] = ms()
                    await send(ws, {"type": "state", "value": "speaking"})
                await send(ws, (np.clip(item, -1, 1) * 32767).astype("<i2").tobytes())
                samples += len(item)
        finally:
            await asyncio.to_thread(gen.close)  # keeps the turn in history even if we stopped early
    metrics["total"] = ms()  # generation done; the browser may still be playing
    await send(ws, {"type": "metrics", "route": "chat", "ms": metrics})
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
                self.busy = True
                prev, self.cancel, self.played = self.task, asyncio.Event(), asyncio.Event()
                self.task = asyncio.create_task(self._turn(speech, prev, self.cancel, self.played))

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
                with contextlib.suppress(ValueError):
                    if json.loads(msg["text"]) == {"type": "playback", "value": "done"}:
                        listener.playback_done()
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
