"""Breakfast voice agent backend, slice 1: voice chat only (no ASR / router / search yet).

  .venv/bin/python server.py      then open http://localhost:8000

WebSocket /ws (technical design §11, but PCM16 instead of float32: the convai SDK worklets' format, half the bytes):
  ↑ binary: one utterance, int16 LE 16kHz mono
  ↓ binary: reply audio, int16 LE 24kHz mono, ~80ms chunks, streamed as they're decoded
  ↓ text:   {"type": "state", "value": "thinking"|"speaking"|"idle"}, {"type": "caption", "text": piece},
            {"type": "metrics", "route": "chat", "ms": {...}}
"""
import asyncio
import contextlib
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

MAX_UTTERANCE_S = 30
AGENT = None  # VoiceAgent, loaded at startup; tests set a fake before starting the app
MODEL_LOCK = asyncio.Lock()
SESSION = {"ws": None}  # single session (design §1): a new connection replaces the old one


@contextlib.asynccontextmanager
async def lifespan(app):
    global AGENT
    if AGENT is None:
        from voice_agent import VoiceAgent

        AGENT = await asyncio.to_thread(VoiceAgent)
    yield


app = FastAPI(lifespan=lifespan)


async def send(ws, msg):
    with contextlib.suppress(Exception):  # dead socket mid-turn: finish the turn, drop the sends
        await (ws.send_bytes(msg) if isinstance(msg, bytes) else ws.send_json(msg))


async def turn(ws, pcm):
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000)
    metrics = {}
    await send(ws, {"type": "state", "value": "thinking"})
    async with MODEL_LOCK:
        gen = AGENT.reply(audio=pcm)
        try:
            # one next() per thread hop, so each chunk goes out as soon as it's decoded
            while (item := await asyncio.to_thread(next, gen, None)) is not None:
                if isinstance(item, str):
                    await send(ws, {"type": "caption", "text": item})
                    continue
                if "voice_first_audio" not in metrics:
                    metrics["voice_first_audio"] = ms()
                    await send(ws, {"type": "state", "value": "speaking"})
                await send(ws, (np.clip(item, -1, 1) * 32767).astype("<i2").tobytes())
        finally:
            await asyncio.to_thread(gen.close)  # keeps the turn in history even if we stopped early
    metrics["total"] = ms()  # generation done; the browser may still be playing
    await send(ws, {"type": "metrics", "route": "chat", "ms": metrics})
    await send(ws, {"type": "state", "value": "idle"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    old, SESSION["ws"] = SESSION["ws"], ws
    if old is not None:
        with contextlib.suppress(Exception):
            await old.close(code=4000)  # second tab takes over (design §14)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data and len(data) % 2 == 0 and len(data) <= MAX_UTTERANCE_S * 16_000 * 2:
                await turn(ws, np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768)
    except WebSocketDisconnect:
        pass
    finally:
        if SESSION["ws"] is ws:
            SESSION["ws"] = None


app.mount("/", StaticFiles(directory="web", html=True), name="web")  # after /ws so it doesn't shadow it

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
