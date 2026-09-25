"""No-model checks for server.py's WebSocket protocol and server-side VAD: python test_server.py"""
import json

import numpy as np
from fastapi.testclient import TestClient

import server


class FakeAgent:
    def __init__(self):
        self.heard, self.speak = [], True

    def reply(self, audio=None, note=None):
        self.heard.append(audio)
        yield "hi"
        if self.speak:
            yield np.full(1920, 0.5, np.float32)


def pcm16(x):
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def stream(ws, *parts):  # send audio like the browser: 100 ms frames
    x = np.concatenate(parts)
    for i in range(0, len(x), 1600):
        ws.send_bytes(pcm16(x[i:i + 1600]))


def until(ws, want):
    msgs = []
    while True:
        m = ws.receive()
        msgs.append(m["bytes"] if m.get("bytes") is not None else json.loads(m["text"]))
        if msgs[-1] == want:
            return msgs


silence = lambda s: np.zeros(int(16_000 * s), np.float32)
tone = lambda s: (0.3 * np.sin(np.arange(int(16_000 * s)) * 0.1)).astype(np.float32)
kinds = lambda msgs: [m["type"] + ":" + str(m.get("value", "")) if isinstance(m, dict) else "audio" for m in msgs]
LISTENING = {"type": "state", "value": "listening"}
DONE = json.dumps({"type": "playback", "value": "done"})

server.AGENT = fake = FakeAgent()
server.VAD = lambda block: float(np.abs(block).max() > 0.05)  # loud = speech
with TestClient(server.app) as client:
    assert "Start" in client.get("/").text  # static page served at /

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json() == LISTENING
        stream(ws, silence(0.5), tone(1.0), silence(1.0))
        msgs = []
        while not (msgs and isinstance(msgs[-1], dict) and msgs[-1]["type"] == "metrics"):
            m = ws.receive()
            msgs.append(m["bytes"] if m.get("bytes") is not None else json.loads(m["text"]))
        assert kinds(msgs) == ["state:hearing", "state:thinking", "caption:", "state:speaking", "audio", "metrics:"], kinds(msgs)
        assert np.array_equal(np.frombuffer(msgs[4], "<i2"), np.full(1920, int(0.5 * 32767), np.int16))
        assert {"voice_first_audio", "total"} <= msgs[5]["ms"].keys()
        heard = len(fake.heard[0]) / 16_000  # 1 s of speech + 0.5 s end silence + lead-in, minus framing
        assert 1.4 <= heard <= 2.0, heard

        stream(ws, tone(1.0), silence(1.0))  # agent is "speaking": the mic is ignored (half duplex)
        ws.send_bytes(b"\x00\x01\x02")  # odd byte count, not PCM16: ignored, connection survives
        ws.send_text(DONE)  # browser finished playing
        assert ws.receive_json() == LISTENING and len(fake.heard) == 1

        stream(ws, tone(0.8), silence(0.8))  # listening again: next turn
        until(ws, {"type": "state", "value": "speaking"})
        assert len(fake.heard) == 2
        ws.send_text(DONE)
        until(ws, LISTENING)

        fake.speak = False  # a reply with no audio: no playback report comes, back to listening anyway
        stream(ws, tone(0.8), silence(0.8))
        assert kinds(until(ws, LISTENING)) == ["state:hearing", "state:thinking", "caption:", "metrics:", "state:listening"]
        fake.speak = True

        with client.websocket_connect("/ws") as ws2:  # second tab takes over, old one closed with 4000
            assert ws.receive() == {"type": "websocket.close", "code": 4000, "reason": ""}
            assert ws2.receive_json() == LISTENING
print("ok")
