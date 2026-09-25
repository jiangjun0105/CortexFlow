"""No-model checks for server.py's WebSocket protocol: python test_server.py"""
import numpy as np
from fastapi.testclient import TestClient

import server


class FakeAgent:
    def __init__(self):
        self.heard = []

    def reply(self, audio=None, note=None):
        self.heard.append(audio)
        yield "hi"
        yield np.full(1920, 0.5, np.float32)


def turn_messages(ws):
    msgs = []
    while True:
        m = ws.receive()
        msgs.append(m["bytes"] if m.get("bytes") is not None else __import__("json").loads(m["text"]))
        if msgs[-1] == {"type": "state", "value": "idle"}:
            return msgs


server.AGENT = fake = FakeAgent()
with TestClient(server.app) as client:
    assert "Hold to talk" in client.get("/").text  # static page served at /

    with client.websocket_connect("/ws") as ws:
        pcm = np.linspace(-1, 1, 16_000, dtype=np.float32)
        ws.send_bytes((pcm * 32767).astype("<i2").tobytes())  # PCM16 on the wire
        msgs = turn_messages(ws)
        kinds = [m["type"] + ":" + str(m.get("value", "")) if isinstance(m, dict) else "audio" for m in msgs]
        assert kinds == ["state:thinking", "caption:", "state:speaking", "audio", "metrics:", "state:idle"], kinds
        assert msgs[1]["text"] == "hi"
        assert np.array_equal(np.frombuffer(msgs[3], "<i2"), np.full(1920, int(0.5 * 32767), np.int16))
        assert msgs[4]["route"] == "chat" and {"voice_first_audio", "total"} <= msgs[4]["ms"].keys()
        assert np.abs(fake.heard[0] - pcm).max() < 1e-4  # the agent got what was sent (PCM16 rounding)

        ws.send_bytes(b"\x00\x01\x02")  # odd byte count, not PCM16: ignored, connection survives
        ws.send_bytes((pcm * 32767).astype("<i2").tobytes())
        assert turn_messages(ws)[-1] == {"type": "state", "value": "idle"} and len(fake.heard) == 2

        with client.websocket_connect("/ws") as ws2:  # second tab takes over, old one closed with 4000
            closed = ws.receive()
            assert closed == {"type": "websocket.close", "code": 4000, "reason": ""}, closed
            ws2.send_bytes((pcm * 32767).astype("<i2").tobytes())
            assert turn_messages(ws2)[-1] == {"type": "state", "value": "idle"}
print("ok")
