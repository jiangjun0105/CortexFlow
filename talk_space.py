"""Record from mic, send to the CortexFlow Space voice chat, play the spoken reply.

  python talk_space.py      # Enter to start/stop recording, q to quit
"""
import os
import tempfile
import json
import time

import httpx

import numpy as np
import sounddevice as sd
import soundfile as sf
from gradio_client import Client, handle_file

SR = 16_000
token = os.environ.get("HF_API_KEY") or open(".env").read().split("HF_API_KEY=")[1].split()[0]
client = Client("jiangjun0105/CortexFlow", token=token, verbose=False)


def server_timing(after):
    """Latest TIMING line the Space logged for a request received after `after` (epoch s)."""
    found = None
    try:
        with httpx.stream("GET", "https://huggingface.co/api/spaces/jiangjun0105/CortexFlow/logs/run",
                          headers={"Authorization": f"Bearer {token}"}, timeout=3) as r:
            for line in r.iter_lines():  # the log stream never closes; stop on read timeout
                if "TIMING" in line:
                    t = dict(kv.split("=") for kv in json.loads(line[5:])["data"].split()[1:])
                    if float(t["recv"]) >= after:
                        found = {k: float(v) for k, v in t.items()}
    except httpx.ReadTimeout:
        pass
    return found

while input("\n[Enter] to talk, q to quit: ").strip().lower() != "q":
    frames = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda data, *_: frames.append(data.copy())):
        input("recording... [Enter] to stop")
    t = time.time()  # you stopped speaking
    if not frames:
        continue
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, np.concatenate(frames), SR)

    print("thinking...", flush=True)
    sent = time.time()
    reply = client.predict(handle_file(path), api_name="/chat")
    returned = time.time()
    if not reply:
        print("(no audio returned)")
        continue
    audio, sr = sf.read(reply, dtype="float32")
    play = time.time()
    print(f"(waited {play - t:.1f}s from end of speech to playback, {len(audio) / sr:.1f}s reply)")
    srv = server_timing(sent - 5)  # 5s slack for clock skew
    if srv:
        print(f"  save wav        {sent - t:5.2f}s\n"
              f"  upload + queue  {srv['recv'] - sent:5.2f}s\n"
              f"  GPU attach      {srv['gpu'] - srv['recv']:5.2f}s\n"
              f"  prep input      {srv['gen_start'] - srv['gpu']:5.2f}s\n"
              f"  to first audio  {srv['first_audio'] - srv['gen_start']:5.2f}s   <- streaming could start playback here\n"
              f"  rest of reply   {srv['done'] - srv['first_audio']:5.2f}s  (generate + decode + save)\n"
              f"  return + dl     {returned - srv['done']:5.2f}s\n"
              f"  read wav        {play - returned:5.2f}s")
    sd.play(audio, sr)
    sd.wait()
