"""Does LFM2.5-Audio follow the notes? Run on the GPU: .venv/bin/python try_notes.py"""
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from lfm_audio import ChatState
from notes import REASSURE, SYSTEM, announce
from voice_agent import VoiceAgent

wav, sr = sf.read("question.wav", dtype="float32", always_2d=True)
pcm = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()
demo = json.load(open("fixtures/demo.json"))
dishes = {"view": "dishes", "meals": demo["meals"]}

agent = VoiceAgent(system=SYSTEM)
os.makedirs("voice_debug", exist_ok=True)


def fresh():
    agent.chat = ChatState(agent.processor)
    agent.chat.new_turn("system")
    agent.chat.add_text(SYSTEM)
    agent.chat.end_turn()


def run(name, **kw):
    t0, first, text, audio = time.perf_counter(), None, [], []
    for p in agent.reply(**kw):
        if isinstance(p, str):
            text.append(p)
        else:
            first = first or time.perf_counter()
            audio.append(p)
    ms = f"{(first - t0) * 1000:.0f}ms" if first else "no audio"
    sf.write(f"voice_debug/notes_{name}.wav", np.concatenate(audio) if audio else np.zeros(1), 24000)
    print(f"[{name}] first audio {ms}: {''.join(text)!r}", flush=True)


fresh(); run("plain", audio=pcm)
for i in range(3):
    fresh(); run(f"reassureA{i}", audio=pcm, note=REASSURE)
for i in range(3):
    fresh()
    agent.chat.new_turn("system"); agent.chat.add_text(REASSURE); agent.chat.end_turn()
    run(f"reassureB{i}", audio=pcm)
fresh(); run("announce", note=announce(dishes))
