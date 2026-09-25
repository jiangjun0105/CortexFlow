import time
import numpy as np, soundfile as sf, torch, torchaudio
import asr


def load(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16_000).numpy()


def timed(pcm):
    t = time.perf_counter()
    text = asr.transcribe(pcm)
    return text, (time.perf_counter() - t) * 1000


asr.warmup()

text, ms = timed(load("question.wav"))
print("question.wav", repr(text), f"{ms:.0f} ms")
assert len(text.split()) >= 3

for f in ("followup.wav", "q10.wav"):
    t, m = timed(load(f))
    print(f, repr(t), f"{m:.0f} ms")

assert len(asr.transcribe(np.zeros(16_000, np.float32))) < 20  # whisper may hallucinate a word on silence
assert asr.transcribe(np.zeros(0, np.float32)) == ""  # never raises
print("ok")
