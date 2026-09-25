"""No-model checks for the voice-mode helpers: python test_lfm_audio.py"""
import numpy as np

from lfm_audio import BLOCK_S, MIC_SR, trim_pauses, utterance


def tone(secs, sr, amp=0.5):
    return (amp * np.sin(np.arange(int(secs * sr)) * 2 * np.pi * 220 / sr)).astype("float32")


def silence(secs, sr):
    return np.zeros(int(secs * sr), "float32")


# 80ms reply chunks: 9 leading silent dropped, 8-chunk pause capped at 3, speech untouched
sr, n = 24_000, 1920
chunks = lambda sig: [sig[i:i + n] for i in range(0, len(sig), n)]
x = chunks(np.concatenate([silence(0.72, sr), tone(0.8, sr, 0.05), silence(0.64, sr), tone(0.8, sr, 0.05)]))
y = list(trim_pauses(x))
assert len(y) == 10 + 3 + 10, len(y)
assert np.abs(y[0]).max() > 0.01  # starts with speech

# VAD state machine (fake prob = "is there signal"): ignores a short click, captures the
# utterance with preroll, ends after 0.5s of silence, caps at max_s
n = int(MIC_SR * BLOCK_S)
blocks = lambda sig: [sig[i:i + n] for i in range(0, len(sig) - n + 1, n)]
prob = lambda b: float(np.abs(b).max() > 0.01)
mic = np.concatenate([silence(0.5, MIC_SR), tone(0.06, MIC_SR), silence(0.5, MIC_SR),
                      tone(1.5, MIC_SR, 0.1), silence(1.0, MIC_SR), tone(1.0, MIC_SR)])
got = utterance(iter(blocks(mic)), prob)
assert got is not None
assert 1.5 + 0.5 <= len(got) / MIC_SR <= 1.5 + 0.5 + 0.32 + 0.1, len(got) / MIC_SR  # speech + end tail + preroll
assert np.abs(got[: int(0.3 * MIC_SR)]).max() < 0.2  # preroll is the quiet lead-in, not the 0.5 click
assert utterance(iter(blocks(silence(2, MIC_SR))), prob) is None
assert abs(len(utterance(iter(blocks(tone(15, MIC_SR, 0.1))), prob)) / MIC_SR - 10) < 0.1  # max_s cap
print("ok")
