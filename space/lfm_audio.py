"""Try LiquidAI/LFM2.5-Audio-1.5B locally.

  python lfm_audio.py tts "Hello there" -o out.wav [--voice "UK female"]
  python lfm_audio.py asr in.wav
  python lfm_audio.py chat in.wav -o answer.wav     # speech in -> text + speech out
  python lfm_audio.py voice                         # live multi-turn voice chat via mic/speakers
"""
import argparse
import itertools
import logging
import queue
import time
from collections import deque
from functools import lru_cache

import numpy as np
import soundfile as sf

# harmless notices, silenced before torch loads: torch.distributed's macOS note (pulled in by liquid_audio),
# and transformers' "causal_conv1d not installed" (a CUDA-only kernel; the PyTorch fallback is what runs on mps)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality

REPO = "LiquidAI/LFM2.5-Audio-1.5B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ponytail: liquid_audio hardcodes .cuda() on the detokenizer; redirect to DEVICE until upstream fixes it
torch.nn.Module.cuda = lambda self, *a, **k: self.to(DEVICE)


@lru_cache
def load():
    processor = LFM2AudioProcessor.from_pretrained(REPO, device=DEVICE).eval()
    model = LFM2AudioModel.from_pretrained(REPO, device=DEVICE).eval()
    return processor, model


MIC_SR, BLOCK = 16_000, 512  # silero needs 512-sample (32ms) frames at 16kHz
BLOCK_S = BLOCK / MIC_SR


def squash_silence(x, sr=24_000, max_gap=0.25, frame=0.02):
    """Drop leading silence and cap pauses at max_gap (the model emits ~0.5-1s ones)."""
    n = int(sr * frame)
    frames = x[: len(x) // n * n].reshape(-1, n)
    energy = np.sqrt((frames ** 2).mean(1))
    silent = energy < 0.1 * energy.max()
    keep, run = [], 0
    for i, s in enumerate(silent):
        run = run + 1 if s else 0
        keep.append(not s or (run * frame <= max_gap and run <= i))  # run == i+1 means leading silence
    return frames[keep].ravel()


def utterance(blocks, speech_prob, start=0.5, stop=0.3, min_speech_s=0.12, end_silence_s=0.5,
              preroll_s=0.32, max_s=10.0):
    """First utterance in a stream of mic blocks, or None if the stream ends.

    Same state machine and defaults as sb_convai's VADDetector: begins after min_speech_s with
    prob >= start (keeping preroll_s before it), ends after end_silence_s with prob < stop, or at max_s.
    """
    pre = deque(maxlen=round((preroll_s + min_speech_s) / BLOCK_S))
    speech, loud, quiet = None, 0, 0
    for b in blocks:
        p = speech_prob(b)
        if speech is None:
            pre.append(b)
            loud = loud + 1 if p >= start else 0
            if loud * BLOCK_S >= min_speech_s:
                speech = list(pre)
                print("(hearing you...)", flush=True)
        else:
            speech.append(b)
            quiet = 0 if p >= stop else quiet + 1
            if quiet * BLOCK_S >= end_silence_s or len(speech) * BLOCK_S >= max_s:
                return np.concatenate(speech)
    return None


def mic_blocks(sd):
    q = queue.Queue()
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", blocksize=BLOCK,
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        while True:
            yield q.get()


@torch.no_grad()
def voice(debug=False):
    """Hands-free multi-turn speech chat: talk whenever, reply plays through speakers. Ctrl+C to quit.

    debug: print mic levels + reply text, save voice_debug/turnN_{heard,reply}.wav
    """
    import os
    import warnings

    import sounddevice as sd
    from silero_vad import load_silero_vad

    processor, model = load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.jit.load deprecation inside silero
        vad = load_silero_vad()
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()
    print("ready; Ctrl+C to quit")
    if debug:
        os.makedirs("voice_debug", exist_ok=True)

    for turn in itertools.count(1):
        print("\nlistening...", flush=True)
        vad.reset_states()
        blocks = mic_blocks(sd)  # mic is closed while the reply plays, so it can't hear itself
        speech = utterance(blocks, lambda b: float(vad(torch.from_numpy(b), MIC_SR)))
        blocks.close()
        print(f"(heard {len(speech) / MIC_SR:.1f}s, thinking...)", flush=True)
        if debug:
            sf.write(f"voice_debug/turn{turn}_heard.wav", speech, MIC_SR)
            print(f"  [debug] mic peak {np.abs(speech).max():.3f} rms {np.sqrt(np.mean(speech ** 2)):.4f}")
        wav = torch.from_numpy(speech).unsqueeze(0)

        chat.new_turn("user")
        chat.add_audio(wav, MIC_SR)
        chat.end_turn()
        chat.new_turn("assistant")

        # text tokens are the model's script for the speech; kept for history, not shown
        text, audio, modality = [], [], []
        for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
            if t.numel() == 1:
                text.append(t)
                modality.append(LFMModality.TEXT)
            else:
                audio.append(t)
                modality.append(LFMModality.AUDIO_OUT)

        # keep the reply in history so the next turn has context (same as liquid_audio's demo)
        if modality:
            empty = torch.empty((8, 0), dtype=torch.long, device=processor.device)
            chat.append(text=torch.stack(text, 1) if text else empty[:1],
                        audio_out=torch.stack(audio, 1) if audio else empty,
                        modality_flag=torch.tensor(modality))
        chat.end_turn()

        if debug:
            print(f"  [debug] reply text: {processor.text.decode(torch.cat(text)) if text else ''!r}")
        if len(audio) <= 1:
            print(f"(model returned no audio; text: {processor.text.decode(torch.cat(text)) if text else ''!r})")
            continue
        # ponytail: plays after full generation (~2x realtime); stream chunks if first-audio latency matters
        reply = squash_silence(processor.decode(torch.stack(audio[:-1], 1).unsqueeze(0)).float().cpu()[0].numpy())
        if debug:
            sf.write(f"voice_debug/turn{turn}_reply.wav", reply, 24_000)
        sd.play(reply, 24_000)
        sd.wait()


def run(system, user_text=None, wav_path=None, out=None, mode="sequential", **gen):
    processor, model = load()

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(system)
    chat.end_turn()
    chat.new_turn("user")
    if wav_path:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        chat.add_audio(torch.from_numpy(wav.mean(1)).unsqueeze(0), sr)  # mono
    if user_text:
        chat.add_text(user_text)
    chat.end_turn()
    chat.new_turn("assistant")

    generate = model.generate_interleaved if mode == "interleaved" else model.generate_sequential
    audio, text = [], ""
    timing = {"gen_start": time.time(), "first_audio": None}  # epoch seconds
    with torch.no_grad():
        for t in generate(**chat, max_new_tokens=512, **gen):
            if t.numel() == 1:
                text += processor.text.decode(t)
                print(processor.text.decode(t), end="", flush=True)
            else:
                timing["first_audio"] = timing["first_audio"] or time.time()
                audio.append(t)
    print()

    if out and len(audio) > 1:
        codes = torch.stack(audio[:-1], 1).unsqueeze(0)  # last frame is end-of-audio
        sf.write(out, processor.decode(codes).float().cpu()[0].numpy(), 24_000)
        print(f"saved {out}")
        return text.removesuffix("<|im_end|>"), out, timing
    return text.removesuffix("<|im_end|>"), None, timing


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["tts", "asr", "chat", "voice"])
    p.add_argument("input", nargs="?", help="text for tts, wav path for asr/chat")
    p.add_argument("-o", "--out", default="out.wav")
    p.add_argument("--voice", default="US female", help="US male | US female | UK male | UK female")
    p.add_argument("--debug", action="store_true", help="voice: print levels/reply text, save wavs to voice_debug/")
    a = p.parse_args()
    print(f"device: {DEVICE}")

    if a.task == "voice":
        try:
            voice(a.debug)
        except KeyboardInterrupt:
            print("\nbye")
    elif a.input is None:
        p.error(f"{a.task} needs an input")
    elif a.task == "tts":
        run(f"Perform TTS. Use the {a.voice} voice.", user_text=a.input, out=a.out,
            audio_temperature=0.8, audio_top_k=64)
    elif a.task == "asr":
        run("Perform ASR.", wav_path=a.input)
    else:
        run("Respond with interleaved text and audio.", wav_path=a.input, out=a.out,
            mode="interleaved", audio_temperature=1.0, audio_top_k=4)
