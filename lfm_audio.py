"""Try LiquidAI/LFM2.5-Audio-1.5B locally.

  python lfm_audio.py tts "Hello there" -o out.wav [--voice "UK female"]
  python lfm_audio.py asr in.wav
  python lfm_audio.py chat in.wav -o answer.wav     # speech in -> text + speech out
  python lfm_audio.py voice [--no-show-text] [--no-transcript] [--debug]   # live multi-turn voice chat
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


def trim_pauses(chunks, max_silent=3, floor=0.003):
    """Drop leading silent chunks and cap pauses at max_silent chunks (80ms each).

    The model pads replies with ~0.5-1s of silence (chunk rms <= 0.002; speech is 0.02-0.06).
    Text pieces (str) mixed into the stream pass straight through.
    """
    started, run = False, 0
    for c in chunks:
        if isinstance(c, str):
            yield c
            continue
        if np.sqrt(np.mean(c ** 2)) < floor:
            run += 1
            if not started or run > max_silent:
                continue
        else:
            started, run = True, 0
        yield c


class Speaker:
    """Streams float32 chunks to the default output as they arrive.

    Same queue + audio-thread idea as convai_sdk's DefaultAudioInterface, but callback-driven:
    an underrun plays silence instead of blocking, and there's no 16kHz resampling.
    """

    def __init__(self, sd, sr=24_000):
        self.q, self.buf = queue.Queue(), np.zeros(0, np.float32)
        self.first_played = None  # perf_counter() when a sample first reaches the device; reset per reply
        self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", callback=self._fill)
        self.stream.start()

    def _fill(self, out, frames, *_):
        while len(self.buf) < frames and not self.q.empty():
            self.buf = np.concatenate([self.buf, self.q.get_nowait()])
        n = min(frames, len(self.buf))
        if n and self.first_played is None:
            self.first_played = time.perf_counter()
        out[:n, 0], out[n:, 0] = self.buf[:n], 0
        self.buf = self.buf[n:]

    def drain(self):
        """Block until everything queued has been played."""
        while not self.q.empty() or len(self.buf):
            time.sleep(0.02)
        time.sleep(self.stream.latency)


class Endpointer:
    """Push-based VAD state machine: feed() one 512-sample block and its speech prob at a time.

    Same state machine and defaults as sb_convai's VADDetector: begins after min_speech_s with
    prob >= start (keeping preroll_s before it), ends after end_silence_s with prob < stop, or at max_s.
    """

    def __init__(self, start=0.5, stop=0.3, min_speech_s=0.12, end_silence_s=0.5, preroll_s=0.32, max_s=10.0):
        self.start, self.stop, self.min_speech_s = start, stop, min_speech_s
        self.end_silence_s, self.max_s = end_silence_s, max_s
        self.pre = deque(maxlen=round((preroll_s + min_speech_s) / BLOCK_S))
        self.reset()

    def reset(self):
        self.pre.clear()
        self.speech, self.loud, self.quiet = None, 0, 0

    @property
    def speaking(self):
        return self.speech is not None

    def feed(self, block, prob):
        """Returns the whole utterance (float32 16kHz) on the block that ends it, else None."""
        if self.speech is None:
            self.pre.append(block)
            self.loud = self.loud + 1 if prob >= self.start else 0
            if self.loud * BLOCK_S >= self.min_speech_s:
                self.speech = list(self.pre)
            return None
        self.speech.append(block)
        self.quiet = 0 if prob >= self.stop else self.quiet + 1
        if self.quiet * BLOCK_S >= self.end_silence_s or len(self.speech) * BLOCK_S >= self.max_s:
            speech = np.concatenate(self.speech)
            self.reset()
            return speech
        return None


def utterance(blocks, speech_prob, **endpointer_kw):
    """First utterance in a stream of mic blocks, or None if the stream ends (pull-style Endpointer)."""
    ep = Endpointer(**endpointer_kw)
    for b in blocks:
        was_speaking = ep.speaking
        if (speech := ep.feed(b, speech_prob(b))) is not None:
            return speech
        if ep.speaking and not was_speaking:
            print("(hearing you...)", flush=True)
    return None


def mic_blocks(sd):
    q = queue.Queue()
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", blocksize=BLOCK,
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        while True:
            yield q.get()


def transcribe(processor, model, wav, sr):
    """What was said in wav, via the same model in ASR mode (chat mode never transcribes the user)."""
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Perform ASR.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(wav, sr)
    chat.end_turn()
    chat.new_turn("assistant")
    toks = [t for t in model.generate_sequential(**chat, max_new_tokens=512) if t.numel() == 1]
    return processor.text.decode(torch.cat(toks), skip_special_tokens=True).strip() if toks else ""


@torch.no_grad()
def voice(debug=False, show_text=True, transcript=True):
    """Hands-free multi-turn speech chat: talk whenever, reply plays through speakers. Ctrl+C to quit.

    show_text: stream the agent's reply text as it's generated (it runs ahead of the speech)
    transcript: print what you said (ASR runs after generation, while the reply is still playing)
    debug: print mic levels + reply text, save voice_debug/turnN_{heard,reply}.wav
    """
    import os
    import warnings

    import sounddevice as sd
    from silero_vad import load_silero_vad

    from voice_agent import VoiceAgent  # imports this module, so import it lazily

    agent = VoiceAgent()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.jit.load deprecation inside silero
        vad = load_silero_vad()
    speaker = Speaker(sd)
    print("ready; Ctrl+C to quit")
    if debug:
        os.makedirs("voice_debug", exist_ok=True)

    for turn in itertools.count(1):
        print("\nlistening...", flush=True)
        vad.reset_states()
        blocks = mic_blocks(sd)  # mic is closed while the reply plays, so it can't hear itself
        speech = utterance(blocks, lambda b: float(vad(torch.from_numpy(b), MIC_SR)))
        blocks.close()
        t0 = time.perf_counter()  # end of speech as decided by the VAD
        print(f"(heard {len(speech) / MIC_SR:.1f}s, thinking...)", flush=True)
        if debug:
            sf.write(f"voice_debug/turn{turn}_heard.wav", speech, MIC_SR)
            print(f"  [debug] mic peak {np.abs(speech).max():.3f} rms {np.sqrt(np.mean(speech ** 2)):.4f}")
        played, marks, printed = [], {}, False  # marks: event -> perf_counter(), first occurrence only
        speaker.first_played = None
        for item in agent.reply(audio=speech):
            if isinstance(item, str):
                if show_text:
                    print(("" if printed else "agent: ") + item, end="", flush=True)
                    printed = True
            else:
                marks.setdefault("first bytes to speaker", time.perf_counter())  # first non-silent chunk
                speaker.q.put(item)
                played.append(item)
        marks.update(agent.marks)
        if printed:
            print(flush=True)
        if transcript:  # generation is done; the reply keeps playing from the speaker's queue meanwhile
            said = transcribe(agent.processor, agent.model, torch.from_numpy(speech).unsqueeze(0), MIC_SR)
            marks["transcript ready"] = time.perf_counter()
            print(f'you said: "{said}"', flush=True)
        speaker.drain()
        marks["playback done"] = time.perf_counter()
        if speaker.first_played:
            marks["first sound played"] = speaker.first_played
        secs = sum(map(len, played)) / 24_000
        print("  [latency] from end of speech (VAD adds 0.5s silence wait before this): "
              + " | ".join(f"{k} {v - t0:.2f}s" for k, v in sorted(marks.items(), key=lambda kv: kv[1]))
              + f" | reply {secs:.1f}s of audio", flush=True)

        if debug:
            print(f"  [debug] reply text: {agent.last_text!r}")
        if not played:
            print(f"(model returned no audio; text: {agent.last_text!r})")
        elif debug:
            sf.write(f"voice_debug/turn{turn}_reply.wav", np.concatenate(played), 24_000)


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
    p.add_argument("--show-text", action=argparse.BooleanOptionalAction, default=True,
                   help="voice: stream the agent's reply text")
    p.add_argument("--transcript", action=argparse.BooleanOptionalAction, default=True,
                   help="voice: print what you said (ASR, during playback)")
    a = p.parse_args()
    print(f"device: {DEVICE}")

    if a.task == "voice":
        try:
            voice(a.debug, a.show_text, a.transcript)
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
