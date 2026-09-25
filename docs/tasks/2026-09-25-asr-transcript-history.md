---
id: 2026-09-25-asr-transcript-history
title: "Explore ASR transcripts to replace historical user audio in LFM2.5-Audio chat context"
created: 2026-09-25T15:24
status: open
priority: medium
type: task
depends_on: []
related: []
branch: ""
pr: ""
---

# Explore ASR transcripts to replace historical user audio in LFM2.5-Audio chat context

## Context

CortexFlow runs `LiquidAI/LFM2.5-Audio-1.5B` on a Hugging Face ZeroGPU Space
(`jiangjun0105/CortexFlow`) with a speech-in / speech-out "Voice chat" endpoint (`/chat`).
`talk_space.py` records from the mic, calls `/chat`, plays the reply, and prints a per-step
latency breakdown. The Space is currently **stateless**: every request is a single turn with no history.

While exploring multi-turn conversations we found that audio is ~4x more expensive in the
context than the same content as text, and asked whether storing past user turns as text
(transcripts) instead of audio would reduce latency. The model does not emit the user's
transcript as a by-product — its interleaved text tokens are its *own* reply — so a
transcript needs a separate ASR step.

## Problem

- Audio costs a fixed **12.5 context positions/second** (mel at 10 ms stride, 8x subsampling
  → 80 ms per position), including silence. The same speech as text is ~2.5–3 tokens/second.
- Input audio also runs through the audio encoder (conformer) every time the context is rebuilt.
- Assistant speech (`audio_out`) is also 12.5 positions/s (8 codebooks summed into one position per 80 ms frame).
- Context limit is 128k positions (`max_position_embeddings`), but trained conversation length is
  likely much shorter; quality/memory/latency degrade as history grows.

Measured baseline (single turn, `question.wav` ~3 s, back-to-back requests on ZeroGPU A10G):

| Step | Run 1 | Run 2 |
|---|---|---|
| upload + queue | 0.76 s | 0.71 s |
| GPU attach (ZeroGPU) | 1.09 s | 1.46 s |
| prep input | 0.51 s | 0.56 s |
| gen start → first audio frame | 1.51 s | 0.93 s |
| rest of reply (generate + decode + save) | 3.38 s | 7.03 s |
| return + download | 0.70 s | 0.90 s |
| **end of speech → first audio frame** | **3.88 s** | **3.65 s** |

Expected impact (unverified): negligible for 1–3 turns; possibly 0.5–1 s+ off time-to-first-audio
for 10+ turns, plus lower memory. Reply generation time and ZeroGPU overhead (~2.5–3 s) are unaffected.

## Desired Behavior

1. The Voice chat keeps multi-turn history: a follow-up question can refer to an earlier turn
   ("what about the second one?") and the model answers in context.
2. After each turn, the user's speech is transcribed by an ASR backend, and the transcript is
   available (printed in `talk_space.py` and/or returned by the Space) before the next turn starts.
3. Transcription does not add to the wait before the reply plays — it runs after the reply
   audio is ready/sent, or in parallel off the critical path.
4. History can be run in two selectable modes for comparison:
   - **audio history** — past user turns stored as audio, past assistant turns as audio + text
   - **text history** — past user turns stored as ASR transcript, past assistant turns as their text tokens only;
     the **current** user turn is always audio
5. A benchmark run of the same scripted multi-turn conversation (e.g. 10 turns of pre-recorded
   wavs) in both modes reports, per turn: context size (positions), prep time,
   time to first audio frame, total generation time, and end-to-end wait.
6. If ASR fails or returns empty text for a turn, that turn falls back to being stored as audio
   (history is never silently dropped).
7. At least two ASR backends are compared on accuracy (spot-check vs known script) and latency:
   - LFM2.5-Audio itself (`"Perform ASR."` — already exposed as `/asr`, zero extra model weight)
   - one dedicated offline model (e.g. Whisper / faster-whisper / Parakeet) **or** one ASR API

## Key Files

| File | Purpose |
|------|---------|
| `lfm_audio.py` | Core model wrapper: `load()` (cached), `run()` builds a `ChatState` and returns `(text, wav_path, timing)`; `voice()` is the local multi-turn loop that already appends assistant turns to history via `chat.append(...)` |
| `space/app.py` | Gradio app deployed to the Space; `/tts`, `/asr`, `/chat`; `chat()` logs a `TIMING ...` line (recv/gpu/gen_start/first_audio/done) |
| `space/lfm_audio.py` | Copy of `lfm_audio.py` uploaded to the Space — keep in sync |
| `space/requirements.txt` | Space deps (`liquid-audio==1.3.0`, `soundfile`) |
| `talk_space.py` | Local client: Enter-to-talk, calls `/chat`, plays reply, fetches the Space's `TIMING` log line and prints a latency breakdown |
| `.venv/lib/python3.12/site-packages/liquid_audio/processor.py` | `ChatState`: separate `text`, `audio_in`, `audio_out`, `modality_flag` tensors; `add_text`, `add_audio`, `append`, `new_turn`, `end_turn` |
| `.venv/lib/python3.12/site-packages/liquid_audio/utils.py` | `mel2emb_len` — mel frames → embedding positions (floor div by 8) |
| `q10.wav`, `question.wav`, `followup.wav` | Test inputs (q10 ≈ 11 s, question ≈ 3 s) |

## Suggested Approach

### History representation

Keep history as a plain Python list of turns and **rebuild a fresh `ChatState` per request**
(slicing an existing `ChatState` is awkward — text and audio live in separate tensors ordered by
`modality_flag`). Each turn: `{"role", "audio": wav|None, "text": str|None}`. The builder emits
`add_audio` or `add_text` per turn depending on mode. For assistant turns in audio mode, replay
`audio_out` codes + text via `chat.append(...)` as `voice()` does.

Where state lives: either client-side (client sends history each request — simplest with a
stateless Space, but re-uploads audio in audio mode) or Gradio `gr.State` per session on the Space.

### ASR backend options

- **Option A — LFM2.5-Audio `"Perform ASR."`**: no new deps; run in the same `@spaces.GPU` call
  right after the reply (one GPU attach, not two). Already verified accurate on `question.wav`.
- **Option B — offline dedicated model** (faster-whisper small/base, Parakeet): run on CPU locally
  or on the Space; independent of the GPU call.
- **Option C — ASR API**: zero GPU cost, adds network latency and a key.

### Recommendation

Start with Option A inside the same GPU call (return audio first, transcript second — pairs with
future streaming). Add one of B/C only as the comparison point.

> **Note:** the latency gain from text history is an estimate; LFM2 is mostly convolutional with
> some attention layers, so per-step decode cost grows only mildly with context. Measure before
> concluding.

## Implementation Approach

- **Artifact type:** history builder + mode switch in `lfm_audio.py`/`space/app.py`; a
  benchmark script (e.g. `bench_history.py`) that replays N pre-recorded user wavs in both modes
  and prints a per-turn table; small update to `talk_space.py` to show the transcript.
- **Extend existing:** reuse `run()`'s `timing` dict and the Space's `TIMING` log line /
  `server_timing()` in `talk_space.py`; reuse `voice()`'s `chat.append(...)` pattern for assistant turns.
- **Do not:** put ASR on the critical path before the reply; drop turns on ASR failure; compare
  modes on different inputs (use the same scripted wavs); run benchmark turns concurrently on ZeroGPU
  (they queue and skew timings); forget to copy `lfm_audio.py` into `space/` before uploading.

## Acceptance Criteria

- [ ] Desired behaviors 1–7 implemented and verified on the live Space
- [ ] Benchmark table for ≥10 turns in both modes, with context size and time-to-first-audio per turn
- [ ] ASR comparison: accuracy spot-check + latency for ≥2 backends
- [ ] Written conclusion: does text history measurably reduce latency, and from which turn count
- [ ] Existing `/tts`, `/asr`, and single-turn `talk_space.py` still work

## Notes

- Upload to the Space with `HfApi().upload_folder(...)` / `upload_file(...)` using `HF_API_KEY`
  from `.env` — `hf upload` fails with 402 because it calls `create_repo` first.
- Space restart after upload takes ~3 min; watch `/api/spaces/jiangjun0105/CortexFlow/runtime`.
- ZeroGPU adds ~2.5–3 s per request (upload/queue ~0.7 s, GPU attach ~1–1.5 s, return ~0.7 s)
  regardless of history mode — report it separately so it doesn't mask the difference.
- Keeping audio for history preserves tone/emotion/emphasis that transcripts lose; note any
  observed change in answer quality between modes.
- Streaming playback (est. ~4 s end-of-speech → first sound on ZeroGPU) is a separate, larger
  latency win — not in scope here.
