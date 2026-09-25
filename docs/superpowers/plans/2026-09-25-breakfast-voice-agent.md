# Breakfast Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local demo web page where I talk to a voice agent that finds breakfasts, videos and step cards and shows them in the right-hand column.

**Architecture:** One FastAPI process with one WebSocket. Per turn: mlx-whisper ASR → Jev router → either the LFM voice agent replies, or the voice agent reassures while the search agent or recommender fetches JSON for the stage. Specs: `docs/superpowers/specs/2026-09-25-breakfast-voice-agent-{technical,frontend}-design.md`. **Read both before starting any task.**

**Tech Stack:** Python 3.12 (`.venv`, managed with `uv`), FastAPI + uvicorn, liquid-audio (LFM2.5-Audio-1.5B, torch/mps), mlx-whisper, TypeSafe Jev (HTTP), Anthropic SDK (`claude-sonnet-5`), yt-dlp, vanilla HTML/CSS/JS.

---

## Current state

| Have | Where |
|---|---|
| LFM load + generation + streaming decode loop | `lfm_audio.py` — `load()`, `run()`, `trim_pauses()`, streaming loop inside `voice()` |
| No-model helper tests | `test_lfm_audio.py` (`python test_lfm_audio.py`) |
| Jev call pattern (reference only, other repo) | `~/Projects/sb/apps/server/convai/sb_convai/services/agent_runtime/retrieval_gate.py`, `~/Projects/sb/apps/server/convai/scripts/bench_jev_gate.py` |
| Sample speech | `question.wav`, `followup.wav`, `q10.wav` |

Missing: everything else. Packages missing from `.venv`: fastapi, uvicorn, mlx-whisper, anthropic, yt-dlp. Keys missing from `.env`: `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`.

Not used by this plan: `talk_space.py`, `space/` (HF Space), `Duplex`/`utterance` (VAD and barge-in are out of scope).

## Waves

```
Wave 0 (sequential, lead)   T0 setup: deps, keys, fixtures, contracts
          │
Wave 1 (parallel, 7 agents) T1 router   T2 voice agent   T3 ASR   T4 search agent   T5 server   T6a web shell   T6b stage views
          │                  (Jev key)    (GPU)            (GPU)    (Claude key)      (fakes)     (mock mode)     (dev harness)
Wave 2 (sequential, lead)   T7 integration: real modules into server, frontend on live server, demo run
```

**Wave-1 agents do not commit.** They share one working tree, so parallel commits would race on the git index. Each agent reports its files, and the lead commits each task separately. Wave-1 tasks touch **disjoint files**, and each depends only on the contracts below plus `fixtures/demo.json`. Every task has a check that runs on its own. T2 and T3 both load models on the GPU; running them at the same time is fine on an M-series Mac (LFM ~3 GB + whisper-turbo ~1.6 GB).

## Contracts (pinned — do not change without the lead)

```python
# asr.py
def warmup() -> None: ...
def transcribe(pcm: np.ndarray) -> str: ...          # float32 mono 16 kHz in; "" on failure, never raises

# router.py
ROUTES = ["recommend", "video", "steps", "next", "back", "repeat", "seek", "show_video", "play", "pause"]
LOOKUPS = {"recommend", "video", "steps"}
def pick(scores: dict[str, float], text: str, session: dict) -> tuple[str, int | None]: ...
    # pure: winner route (or "chat") + target index (dish/video/step), guards applied.
    # Can also return ("goto", step_index) — from the "step N" regex, not a Jev question.
async def route(text: str, session: dict) -> tuple[str, int | None, dict[str, float]]: ...
    # Jev call + pick; ("chat", None, {}) on any error or 1 s timeout

# voice_agent.py
REASSURE: str; FAILED: str
def announce(view: dict) -> str: ...                 # builds the ANNOUNCE note from a view payload
class VoiceAgent:
    def __init__(self) -> None: ...                  # loads LFM + warms mimi; slow, call once
    def reply(self, audio: np.ndarray | None = None, note: str | None = None) -> Iterator[str | np.ndarray]: ...
        # sync generator, run in a thread by the server. yields str = text delta,
        # np.ndarray = float32 24 kHz audio chunk (~80 ms, already trim_pauses'd).
        # audio is 16 kHz float32. Appends the turn to its own ChatState history.

# search_agent.py   (raise on failure; results cached under cache/)
async def video(query: str) -> dict: ...             # {"view":"video","dish":str,"main":Video,"alternates":[Video]}
async def steps(video_id: str, query: str) -> dict: ...  # {"view":"steps","dish":str,"video_id":str,"steps":[Step]}

# recommender.py
async def recommend(query: str) -> dict: ...         # {"view":"dishes","meals":[Meal]*3}
```

- `Video = {id, title, minutes, thumb}`
- `Step = {title, icon, tags: [str], detail, video_start: int | None}`
- `Meal = {id, name, image, minutes, difficulty, why}`

Session dict: exactly technical spec §12. WebSocket messages: technical spec §11. The server sends a module result as `{"type": "view", **payload}`.

## File map

| File | Task | Responsibility |
|---|---|---|
| `requirements.txt`, `.gitignore`, `fixtures/demo.json` | T0 | deps, ignore `cache/`, shared demo data |
| `router.py`, `test_router.py` | T1 | Jev questions, `pick` guards, benchmark |
| `voice_agent.py` | T2 | LFM session, notes, streaming |
| `asr.py`, `test_asr.py` | T3 | mlx-whisper wrapper |
| `search_agent.py`, `test_search_agent.py` | T4 | yt-dlp + Claude steps + cache |
| `server.py`, `recommender.py`, `test_server.py` | T5 | FastAPI, WS, SESSION, `turn()`, nav |
| `web/index.html`, `web/app.js`, `web/style.css`, `web/mock.js` | T6a | shell: layout, orb, audio, socket, metrics, mock |
| `web/views.js`, `web/views.css`, `web/views-dev.html` | T6b | stage views, YouTube player, transitions |

Test style: plain `assert` scripts run with `.venv/bin/python test_x.py`, matching `test_lfm_audio.py`. Use pytest only where async/TestClient needs it (`test_server.py`).

---

### Task 0: Setup (lead, before wave 1)

**Files:** Create `requirements.txt`, `fixtures/demo.json`. Modify `.gitignore`.

- [ ] **Step 1: Dependencies**

`requirements.txt`:
```
liquid-audio==1.3.0
soundfile
fastapi
uvicorn[standard]
mlx-whisper
anthropic
yt-dlp
httpx
pytest
```
Run: `uv pip install --python .venv/bin/python -r requirements.txt`
Expected: `.venv/bin/python -c "import fastapi, mlx_whisper, anthropic, yt_dlp"` exits 0.

- [ ] **Step 2: Keys.** You add `TYPESAFE_API_KEY=` and `ANTHROPIC_API_KEY=` to `.env` (already gitignored). Code reads keys from `os.environ` first, then from `.env` using the same one-liner as `talk_space.py`.

- [ ] **Step 3: `.gitignore`.** Append `cache/`.

- [ ] **Step 4: `fixtures/demo.json`.** Real data for one shakshuka run. Fetch a real embeddable video ID with `yt-dlp --flat-playlist -j "ytsearch3:shakshuka recipe"`, so the mock and fake modules show a working player:
```json
{
  "meals": [ {"id": "shakshuka", "name": "Shakshuka", "image": null, "minutes": 25, "difficulty": "easy", "why": "Trending on TikTok this week"},
             {"id": "dutch-baby", "name": "Dutch Baby", "image": null, "minutes": 30, "difficulty": "easy", "why": "Big on Instagram brunch posts"},
             {"id": "avocado-toast", "name": "Avocado Toast", "image": null, "minutes": 10, "difficulty": "easy", "why": "Always a favourite"} ],
  "video": {"dish": "Shakshuka", "main": {"id": "<real id>", "title": "<real title>", "minutes": 8, "thumb": "https://i.ytimg.com/vi/<real id>/hqdefault.jpg"},
            "alternates": [ "<2 more real entries>" ]},
  "steps": [ {"title": "Heat the oil", "icon": "🫒", "tags": ["medium heat", "1 min"], "detail": "Warm 2 tbsp olive oil in a wide skillet.", "video_start": 30},
             "<5 more steps>" ]
}
```

- [ ] **Step 5: Commit** the specs, the plan and the setup files: `git add docs requirements.txt fixtures .gitignore && git commit -m "chore: breakfast agent specs, plan, deps, fixtures"`

---

### Task 1: Router (Jev) — `router.py`, `test_router.py`

**Reference:** technical spec §8, and the `sb` files listed under *Current state* (copy the request shape and the fail-open style).

- [ ] **Step 1: Offline tests for `pick`** (fake scores, no network). Write these first in `test_router.py`:
```python
from router import pick
S = lambda **kw: {"screen": "welcome", "meals": [], "video": None, "steps": [], "step": 0, **kw}
meals = [{"name": n} for n in ("Shakshuka", "Dutch Baby", "Avocado Toast")]
vid = {"main": {"id": "x"}, "alternates": []}

assert pick({"recommend": .9}, "what's popular", S()) == ("recommend", None)
assert pick({"recommend": .3}, "thanks!", S()) == ("chat", None)               # nothing >= .5
assert pick({"video": .9, "dish_1": .8}, "the second one", S(screen="dishes", meals=meals)) == ("video", 1)
assert pick({"steps": .9}, "hard to follow", S()) == ("chat", None)            # guard: no video
assert pick({"steps": .9}, "hard to follow", S(video=vid)) == ("steps", None)
assert pick({"next": .9}, "next", S()) == ("chat", None)                       # guard: no steps
assert pick({"next": .9}, "next", S(steps=[{}] * 6, video=vid)) == ("next", None)
assert pick({}, "go to step four", S(steps=[{}] * 6, video=vid)) == ("goto", 3) # regex, 0-based
```
Note: `goto` is a route produced only by the step regex. Add it to `ROUTES` handling in `pick` (not a Jev question).

- [ ] **Step 2:** Run `.venv/bin/python test_router.py`. Expected: ImportError / fail.
- [ ] **Step 3:** Implement `pick`, plus `QUESTIONS` (one dict of `{id: {instructions, criteria}}` for every route in spec §8, and `dish_i`/`video_i` built from the session).
- [ ] **Step 4:** Run the offline tests. Expected: pass.
- [ ] **Step 5: `route()`.** One `httpx.AsyncClient` shared for the whole process. POST `{state, model: "jev-latest", questions}` to `https://api.typesafe.ai/v1/systemone`. Timeout 1 s; any exception returns `("chat", None, {})`. `state` is built exactly as in spec §8.
- [ ] **Step 6: Live scenarios + benchmark** (skip with a printed notice if there's no `TYPESAFE_API_KEY`). In `test_router.py` under `if os.environ.get("TYPESAFE_API_KEY")`, cover at least:

| Text | Screen | Expected |
|---|---|---|
| "I want to cook breakfast for my wife, what's popular lately?" | welcome | recommend |
| "let's do the shakshuka" | dishes | video, 0 |
| "the second one looks good" | dishes | video, 1 |
| "that's hard to follow" | video | steps |
| "okay next" / "wait go back" / "say that again" | steps | next / back / repeat |
| "show me that part" | steps | seek |
| "thanks, she's going to love it" | steps | chat |
| "pause it" | video | pause |

It prints accuracy and p50/p95 latency over 10 spaced calls.
Expected: ≥ 9/10 correct and p50 ≤ 200 ms. **Report the numbers to the lead.** If p50 > 200 ms, say so; the fallback (spec §8) is a second call just for targets.
- [ ] **Step 7:** Commit `router.py test_router.py` with message `feat: jev router`.

---

### Task 2: Voice agent — `voice_agent.py`

**Reference:** technical spec §6. Reuse `lfm_audio.load()` and `trim_pauses()`. Copy the streaming loop and the history `chat.append(...)` from `lfm_audio.voice()` (don't import `voice()` itself). Do **not** modify `lfm_audio.py`.

- [ ] **Step 1:** Implement `VoiceAgent` per the contract, with the system prompt and the `REASSURE` / `FAILED` / `announce()` text from spec §6. `announce(view)` builds the summary from `view["view"]`:
  - `dishes` → "3 breakfasts: A, B, C"
  - `video` → "a N-minute video: TITLE"
  - `steps` → "N steps; step 1 is TITLE"
- [ ] **Step 2: Notes check (the spec's "verify during build").** Put the audio and the note in the same user turn: `chat.add_audio(...)` then `chat.add_text(note)`.
- [ ] **Step 3: Smoke script** under `if __name__ == "__main__":`. It runs `question.wav` three ways: no note, with `REASSURE`, and text-only with `announce(fixtures/demo.json meals view)`. For each run it prints the reply text and the ms to the first audio chunk, and writes `voice_debug/va_{plain,reassure,announce}.wav`.
  Run: `.venv/bin/python voice_agent.py`
  Expected:
  - all three produce audio
  - the reassure reply names no dish
  - the announce reply mentions at least one of the three meal names
  - first-chunk latency is printed

  If the reassure reply ignores the note in about 2 of 3 tries, switch to a separate `system` turn for the note, re-run, and report which variant works.
- [ ] **Step 4: No-model check.** Add `announce()` asserts, for the three view kinds, to the top of the smoke script's output, or to a small `test_voice_agent.py` that imports only `announce`. `announce` must not load the model at import time, so keep `load()` inside `__init__`.
- [ ] **Step 5:** Commit `voice_agent.py` (+ test) with message `feat: streaming voice agent with notes`.

---

### Task 3: ASR — `asr.py`, `test_asr.py`

**Reference:** technical spec §7.

- [ ] **Step 1: Test first:**
```python
import time, numpy as np, soundfile as sf, torch, torchaudio
import asr
asr.warmup()
wav, sr = sf.read("question.wav", dtype="float32", always_2d=True)
pcm = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16_000).numpy()
t = time.perf_counter(); text = asr.transcribe(pcm); ms = (time.perf_counter() - t) * 1000
print(repr(text), f"{ms:.0f} ms")
assert len(text.split()) >= 3
assert len(asr.transcribe(np.zeros(16_000, np.float32))) < 20  # whisper may hallucinate a word on silence
assert asr.transcribe(np.zeros(0, np.float32)) == ""       # never raises
```
- [ ] **Step 2:** Run it. Expected: fail (no `asr`).
- [ ] **Step 3:** Implement with `mlx_whisper.transcribe(pcm, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en")["text"].strip()`. Wrap it in `try/except`, returning `""`. `warmup()` transcribes 1 s of zeros.
- [ ] **Step 4:** Run it. Expected: pass. **Report the latency to the lead** (target ≤ 400 ms for a short clip).
- [ ] **Step 5:** Commit with message `feat: mlx-whisper asr`.

---

### Task 4: Search agent — `search_agent.py`, `test_search_agent.py`

**Reference:** technical spec §9. Model: `claude-sonnet-5`. Blocking yt-dlp and Anthropic calls run via `asyncio.to_thread`, or use `anthropic.AsyncAnthropic`.

- [ ] **Step 1: Offline tests first** for the pure helpers:
  - `parse_steps(text) -> list[Step]`: extracts the JSON array even inside ```json fences, and raises `ValueError` on garbage.
  - `captions_to_lines(json3: dict) -> str`: produces `"[95s] now add the onion"` lines.
  - cache round-trip: `cached(key, fn)` writes `cache/<key>.json`, a second call doesn't call `fn`, and a failure isn't cached.
- [ ] **Step 2:** Run. Expected: fail.
- [ ] **Step 3:** Implement the helpers. Run. Expected: pass.
- [ ] **Step 4:** Implement `video(query)`: `ytsearch5:{query} recipe`, `extract_flat`, keep entries under 20 min, first = main, up to 3 alternates, `dish = query` title-cased.
- [ ] **Step 5:** Implement `steps(video_id, query)`:
  1. auto-captions json3 via yt-dlp to a temp dir
  2. `captions_to_lines`
  3. one Claude call using the spec §9 prompt
  4. `parse_steps`, retrying once on `ValueError`

  With no captions, the prompt uses only the title and sets `video_start` to null.
- [ ] **Step 6: Live check** (skip if there's no `ANTHROPIC_API_KEY`). `asyncio.run(video("shakshuka"))`, then `steps(main.id, "shakshuka")`. Print the titles and timestamps.
  Expected:
  - at least one alternate
  - 5–8 steps
  - `video_start` values rise in order
  - a second run is instant (cache hit)
- [ ] **Step 7:** Commit with message `feat: search agent (yt-dlp + claude steps)`.

---

### Task 5: Server — `server.py`, `recommender.py`, `test_server.py`

**Reference:** technical spec §5, §11–14. Build against **fakes**. Module functions are looked up as module attributes (`server.asr`, `server.router`, …), so tests and T7 can swap them.

- [ ] **Step 1: `recommender.py`** returns `{"view": "dishes", "meals": fixtures["meals"]}` from `fixtures/demo.json`. Mark it: `# ponytail: fixture stub; swap body for the social-media service call`.
- [ ] **Step 2: Test first** (`test_server.py`, pytest + `fastapi.testclient.TestClient`). Fakes:
  - `asr.transcribe` returns preset text
  - `router.route` returns a preset route
  - a `VoiceAgent` fake yields `"hi"` plus one 1920-sample zeros chunk
  - search fakes return the fixture views

  Cases:
  1. route `chat` → receive `state:thinking`, `caption`, ≥ 1 binary frame, `metrics` with `route=="chat"`, `state:idle`.
  2. route `recommend` → a `view` with `view=="dishes"` arrives, and the fake voice agent was called twice: first with note `REASSURE`, then with `announce(...)`.
  3. route `next` with steps loaded → `control goto_step index 1`, voice agent **not** called.
  4. The module raises → the voice agent is called with `FAILED`, and no `view` is sent.
  5. `{"type":"action","name":"select_dish","id":"shakshuka"}` → runs the `video` lookup without ASR or router.
  6. On connect with `SESSION.screen=="dishes"` → the current view is replayed first.
- [ ] **Step 3:** Run `.venv/bin/python -m pytest test_server.py -q`. Expected: fail.
- [ ] **Step 4:** Implement `server.py`:
  - FastAPI with `/` serving `web/` via `StaticFiles(html=True)`, and `/ws`.
  - The `SESSION` dict, and `turn()` following spec §5.
  - `speak()` runs `VoiceAgent.reply` in a thread and forwards text as `caption` and chunks as binary `float32` bytes. It appends to `SESSION.transcript` and `cache/transcript-YYYY-MM-DD.jsonl`.
  - `MODEL_LOCK` around voice-agent calls, and `Timer` for metrics (ms from PCM receipt).
  - A new connection replaces the old one and closes it with code 4000.
  - `build_query` per spec §10.
  - Models load at startup via FastAPI lifespan, skipped when `FAKE=1`.
  - `__main__` runs uvicorn on port 8000.
- [ ] **Step 5:** Run the tests. Expected: pass.
- [ ] **Step 6:** Commit with message `feat: websocket server and turn orchestration`.

---

### Frontend split: T6a (shell) and T6b (stage views)

T6a and T6b run in parallel against the same frontend spec and the interface below. Neither edits the other's files.

**Shared design tokens:** CSS variables are defined once by T6a in `web/style.css` `:root`, and T6b uses them without redefining:
```css
--bg: #17130f; --panel: #221c16; --ink: #f6eee4; --muted: #a8998a;
--accent: #f2b33d; --accent-2: #e8793a; --radius: 20px;
--font: "Inter", system-ui, sans-serif; --step-size: 34px;
```

**Interface** (`web/views.js`, owned by T6b, loaded before `app.js`):
```js
window.Views = {
  mount(stageEl),           // called once by app.js with the right-column element
  show(msg),                // msg = {type:"view", view:"welcome"|"dishes"|"video"|"steps", ...}; caches per view
  control(msg),             // msg = {type:"control", name:"goto_step"|"video", ...}
  pauseVideo(),             // app.js calls this when the mic opens or agent audio starts
  onAction: (action) => {}, // app.js sets it; Views calls it for clicks/keys: {type:"action", name, ...}
};
```
`app.js` never touches stage DOM, and `views.js` never touches the socket or the audio.

---

### Task 6a: Frontend shell — `web/index.html`, `web/style.css`, `web/app.js`, `web/mock.js`

**Reference:** frontend spec §2, §3, §5, §6, §8, §10, and technical spec §11.

- [ ] **Step 1: `index.html`:**
  - the left column (orb, caption, mic button, metrics strip) and an empty `<section id="stage">`
  - loads `https://www.youtube.com/iframe_api`, `views.js`, `mock.js` and `app.js`, plus both `style.css` and `views.css`
  - until T6b lands, create a 10-line `web/views.js` stub that implements the interface by printing `msg.view` into the stage. T6b overwrites it.
- [ ] **Step 2: `style.css`:**
  - the `:root` tokens above
  - two columns (40/60), stacking below 900 px with the avatar as a top strip
  - the orb's four states as CSS classes on `<body data-state=…>`, with volume from a `--level` CSS variable set by JS
  - the caption with a word fade-in, the mic button and the metrics strip
- [ ] **Step 3: `app.js` socket and dispatch:**
  - WebSocket `ws://${location.host}/ws` with `binaryType = "arraybuffer"`, reconnect with backoff, and a "reconnecting…" pill.
  - `handle(msg)`:
    - `state` → body `data-state`
    - `caption` → append text
    - `view` / `control` → `Views`
    - `metrics` → strip
    - `error` → caption
  - `Views.onAction = send`. If thinking runs past 15 s, show "Still looking…".
- [ ] **Step 4: Audio.**
  - Push-to-talk: the mic button (pointer down/up) and Space hold/release.
  - Capture via `getUserMedia` + `new AudioContext({sampleRate: 16000})` + an AudioWorklet (inline Blob module) collecting float32. On release, send one `Float32Array.buffer` binary frame.
  - Set `listening` and call `Views.pauseVideo()` when the mic opens. If mic permission is denied, grey out the button with a tooltip.
  - Playback: binary frames → `Float32Array` → an `AudioBuffer` at 24 kHz, scheduled back to back on one `AudioContext` (`nextTime = max(nextTime, ctx.currentTime)`). Call `Views.pauseVideo()` when the first chunk arrives, and set `idle` when the last scheduled buffer ends.
  - An `AnalyserNode` on the mic and on the playback feeds `--level` for the orb.
- [ ] **Step 5: Metrics strip:** the route and target text, plus one bar per `ms` key (asr, jev, voice_first_audio, module, announce_first_audio) scaled to `total`. `M` toggles it.
- [ ] **Step 6: `mock.js`:**
  - When `?mock=1` is set, `app.js` uses a fake socket from `mock.js` instead of a real one. It exposes the same `onmessage` / `send` and replays the frontend spec §1 demo.
  - Data comes from `fetch("../fixtures/demo.json")`: meals (as `{view:"dishes", meals}`), `video` and `steps`, then `next` → `goto_step 1` and `seek`.
  - It sends `state`, `caption` and fake `metrics` with 1–3 s delays. Silent audio chunks (`new Float32Array(1920)`) exercise playback.
  - It advances automatically, one step every 3 s, or on the `N` key.
- [ ] **Step 7: Check.**
  - Run `.venv/bin/python -m http.server -d . 8001` and open `http://localhost:8001/web/?mock=1`.
  - The whole script plays with no console errors, the orb changes through all four states and `M` toggles the strip.
  - It works at 1440 and 390 px wide.
  - Push-to-talk on the real mic sends a binary frame: log the byte length in mock mode.
- [ ] **Step 8:** Report to the lead: files changed, the check result, and anything that deviated. Don't commit; the lead commits.

---

### Task 6b: Stage views — `web/views.js`, `web/views.css`

**Reference:** frontend spec §4 and §7. Message shapes come from `fixtures/demo.json` (`video` and `steps` entries are full view payloads; dishes = `{view:"dishes", meals}`).

- [ ] **Step 1: A standalone harness `web/views-dev.html`.** It loads `style.css` (tokens only; if T6a hasn't created it yet, copy the `:root` tokens inline in the harness), `views.css`, the YouTube iframe API and `views.js`. It has buttons that call `Views.show(...)` / `Views.control(...)` with fixture data, and logs `onAction` calls. This is how the views are built without T6a.
- [ ] **Step 2: `welcome`:** the greeting "Good morning — what are we cooking?" plus 3 example chips. Clicking one calls `onAction({type:"text", text})`.
- [ ] **Step 3: `dishes`:**
  - 3 cards in a row with a staggered fly-in: image or gradient placeholder with emoji, name, minutes, difficulty tag, and a "why" line.
  - Clicking a card calls `onAction({type:"action", name:"select_dish", id})`.
  - The selected card grows and the others fade out. This plays when the next `video` view arrives *after* a dish was selected, or immediately on click.
- [ ] **Step 4: `video`:**
  - A YouTube player via `new YT.Player` (wait for `onYouTubeIframeAPIReady`), 16:9 and autoplaying, with alternates as thumbnail rows.
  - Clicking an alternate swaps the main video and calls `onAction({type:"action", name:"select_video", id})`.
  - `onError` automatically skips to the next alternate; with none left, show the thumbnail plus an "open on YouTube" link.
  - The player position is remembered when leaving the view.
- [ ] **Step 5: `steps`:**
  - One big card: icon, title (`--step-size`), tag chips and detail, plus progress dots, previous/next hint titles and a "▶ back to video" chip.
  - Entry transition from video: the video shrinks and slides away while the cards deal in from the right.
  - ←/→ keys and arrows call `onAction({type:"action", name:"goto_step", index})` and update locally.
  - The chip calls `onAction({type:"action", name:"show_video"})`.
- [ ] **Step 6: `control`:**
  - `goto_step` → animate to the index.
  - `video` `play` / `pause` / `seek t` → act on the player.
  - `seek` while on steps: the server sends `view: video` first, then `seek`, so just seek the player.
- [ ] **Step 7: Caching** per frontend spec §7: switching video → steps → video resumes the video position, and steps keep their index.
- [ ] **Step 8: Check.** Using the harness, cycle through every view and control with no console errors. Check all transitions and the 390 px layout. Report files and the result to the lead. Don't commit.

---

### Task 7: Integration (lead, after wave 1)

- [ ] **Step 1:** Remove the fakes from `server.py`'s startup path and load the real `asr`, `VoiceAgent`, `router`, `search_agent` and `recommender`. Run `test_server.py` again (it still uses fakes). Expected: pass.
- [ ] **Step 2:** Serve `fixtures/` statically so `mock.js` works under the live server too.
- [ ] **Step 3: Live run.** Start with `.venv/bin/python server.py`, open `http://localhost:8000`, and speak the full demo script from frontend spec §1.
  Expected:
  - each step lands on the right view
  - reassure lines never name dishes
  - the metrics strip shows ASR ≤ 400 ms and Jev ≤ 200 ms
  - voice first audio ≈ 1 s after release
- [ ] **Step 4:** Fix what the live run shows. Tune the Jev instructions in `router.py` against `test_router.py`.
- [ ] **Step 5:** Commit with message `feat: wire real modules, first live demo`.

---

## Risks the lead should watch

| Risk | Where it shows | Fallback |
|---|---|---|
| Jev with ~10 questions > 200 ms | T1 step 6 | two calls: route, then target |
| LFM ignores notes | T2 step 3 | separate system turn; if that fails too, the server plays TTS of a fixed reassure line (`lfm_audio.run` TTS) |
| LFM + whisper on the GPU at the same time | T2/T3 in parallel, and live | they run in sequence per turn anyway; only test runs overlap |
| yt-dlp captions blocked or rate-limited | T4 step 6 | title-only steps (already in spec), or the cache |
| `claude-sonnet-5` step JSON too slow | T4 | cache warmed by one run before the demo |
