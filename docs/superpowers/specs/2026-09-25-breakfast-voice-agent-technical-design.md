# Breakfast Voice Agent — Technical Design

Date: 2026-09-25
Status: Locked for v1 demo
Companion: `2026-09-25-breakfast-voice-agent-frontend-design.md` (layout, views, visuals)
Goal: a quick local demo with one user and one browser tab, running on my Mac.

## 1. Decisions

| Decision | Why |
|---|---|
| **Single session**, all state in one global dict | Demo only. No IDs, no DB. A new WebSocket replaces the old one. |
| **Three modules**: voice agent, search agent, recommender | Each has one job and a text/JSON boundary, so they can be swapped out on their own. |
| **Voice agent = LFM2.5-Audio-1.5B (local)** | Good at natural speech in and out. Not trusted for facts, search or structured output. |
| **ASR = mlx-whisper (large-v3-turbo, local)** | ~200–400 ms per utterance, so the transcript is ready before the voice agent needs to start. |
| **Router = Jev (TypeSafe System One)** | ~100 ms typed yes/no judgments on the transcript plus context. Pattern proven in `sb` (`retrieval_gate.py`, `bench_jev_gate.py`). |
| **Route first, then speak** | Jev runs *before* the voice agent, so the voice agent always knows whether a lookup is in flight. Costs ~100 ms and prevents it from making up answers. |
| **One WebSocket**, raw PCM audio both ways | One command to start. No encoding and no CORS. |
| **Push-to-talk** | Avoids echo problems. VAD and barge-in come later. |

## 2. Modules

| # | Module | In | Out | Implementation |
|---|---|---|---|---|
| 1 | **Voice agent** | audio (+ optional text instruction) | streamed audio + text | LFM2.5-Audio via `lfm_audio.py`, one persistent `ChatState` |
| 2 | **Search agent** | text query (+ current video) | JSON: `video` *or* `steps` | External web agent (different model, built separately). v1 uses a stub with fixtures. |
| 3 | **Recommender** | text query | JSON: 3 meals | Separate service, based on social-media data. v1 uses a stub with sample data. |
| — | **ASR** | audio | user text | `mlx_whisper.transcribe` |
| — | **Router** | transcript + screen state | route + target | Jev `POST https://api.typesafe.ai/v1/systemone` |

Modules 2 and 3 are plain `async def fn(query: str, **ctx) -> dict`. The orchestrator doesn't care whether they're local code, HTTP or a stub.

## 3. Architecture

```
┌───────── Browser ─────────┐  WS /ws   ┌──────────────────── server.py ────────────────────────┐
│ mic → 16k float32 PCM ────┼─binary──▶ │ turn(pcm)                                             │
│ ◀── 24k float32 PCM chunks┼─binary─── │   1. ASR (mlx-whisper) ──▶ TRANSCRIPT += user text    │
│ JSON (views, state,  ◀───▶┼──text───▶ │   2. Router (Jev) ──▶ route                           │
│       metrics, actions)   │           │   3a. route=chat ──▶ Voice agent (normal reply)       │
└───────────────────────────┘           │   3b. route=lookup ──┬▶ Voice agent ("reassure")      │
      GET / → web/ static               │                      └▶ Search agent / Recommender    │
                                        │                         ──▶ view JSON ──▶ browser     │
                                        │                         ──▶ Voice agent ("announce")  │
                                        │   4. metrics ──▶ browser                              │
                                        │ SESSION: transcript, screen state, ws                 │
                                        └───────────────────────────────────────────────────────┘
```

## 4. Files

```
server.py          FastAPI, WebSocket, SESSION, turn() orchestration, nav handling
voice_agent.py     VoiceAgent: persistent LFM ChatState, streaming reply(audio=None, text=None)
router.py          Jev call: questions, route selection, guards
search_agent.py    video(query) / steps(video_id, query)  — stub over fixtures; client for the external web agent later
recommender.py     recommend(query) — stub returning fixtures until the real service exists
lfm_audio.py       unchanged: load(), trim_pauses(); voice_agent.py imports from it
web/               index.html, app.js, style.css, mock.js
cache/             search/recommender results as JSON (gitignored)
test_router.py     labelled route scenarios (see §12)
```

Run: `TYPESAFE_API_KEY=… python server.py`, then open `http://localhost:8000`.
New dependencies: `fastapi`, `uvicorn`, `mlx-whisper`, `httpx`.

## 5. The turn

```
end of speech (you release the mic)
   0 ms  ├ ASR (mlx-whisper) ~300 ms ┤
 300 ms                              ├ Jev ~100 ms ┤ route known ≈ 400 ms
 400 ms                                            │
   route = chat ───────────────────────────────────┼─▶ voice agent: reply to your audio
                                                   │     first audio ≈ 400 + ~700 ms
   route = recommend | video | steps ──────────────┼─▶ voice agent: your audio + REASSURE note
                                                   │     "Sure, pulling that up…"   ┐ in parallel
                                                   └─▶ module 2/3 (network) ────────┘ GPU ∥ network
                                                         ──▶ view JSON ──▶ right column
                                                         ──▶ voice agent: ANNOUNCE note
   route = next | back | repeat | goto | seek | show_video | play | pause
                                                   └─▶ control/view message only, no speech
```

```python
async def turn(pcm):
    t = Timer()
    send(state="thinking")
    text = await t.run("asr", asr, pcm)                     # "" on failure
    SESSION.transcript.append(("user", text))
    route, target, scores = await t.run("jev", router.route, text, SESSION)

    if route == "chat":
        await speak(audio=pcm)
    elif route in LOOKUPS:                                    # recommend / video / steps
        query = build_query(text, SESSION)
        job = asyncio.create_task(t.run("module", LOOKUPS[route], query, SESSION))
        await speak(audio=pcm, note=REASSURE)                 # plays while the module works
        view = await job
        if view: send(view); SESSION.update(view)
        await speak(note=announce(view))                      # or apologise on error
    else:
        send(NAV[route](target, SESSION))                     # no speech, instant

    send(metrics=t.report(route, scores))
    send(state="idle")
```

- `speak()` streams voice-agent audio chunks to the browser as they're decoded and adds the reply text to `SESSION.transcript`.
- **Why `speak(audio=pcm)` and not text:** the voice agent hears your actual voice (tone, "for my wife"), while Whisper's text drives only the routing and the queries.
- If the module finishes before the reassure line has played, the announce line is simply queued after it. The browser plays chunks in order.

## 6. Voice agent (module 1)

`VoiceAgent` wraps one LFM `ChatState` that lives for the whole session, so the history carries across turns (same approach as `lfm_audio.voice()`).

**System prompt (fixed):**
> You are a warm, brief kitchen helper speaking out loud. A separate system finds dishes, recipes and videos and shows them on the screen. Never name dishes, give recipe steps, or state facts you'd need to look up — the screen does that. Keep every reply to one or two short sentences.

**Notes** are extra text added to the user turn, next to the audio:

| Note | When | Text |
|---|---|---|
| `REASSURE` | lookup started | "[A lookup for this is running and will appear on screen. Don't answer it — just tell them you're on it, in one short sentence.]" |
| `ANNOUNCE` | lookup done | "[The screen now shows: {summary}. Tell them in one sentence and invite them to pick or continue.]" |
| `FAILED` | lookup failed | "[The lookup failed. Apologise briefly and suggest trying again.]" |

`{summary}` is built by the server from the JSON, e.g. "3 breakfasts: Shakshuka, Dutch Baby, Avocado Toast" or "a 8-minute video: Easy Shakshuka" or "6 steps; step 1 is Heat the oil". That way the voice agent can mention real items without having to know them.

**Streaming:** `generate_interleaved` + `mimi.streaming` decode → `trim_pauses` → each 80 ms chunk (24 kHz float32) is sent as a binary frame. This is the same loop as `lfm_audio.voice()`, just sending over the socket instead of playing locally.

**Verify during build:** whether a text note can sit in the same user turn as the audio (`chat.add_audio(...)` then `chat.add_text(note)`). If LFM ignores it, add the note as a separate `system` turn instead.

The model is loaded once at startup and warmed up with the same mimi warm-up as `voice()`. Calls run in `asyncio.to_thread` under one `MODEL_LOCK`.

## 7. ASR

```python
mlx_whisper.transcribe(pcm, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                       language="en")["text"].strip()
```
- The input is the same 16 kHz float32 array the browser sent, so no conversion is needed.
- It's warmed up once at startup with 1 s of silence.
- It runs before the voice agent starts, so it doesn't compete with LFM for the GPU.
- **Fallback:** an empty transcript or an exception gives `text = ""`. Jev is skipped, the route is `chat`, and the voice agent still answers from the audio, so the conversation never breaks.

## 8. Router (Jev)

One Jev call per turn. Several `noul` questions go in the same request's `questions` dict.

**`state` sent to Jev:**
```jsonc
{
  "screen": "steps",                                   // welcome | dishes | video | steps
  "dishes_on_screen": ["Shakshuka", "Dutch Baby", "Avocado Toast"],
  "videos_on_screen": ["Easy Shakshuka", "Shakshuka for beginners"],
  "current_step": "2 of 6: Soften the onion",
  "recent_conversation": "User: …\nAssistant: …",     // last 4 turns of SESSION.transcript
  "latest_user_message": "that's hard to follow"
}
```

**Route questions** (winner = highest `noul` ≥ 0.5; none ≥ 0.5 → `chat`):

| id | Instruction (short form) |
|---|---|
| `recommend` | wants breakfast / meal ideas or what's popular |
| `video` | wants to see how to cook a dish (a dish is chosen or named) |
| `steps` | finds the video/recipe hard to follow, wants it broken into steps |
| `next` / `back` / `repeat` | moving through the step cards |
| `seek` | wants to see the current step in the video |
| `show_video` | wants the video again |
| `play` / `pause` | video playback |

**Target questions** (only sent when relevant to the screen):
- `dish_0 … dish_n`: "refers to dish N on screen" (asked on `dishes` screen). This handles "the first one", "the eggy one" and a misheard "shakshuka".
- `video_0 … video_n`: the same for alternate videos.
- "Step 4" is taken from the text with a regex (`step (\d+|one…ten)`), because Jev can't return numbers.

**Guards** (applied after Jev; each one turns an invalid route into `chat`):
- `steps` / `seek` / `show_video` / `play` / `pause` need a video loaded.
- `next` / `back` / `repeat` need steps loaded.
- `video` needs a dish: a `dish_i` ≥ 0.5, or a chosen dish, or else the raw text is used as the dish query.

**Failure:** a Jev timeout (1 s) or error gives `chat`. This fails open, like `retrieval_gate.py`.

Instructions and criteria live in `router.py` as one dict, and they're the "system prompt" to tune when `test_router.py` misfires.

**Verify first (build step 1):** that asking about 10 questions in one call still takes around 100 ms. If it doesn't, split into two calls: route first, then target only when needed.

## 9. Search agent (module 2)

**External: built separately, on a different model; not part of this project.** This project owns only the interface and a stub.

**Raw output from the web agent (the real format, from its example output):**
```jsonc
// video
{ "video_url": "https://www.youtube.com/watch?v=Km7KRbKVu88",
  "description": "This classic French toast recipe is so easy… {\"name\":\"Easy French Toast Recipe\"}" }
// steps: format not yet known
```

**`search_agent.py` adapts raw output into the internal view shapes.** Only this file knows the raw format:
```jsonc
// video(query) →
{ "view": "video", "dish": "<query>",
  "main": { "id": "Km7KRbKVu88", "title": "Easy French Toast Recipe", "minutes": null,
            "thumb": "https://i.ytimg.com/vi/Km7KRbKVu88/hqdefault.jpg",
            "description": "This classic French toast recipe is so easy…" },
  "alternates": [] }
```

Video adapter rules:
- `id` comes from `video_url`, parsed as the `v=` query param, the last path segment for `youtu.be/…`, or `/shorts/…`. A URL that doesn't parse raises an error (module failure).
- `title` comes from a trailing `{"name": …}` JSON blob in `description` if present, stripped from the description. Otherwise it's the first sentence of the description, truncated to 60 chars.
- `thumb` is built from `id`, `minutes` is `null`, and `alternates` is empty. The web agent returns only one video.

```jsonc
// steps(video_id, query) → (internal shape; the adapter maps raw output to this once the raw format is known)
{ "view": "steps", "dish": "French toast", "video_id": "Km7KRbKVu88",
  "steps": [ { "title": "≤5 words", "icon": "🥣", "tags": ["2 min"],
               "detail": "≤2 short sentences", "video_start": 20 /* seconds, or null */ } ] }  // 5-8 steps
```
The server calls both with a 20 s timeout, and any exception counts as a module failure (§14).

v1: the stub adapts `fixtures/demo.json` `raw.video`, and returns `steps` from the fixture. When the web agent exists, only the fetch changes and the adapters stay. The server caches results in `cache/` by query or video ID.

## 10. Recommender (module 3)

`recommender.recommend(query)` → `view: dishes`

**Raw output from the service (the real format):**
```jsonc
[ { "image_url": "https://…/fluffy-french-toast-hero.jpg",
    "description": "A small scoop of flour makes this the best French toast recipe! …" } ]   // 3 items
```

**`recommender.py` adapts it to:**
```jsonc
{ "view": "dishes",
  "meals": [ { "id": 0, "name": null, "image": "https://…", "description": "A small scoop of flour…" } ] }
```
- `id` = the list index.
- `name` = `null`, because the service sends none. Everything downstream uses `label(meal) = name or the first sentence of description`: Jev's `dishes_on_screen`, the ANNOUNCE summary and the query sent to the search agent.
- Items without `image_url` or `description` are dropped, and an empty result counts as a module failure.

v1: the stub adapts `fixtures/demo.json` `raw.recommend`. When the service is ready, only the fetch changes.

`build_query(text, SESSION)` = the latest user text plus the last 2 user turns, e.g. "breakfast for my wife; most popular recently". Modules 2 and 3 both receive this.

## 11. WebSocket protocol

**Binary frames = audio**
- ↑ one utterance: float32 little-endian, 16 kHz mono (browser `AudioContext({sampleRate: 16000})`).
- ↓ streamed reply chunks: float32 little-endian, 24 kHz mono, ~80 ms each. The browser schedules them back to back on an `AudioContext` (24 kHz) and measures the level for the orb.

**Text frames = JSON**

| Dir | `type` | Payload |
|---|---|---|
| ↑ | `text` | `text`. Typed input or welcome chip: skips ASR, still routed. |
| ↑ | `action` | `select_dish {id}` · `select_video {id}` · `goto_step {index}` · `show_steps` · `show_video` |
| ↓ | `state` | `thinking` · `speaking` · `idle` |
| ↓ | `caption` | `text` (voice-agent text, sent as it streams) |
| ↓ | `view` | `dishes {meals}` · `video {dish, main, alternates}` · `steps {dish, video_id, steps}` |
| ↓ | `control` | `goto_step {index}` · `video {cmd: play\|pause\|seek, t}` |
| ↓ | `metrics` | see §13 |
| ↓ | `error` | `text` |

Clicks (`action`) go straight to the same NAV/LOOKUP handlers as the matching voice routes, with no ASR and no Jev.
The browser sets `listening` itself when the mic opens and `idle` once playback of the last chunk ends.
On connect, the server replays the current `view` so a refresh restores the stage.

## 12. Session state

```python
SESSION = {
    "transcript": [],     # [("user"|"assistant", text)] — saved user input + agent replies
    "screen": "welcome",
    "meals": [], "dish": None,
    "video": None,        # {"main", "alternates"}
    "steps": [], "step": 0,
    "ws": None,
}
```
The transcript is also appended to `cache/transcript-YYYY-MM-DD.jsonl`, so user input is saved across restarts.

## 13. Metrics

Sent after every turn and shown in the metrics strip under the avatar (toggled with `M`, visible by default):
```jsonc
{ "type": "metrics",
  "route": "video", "target": "dish_0",
  "scores": {"recommend": 0.04, "video": 0.94, "steps": 0.07, "...": 0},
  "ms": { "asr": 310, "jev": 104, "voice_first_audio": 820,
          "module": 2100, "announce_first_audio": 3300, "total": 4200 } }
```
All timings are measured from the end of speech (when the server receives the PCM). The strip shows the route plus one bar per stage, so it's visible which model the time went to.

## 14. Error handling

| Failure | Behaviour |
|---|---|
| ASR fails or is empty | route `chat`; voice agent answers from audio |
| Jev timeout or error | route `chat` (fails open) |
| Module 2/3 error, timeout (20 s) or bad JSON after one retry | `FAILED` note → voice agent apologises; screen unchanged |
| Video won't embed | frontend skips to the next alternate |
| Route guard fails | `chat` |
| Voice agent error | send `error`, `state: idle`; the next turn works normally |
| WebSocket drops mid-turn | turn finishes, and sends to the dead socket are dropped; reconnect replays the view |
| Second tab connects | it takes over; the old socket is closed with code 4000 |

Failures are never cached.

## 15. Testing

- **`test_router.py`**: labelled scenarios in the style of `bench_jev_gate.py`, each with an utterance plus screen state and the expected route or target. It covers the full demo script, the guard cases and small talk ("thanks, she'll love it" → `chat`). It also prints Jev p50/p95 for about 10 questions in one call, which is the build step 1 check. The guards are also tested offline with fake scores.
- **Frontend**: `?mock=1` replays the whole demo from `web/mock.js` with no server.
- **End-to-end**: the scripted demo is run once by voice before every demo. It watches the metrics strip and warms `cache/`.

## 16. Build order

1. `test_router.py` against live Jev: confirm latency and accuracy for many questions in one call.
2. `voice_agent.py`: streaming reply, and a check that notes work (§6).
3. `server.py` + protocol, with the stub recommender and a fake search agent.
4. Frontend against `?mock=1`, then against the server.
5. Plug in the real web agent and recommender when they exist.

## 17. Out of scope (v1)

- Multi-session, auth, accounts
- Hands-free VAD and barge-in (`lfm_audio.voice()` has both for later)
- Speculatively starting the voice agent before Jev (saves ~100 ms, not worth it)
- Character video avatar, step timers
- Recipe-site steps merged with video timestamps
