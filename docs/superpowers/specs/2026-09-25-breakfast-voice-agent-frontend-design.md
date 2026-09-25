# Breakfast Voice Agent — Frontend Design

Date: 2026-09-25
Status: Draft, awaiting review
Scope: the web page only. The backend (intent detection, search, transcript → steps) gets its own spec.

## 1. Goal

A single web page where I talk to a voice agent to plan and cook breakfast for my wife.
The agent talks back, and everything it finds shows up visually on the page instead of in a chat log.

Success = I can run this whole conversation hands-free:

1. "I want to cook breakfast for my wife. What's popular lately?" → trending dish cards appear
2. "Let's do the shakshuka. How do I cook it?" → a video of the dish appears
3. "That's hard to follow." → the video turns into step-by-step cards
4. "Next." / "Go back." / "Repeat that." / "Show me this part." → move between steps, jump the video

## 2. Layout

Two columns, full viewport, no page scroll.

```
┌───────────────────────┬──────────────────────────────┐
│                       │                              │
│        AVATAR         │           STAGE              │
│   (orb, animated by   │   (one view at a time:       │
│    agent state)       │    welcome / dishes /        │
│                       │    video / steps)            │
│  caption: last thing  │                              │
│  the agent said       │                              │
│                       │                              │
│      [ 🎤 mic ]       │                              │
│ → video  asr▮ jev▮ …  │                              │
└───────────────────────┴──────────────────────────────┘
        ~40% width                 ~60% width
```

- **Left (avatar column):** the avatar, one caption line and the mic button. Nothing else.
- **Right (stage):** exactly one view at a time. New views replace old ones with a transition. No history or scrollback.
- **Narrow screens (< 900px):** the avatar shrinks to a small strip on top and the stage takes the rest. This is for phones propped up in the kitchen.
- **Theme:** warm and calm. Dark background by default (easy to read from across a kitchen), one warm accent colour (e.g. egg-yolk amber), large type. Step text is at least 32px.

## 3. Avatar

### v1: animated orb
A glowing orb drawn with CSS plus a `<canvas>` or SVG. It has four states:

| State       | When                                     | Look                                   |
|-------------|------------------------------------------|----------------------------------------|
| `idle`      | nothing happening                        | slow breathing pulse, dim              |
| `listening` | mic is open                              | brighter, ring reacts to mic volume    |
| `thinking`  | request sent, waiting for reply          | slow swirl / shimmer                   |
| `speaking`  | agent audio playing                      | pulses with output audio volume        |

Volume comes from a Web Audio `AnalyserNode` on the mic stream (listening) or on the playback element (speaking).

### Later: character avatar
Pre-generate four short looping clips of one character (e.g. a friendly chef), one per state, with a video-generation model. The avatar component then swaps `<video src>` by state instead of drawing the orb. The state machine does not change, so this is a drop-in replacement.

### Caption
The caption shows the last sentence the agent said ("Sure, pulling that up — just a few seconds…"). It fades in word by word while the agent speaks. There's only one line, and old captions disappear.

## 4. Stage views

### 4.0 Welcome (initial)
A short greeting ("Good morning — what are we cooking?") plus 2–3 example prompts as faint chips. Clicking a chip sends it as text, which also makes a handy test shortcut.

### 4.1 Dishes — "What's popular?"
```
┌──────────────────────────────────────────┐
│  Trending breakfasts this week           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  [img]   │ │  [img]   │ │  [img]   │  │
│  │ Shakshuka│ │ Dutch    │ │ Avocado  │  │
│  │ 25 min   │ │ Baby     │ │ Toast    │  │
│  │ easy     │ │ 30 min   │ │ 10 min   │  │
│  └──────────┘ └──────────┘ └──────────┘  │
│  Say "the first one" or the dish name    │
└──────────────────────────────────────────┘
```
- Exactly 3 cards in a row (from the recommender). Each card shows an image, name, total time, difficulty tag and a one-line "why it's popular".
- Cards fly in with a short stagger.
- **Selection:** when a dish is chosen (by voice, or by clicking the card), that card grows and the rest fade out. Then the video view loads.

### 4.2 Video — "How do I cook it?"
```
┌──────────────────────────────────────────┐
│  Shakshuka                               │
│  ┌────────────────────────────────────┐  │
│  │          ▶ YouTube player          │  │
│  └────────────────────────────────────┘  │
│  Other videos:                           │
│  [thumb] Easy Shakshuka · 8 min          │
│  [thumb] Shakshuka for beginners · 12 min│
└──────────────────────────────────────────┘
```
- Main player: YouTube IFrame Player API (`https://www.youtube.com/embed/<id>`), 16:9, autoplay.
- Below it, 2–3 alternates. Clicking one (or saying "the second one") swaps the main video.
- **Audio rule:** the video pauses automatically whenever the mic opens or the agent speaks, so its sound never leaks into the mic. It does not resume on its own; I say "play" or click.

### 4.3 Steps — "That's hard to follow"
```
┌──────────────────────────────────────────┐
│  Shakshuka          Step 2 of 6  ●●○○○○  │
│  ┌────────────────────────────────────┐  │
│  │  🧅  Soften the onion              │  │
│  │                                    │  │
│  │  [medium heat] [5 min]             │  │
│  │  Stir until it's clear, not brown  │  │
│  └────────────────────────────────────┘  │
│   ← heat oil               add garlic →  │
│                          [▶ back to video]│
└──────────────────────────────────────────┘
```
- One big card per step: icon/emoji, short title, tags (heat, time, key ingredient) and one or two lines of detail.
- A progress dots row, plus previous and next step titles as small hints.
- **Entry transition:** the video shrinks and slides away while the cards deal in from the right. This is the demo's signature moment, so it's worth polishing.
- **Navigation:** voice ("next", "back", "repeat", "go to step 4"), on-screen arrows, or the keyboard ← / →.
- **"Show me this part":** returns to the video view and seeks to the current step's `video_start`.
- **"Back to video" chip:** returns to the video at its last position. Returning to steps keeps the current step.
- Step content comes from the video's transcript (the backend turns it into steps), so each step carries a timestamp.

## 5. Interaction model

- **Push-to-talk first:** click the mic button or hold Space to talk, and release to send. This matches how `talk_space.py` works today and avoids echo problems.
- Hands-free voice detection (auto start/stop) is a later upgrade, because the backend already has silero VAD experience in `lfm_audio.py`.
- Every voice action also has a click or keyboard equivalent. That covers when the kitchen is loud, and it makes testing easy.

## 6. Frontend ↔ backend contract

The frontend is dumb: it renders what the backend tells it. It keeps one WebSocket open to the backend.
Audio travels as **binary frames**: up = one utterance of raw float32 16kHz PCM, down = streamed ~80ms chunks of float32 24kHz PCM, played back-to-back on an `AudioContext` (see the technical design §11). Everything below is a JSON text frame. The technical design §11 is the source of truth for the protocol.

**Frontend → backend**
```jsonc
{ "type": "text",  "text": "what's popular for breakfast?" } // welcome chips / debug
{ "type": "action", "name": "select_dish", "id": "shakshuka" } // clicks
{ "type": "action", "name": "goto_step", "index": 3 }
```

**Backend → frontend**
```jsonc
{ "type": "state",   "value": "thinking" }                 // idle|listening|thinking|speaking
{ "type": "caption", "text": "Sure, pulling that up…" }
{ "type": "view",    "view": "dishes", "meals": [            // always 3, from the recommender
    { "id": "shakshuka", "name": "Shakshuka", "image": "https://…",
      "minutes": 25, "difficulty": "easy", "why": "Trending on TikTok this week" } ] }
{ "type": "view",    "view": "video", "dish": "Shakshuka",
  "main": { "id": "abc123", "title": "…", "minutes": 8, "thumb": "https://…" },
  "alternates": [ { "id": "…", "title": "…", "minutes": 12, "thumb": "…" } ] }
{ "type": "view",    "view": "steps", "dish": "Shakshuka", "video_id": "abc123",
  "steps": [ { "title": "Soften the onion", "icon": "🧅",
               "tags": ["medium heat", "5 min"], "detail": "Stir until…",
               "video_start": 95 } ] }
{ "type": "control", "name": "goto_step", "index": 1 }      // voice "next"/"back"
{ "type": "control", "name": "video", "cmd": "play" | "pause" | "seek", "t": 95 }
{ "type": "error",   "text": "Couldn't find videos for that, try another dish?" }
{ "type": "metrics", "route": "video", "scores": {…}, "ms": {"asr": 310, "jev": 104, …} }
```

The frontend sets `listening` itself when the mic opens, and `idle` once the last audio chunk finishes playing. The backend sends `thinking` and `speaking`.

**Metrics strip:** a thin strip under the mic button shows the last turn's route (e.g. `→ video · dish 0`) and one horizontal bar per stage (ASR, Jev, voice agent, module), labelled in ms. The `M` key toggles it; it's visible by default.

## 7. Frontend state

```
agentState: idle | listening | thinking | speaking
view:       welcome | dishes | video | steps
dishes[], selectedDish, video{main, alternates, position}, steps[], stepIndex
```
- Views are cached. Going video → steps → video keeps the video position, and going back to steps keeps `stepIndex`.
- A new `view` message of the same type replaces the cached data (for example, a new dish search).

## 8. Error and edge states

| Case                          | What the page shows                                             |
|-------------------------------|-----------------------------------------------------------------|
| Mic permission denied         | Mic button greyed out, tooltip explaining; text chips still work |
| WebSocket drops               | Small "reconnecting…" pill top-right; auto-retry with backoff   |
| Backend `error` message       | Agent caption shows the text; the stage keeps the current view  |
| Search returns nothing        | Stage keeps previous view; agent says so                         |
| Video won't embed / errors    | Auto-skip to the next alternate; if none, show thumbnail + "open on YouTube" link |
| Thinking > 15s                | Caption changes to "Still looking…" so it doesn't feel frozen   |
| Missing dish image            | Warm gradient placeholder with the dish emoji                    |

## 9. Tech choices

- **One `index.html` plus one `app.js` plus one `style.css`**, vanilla JS, no build step. There are four views and one socket, so a framework isn't needed. Revisit if the page grows past that.
- Served by the same Python backend (FastAPI static files), so there's one process and no CORS.
- YouTube IFrame Player API for playback control. Web Audio API for mic capture and volume meters.
- Transitions use CSS (`transform`/`opacity` plus `@keyframes`), and the View Transitions API where the browser supports it.

## 10. Testing

- **Mock mode:** opening `index.html?mock=1` skips the WebSocket and replays a scripted list of backend messages (the full breakfast conversation from §1) from `mock.js`, with delays. This lets the whole UI be built and demoed before the backend exists. It's also the regression check: run the script and every view and transition must render without console errors.
- Manual check on a laptop and on a phone-sized viewport.

## 11. Out of scope for v1

- Character video avatar (the orb ships first; see §3)
- Hands-free VAD (push-to-talk first)
- Timers on step cards
- Recipe-website steps merged with video timestamps
- Conversation history, accounts, saved recipes
