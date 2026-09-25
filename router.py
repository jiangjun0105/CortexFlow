"""Jev router: one TypeSafe System One call per turn -> (route, target). Spec §8."""
import asyncio
import os
import re
from pathlib import Path

import httpx

ROUTES = ["recommend", "video", "steps", "next", "back", "repeat", "seek", "show_video", "play", "pause"]
LOOKUPS = {"recommend", "video", "steps"}
URL = "https://api.typesafe.ai/v1/systemone"
NEEDS_VIDEO = {"steps", "seek", "show_video", "play", "pause"}
NEEDS_STEPS = {"next", "back", "repeat"}
NO = "Anything else: a different request, chit-chat, thanks, or an acknowledgment."

# The tunable "system prompt": {id: (instructions, true criterion)}.
QUESTIONS = {
    "recommend": ("Does the user want breakfast or meal ideas, suggestions, or what's popular?",
                  "The latest message asks for dish ideas, recommendations, or what's popular/trending to cook."),
    "video": ("Does the user want to see how to cook a specific dish (choosing one on screen or naming one)?",
              "The latest message picks a dish (e.g. 'let's do X', 'the second one', 'that one looks good') or asks how to make a named dish."),
    "steps": ("Does the user find the video or recipe hard to follow and want it broken into steps?",
              "The latest message says the video/recipe is confusing, too fast, or hard to follow, or asks for step-by-step instructions, "
              "and the step cards are not already on screen (screen is not 'steps')."),
    "next": ("Does the user want to move to the next step?",
             "The latest message asks to go forward to the next step ('next', 'okay next', 'done, what now')."),
    "back": ("Does the user want to go back to the previous step?",
             "The latest message asks to go back to the previous step."),
    "repeat": ("Does the user want the current step repeated?",
               "The latest message asks to hear or see the current step again ('say that again', 'repeat')."),
    "seek": ("Does the user want to see the current step in the video?",
             "The latest message asks to be shown the part of the video for the current step ('show me that part')."),
    "show_video": ("Does the user want to go back to watching the whole video?",
                   "The latest message asks to see or watch the video again."),
    "play": ("Does the user want the video to play or resume?",
             "The latest message asks to play, resume, or continue the video."),
    "pause": ("Does the user want the video paused or stopped?",
              "The latest message asks to pause, stop, or hold the video."),
}
WORDS = {w: i + 1 for i, w in enumerate("one two three four five six seven eight nine ten".split())}
STEP_RE = re.compile(r"\bstep\s+(\d+|" + "|".join(WORDS) + r")\b", re.I)


def _label(meal):
    return meal["name"] or re.split(r"(?<=[.!?])\s", meal["description"].strip(), maxsplit=1)[0]


def _best(scores, prefix):
    hits = [(v, int(k[len(prefix):])) for k, v in scores.items() if k.startswith(prefix) and v >= 0.5]
    return max(hits)[1] if hits else None


def pick(scores, text, session):
    steps = session.get("steps") or []
    m = STEP_RE.search(text)
    if m and steps:
        n = int(m[1]) if m[1].isdigit() else WORDS[m[1].lower()]
        if 1 <= n <= len(steps):
            return "goto", n - 1
    r = max(ROUTES, key=lambda k: scores.get(k, 0))
    if scores.get(r, 0) < 0.5:
        return "chat", None
    if (r in NEEDS_VIDEO and not session.get("video")) or (r in NEEDS_STEPS and not steps):
        return "chat", None
    if r == "video":  # no dish_i hit -> server uses chosen dish or the raw text as the query
        return "video", _best(scores, "dish_")
    if r == "show_video":
        return r, _best(scores, "video_")
    return r, None


def _q(instructions, true):
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": NO}}


def _questions(session):
    qs = {k: _q(*v) for k, v in QUESTIONS.items()}
    if session.get("screen") == "dishes":
        for i, meal in enumerate(session.get("meals") or []):
            qs[f"dish_{i}"] = _q(f"Does the user's latest message refer to dish {i + 1} on screen: '{_label(meal)}'?",
                                 f"The user means this dish (by position, name, description, or a mishearing of it).")
    for i, v in enumerate((session.get("video") or {}).get("alternates") or []):
        qs[f"video_{i}"] = _q(f"Does the user's latest message refer to alternate video {i + 1}: '{v.get('title')}'?",
                              "The user means this video (by position or title).")
    return qs


def _state(text, session):
    video = session.get("video") or {}
    steps = session.get("steps") or []
    step = session.get("step", 0)
    cur = f"{step + 1} of {len(steps)}: {steps[step].get('title', '')}" if 0 <= step < len(steps) else None
    tail = session.get("transcript", [])[-4:]
    return {
        "screen": session.get("screen", "welcome"),
        "dishes_on_screen": [_label(m) for m in session.get("meals") or []] if session.get("screen") == "dishes" else [],
        "videos_on_screen": [v.get("title") for v in [video.get("main"), *video.get("alternates", [])] if v],
        "current_step": cur,
        "recent_conversation": "\n".join(f"{r.capitalize()}: {t}" for r, t in tail),
        "latest_user_message": text,
    }


def _key():
    if k := os.environ.get("TYPESAFE_API_KEY"):
        return k
    env = Path(__file__).with_name(".env")
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


KEY = _key()
_client = None


async def route(text, session):
    global _client
    try:
        _client = _client or httpx.AsyncClient(timeout=1.0)
        r = await asyncio.wait_for(_client.post(
            URL, headers={"Authorization": f"Bearer {KEY}"},
            json={"state": _state(text, session), "model": "jev-latest", "questions": _questions(session)}), 1.0)
        r.raise_for_status()
        scores = {k: float(v["noul"]) for k, v in r.json()["answers"].items()}
    except Exception as e:
        print(f"[router] jev failed: {e!r}")
        return "chat", None, {}
    return (*pick(scores, text, session), scores)
