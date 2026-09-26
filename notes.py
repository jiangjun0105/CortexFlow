"""Voice-agent prompt text (spec §6): the system prompt and the notes added to a user turn."""
import re

SYSTEM = ("You are a warm, brief kitchen helper speaking out loud. A separate system finds dishes, "
          "recipes and videos and shows them on the screen. Never name dishes, give recipe steps, "
          "or state facts you'd need to look up — the screen does that. "
          "Keep every reply to one or two short sentences.")

# ponytail: tuned by try_notes.py against LFM2.5-Audio; near-identical rewordings (e.g. "That is being looked up")
# make it answer the question instead. Re-run try_notes.py after any change.
REASSURE = ("[The answer is being looked up and will appear on screen. Don't answer yourself. "
            "Just say something like 'On it, one sec!']")

FAILED = "[The lookup failed. Apologise briefly and suggest trying again.]"


def label(meal):
    return meal.get("name") or re.split(r"(?<=[.!?])\s", meal["description"].strip(), 1)[0]


def announce(view):
    """What the screen now shows, and what one sentence to say about it: the ask differs per screen."""
    kind = view["view"]
    if kind == "dishes":
        summary = f"{len(view['meals'])} breakfasts: " + "; ".join(label(m) for m in view["meals"])
        ask = "ask which one they'd like to make"
    elif kind == "video":
        main = view["main"]
        summary = (f"a {main['minutes']}-minute video: " if main.get("minutes") else "a video: ") + main["title"]
        ask = "say the video is playing, and they can ask for step-by-step cards if it's hard to follow"
    elif kind == "steps":
        summary = f"{len(view['steps'])} steps; step 1 is {view['steps'][0]['title']}"
        ask = "say step 1 out loud, and that they can say 'next' when ready"
    else:
        raise ValueError(kind)
    return f"[The screen now shows {summary.rstrip('.')}. In one sentence, {ask}.]"
