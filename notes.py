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
    kind = view["view"]
    if kind == "dishes":
        summary = f"{len(view['meals'])} breakfasts: " + "; ".join(label(m) for m in view["meals"])
    elif kind == "video":
        main = view["main"]
        summary = (f"a {main['minutes']}-minute video: " if main.get("minutes") else "a video: ") + main["title"]
    elif kind == "steps":
        summary = f"{len(view['steps'])} steps; step 1 is {view['steps'][0]['title']}"
    else:
        raise ValueError(kind)
    return f"[The screen now shows: {summary.rstrip(".")}. Tell them in one sentence and invite them to pick or continue.]"
