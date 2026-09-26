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


# Notes tell the model what is happening, not what to say; it replies in its own words.
def reassure(route, dish=None):
    """Status when Jev triggers a lookup, while the web agent works."""
    what = {"recommend": "trending breakfasts online",
            "video": f"a cooking video for {dish}" if dish else "a cooking video online",
            "steps": "step-by-step cards for this recipe"}[route]
    return f"[Status: you are now searching for {what}. The results will appear on the screen in a few seconds.]"


def announce(view):
    """Status when the web agent's result is on screen."""
    kind = view["view"]
    if kind == "dishes":
        names = [label(m) for m in view["meals"]]
        found = f"{len(names)} trending breakfasts: " + ", ".join(names)
    elif kind == "video":
        found = f"a cooking video, {view['main']['title'].rstrip('.')}, now playing"
    elif kind == "steps":
        found = f"{len(view['steps'])} step-by-step cards, starting with {view['steps'][0]['title']}"
    else:
        raise ValueError(kind)
    return f"[Status: the search finished. The screen now shows {found}.]"
