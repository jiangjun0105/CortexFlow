"""Voice-agent prompt text (spec §6): the system prompt and the notes added to a user turn."""
import re

SYSTEM = ("Respond with interleaved text and audio. "  # keep first: LFM's conversation mode (without it, it transcribes)
          "You are a friendly breakfast helper, talking out loud with the user. You can search the web: when the "
          "user asks for breakfast ideas, recipes or cooking videos, a search runs automatically and the results "
          "appear on the screen next to you. Status messages tell you what is being searched and what the screen "
          "shows; talk about them naturally. Keep replies short and conversational.")
# ponytail: picked by A/B on real clips; naming "a separate search agent" made LFM copy its previous reply verbatim

FAILED = "[The lookup failed. Apologise briefly and suggest trying again.]"


def label(meal):
    return meal.get("name") or re.split(r"(?<=[.!?])\s", meal["description"].strip(), 1)[0]


# Notes tell the model what is happening, not what to say; it replies in its own words.
def reassure(route, dish=None):
    """Status when Jev triggers a lookup, while the web agent works."""
    what = {"recommend": "trending breakfasts",
            "video": f"a cooking video for {dish}" if dish else "a cooking video online",
            "steps": "step-by-step cards for this recipe"}[route]
    return f"[Status: now searching online for {what}. The results will appear on the screen in a few seconds.]"


def announce(view):
    """Status when the web agent's result is on screen."""
    kind = view["view"]
    if kind == "dishes":
        names = [label(m) for m in view["meals"]]
        found = f"{len(names)} trending breakfasts: " + ", ".join(names)
    elif kind == "video":
        found = f"a cooking video, {view['main']['title'].rstrip('.')}, ready to play"
    elif kind == "steps":
        found = f"{len(view['steps'])} step-by-step cards, starting with {view['steps'][0]['title']}"
    else:
        raise ValueError(kind)
    return f"[Status: the search finished. The screen now shows {found}.]"
