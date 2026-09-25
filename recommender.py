"""Adapts web_agent.py dish cards into the `dishes` view (spec §10)."""
import asyncio
import re

from search_agent import cached, web

# ponytail: fixed names until the social-media recommender exists
DISHES = ["French toast", "Miso soup", "Shakshuka"]


async def recommend(query: str) -> dict:
    # names are fixed, so cache on them, not on however the user phrased the ask
    return await _cards(", ".join(DISHES))


@cached
async def _cards(key: str) -> dict:
    find = web().find_dish_card
    raws = await asyncio.gather(*(asyncio.to_thread(find, n) for n in DISHES), return_exceptions=True)
    meals = [
        {"id": i, "name": name, "image": raw["image_url"], "description": raw["description"]}
        for i, (name, raw) in enumerate(zip(DISHES, raws))
        if isinstance(raw, dict) and raw.get("image_url") and raw.get("description")
    ]
    if not meals:
        raise RuntimeError(f"all dish lookups failed: {raws}")
    return {"view": "dishes", "meals": meals}


def label(meal: dict) -> str:
    return meal.get("name") or re.split(r"(?<=[.!?])\s", meal.get("description") or "", maxsplit=1)[0]
