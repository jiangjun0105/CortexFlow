"""Adapts web_agent.py dish cards into the `dishes` view (spec §10).

Dish names come from the RawTree social-media trends that send_recs.py POSTs to /recommendations
(saved in cache/recommendations.json); until one arrives, a fixed list.
"""
import asyncio
import json
import re

from search_agent import CACHE, cached, web

FALLBACK = ["French toast", "Miso soup", "Shakshuka"]
PUSHED = CACHE / "recommendations.json"


def dishes() -> list[str]:
    """Top-3 trend names from the latest send_recs.py push, else the fallback list."""
    try:
        rows = json.loads(PUSHED.read_text())["recommendations"]
        names = [str(r["trend"]).strip() for r in rows if str(r.get("trend") or "").strip()][:3]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        names = []
    return names or FALLBACK


def save_push(payload: dict) -> list[str]:
    """Store a send_recs.py payload; returns the dish names it yields (raises ValueError if unusable)."""
    rows = payload.get("recommendations")
    if not isinstance(rows, list) or not any(isinstance(r, dict) and r.get("trend") for r in rows):
        raise ValueError("expected {'recommendations': [{'trend': ...}, ...]}")
    CACHE.mkdir(exist_ok=True)
    PUSHED.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return dishes()


async def recommend(query: str) -> dict:
    # cache on the dish names, not on however the user phrased the ask
    return await _cards(" | ".join(dishes()))


@cached
async def _cards(key: str) -> dict:
    names = key.split(" | ")
    find = web().find_dish_card
    raws = await asyncio.gather(*(asyncio.to_thread(find, n) for n in names), return_exceptions=True)
    meals = [
        {"id": i, "name": name, "image": raw["image_url"], "description": raw["description"]}
        for i, (name, raw) in enumerate(zip(names, raws))
        if isinstance(raw, dict) and raw.get("image_url") and raw.get("description")
    ]
    if not meals:
        raise RuntimeError(f"all dish lookups failed: {raws}")
    return {"view": "dishes", "meals": meals}


def label(meal: dict) -> str:
    return meal.get("name") or re.split(r"(?<=[.!?])\s", meal.get("description") or "", maxsplit=1)[0]
