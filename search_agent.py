"""Adapts web_agent.py (Nimble) raw output into the view payloads (spec §9)."""
import asyncio
import functools
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
FIXTURE = json.loads((ROOT / "fixtures" / "demo.json").read_text())


def web():
    """Lazy import: web_agent builds its Nimble client at import time and reads only local.env."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import web_agent
    return web_agent


def cached(fn):
    """JSON cache at cache/<fn>-<slug>.json keyed on the first arg; only successes are stored."""
    @functools.wraps(fn)
    async def wrapper(arg, *rest):
        slug = re.sub(r"[^a-z0-9]+", "-", arg.lower()).strip("-")[:80] or "empty"
        path = CACHE / f"{fn.__name__}-{slug}.json"
        if path.exists():
            return json.loads(path.read_text())
        result = await fn(arg, *rest)
        CACHE.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    return wrapper


def youtube_id(url: str) -> str:
    u = urlparse(url)
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        vid = u.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        m = re.match(r"/(?:shorts|embed)/([^/?#]+)", u.path)
        vid = m.group(1) if m else (parse_qs(u.query).get("v") or [""])[0]
    else:
        vid = ""
    if not vid:
        raise ValueError(f"not a YouTube video URL: {url}")
    return vid


def adapt_video(raw: dict, dish: str) -> dict:
    vid = youtube_id(raw["video_url"])
    desc = (raw.get("description") or "").strip()
    title = None
    m = re.search(r"\s*(\{\s*\"name\"\s*:.*\})\s*$", desc)
    if m:
        try:
            title = json.loads(m.group(1))["name"]
            desc = desc[: m.start()].strip()
        except (ValueError, KeyError):
            pass
    if not title:
        title = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0][:60] or dish
    return {
        "view": "video",
        "dish": dish,
        "main": {"id": vid, "title": title, "minutes": None,
                 "thumb": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg", "description": desc},
        "alternates": [],
    }


@cached
async def video(query: str) -> dict:
    raw = await asyncio.to_thread(web().find_dish_video, query)
    return adapt_video(raw, query)


async def steps(video_id: str, query: str) -> dict:
    # ponytail: fixture steps; nothing generates steps yet
    return {**FIXTURE["steps"], "video_id": video_id, "dish": query}
