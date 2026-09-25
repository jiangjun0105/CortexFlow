"""
Nimble web agent for dish discovery and cooking-video lookup.

The agent has two workflows:

TASK 1 - Dish cards
    Input: exactly 3 dish names.
    Output: JSON array with exactly 3 objects, each containing:
        - image_url
        - description

TASK 2 - Selected dish video
    Input: 1 dish name.
    Output: JSON object containing:
        - video_url
        - description

LiquidAI is intentionally NOT used here. It produces the text request upstream.
This module is responsible for web retrieval through Nimble.

Requirements:
    Python 3.10+

Install:
    pip install nimble_python python-dotenv beautifulsoup4

Create local.env next to this file:
    NIMBLE_API_KEY=your-api-key

CLI examples:
    python web_agent.py images "French toast" "Miso soup" "Shakshuka"
    python web_agent.py video "French toast"
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse

import nimble_python
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from nimble_python import Nimble


# ---------------------------------------------------------------------------
# Local environment
# ---------------------------------------------------------------------------

ENV_FILE = Path(__file__).resolve().parent / "local.env"
load_dotenv(ENV_FILE)


# ---------------------------------------------------------------------------
# Nimble client
# ---------------------------------------------------------------------------


def create_nimble_client(
    api_key: Optional[str] = None,
    *,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> Nimble:
    """Create an authenticated Nimble client."""
    resolved_key = api_key or os.getenv("NIMBLE_API_KEY")

    if not resolved_key:
        raise RuntimeError(
            f"NIMBLE_API_KEY is not set. Create {ENV_FILE.name} next to "
            "this script with:\n\nNIMBLE_API_KEY=your-api-key"
        )

    return Nimble(
        api_key=resolved_key,
        timeout=timeout,
        max_retries=max_retries,
    )


nimble = create_nimble_client()


# ---------------------------------------------------------------------------
# Low-level Nimble helpers
# ---------------------------------------------------------------------------


def search_web(
    query: str,
    *,
    max_results: int = 10,
    search_depth: str = "standard",
    full_content: bool = False,
    focus: Optional[str] = None,
    country: Optional[str] = None,
    include_domains: Optional[Iterable[str]] = None,
    exclude_domains: Optional[Iterable[str]] = None,
    time_range: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Any:
    """Run a live Nimble web search."""
    if not query.strip():
        raise ValueError("query must not be empty")

    params: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "full_content": full_content,
    }

    if focus is not None:
        params["focus"] = focus
    if country is not None:
        params["country"] = country
    if include_domains is not None:
        params["include_domains"] = list(include_domains)
    if exclude_domains is not None:
        params["exclude_domains"] = list(exclude_domains)
    if time_range is not None:
        params["time_range"] = time_range
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date

    return nimble.search(**params)


def extract_url(
    url: str,
    *,
    render: bool | str = "auto",
    driver: Optional[str] = None,
    formats: Optional[Iterable[str]] = None,
    country: Optional[str] = None,
    locale: Optional[str] = None,
    parse: Optional[bool] = None,
    parser: Optional[dict[str, Any]] = None,
    browser_actions: Optional[list[dict[str, Any]]] = None,
) -> Any:
    """Fetch a specific public URL through Nimble Extract."""
    if not url.strip():
        raise ValueError("url must not be empty")

    params: dict[str, Any] = {
        "url": url,
        "render": render,
    }

    if driver is not None:
        params["driver"] = driver
    if formats is not None:
        params["formats"] = list(formats)
    if country is not None:
        params["country"] = country
    if locale is not None:
        params["locale"] = locale
    if parse is not None:
        params["parse"] = parse
    if parser is not None:
        params["parser"] = parser
    if browser_actions is not None:
        params["browser_actions"] = browser_actions

    return nimble.extract.run(**params)


def get_search_results(
    query: str,
    *,
    max_results: int = 10,
    search_depth: str = "standard",
    full_content: bool = False,
    include_domains: Optional[Iterable[str]] = None,
    exclude_domains: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Return Nimble search results as plain dictionaries."""
    response = search_web(
        query,
        max_results=max_results,
        search_depth=search_depth,
        full_content=full_content,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )

    rows: list[dict[str, Any]] = []
    for item in response.results:
        rows.append(
            {
                "title": getattr(item, "title", None),
                "url": getattr(item, "url", None),
                "description": getattr(item, "description", None),
                "content": getattr(item, "content", None),
                "metadata": getattr(item, "metadata", None),
            }
        )

    return rows


def get_page_content(
    url: str,
    *,
    render: bool | str = "auto",
    formats: tuple[str, ...] = ("html", "markdown"),
) -> dict[str, Any]:
    """Extract HTML/Markdown from a page through Nimble."""
    response = extract_url(url, render=render, formats=formats)
    data = getattr(response, "data", None)

    return {
        "url": getattr(response, "url", url),
        "task_id": getattr(response, "task_id", None),
        "status": getattr(response, "status", None),
        "status_code": getattr(response, "status_code", None),
        "html": getattr(data, "html", None) if data else None,
        "markdown": getattr(data, "markdown", None) if data else None,
        "headers": getattr(data, "headers", None) if data else None,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# HTML/media extraction helpers
# ---------------------------------------------------------------------------


def _absolute_url(candidate: Optional[str], page_url: str) -> Optional[str]:
    """Convert relative media URLs to absolute URLs."""
    if not candidate:
        return None

    candidate = candidate.strip()
    if not candidate or candidate.startswith(("data:", "blob:", "javascript:")):
        return None

    return urljoin(page_url, candidate)


def _media_url_from_srcset(srcset: Optional[str], page_url: str) -> Optional[str]:
    """Pick the last/largest-looking URL from a srcset."""
    if not srcset:
        return None

    candidates = []
    for entry in srcset.split(","):
        parts = entry.strip().split()
        if parts:
            url = _absolute_url(parts[0], page_url)
            if url:
                candidates.append(url)

    return candidates[-1] if candidates else None


def _parse_json_ld(soup: BeautifulSoup) -> list[Any]:
    """Parse JSON-LD blocks while tolerating malformed/irrelevant blocks."""
    objects: list[Any] = []

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue

        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue

        if isinstance(parsed, list):
            objects.extend(parsed)
        else:
            objects.append(parsed)

    return objects


def _walk_json(value: Any) -> Iterable[Any]:
    """Recursively walk dictionaries/lists from JSON-LD."""
    yield value

    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _recipe_description_from_html(soup: BeautifulSoup) -> Optional[str]:
    """Prefer Schema.org Recipe.description when present."""
    for root in _parse_json_ld(soup):
        for value in _walk_json(root):
            if isinstance(value, dict):
                value_type = value.get("@type")
                types = value_type if isinstance(value_type, list) else [value_type]
                if "Recipe" in types and value.get("description"):
                    return str(value["description"]).strip()

    return None


def _description_from_html(soup: BeautifulSoup) -> Optional[str]:
    """Extract a useful page description from metadata."""
    recipe_description = _recipe_description_from_html(soup)
    if recipe_description:
        return recipe_description

    candidates = [
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "description"}),
        ("meta", {"name": "twitter:description"}),
    ]

    for tag_name, attrs in candidates:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get("content"):
            value = tag["content"].strip()
            if value:
                return value

    # Fall back to the first substantive paragraph.
    for paragraph in soup.find_all("p"):
        text = " ".join(paragraph.stripped_strings)
        if len(text) >= 80:
            return text

    return None


def _score_image_candidate(url: str, dish_name: str, context: str = "") -> int:
    """Score an image candidate using simple deterministic signals."""
    haystack = f"{url} {context}".lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", dish_name.lower()) if len(t) > 2]

    score = 0
    for token in tokens:
        if token in haystack:
            score += 3

    if any(ext in urlparse(url).path.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
        score += 3

    # Avoid common non-dish images when an alternative exists.
    for bad in ("logo", "icon", "avatar", "favicon", "sprite", "banner", "advert"):
        if bad in haystack:
            score -= 5

    return score


def _extract_image_url_from_page(
    html: str,
    page_url: str,
    dish_name: str,
) -> Optional[str]:
    """Find the most plausible dish image URL on a recipe page."""
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[int, str]] = []

    # 1. Schema.org Recipe.image is usually the cleanest source.
    for root in _parse_json_ld(soup):
        for value in _walk_json(root):
            if not isinstance(value, dict):
                continue

            value_type = value.get("@type")
            types = value_type if isinstance(value_type, list) else [value_type]
            image = value.get("image")

            if "Recipe" in types and image:
                image_values = image if isinstance(image, list) else [image]
                for image_value in image_values:
                    if isinstance(image_value, dict):
                        image_value = image_value.get("url") or image_value.get("contentUrl")
                    absolute = _absolute_url(str(image_value), page_url)
                    if absolute:
                        candidates.append((100, absolute))

    # 2. OpenGraph / Twitter preview image.
    for attrs in (
        {"property": "og:image"},
        {"property": "og:image:url"},
        {"name": "twitter:image"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            absolute = _absolute_url(tag["content"], page_url)
            if absolute:
                candidates.append((90, absolute))

    # 3. Image tags, using alt/title text as context.
    for image in soup.find_all("img"):
        src = image.get("src") or image.get("data-src") or _media_url_from_srcset(
            image.get("srcset") or image.get("data-srcset"), page_url
        )
        absolute = _absolute_url(src, page_url)
        if absolute:
            context = " ".join(
                filter(
                    None,
                    [
                        image.get("alt"),
                        image.get("title"),
                        image.get("class", [None])[0]
                        if isinstance(image.get("class"), list)
                        else image.get("class"),
                    ],
                )
            )
            candidates.append(
                (_score_image_candidate(absolute, dish_name, context), absolute)
            )

    if not candidates:
        return None

    # Deduplicate while preserving the highest score.
    best_by_url: dict[str, int] = {}
    for score, url in candidates:
        best_by_url[url] = max(best_by_url.get(url, -10_000), score)

    return max(best_by_url.items(), key=lambda pair: pair[1])[0]


def _first_youtube_result(results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Select the first YouTube watch/shorts result."""
    for item in results:
        url = (item.get("url") or "").lower()
        if "youtube.com/watch" in url or "youtube.com/shorts/" in url or "youtu.be/" in url:
            return item
    return None


def _clean_description(text: Optional[str]) -> str:
    """Normalize a short frontend description."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip()
    # Keep descriptions concise for cards/frontend output.
    if len(text) > 500:
        text = text[:497].rsplit(" ", 1)[0] + "..."
    return text


# ---------------------------------------------------------------------------
# TASK 1: three dishes -> image URL + dish description
# ---------------------------------------------------------------------------


def find_dish_card(dish_name: str) -> dict[str, str]:
    """Find one dish's image URL and short description."""
    dish_name = dish_name.strip()
    if not dish_name:
        raise ValueError("dish_name must not be empty")

    query = f'"{dish_name}" recipe dish'

    # Search for recipe pages. Social/video results are not useful for this task.
    results = get_search_results(
        query,
        max_results=8,
        search_depth="standard",
        full_content=True,
        exclude_domains=(
            "pinterest.com",
            "youtube.com",
            "facebook.com",
            "instagram.com",
            "tiktok.com",
        ),
    )

    if not results:
        raise LookupError(f"No web results found for dish: {dish_name}")

    # Choose the first result with recipe-like signals and a usable URL.
    selected = results[0]
    for result in results:
        combined = " ".join(
            str(result.get(field) or "")
            for field in ("title", "description", "content", "url")
        ).lower()
        if "recipe" in combined or "ingredients" in combined:
            selected = result
            break

    page_url = selected.get("url")
    if not page_url:
        raise LookupError(f"Search result had no URL for dish: {dish_name}")

    page = get_page_content(
        page_url,
        render="auto",
        formats=("html", "markdown"),
    )

    html = page.get("html") or ""
    soup = BeautifulSoup(html, "html.parser") if html else BeautifulSoup("", "html.parser")

    image_url = _extract_image_url_from_page(
        html,
        page_url,
        dish_name,
    )

    description = _description_from_html(soup)
    if not description:
        description = _clean_description(selected.get("description"))

    if not image_url:
        raise LookupError(
            f"Found a page for {dish_name}, but could not identify a dish image URL."
        )

    if not description:
        description = f"A recipe for {dish_name}."

    return {
        "image_url": image_url,
        "description": _clean_description(description),
    }


def get_three_dish_cards(dish_names: Iterable[str]) -> list[dict[str, str]]:
    """Process exactly three dish names in input order."""
    names = [name.strip() for name in dish_names if name and name.strip()]

    if len(names) != 3:
        raise ValueError(
            f"TASK 1 requires exactly 3 non-empty dish names; received {len(names)}."
        )

    return [find_dish_card(name) for name in names]


def save_three_dish_cards(
    dish_names: Iterable[str],
    output_file: str | Path = "dish_cards.json",
) -> list[dict[str, str]]:
    """Run TASK 1 and save the exact required JSON structure."""
    cards = get_three_dish_cards(dish_names)
    output_path = Path(output_file)
    output_path.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cards


# ---------------------------------------------------------------------------
# TASK 2: selected dish -> cooking video URL + video description
# ---------------------------------------------------------------------------


def find_dish_video(dish_name: str) -> dict[str, str]:
    """Find a cooking video for one selected dish."""
    dish_name = dish_name.strip()
    if not dish_name:
        raise ValueError("dish_name must not be empty")

    query = f'"{dish_name}" cooking video recipe'

    # Prefer YouTube pages because the requested output is a user-playable video URL.
    results = get_search_results(
        query,
        max_results=8,
        search_depth="standard",
        full_content=False,
        include_domains=("youtube.com",),
    )

    selected = _first_youtube_result(results)

    # Fallback: broaden the search if no YouTube result was returned.
    if selected is None:
        fallback_results = get_search_results(
            f'"{dish_name}" cooking tutorial video',
            max_results=10,
            search_depth="standard",
            full_content=False,
        )
        video_hosts = (
            "youtube.com",
            "youtu.be",
            "vimeo.com",
            "dailymotion.com",
        )
        for result in fallback_results:
            url = (result.get("url") or "").lower()
            if any(host in url for host in video_hosts):
                selected = result
                break

    if selected is None or not selected.get("url"):
        raise LookupError(f"No cooking video found for dish: {dish_name}")

    video_url = selected["url"]
    description = _clean_description(selected.get("description"))

    # Try the video page metadata when the search snippet is too thin.
    if len(description) < 40:
        try:
            page = get_page_content(
                video_url,
                render="auto",
                formats=("html", "markdown"),
            )
            html = page.get("html") or ""
            if html:
                soup = BeautifulSoup(html, "html.parser")
                extracted = _description_from_html(soup)
                if extracted:
                    description = _clean_description(extracted)
        except Exception:
            # A video host may block extraction even though the search result is usable.
            pass

    if not description:
        description = f"Cooking video showing how to prepare {dish_name}."

    return {
        "video_url": video_url,
        "description": description,
    }


def save_dish_video(
    dish_name: str,
    output_file: str | Path = "dish_video.json",
) -> dict[str, str]:
    """Run TASK 2 and save the exact required JSON structure."""
    result = find_dish_video(dish_name)
    output_path = Path(output_file)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run either TASK 1 or TASK 2 from the command line."""
    if not ENV_FILE.exists():
        raise SystemExit(
            f"{ENV_FILE.name} was not found. Create it next to web_agent.py "
            "with:\n\nNIMBLE_API_KEY=your-api-key"
        )

    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage:\n"
            '  python web_agent.py images "dish 1" "dish 2" "dish 3"\n'
            '  python web_agent.py video "dish name"'
        )

    task = sys.argv[1].strip().lower()

    try:
        if task == "images":
            if len(sys.argv) != 5:
                raise SystemExit(
                    'TASK 1 requires exactly 3 dish names. Example:\n'
                    'python web_agent.py images "French toast" "Miso soup" "Shakshuka"'
                )

            results = save_three_dish_cards(sys.argv[2:5], "dish_cards.json")
            print(json.dumps(results, ensure_ascii=False, indent=2))
            print("\nSaved to dish_cards.json")

        elif task == "video":
            if len(sys.argv) != 3:
                raise SystemExit(
                    'TASK 2 requires exactly 1 dish name. Example:\n'
                    'python web_agent.py video "French toast"'
                )

            result = save_dish_video(sys.argv[2], "dish_video.json")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("\nSaved to dish_video.json")

        else:
            raise SystemExit('Unknown task. Use "images" or "video".')

    except nimble_python.AuthenticationError as exc:
        raise SystemExit(
            "Nimble authentication failed. Check NIMBLE_API_KEY in local.env."
        ) from exc
    except nimble_python.RateLimitError as exc:
        raise SystemExit("Nimble rate limit reached. Please retry later.") from exc
    except nimble_python.APIConnectionError as exc:
        raise SystemExit(
            "Could not connect to Nimble. Check your network connection."
        ) from exc
    except nimble_python.APIStatusError as exc:
        raise SystemExit(
            f"Nimble API error ({exc.status_code}): {exc.message}"
        ) from exc
    except (LookupError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
