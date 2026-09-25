"""
Nimble web-agent connection layer.

This module contains only the Nimble API integration.
The Japanese breakfast recipe search/extraction logic can be added later.

Requirements:
    Python 3.9+
    pip install nimble_python python-dotenv

Environment:
    Create a local.env file next to this script:

        NIMBLE_API_KEY=your-api-key
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

import nimble_python
from dotenv import load_dotenv
from nimble_python import Nimble


# Load local.env from the same directory as this script.
ENV_FILE = Path(__file__).resolve().parent / "local.env"
load_dotenv(ENV_FILE)


def create_nimble_client(
    api_key: Optional[str] = None,
    *,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> Nimble:
    """Create an authenticated Nimble client.

    Priority:
        1. Explicit api_key argument
        2. NIMBLE_API_KEY from local.env / environment
    """
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


# Reusable application client.
nimble = create_nimble_client()


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
    """Run a real-time Nimble web search."""
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
    render: bool | str = True,
    driver: Optional[str] = None,
    formats: Optional[Iterable[str]] = None,
    country: Optional[str] = None,
    locale: Optional[str] = None,
    parse: Optional[bool] = None,
    parser: Optional[dict[str, Any]] = None,
    browser_actions: Optional[list[dict[str, Any]]] = None,
) -> Any:
    """Fetch a webpage through Nimble Extract."""
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
) -> list[dict[str, Any]]:
    """Return search results as plain dictionaries."""
    response = search_web(
        query,
        max_results=max_results,
        search_depth=search_depth,
        full_content=full_content,
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
    render: bool | str = True,
    formats: tuple[str, ...] = ("markdown",),
) -> dict[str, Any]:
    """Return common page-content fields from Nimble Extract."""
    response = extract_url(
        url,
        render=render,
        formats=formats,
    )

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


def main() -> None:
    """Verify local.env and the Nimble connection.

    Run:
        python web_agent.py

    This performs one inexpensive search.
    """
    print(f"Loading environment from: {ENV_FILE}")

    if not ENV_FILE.exists():
        raise SystemExit(
            f"{ENV_FILE.name} was not found. Create it next to "
            "web_agent.py with:\n\nNIMBLE_API_KEY=your-api-key"
        )

    try:
        results = get_search_results(
            "Breakfast recipes",
            max_results=3,
            search_depth="lite",
        )

        print(f"\nNimble connection OK — received {len(results)} results.\n")

        for index, item in enumerate(results, start=1):
            print(f"{index}. {item['title']}")
            print(f"   {item['url']}")

    except nimble_python.AuthenticationError as exc:
        raise SystemExit(
            "Nimble authentication failed. Check NIMBLE_API_KEY in local.env."
        ) from exc

    except nimble_python.RateLimitError as exc:
        raise SystemExit(
            "Nimble rate limit reached. Please retry later."
        ) from exc

    except nimble_python.APIConnectionError as exc:
        raise SystemExit(
            "Could not connect to Nimble. Check your network connection."
        ) from exc

    except nimble_python.APIStatusError as exc:
        raise SystemExit(
            f"Nimble API error ({exc.status_code}): {exc.message}"
        ) from exc


if __name__ == "__main__":
    main()