"""Alpaca News API client (T2.2) — free Benzinga headlines since 2015.

Fetches news headlines per symbol for the sentiment factor. Free on the Alpaca
data plan (200 calls/min), Benzinga history back to 2015 — so the factor is
backtestable AND forward, with no paid data (consistent with the program's
no-paid-data invariant). Auth uses the same Alpaca key headers; the HTTP getter
is injected so parsing is unit-tested with no network.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

# (status_code, body_text) <- (url, headers)
Fetcher = Callable[[str, dict], "tuple[int, str]"]

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


def parse_news(body: str) -> list[dict]:
    """Parse the Alpaca news JSON into ``[{headline, summary, symbols, created_at}, ...]``."""
    data = json.loads(body)
    out = []
    for a in data.get("news", []):
        out.append({
            "headline": a.get("headline", ""),
            "summary": a.get("summary", ""),
            "symbols": list(a.get("symbols", [])),
            "created_at": a.get("created_at", ""),
        })
    return out


def _default_fetch(url: str, headers: dict) -> "tuple[int, str]":  # pragma: no cover - network
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return e.code, e.read().decode("utf-8", "replace")


def fetch_news(symbols: list[str], start: str, end: str, limit: int = 50,
               fetch: Optional[Fetcher] = None) -> list[dict]:
    """Fetch news for ``symbols`` in ``[start, end]`` (ISO dates).

    Args:
        symbols: Tickers (joined into the ``symbols`` query param).
        start, end: ISO date bounds.
        limit: Max articles per request.
        fetch: Injected HTTP getter (defaults to the real Alpaca-authed GET).

    Returns:
        Parsed article list.

    Raises:
        RuntimeError: On a non-200 response.
    """
    import urllib.parse
    if fetch is None:
        fetch = _default_fetch
    headers = {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
    }
    q = urllib.parse.urlencode({"symbols": ",".join(symbols), "start": start,
                                "end": end, "limit": limit})
    status, body = fetch(f"{NEWS_URL}?{q}", headers)
    if status != 200:
        raise RuntimeError(f"Alpaca news fetch failed: HTTP {status}")
    return parse_news(body)
