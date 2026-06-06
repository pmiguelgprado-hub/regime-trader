"""Lightweight US macro/markets news feed for the dashboard (awareness, NOT trading).

Pulls headlines from free finance RSS feeds so the operator has situational awareness of
what is moving the market (Fed, policy, big-company news). This is **for the human to read**
— it does NOT feed any trading decision. Public headlines are priced in within seconds, so
they carry no tradable edge for a retail bot; the value here is context, not alpha.

Pure parsing (``_parse_rss``) is unit-testable with injected XML; the network fetch is
behind an injectable ``fetcher`` so tests never hit the wire. Uses only the standard library
(no feedparser dependency).
"""

from __future__ import annotations

import logging
from typing import Callable

try:  # defuse XXE / billion-laughs on untrusted RSS when available
    from defusedxml.ElementTree import fromstring as _xml_fromstring
    from defusedxml.ElementTree import ParseError as _XMLParseError
except ImportError:  # stdlib fallback: bounded input (see _http_get) limits blowup
    from xml.etree.ElementTree import ParseError as _XMLParseError
    from xml.etree.ElementTree import fromstring as _xml_fromstring

logger = logging.getLogger(__name__)

MAX_FEED_BYTES = 2_000_000          # cap fetched RSS size (defense vs entity-expansion)

# Free, keyless finance RSS feeds (US markets / macro).
DEFAULT_FEEDS: dict[str, str] = {
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
}


def _parse_rss(xml_text: str, source: str) -> list[dict]:
    """Parse an RSS document into ``[{title, link, published, source}]`` (pure)."""
    out: list[dict] = []
    try:
        root = _xml_fromstring(xml_text)
    except (_XMLParseError, ValueError):
        return out
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub = item.findtext("pubDate") or ""
        if title.strip():
            out.append({"title": title.strip(), "link": link.strip(),
                        "published": pub.strip(), "source": source})
    return out


def _http_get(url: str, timeout: float = 6.0) -> str:
    """Fetch a URL as text (stdlib, browser-ish UA so feeds don't 403)."""
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (regime-trader news panel)"})
    with urlopen(req, timeout=timeout) as resp:          # noqa: S310 (trusted RSS hosts)
        return resp.read(MAX_FEED_BYTES).decode("utf-8", errors="replace")


def fetch_headlines(
    feeds: dict[str, str] | None = None,
    limit: int = 15,
    fetcher: Callable[[str], str] | None = None,
) -> list[dict]:
    """Fetch and merge recent headlines across ``feeds`` (most recent first, best-effort).

    Args:
        feeds: ``{source: rss_url}`` (defaults to :data:`DEFAULT_FEEDS`).
        limit: Max headlines returned.
        fetcher: Injectable ``url -> xml_text`` (defaults to a stdlib HTTP GET); a failing
            feed is skipped, never fatal.

    Returns:
        ``[{title, link, published, source}]`` — newest first where dates parse, capped at
        ``limit``. Empty if every feed fails (the panel then shows a placeholder).
    """
    from email.utils import parsedate_to_datetime

    feeds = feeds or DEFAULT_FEEDS
    get = fetcher or _http_get
    items: list[dict] = []
    for source, url in feeds.items():
        try:
            items.extend(_parse_rss(get(url), source))
        except Exception as exc:                          # network/parse: skip this feed
            logger.warning("news feed %s failed: %s", source, exc)

    def _key(it: dict):
        try:
            return parsedate_to_datetime(it["published"])
        except (TypeError, ValueError):
            return None

    dated = [(it, _key(it)) for it in items]
    dated.sort(key=lambda p: (p[1] is not None, p[1]), reverse=True)
    return [it for it, _ in dated][:limit]
