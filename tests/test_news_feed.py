"""Tests for the dashboard news feed (pure RSS parse + merge; no network)."""

from __future__ import annotations

from monitoring.news_feed import _parse_rss, fetch_headlines

SAMPLE = (
    '<?xml version="1.0"?><rss><channel>'
    '<item><title>Fed holds rates steady</title><link>http://x/1</link>'
    '<pubDate>Mon, 08 Jun 2026 14:00:00 GMT</pubDate></item>'
    '<item><title>Apple earnings beat estimates</title><link>http://x/2</link>'
    '<pubDate>Mon, 08 Jun 2026 15:00:00 GMT</pubDate></item>'
    '</channel></rss>'
)


def test_parse_rss_extracts_items() -> None:
    items = _parse_rss(SAMPLE, "TEST")
    assert len(items) == 2
    assert items[0]["title"] == "Fed holds rates steady"
    assert items[0]["source"] == "TEST"
    assert items[0]["link"] == "http://x/1"


def test_parse_rss_bad_xml_is_empty() -> None:
    assert _parse_rss("<not xml", "TEST") == []


def test_fetch_headlines_merges_and_sorts_newest_first() -> None:
    out = fetch_headlines({"A": "u1", "B": "u2"}, limit=10, fetcher=lambda u: SAMPLE)
    assert len(out) == 4                                  # 2 items × 2 feeds
    assert out[0]["title"] == "Apple earnings beat estimates"   # 15:00 before 14:00


def test_fetch_headlines_skips_failing_feed() -> None:
    def fake(url: str) -> str:
        if url == "bad":
            raise RuntimeError("feed down")
        return SAMPLE

    out = fetch_headlines({"good": "ok", "bad": "bad"}, fetcher=fake)
    assert len(out) == 2 and all(h["source"] == "good" for h in out)
