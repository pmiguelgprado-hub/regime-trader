"""Tests for the news fetch + sentiment factor (T2.2) — no network (injected fetch).

Alpaca News API (free, Benzinga since 2015) -> per-symbol daily sentiment factor.
Scorer is pluggable; the default is a deterministic financial lexicon (no torch),
with FinBERT / local Ollama as documented swap-ins. PIT: only headlines stamped
before the session close count for that day. Pure parsing/scoring is unit-tested.
"""

from __future__ import annotations

from datetime import date

from core import sentiment_factor as sf
from data import news_data as nd


# --- news parsing -----------------------------------------------------------------


def _api_body():
    return (
        '{"news": ['
        '{"headline": "Acme beats earnings, raises guidance", "symbols": ["ACME"],'
        ' "created_at": "2026-06-10T13:00:00Z"},'
        '{"headline": "Acme faces lawsuit, shares slump on weak outlook", "symbols": ["ACME"],'
        ' "created_at": "2026-06-10T20:00:00Z"},'
        '{"headline": "Beta announces buyback", "symbols": ["BETA"],'
        ' "created_at": "2026-06-11T09:00:00Z"}'
        ']}'
    )


def test_parse_news_returns_articles():
    arts = nd.parse_news(_api_body())
    assert len(arts) == 3
    assert arts[0]["headline"].startswith("Acme beats")
    assert "ACME" in arts[0]["symbols"]


def test_fetch_news_uses_injected_getter():
    arts = nd.fetch_news(["ACME"], "2026-06-01", "2026-06-12",
                         fetch=lambda url, headers: (200, _api_body()))
    assert len(arts) == 3


def test_fetch_news_non200_raises():
    import pytest
    with pytest.raises(RuntimeError):
        nd.fetch_news(["ACME"], "2026-06-01", "2026-06-12",
                      fetch=lambda url, headers: (429, "rate limited"))


# --- lexicon scorer ---------------------------------------------------------------


def test_lexicon_positive_headline():
    s = sf.lexicon_score("Acme beats earnings, raises guidance, strong growth")
    assert s > 0


def test_lexicon_negative_headline():
    s = sf.lexicon_score("Acme faces lawsuit, shares slump on weak outlook, misses")
    assert s < 0


def test_lexicon_neutral_headline():
    assert sf.lexicon_score("Acme to hold annual meeting on Tuesday") == 0.0


# --- daily aggregation + PIT ------------------------------------------------------


def test_daily_sentiment_aggregates_per_symbol():
    arts = nd.parse_news(_api_body())
    scores = sf.daily_sentiment(arts, asof=date(2026, 6, 11))
    assert "ACME" in scores and "BETA" in scores
    assert -1.0 <= scores["ACME"] <= 1.0


def test_daily_sentiment_pit_excludes_future_headlines():
    arts = nd.parse_news(_api_body())
    # as of 2026-06-10: BETA's 06-11 headline must not count yet
    scores = sf.daily_sentiment(arts, asof=date(2026, 6, 10))
    assert "BETA" not in scores
    assert "ACME" in scores


def test_rank_by_sentiment_orders_best_first():
    scores = {"GOOD": 0.8, "MEH": 0.0, "BAD": -0.5}
    ranked = sf.rank_by_sentiment(scores)
    assert ranked[0] == "GOOD" and ranked[-1] == "BAD"


def test_custom_scorer_is_pluggable():
    arts = nd.parse_news(_api_body())
    scores = sf.daily_sentiment(arts, asof=date(2026, 6, 11),
                                scorer=lambda text: 1.0)   # constant scorer
    assert scores["ACME"] == 1.0
