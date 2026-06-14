"""News sentiment factor (T2.2) — cross-sectional, point-in-time.

Turns Alpaca/Benzinga headlines into a per-symbol daily sentiment score, then a
cross-sectional ranking. The scorer is **pluggable**: the default is a
deterministic financial lexicon (Loughran-McDonald flavored — no torch, fully
reproducible), with FinBERT (pinned, for backtest) or the local Ollama worker
(qwen, private/free — the AIOS synergy) as documented swap-ins. Re-scoring the
history with a new scorer is a new trial in the research ledger.

PIT discipline: only headlines time-stamped before the session close count for
that day (no look-ahead). Pure + unit-tested.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable, Optional

# Small financial sentiment lexicon (extend as needed; the scorer is swappable).
_POSITIVE = {
    "beats", "beat", "raises", "raise", "raised", "strong", "growth", "surge",
    "surges", "upgrade", "upgraded", "outperform", "record", "buyback", "profit",
    "gains", "gain", "rally", "rallies", "tops", "top", "boost", "boosts",
    "approval", "approved", "wins", "win", "expands", "soars", "soar", "bullish",
}
_NEGATIVE = {
    "miss", "misses", "missed", "lawsuit", "slump", "slumps", "weak", "downgrade",
    "downgraded", "cut", "cuts", "loss", "losses", "plunge", "plunges", "fraud",
    "probe", "recall", "warning", "warns", "bankruptcy", "decline", "declines",
    "falls", "fall", "drops", "drop", "bearish", "halts", "halt", "investigation",
}


def lexicon_score(text: str) -> float:
    """Net sentiment of a headline in [-1, 1] (pos-neg counts / total hits; 0 if none)."""
    words = [w.strip(".,!?:;\"'()").lower() for w in str(text).split()]
    pos = sum(w in _POSITIVE for w in words)
    neg = sum(w in _NEGATIVE for w in words)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _created_date(article: dict) -> Optional[date]:
    raw = article.get("created_at", "")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def daily_sentiment(articles: list[dict], asof: date,
                    scorer: Optional[Callable[[str], float]] = None) -> dict[str, float]:
    """Mean sentiment per symbol over headlines published on or before ``asof`` (PIT).

    Args:
        articles: ``[{headline, symbols, created_at}, ...]`` (from news_data).
        asof: As-of date — articles created after it are excluded (no look-ahead).
        scorer: ``text -> [-1,1]`` (defaults to :func:`lexicon_score`).

    Returns:
        ``{symbol: mean_score}`` over its eligible headlines (symbols with none omitted).
    """
    score = scorer or lexicon_score
    buckets: dict[str, list[float]] = {}
    for a in articles:
        d = _created_date(a)
        if d is None or d > asof:
            continue
        s = score(a.get("headline", ""))
        for sym in a.get("symbols", []):
            buckets.setdefault(sym, []).append(s)
    return {sym: sum(v) / len(v) for sym, v in buckets.items() if v}


def rank_by_sentiment(scores: dict[str, float]) -> list[str]:
    """Symbols ordered most-positive first (ties broken by symbol)."""
    return sorted(scores, key=lambda s: (-scores[s], s))
