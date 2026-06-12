"""Quality (+momentum) cross-sectional ranking — paper-only candidate sleeve.

Implements the one structural Sharpe lever the 2026-06-11 honest review endorsed:
combine a **low-correlation factor** (quality — Novy-Marx gross profitability + low
leverage + profitability) with the live price-momentum book, **vol-targeted**, as a
*separate forward-paper sleeve*. Portfolio Sharpe = f(sleeve Sharpes, correlations);
quality and momentum are the classic complementary pair (AQR, "Quality Minus Junk";
Novy-Marx 2013), so combining them is the only available lever that is **not leverage**.

**This module does NOT touch the frozen baseline/challenger books or their 12-month
forward gate.** It is a candidate evaluated forward, under its own pre-registration
(docs/analysis/2026-06-11-quality-momentum-sleeve-prereg.md).

It **reuses, never duplicates**, the existing machinery:

* :func:`core.fundamental_features.compute_features` — point-in-time quality factors,
  gated on SimFin *Publish Date* <= as-of (no look-ahead).
* :func:`core.cross_sectional_ranking.rank_universe` — price-momentum ranking.
* ``select_top`` / ``select_top_sector_capped`` — top-decile + GICS sector cap.
* ``_overlay_gross`` — the vol-target gross overlay (Barroso-Santa-Clara / Daniel-Moskowitz).
* ``_weight_names`` — equal / inv-vol weighting with per-name caps.

**Honest limitation (same wall as the rest of the project):** SimFin's free history
carries survivorship bias, so this sleeve is for **forward** evaluation against the
pre-registered gate, not a historical-backtest edge claim. Pure + unit-tested; the
SimFin network fetch lives in ``data.simfin_data`` and is injected as loaded blocks.
"""

from __future__ import annotations

from datetime import date
from statistics import fmean, pstdev
from typing import Any, Callable, Optional

import pandas as pd

from core.cross_sectional_ranking import (
    DEFAULT_LOOKBACK,
    DEFAULT_SKIP,
    TOP_DECILE,
    _overlay_gross,
    _resolve_overlay,
    _weight_names,
    rank_universe,
    select_top,
    select_top_sector_capped,
)
from core.fundamental_features import compute_features

# Sign of each quality factor's contribution to the composite (+ = higher is better).
# Profitability/quality up, balance-sheet leverage down (Novy-Marx 2013; Piotroski 2000).
QUALITY_SIGNS: dict[str, float] = {
    "gross_profitability": 1.0,
    "roa": 1.0,
    "roe": 1.0,
    "gross_margin": 1.0,
    "leverage": -1.0,
}


def _zscore(values: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    """Cross-sectional z-score of a ``{symbol: value}`` map.

    Missing (None/NaN) values stay None (the ranker drops them, like momentum does).
    Returns all-zero (None preserved) when fewer than two usable points or zero spread,
    so a degenerate factor contributes nothing rather than exploding.
    """
    usable = [v for v in values.values() if v is not None and v == v]
    if len(usable) < 2:
        return {k: (0.0 if (v is not None and v == v) else None) for k, v in values.items()}
    mu, sd = fmean(usable), pstdev(usable)
    if sd == 0:
        return {k: (0.0 if (v is not None and v == v) else None) for k, v in values.items()}
    return {
        k: ((v - mu) / sd if (v is not None and v == v) else None)
        for k, v in values.items()
    }


def quality_scores(features_by_symbol: dict[str, dict[str, Optional[float]]]) -> dict[str, float]:
    """Composite quality score per symbol = mean of sign-adjusted factor z-scores.

    Args:
        features_by_symbol: ``{symbol: {factor: value_or_None}}`` (from
            :func:`core.fundamental_features.compute_features`). Symbols with no usable
            factor are dropped from the output.

    Returns:
        ``{symbol: composite_z}`` (higher = higher quality). Cross-sectional, so it is
        only meaningful relative to the universe passed in.
    """
    syms = list(features_by_symbol)
    parts: dict[str, list[float]] = {s: [] for s in syms}
    for factor, sign in QUALITY_SIGNS.items():
        raw = {s: features_by_symbol[s].get(factor) for s in syms}
        z = _zscore(raw)
        for s in syms:
            if z[s] is not None:
                parts[s].append(sign * z[s])
    return {s: fmean(p) for s, p in parts.items() if p}


def rank_by_quality(features_by_symbol: dict[str, dict[str, Optional[float]]]) -> list[str]:
    """Symbols ordered best-quality first. Ties break by symbol (R-4 determinism)."""
    scores = quality_scores(features_by_symbol)
    return sorted(scores, key=lambda s: (-scores[s], s))


def combined_rank(momentum_ranked: list[str], quality_score_map: dict[str, float]) -> list[str]:
    """Average-of-ranks combine of momentum and quality (AQR "Value and Momentum").

    Rank-average (not score-average) so the two factors mix on equal footing without one
    factor's scale dominating. Only names present in **both** rankings are kept — a name
    needs a price-momentum score *and* point-in-time fundamentals to enter the book.

    Args:
        momentum_ranked: Symbols best-momentum first (from :func:`rank_universe`).
        quality_score_map: ``{symbol: composite_z}`` (from :func:`quality_scores`).

    Returns:
        Symbols ordered best-combined first (lowest average rank), ties by symbol.
    """
    mom_rank = {s: i for i, s in enumerate(momentum_ranked)}
    qual_ranked = sorted(quality_score_map, key=lambda s: (-quality_score_map[s], s))
    qual_rank = {s: i for i, s in enumerate(qual_ranked)}
    common = mom_rank.keys() & qual_rank.keys()
    avg = {s: (mom_rank[s] + qual_rank[s]) / 2.0 for s in common}
    return sorted(avg, key=lambda s: (avg[s], s))


def make_book_weights_quality(
    frames: dict[str, pd.DataFrame],
    blocks: dict[str, dict[str, Any]],
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    frac: float = TOP_DECILE,
    max_single: float = 0.15,
    max_concurrent: int = 50,
    combine: str = "quality_momentum",
    overlay: "str | None" = "vol_target",
    target_vol: float = 0.12,
    vol_window: int = 126,
    gross_cap: float = 1.0,
    gross_floor: float = 0.0,
    risk_on_gross: float = 1.0,
    risk_off_gross: float = 0.5,
    weighting: str = "equal",
    sector_map: dict[str, str] | None = None,
    max_sector_frac: float = 0.30,
) -> Callable[[pd.Timestamp, float], dict[str, float]]:
    """Build a monthly-rebalanced weight function for the quality(+momentum) sleeve.

    Returns ``weight_fn(ts, vol_rank) -> {symbol: weight}`` — the same contract as
    :func:`core.cross_sectional_ranking.make_book_weights`, so it drops straight into
    ``Backtester.run_portfolio``'s ``weight_fn`` hook and the live rebalance path.

    Each call (monthly-memoized on ``(year, month)`` of ``ts``):

    1. **Point-in-time fundamentals** — for ``asof = ts.date()``, re-derive each name's
       quality features from its SimFin block, keeping only periods *published* on or
       before that date (no look-ahead). Names without eligible fundamentals are dropped.
    2. **Alpha** — ``combine="quality"`` ranks by the quality composite alone;
       ``"quality_momentum"`` (default) average-of-ranks combines it with causal
       price momentum (:func:`rank_universe` sliced up to ``ts``).
    3. **Selection** — top ``frac`` with the GICS sector cap (if ``sector_map`` given).
    4. **Risk overlay** — ``_overlay_gross`` (default ``"vol_target"`` to ~12% annual,
       the only overlay that survived the project's own validation).
    5. **Weights** — equal or inv-vol to the gross budget, per-name capped.

    Args:
        frames: ``{symbol: OHLCV}`` full history (sliced causally per bar).
        blocks: ``{symbol: SimFin company block}`` (from ``data.simfin_data.statements``),
            loaded once; PIT slicing happens per ``asof`` inside, so it is network-free here.
        combine: ``"quality_momentum"`` (default) or ``"quality"``.
        overlay: ``"vol_target"`` (default), ``"hmm"``, ``"both"`` or ``"none"``.
        Other args: as in :func:`core.cross_sectional_ranking.make_book_weights`.

    Returns:
        A stateful ``weight_fn(ts, vol_rank)`` closure (monthly-memoized selection + gross).
    """
    if combine not in ("quality", "quality_momentum"):
        raise ValueError(f"combine must be 'quality' or 'quality_momentum', got {combine!r}")
    mode = _resolve_overlay(overlay, True)
    cache: dict[tuple[int, int], list[str]] = {}

    def weight_fn(ts: pd.Timestamp, vol_rank: float) -> dict[str, float]:
        key = (ts.year, ts.month)
        top = cache.get(key)
        if top is None:
            asof: date = ts.date()
            feats: dict[str, dict[str, Optional[float]]] = {}
            for sym, blk in blocks.items():
                f = compute_features(blk, asof)
                if f is not None:
                    feats[sym] = f
            qmap = quality_scores(feats)
            if combine == "quality":
                ranked = sorted(qmap, key=lambda s: (-qmap[s], s))
            else:
                sliced = {s: df.loc[:ts] for s, df in frames.items() if s in qmap}
                ranked = combined_rank(rank_universe(sliced, lookback, skip), qmap)
            top = (
                select_top_sector_capped(ranked, sector_map, frac, max_sector_frac,
                                         max_n=max_concurrent)
                if sector_map else select_top(ranked, frac)
            )
            cache[key] = top
        gross = _overlay_gross(top, frames, ts, vol_rank, mode, risk_on_gross,
                               risk_off_gross, target_vol, vol_window, gross_cap, gross_floor)
        return _weight_names(gross, top, frames, ts, weighting, max_single, max_concurrent)

    return weight_fn
