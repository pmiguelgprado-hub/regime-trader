"""Cross-sectional momentum ranker (vía C — the v1 return predictor).

The project's HMM is a *volatility classifier*, not a return predictor (proven OOS:
three timing/rotation avenues all modulated beta without adding alpha). The only
non-falsified way to beat the index is a **cross-sectional** signal — predicting which
names outperform *others*, not predicting the index level (near-random-walk). See
docs/analysis/2026-06-04-stock-picking-feasibility.md.

This module is the simplest such return predictor: **cross-sectional momentum (12-1)**,
the Jegadeesh-Titman signal — rank names by their trailing 12-month return *skipping the
most recent month* (the skip avoids short-term reversal). It needs **no training data**
and has **no fitted parameters** (lookback/skip are fixed, not swept — sweeping is the
overfit trap this project escaped). The ML return predictor (Gu-Kelly-Xiu trees/NN) is a
gated v2 that needs a paid point-in-time panel to train.

The ranker is the *alpha* (what to hold); the existing HMM is the *risk overlay* (how much
gross exposure, via core.asset_rotation.vol_target_scale). Kept pure and causal so it is
fully unit-testable with no network and no look-ahead.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

# Daily-bar defaults for the 12-1 momentum window (≈12 months lookback, ≈1 month skip).
DEFAULT_LOOKBACK = 252
DEFAULT_SKIP = 21
TOP_DECILE = 0.10


def momentum_score(
    close: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
) -> float:
    """Cross-sectional momentum (12-1): trailing return skipping recent bars.

    Measures the cumulative return over the window ``[t-lookback, t-skip]`` using the
    close series ending at the decision bar ``t`` (most recent value last). The most
    recent ``skip`` bars are excluded so a short-term spike/reversal does not pollute the
    momentum signal. Strictly causal: uses only past closes.

    Args:
        close: Close-price series ending at the decision bar (most recent last).
        lookback: Bars back to the start of the window (≈12 months).
        skip: Most-recent bars to exclude from the window end (≈1 month).

    Returns:
        The window return as a float, or ``nan`` if there is too little history to
        span the lookback (such names are dropped from the ranking).
    """
    if lookback <= skip or len(close) < lookback + 1:
        return float("nan")
    start = float(close.iloc[-(lookback + 1)])
    end = float(close.iloc[-(skip + 1)])
    if start <= 0.0:
        return float("nan")
    return end / start - 1.0


def rank_universe(
    frames: dict[str, pd.DataFrame],
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
) -> list[str]:
    """Rank a universe of names by descending cross-sectional momentum.

    Names without enough history to score (``nan``) are dropped. Ties break by symbol
    name for determinism (avoids the cross-process non-reproducibility, R-4, that any
    unstable ordering would reintroduce).

    Args:
        frames: ``{symbol: OHLCV}`` with a ``close`` column, each ending at the same
            decision bar.
        lookback: Momentum lookback in bars.
        skip: Momentum skip in bars.

    Returns:
        Symbols ordered best-momentum first.
    """
    scored: list[tuple[str, float]] = []
    for sym, df in frames.items():
        score = momentum_score(df["close"], lookback=lookback, skip=skip)
        if score == score:   # not nan
            scored.append((sym, score))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [sym for sym, _ in scored]


def select_top(ranked: list[str], frac: float = TOP_DECILE) -> list[str]:
    """Select the top ``frac`` of a ranked list (at least one name if non-empty).

    Args:
        ranked: Symbols ordered best-first (from :func:`rank_universe`).
        frac: Fraction to keep (0.10 = top decile).

    Returns:
        The top slice; empty if ``ranked`` is empty.
    """
    if not ranked:
        return []
    import math

    n = max(1, math.ceil(len(ranked) * frac))
    return ranked[:n]


def select_top_sector_capped(
    ranked: list[str],
    sector_map: dict[str, str],
    frac: float = TOP_DECILE,
    max_sector_frac: float = 0.30,
    max_n: int | None = None,
) -> list[str]:
    """Select the top ``frac`` by momentum, capping any single sector's share.

    Walks the momentum ranking best-first and admits a name only while its GICS sector is
    below the per-sector cap (``ceil(max_sector_frac * target_n)`` names). The hottest
    sector still gets the **most** slots (up to the cap) — so the book rides growing
    sectors like semiconductors — but no single sector can dominate the whole book. Names
    skipped for a full sector are passed over in favour of the next-best momentum name in
    another sector.

    Args:
        ranked: Symbols ordered best-momentum first (from :func:`rank_universe`).
        sector_map: ``{symbol: GICS sector}`` (missing -> ``"UNKNOWN"``).
        frac: Target fraction of the universe to hold (0.10 = top decile).
        max_sector_frac: Max share of the *book* any one sector may take (0.30 = 30%).
        max_n: Hard cap on book size (the downstream ``max_concurrent``). When set, the
            target and the per-sector cap are computed against ``min(frac*N, max_n)`` so
            the sector share holds against the *realized* book, not a larger pre-truncation
            target.

    Returns:
        Selected symbols (momentum order), sector-capped. May be shorter than the target
        if too few names clear the cap.
    """
    if not ranked:
        return []
    import math

    target_n = max(1, math.ceil(len(ranked) * frac))
    if max_n:
        target_n = min(target_n, max_n)
    cap = max(1, math.ceil(max_sector_frac * target_n))
    counts: dict[str, int] = {}
    out: list[str] = []
    for sym in ranked:
        sec = sector_map.get(sym, "UNKNOWN")
        if counts.get(sec, 0) >= cap:
            continue                       # sector full -> skip to next-best other sector
        out.append(sym)
        counts[sec] = counts.get(sec, 0) + 1
        if len(out) >= target_n:
            break
    return out


def make_book_weights(
    frames: dict[str, pd.DataFrame],
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    frac: float = TOP_DECILE,
    max_single: float = 0.15,
    max_concurrent: int = 50,
    risk_on_gross: float = 1.0,
    risk_off_gross: float = 0.5,
    use_overlay: bool = True,
    sector_map: dict[str, str] | None = None,
    max_sector_frac: float = 0.30,
) -> Callable[[pd.Timestamp, float], dict[str, float]]:
    """Build a monthly-rebalanced weight function for the cross-sectional book.

    Returns ``weight_fn(ts, vol_rank) -> {symbol: weight}`` suitable for
    :meth:`Backtester.run_portfolio`'s ``weight_fn`` hook and reusable live. Each call:

    1. **Monthly rebalance** — the top-decile selection is recomputed only when the
       ``(year, month)`` of ``ts`` changes and held otherwise (realistic turnover, not a
       daily churn that costs would eat). The ranking uses each name's closes sliced
       **up to** ``ts`` (strictly causal — no look-ahead).
    2. **Alpha** — :func:`rank_universe` + :func:`select_top` pick the momentum leaders.
    3. **Risk overlay** — :func:`~core.asset_rotation.regime_gross_scale` turns the HMM
       ``vol_rank`` into a total gross multiplier (full in risk-on, de-risked in
       risk-off). ``use_overlay=False`` pins gross to 1.0 (the *naked ranker* the gate
       compares against).
    4. **Weights** — equal-weight the selected names to that gross, capped per name via
       the existing :func:`~core.portfolio.portfolio_target_weights`.

    Args:
        frames: ``{symbol: OHLCV}`` full history (sliced causally per bar).
        lookback, skip, frac: Momentum window + top-fraction (fixed, un-swept).
        max_single, max_concurrent: Per-name and count caps.
        risk_on_gross, risk_off_gross: Overlay gross in the low/high-vol tiers.
        use_overlay: If False, gross is always 1.0 (naked ranker).

    Returns:
        A stateful ``weight_fn(ts, vol_rank)`` closure (monthly-memoized).
    """
    from core.asset_rotation import regime_gross_scale
    from core.portfolio import portfolio_target_weights

    cache: dict[tuple[int, int], list[str]] = {}

    def weight_fn(ts: pd.Timestamp, vol_rank: float) -> dict[str, float]:
        key = (ts.year, ts.month)
        top = cache.get(key)
        if top is None:
            sliced = {s: df.loc[:ts] for s, df in frames.items()}
            ranked = rank_universe(sliced, lookback, skip)
            top = (select_top_sector_capped(ranked, sector_map, frac, max_sector_frac,
                                            max_n=max_concurrent)
                   if sector_map else select_top(ranked, frac))
            cache[key] = top
        gross = regime_gross_scale(vol_rank, risk_on_gross, risk_off_gross) if use_overlay else 1.0
        return portfolio_target_weights(gross, top, max_single, max_concurrent)

    return weight_fn


def compute_book_targets(
    frames: dict[str, pd.DataFrame],
    vol_rank: float,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    frac: float = TOP_DECILE,
    max_single: float = 0.15,
    max_concurrent: int = 50,
    risk_on_gross: float = 1.0,
    risk_off_gross: float = 0.5,
    use_overlay: bool = True,
    sector_map: dict[str, str] | None = None,
    max_sector_frac: float = 0.30,
) -> dict[str, float]:
    """One-shot target weights for the live/paper rebalance (single decision now).

    The live counterpart of :func:`make_book_weights`: rank the universe on the latest
    available closes, take the top decile, scale gross by the current HMM ``vol_rank``,
    and equal-weight under the caps. Pure (no network) so the rebalance decision is
    unit-testable; the orchestration that fetches data and submits orders lives in
    ``main.run_rebalance``.

    Args:
        frames: ``{symbol: OHLCV}`` ending at the latest closed bar.
        vol_rank: Current market regime volatility rank in ``[0, 1]`` (HMM on the proxy).
        lookback, skip, frac, max_single, max_concurrent, risk_on_gross, risk_off_gross,
        use_overlay: As in :func:`make_book_weights`.

    Returns:
        ``{symbol: target_weight}`` for the selected top-decile names.
    """
    from core.asset_rotation import regime_gross_scale
    from core.portfolio import portfolio_target_weights

    ranked = rank_universe(frames, lookback, skip)
    top = (select_top_sector_capped(ranked, sector_map, frac, max_sector_frac,
                                    max_n=max_concurrent)
           if sector_map else select_top(ranked, frac))
    gross = regime_gross_scale(vol_rank, risk_on_gross, risk_off_gross) if use_overlay else 1.0
    return portfolio_target_weights(gross, top, max_single, max_concurrent)


def targets_to_orders(
    targets: dict[str, float],
    equity: float,
    prices: dict[str, float],
) -> list[dict]:
    """Turn target weights into a concrete, whole-share order plan (pure).

    The rebalance decision made executable: weight × equity = target notional;
    notional ÷ price = whole shares (floored — no fractional shares). Names without a
    usable price are skipped. Kept pure so the rebalance plan is unit-testable without a
    broker; ``main.run_rebalance`` wraps this with live data + (gated) submission.

    Args:
        targets: ``{symbol: target_weight}`` from :func:`compute_book_targets`.
        equity: Account equity to allocate.
        prices: ``{symbol: last_price}``.

    Returns:
        List of ``{symbol, weight, notional, price, shares}`` (shares > 0 only), sorted
        by descending notional.
    """
    import math

    plan: list[dict] = []
    for sym, w in targets.items():
        price = prices.get(sym, 0.0)
        if price <= 0.0:
            continue
        notional = w * equity
        shares = int(math.floor(notional / price))
        if shares <= 0:
            continue
        plan.append({"symbol": sym, "weight": w, "notional": notional,
                     "price": price, "shares": shares})
    plan.sort(key=lambda o: -o["notional"])
    return plan


def plan_rebalance_orders(
    target_shares: dict[str, int],
    held_shares: dict[str, int],
) -> list[dict]:
    """Diff target vs held positions into a rebalance order list (pure).

    For each name, trade the share delta: buy to increase, sell to reduce. Names held
    but **not** in the target (dropped from the book, or never part of it) get a full
    liquidating sell (target 0). **Sells are emitted before buys** so the freed cash
    covers the buys — 49 buys at ~98% gross would otherwise exhaust buying power before
    the sells settle and the broker would reject the tail.

    Args:
        target_shares: ``{symbol: desired_shares}`` (from the book plan).
        held_shares: ``{symbol: currently_held_shares}`` (from the broker).

    Returns:
        Ordered ``[{symbol, side, qty}, ...]`` — all sells first, then buys; names with
        zero delta are omitted.
    """
    sells: list[dict] = []
    buys: list[dict] = []
    for sym in sorted(set(target_shares) | set(held_shares)):
        delta = int(target_shares.get(sym, 0)) - int(held_shares.get(sym, 0))
        if delta < 0:
            sells.append({"symbol": sym, "side": "sell", "qty": -delta})
        elif delta > 0:
            buys.append({"symbol": sym, "side": "buy", "qty": delta})
    return sells + buys
