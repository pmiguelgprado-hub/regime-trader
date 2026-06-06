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


DEFAULT_EST_WINDOW = 504


def residual_momentum_score(
    close: pd.Series,
    market_close: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    est_window: int = DEFAULT_EST_WINDOW,
    min_obs: int = 60,
) -> float:
    """Idiosyncratic (residual) momentum: market-model residual 12-1, IR-standardized.

    The Blitz-Huij-Martens (2011) / Chaves (2016) iMOM signal, market-model version:

    1. Estimate the name's market beta + alpha by OLS ``r_stock = a + b·r_market + e`` over
       a **longer estimation window** (``est_window`` ≈ 24 months ending at ``t``).
    2. Form residual returns over the **recent scoring sub-window** ``[t-lookback, t-skip]``
       (≈ the 12-1 window) using those estimated coefficients.
    3. Score = ``mean(residual) / std(residual)`` over the scoring window — an
       information-ratio of *recent firm-specific* return.

    The estimation/scoring split is essential: OLS residuals over their own fit window sum
    to zero by construction, so the signal must score a recent **sub-window** against
    coefficients fit on a longer one. The signal captures recent idiosyncratic
    outperformance (slow information diffusion) without the market-beta exposure that makes
    raw momentum crash on sharp reversals — ~2x the Sharpe of conventional momentum with
    crash risk ≈ eliminated, replicated across 21 countries incl. Japan.

    Strictly causal: uses only closes up to the decision bar; the most recent ``skip``
    return bars are excluded (short-term reversal), mirroring :func:`momentum_score`.

    Args:
        close: Name's close series ending at the decision bar (most recent last).
        market_close: Market-proxy (e.g. SPY) close series, same convention.
        lookback: Bars back to the scoring-window start (≈12 months).
        skip: Most-recent return bars to exclude (≈1 month).
        est_window: Estimation-window length for the beta/alpha fit (≈24 months).
        min_obs: Minimum bars required in each of the estimation and scoring windows.

    Returns:
        Standardized residual momentum (``mean(resid) / std(resid)``), or ``nan`` if there
        is too little history or the regression/residuals are degenerate (dropped names).
    """
    import numpy as np

    if lookback <= skip:
        return float("nan")
    aligned = pd.concat(
        [close.pct_change(), market_close.pct_change()], axis=1, join="inner"
    ).dropna()
    if len(aligned) < max(lookback, min_obs):
        return float("nan")

    est = aligned.iloc[-est_window:] if est_window else aligned
    if len(est) < min_obs:
        return float("nan")
    ye = est.iloc[:, 0].to_numpy()
    xe = est.iloc[:, 1].to_numpy()
    xm = float(xe.mean())
    var_x = float(((xe - xm) ** 2).sum())
    if var_x <= 0.0:
        return float("nan")
    beta = float(((xe - xm) * (ye - ye.mean())).sum() / var_x)
    alpha = float(ye.mean() - beta * xm)

    window = aligned.iloc[-lookback:]
    if skip > 0:
        window = window.iloc[:-skip]
    if len(window) < min_obs:
        return float("nan")
    ys = window.iloc[:, 0].to_numpy()
    xs = window.iloc[:, 1].to_numpy()
    resid = ys - (alpha + beta * xs)
    sd = float(np.std(resid, ddof=1))
    if not (sd > 0.0):
        return float("nan")
    return float(resid.mean() / sd)


def rank_universe_residual(
    frames: dict[str, pd.DataFrame],
    market_close: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    est_window: int = DEFAULT_EST_WINDOW,
) -> list[str]:
    """Rank a universe by descending **residual** (idiosyncratic) momentum.

    The challenger counterpart of :func:`rank_universe`: identical contract (drop
    un-scoreable names, ties break by symbol for determinism / R-4 reproducibility) but
    scores with :func:`residual_momentum_score` against a shared market proxy.

    Args:
        frames: ``{symbol: OHLCV}`` with a ``close`` column, each ending at the decision bar.
        market_close: Market-proxy close series (e.g. SPY), ending at the same bar.
        lookback: Momentum lookback in bars.
        skip: Momentum skip in bars.

    Returns:
        Symbols ordered best residual-momentum first.
    """
    scored: list[tuple[str, float]] = []
    for sym, df in frames.items():
        score = residual_momentum_score(df["close"], market_close, lookback, skip, est_window)
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


def _overlay_gross(
    top: list[str],
    frames: dict[str, pd.DataFrame],
    ts: "pd.Timestamp | None",
    vol_rank: float,
    overlay: str,
    risk_on_gross: float,
    risk_off_gross: float,
    target_vol: float,
    vol_window: int,
    gross_cap: float,
    gross_floor: float,
) -> float:
    """Shared gross-exposure overlay for the book (used by baseline + challenger).

    Maps the chosen ``overlay`` mode to a total gross multiplier:

    * ``"none"``       — 1.0 (naked ranker; the gate's incremental-value control).
    * ``"hmm"``        — :func:`~core.asset_rotation.regime_gross_scale` (binary vol tier).
    * ``"vol_target"`` — :func:`~core.asset_rotation.vol_target_scale` so the book's realized
      vol approaches ``target_vol`` (Barroso-Santa-Clara / Daniel-Moskowitz constant-vol).
    * ``"both"``       — product of the two, capped.

    The book-vol estimate is the equal-weight trailing return of the selected names, sliced
    causally to ``ts`` (or the full series when ``ts`` is None, i.e. the one-shot live path).
    """
    from core.asset_rotation import regime_gross_scale, vol_target_scale

    hmm_g = regime_gross_scale(vol_rank, risk_on_gross, risk_off_gross)
    if overlay == "none":
        return 1.0
    if overlay == "hmm":
        return hmm_g
    vol_g = gross_cap
    if top:
        rets = pd.DataFrame(
            {s: (frames[s]["close"].loc[:ts] if ts is not None else frames[s]["close"]).pct_change()
             for s in top}
        ).dropna(how="all")
        book_ret = rets.tail(vol_window).mean(axis=1).dropna()
        vol_g = vol_target_scale(book_ret.to_numpy(), target_vol, gross_cap, gross_floor)
    if overlay == "vol_target":
        return vol_g
    if overlay == "both":
        return min(gross_cap, hmm_g * vol_g)
    return 1.0


def _resolve_overlay(overlay: "str | None", use_overlay: bool) -> str:
    """Back-compat: map the legacy ``use_overlay`` bool to an overlay mode when unset."""
    if overlay is not None:
        return overlay
    return "hmm" if use_overlay else "none"


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
    overlay: "str | None" = None,
    target_vol: float = 0.12,
    vol_window: int = 126,
    gross_cap: float = 1.0,
    gross_floor: float = 0.0,
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
    from core.portfolio import portfolio_target_weights

    mode = _resolve_overlay(overlay, use_overlay)
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
        gross = _overlay_gross(top, frames, ts, vol_rank, mode, risk_on_gross,
                               risk_off_gross, target_vol, vol_window, gross_cap, gross_floor)
        return portfolio_target_weights(gross, top, max_single, max_concurrent)

    return weight_fn


def make_book_weights_challenger(
    frames: dict[str, pd.DataFrame],
    market_close: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    est_window: int = DEFAULT_EST_WINDOW,
    frac: float = TOP_DECILE,
    max_single: float = 0.15,
    max_concurrent: int = 50,
    overlay: str = "vol_target",
    risk_on_gross: float = 1.0,
    risk_off_gross: float = 0.5,
    target_vol: float = 0.12,
    vol_window: int = 126,
    gross_cap: float = 1.0,
    gross_floor: float = 0.0,
    sector_map: dict[str, str] | None = None,
    max_sector_frac: float = 0.30,
) -> Callable[[pd.Timestamp, float], dict[str, float]]:
    """Build the **challenger** monthly weight function (residual momentum + vol-target).

    Same ``weight_fn(ts, vol_rank) -> {symbol: weight}`` contract as
    :func:`make_book_weights`, but two evidence-backed swaps over the frozen baseline:

    1. **Alpha = residual momentum** (:func:`rank_universe_residual`) instead of raw 12-1 —
       the idiosyncratic-momentum signal (Chaves 2016; Blitz-Huij-Martens 2011) that
       roughly doubles Sharpe and removes crash risk in the literature.
    2. **Risk overlay = volatility targeting** (Daniel-Moskowitz 2016; Barroso-Santa-Clara
       2015) via the existing :func:`~core.asset_rotation.vol_target_scale`, scaling gross
       so the book's realized vol approaches ``target_vol``. ``overlay`` selects how the
       gross is set, so the pre-registered eval can attribute the edge:

       * ``"none"``       — gross pinned to 1.0 (naked residual ranker).
       * ``"hmm"``        — the baseline HMM ``regime_gross_scale`` overlay.
       * ``"vol_target"`` — constant-vol-target scaling (default).
       * ``"both"``       — product of the HMM and vol-target multipliers (capped).

    Gross is set **at each monthly rebalance** (and held through the month, with the
    selection), matching how a monthly book actually re-levers — not a daily churn.
    Strictly causal: the ranker, the proxy, and the book-vol estimate are all sliced
    ``up to`` ``ts``.

    Args:
        frames: ``{symbol: OHLCV}`` full history (sliced causally per bar).
        market_close: Market-proxy close series for the residual regression (e.g. SPY).
        lookback, skip, frac: Momentum window + top-fraction (fixed, un-swept).
        max_single, max_concurrent: Per-name and count caps.
        overlay: Gross-overlay mode (see above).
        risk_on_gross, risk_off_gross: HMM-overlay gross in the low/high-vol tiers.
        target_vol, vol_window, gross_cap, gross_floor: Vol-target overlay knobs.
        sector_map, max_sector_frac: Sector-cap selection (as in :func:`make_book_weights`).

    Returns:
        A stateful ``weight_fn(ts, vol_rank)`` closure (monthly-memoized selection + gross).
    """
    from core.portfolio import portfolio_target_weights

    cache: dict[tuple[int, int], tuple[list[str], float]] = {}

    def weight_fn(ts: pd.Timestamp, vol_rank: float) -> dict[str, float]:
        key = (ts.year, ts.month)
        memo = cache.get(key)
        if memo is None:
            sliced = {s: df.loc[:ts] for s, df in frames.items()}
            ranked = rank_universe_residual(sliced, market_close.loc[:ts], lookback, skip,
                                            est_window)
            top = (select_top_sector_capped(ranked, sector_map, frac, max_sector_frac,
                                            max_n=max_concurrent)
                   if sector_map else select_top(ranked, frac))
            gross = _overlay_gross(top, frames, ts, vol_rank, overlay, risk_on_gross,
                                   risk_off_gross, target_vol, vol_window, gross_cap, gross_floor)
            cache[key] = (top, gross)
        else:
            top, gross = memo
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
    overlay: "str | None" = None,
    target_vol: float = 0.12,
    vol_window: int = 126,
    gross_cap: float = 1.0,
    gross_floor: float = 0.0,
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
    from core.portfolio import portfolio_target_weights

    ranked = rank_universe(frames, lookback, skip)
    top = (select_top_sector_capped(ranked, sector_map, frac, max_sector_frac,
                                    max_n=max_concurrent)
           if sector_map else select_top(ranked, frac))
    mode = _resolve_overlay(overlay, use_overlay)
    gross = _overlay_gross(top, frames, None, vol_rank, mode, risk_on_gross,
                           risk_off_gross, target_vol, vol_window, gross_cap, gross_floor)
    return portfolio_target_weights(gross, top, max_single, max_concurrent)


def compute_book_targets_challenger(
    frames: dict[str, pd.DataFrame],
    market_close: pd.Series,
    vol_rank: float,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    est_window: int = DEFAULT_EST_WINDOW,
    frac: float = TOP_DECILE,
    max_single: float = 0.15,
    max_concurrent: int = 50,
    overlay: str = "vol_target",
    risk_on_gross: float = 1.0,
    risk_off_gross: float = 0.5,
    target_vol: float = 0.12,
    vol_window: int = 126,
    gross_cap: float = 1.0,
    gross_floor: float = 0.0,
    sector_map: dict[str, str] | None = None,
    max_sector_frac: float = 0.30,
) -> dict[str, float]:
    """One-shot **challenger** target weights for the live/paper rebalance.

    The live counterpart of :func:`make_book_weights_challenger` (as
    :func:`compute_book_targets` is to :func:`make_book_weights`): rank the universe by
    residual momentum against ``market_close``, take the top decile (sector-capped), set
    gross by the chosen ``overlay`` mode, and equal-weight under the caps. Pure (no
    network) so the rebalance decision is unit-testable; ``main.run_rebalance --challenger``
    wraps it with live data + (gated) submission.

    Args:
        frames: ``{symbol: OHLCV}`` ending at the latest closed bar.
        market_close: Market-proxy close series for the residual regression.
        vol_rank: Current market regime volatility rank in ``[0, 1]`` (HMM on the proxy).
        overlay: Gross-overlay mode (``none`` | ``hmm`` | ``vol_target`` | ``both``).
        Other args: as in :func:`make_book_weights_challenger`.

    Returns:
        ``{symbol: target_weight}`` for the selected top-decile names.
    """
    from core.portfolio import portfolio_target_weights

    ranked = rank_universe_residual(frames, market_close, lookback, skip, est_window)
    top = (select_top_sector_capped(ranked, sector_map, frac, max_sector_frac,
                                    max_n=max_concurrent)
           if sector_map else select_top(ranked, frac))
    gross = _overlay_gross(top, frames, None, vol_rank, overlay, risk_on_gross,
                           risk_off_gross, target_vol, vol_window, gross_cap, gross_floor)
    return portfolio_target_weights(gross, top, max_single, max_concurrent)


def book_targets_fixed_selection(
    frames: dict[str, pd.DataFrame],
    selected: list[str],
    vol_rank: float,
    overlay: str = "vol_target",
    risk_on_gross: float = 1.0,
    risk_off_gross: float = 0.5,
    target_vol: float = 0.12,
    vol_window: int = 126,
    gross_cap: float = 1.0,
    gross_floor: float = 0.0,
    max_single: float = 0.15,
    max_concurrent: int = 50,
) -> dict[str, float]:
    """Re-scale a FIXED book selection to today's overlay gross (intra-month daily run).

    The cross-sectional **selection** is a slow monthly signal (momentum 12-1) — re-ranking
    it daily would churn the book and pay slippage for noise. But the **risk overlay** (the
    constant-vol target) reads the market every day: this keeps the month's selected names
    and only re-scales total gross to today's realized vol — de-risking when vol spikes,
    re-risking when it subsides. That is what "attentive every day" should mean for a
    monthly-rebalanced book: daily risk management, monthly name turnover. Pure + causal.

    Args:
        frames: ``{symbol: OHLCV}`` ending at the latest closed bar (≥ the selected names).
        selected: The month's selected book names (from the last monthly re-rank).
        vol_rank: Current market regime volatility rank in ``[0, 1]`` (HMM on the proxy).
        overlay: Gross-overlay mode (``none`` | ``hmm`` | ``vol_target`` | ``both``).
        Other args: as in :func:`compute_book_targets`.

    Returns:
        ``{symbol: target_weight}`` for the (still-priced) selected names, re-scaled to the
        overlay gross. Names no longer in ``frames`` are dropped.
    """
    from core.portfolio import portfolio_target_weights

    present = [s for s in selected if s in frames]
    gross = _overlay_gross(present, frames, None, vol_rank, overlay, risk_on_gross,
                           risk_off_gross, target_vol, vol_window, gross_cap, gross_floor)
    return portfolio_target_weights(gross, present, max_single, max_concurrent)


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


def drop_open_order_symbols(orders: list[dict], open_symbols: set[str]) -> list[dict]:
    """Drop any planned order whose symbol already has an open (unfilled) order.

    The idempotency guard for the monthly execute path. The rebalance diff is computed
    from *held* positions; a still-pending order from a prior run is not yet a position,
    so the diff would re-issue it and **double-submit**. Skipping every symbol with a live
    order (regardless of side) makes a re-run within the fill gap a no-op for those names —
    they settle on the next run once filled. Required before un-gating monthly auto-execute.

    Args:
        orders: Planned orders from :func:`plan_rebalance_orders`.
        open_symbols: Symbols that currently have an open order on the account.

    Returns:
        The orders whose symbol has no pending order (input order preserved).
    """
    return [o for o in orders if o["symbol"] not in open_symbols]
