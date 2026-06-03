"""Honest benchmarks for the cross-asset rotation (vía B).

A regime rotation that beats *buy & hold on one index* but loses to a static
60/40 or risk-parity book has no edge — it just rebrands diversification. So the
rotation must clear those bars. Critically, these benchmarks run through the
**same per-bar cost engine** as ``Backtester.run_rotation``: identical slippage on
turnover and identical risk-free credit on idle cash. A frictionless benchmark
would rig the comparison in the rotation's favour (the rotation rebalances more),
which is exactly the cost/cash confound that flipped a prior validation 0/5->1/5
(see docs/analysis/2026-06-03-reentry-validation.md).
"""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd


def simulate_portfolio(
    frames: dict[str, pd.DataFrame],
    idx: pd.Index,
    weight_fn: Callable[[int, list[float]], dict[str, float]],
    slippage_pct: float,
    rf_daily: float,
    initial_capital: float = 100000.0,
    return_weights: bool = False,
):
    """Per-bar weight-driven portfolio simulator (shared cost engine).

    Mirrors the realization/cost mechanics of ``Backtester.run_rotation`` exactly:
    prior weights earn this bar's return, idle cash ``1 - sum(weights)`` earns
    ``rf_daily``, and turnover is charged ``slippage_pct``. The only thing that
    varies between strategies is ``weight_fn``.

    Args:
        frames: ``{symbol: OHLCV}``.
        idx: Bars to simulate over (aligns benchmark to the rotation's OOS index).
        weight_fn: ``(t, port_return_history) -> {symbol: target_weight}`` for the
            NEXT bar. Missing symbols default to 0; the unallocated remainder is cash.
        slippage_pct: Slippage rate per unit turnover.
        rf_daily: Daily risk-free rate credited on idle cash (0 to disable).
        initial_capital: Starting equity.
        return_weights: If True, also return the per-bar weight DataFrame.

    Returns:
        Equity ``Series`` (or ``(equity, weights_df)`` if requested).
    """
    symbols = list(frames)
    rets = {
        s: frames[s]["close"].pct_change().reindex(idx).fillna(0.0).to_numpy()
        for s in symbols
    }
    equity = initial_capital
    eq_val: list[float] = []
    weight_rows: list[dict] = []
    port_hist: list[float] = []
    prev_w = {s: 0.0 for s in symbols}

    for t in range(len(idx)):
        risky_ret = sum(prev_w[s] * rets[s][t] for s in symbols)
        cash_w = 1.0 - sum(prev_w.values())
        port_ret = risky_ret + cash_w * rf_daily
        equity *= (1.0 + port_ret)
        port_hist.append(port_ret)

        raw = weight_fn(t, port_hist)
        w = {s: float(raw.get(s, 0.0)) for s in symbols}
        turnover = sum(abs(w[s] - prev_w[s]) for s in symbols)
        equity *= (1.0 - turnover * slippage_pct)
        prev_w = w
        eq_val.append(equity)
        weight_rows.append(dict(prev_w))

    eq = pd.Series(eq_val, index=idx, name="equity")
    if return_weights:
        return eq, pd.DataFrame(weight_rows, index=idx)
    return eq


def static_mix_returns(
    frames: dict[str, pd.DataFrame],
    target: dict[str, float],
    idx: pd.Index,
    slippage_pct: float,
    rf_daily: float,
    initial_capital: float = 100000.0,
    return_weights: bool = False,
):
    """Static target-weight book rebalanced every bar (e.g. 60/40 SPY/TLT).

    Args:
        frames: ``{symbol: OHLCV}`` (must include every key in ``target``).
        target: Fixed target weights, e.g. ``{"SPY": 0.6, "TLT": 0.4}``.
        idx: Bars to simulate over.
        slippage_pct: Slippage per unit turnover (matched to the rotation).
        rf_daily: Risk-free credit on idle cash (matched to the rotation).
        initial_capital: Starting equity.
        return_weights: If True, also return weights.

    Returns:
        Equity ``Series`` (or ``(equity, weights_df)``).
    """
    sub = {s: frames[s] for s in target}
    return simulate_portfolio(
        sub, idx, lambda t, h: target, slippage_pct, rf_daily,
        initial_capital, return_weights,
    )


def risk_parity_returns(
    frames: dict[str, pd.DataFrame],
    idx: pd.Index,
    slippage_pct: float,
    rf_daily: float,
    lookback: int = 60,
    initial_capital: float = 100000.0,
    return_weights: bool = False,
):
    """Inverse-volatility (risk-parity) book, gross 1.0, rebalanced every bar.

    Each bar, weights are proportional to ``1 / trailing_vol`` of each symbol over
    the trailing ``lookback`` window, normalized to sum to 1. Before enough history
    accrues, falls back to equal weight.

    Args:
        frames: ``{symbol: OHLCV}``.
        idx: Bars to simulate over.
        slippage_pct: Slippage per unit turnover (matched to the rotation).
        rf_daily: Risk-free credit on idle cash (matched to the rotation).
        lookback: Trailing window for the volatility estimate.
        initial_capital: Starting equity.
        return_weights: If True, also return weights.

    Returns:
        Equity ``Series`` (or ``(equity, weights_df)``).
    """
    symbols = list(frames)
    sym_rets = {
        s: frames[s]["close"].pct_change().reindex(idx).fillna(0.0).to_numpy()
        for s in symbols
    }

    def weight_fn(t: int, _hist: list[float]) -> dict[str, float]:
        lo = max(0, t - lookback)
        inv = {}
        for s in symbols:
            window = sym_rets[s][lo : t + 1]
            if len(window) < 2:
                inv[s] = 1.0
                continue
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
            sd = math.sqrt(var)
            inv[s] = (1.0 / sd) if sd > 0 else 0.0
        total = sum(inv.values())
        if total <= 0:
            return {s: 1.0 / len(symbols) for s in symbols}  # equal-weight fallback
        return {s: inv[s] / total for s in symbols}

    return simulate_portfolio(
        frames, idx, weight_fn, slippage_pct, rf_daily, initial_capital, return_weights,
    )
