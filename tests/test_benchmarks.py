"""Tests for the honest rotation benchmarks (60/40 + risk-parity).

The benchmarks share the rotation's per-bar cost engine; these tests pin the
matched cost/cash treatment and the inverse-vol weighting that make the
comparison fair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.benchmarks import (
    risk_parity_returns,
    simulate_portfolio,
    static_mix_returns,
)


def _series(values) -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"close": values}, index=idx)


def _const_growth(n: int, r: float, start: float = 100.0) -> pd.DataFrame:
    closes = [start * (1.0 + r) ** i for i in range(n)]
    return _series(closes)


# ------------------------------------------------------------------- cash term ---
def test_all_cash_grows_at_rf():
    """Empty target -> 100% cash -> equity compounds at rf when credited."""
    frames = {"SPY": _const_growth(50, 0.001)}
    idx = frames["SPY"].index
    eq = static_mix_returns(frames, {}, idx, slippage_pct=0.0, rf_daily=0.0002)
    # cash_w=1 from bar 0 (no prior risky weight) -> all 50 bars credit rf
    assert eq.iloc[-1] == pytest.approx(100000.0 * (1.0002 ** 50), rel=1e-9)


def test_cash_not_credited_when_rf_zero():
    frames = {"SPY": _const_growth(30, 0.0)}
    idx = frames["SPY"].index
    eq = static_mix_returns(frames, {}, idx, slippage_pct=0.0, rf_daily=0.0)
    assert eq.iloc[-1] == pytest.approx(100000.0, rel=1e-12)


# ----------------------------------------------------------------- static mix ---
def test_static_6040_tracks_weighted_return():
    """60/40 on two assets earns the weighted blend (no slippage, no cash)."""
    spy = _const_growth(40, 0.002)
    tlt = _const_growth(40, 0.0005)
    frames = {"SPY": spy, "TLT": tlt}
    idx = spy.index
    eq = static_mix_returns(
        frames, {"SPY": 0.6, "TLT": 0.4}, idx, slippage_pct=0.0, rf_daily=0.0
    )
    # prior weights earn each bar; constant-growth -> blended per-bar return
    blended = 0.6 * 0.002 + 0.4 * 0.0005
    assert eq.iloc[-1] == pytest.approx(100000.0 * (1.0 + blended) ** 39, rel=1e-9)


def test_static_is_deterministic():
    frames = {"SPY": _const_growth(40, 0.001), "TLT": _const_growth(40, 0.0005)}
    idx = frames["SPY"].index
    tgt = {"SPY": 0.6, "TLT": 0.4}
    a = static_mix_returns(frames, tgt, idx, 0.0005, 0.0001)
    b = static_mix_returns(frames, tgt, idx, 0.0005, 0.0001)
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------- risk parity ---
def test_risk_parity_overweights_low_vol_asset():
    """Inverse-vol: the calmer asset gets the larger weight."""
    rng = np.random.default_rng(1)
    n = 200
    calm = _series(100.0 * np.cumprod(1 + rng.normal(0, 0.002, n)))
    wild = _series(100.0 * np.cumprod(1 + rng.normal(0, 0.02, n)))
    frames = {"CALM": calm, "WILD": wild}
    idx = calm.index
    _, w = risk_parity_returns(
        frames, idx, slippage_pct=0.0, rf_daily=0.0, lookback=60, return_weights=True
    )
    tail = w.iloc[-1]
    assert tail["CALM"] > tail["WILD"]
    assert tail["CALM"] + tail["WILD"] == pytest.approx(1.0, abs=1e-9)


def test_risk_parity_weights_sum_to_one():
    frames = {"A": _const_growth(120, 0.001), "B": _const_growth(120, 0.0008)}
    idx = frames["A"].index
    _, w = risk_parity_returns(frames, idx, 0.0, 0.0, return_weights=True)
    sums = w.sum(axis=1)
    assert (abs(sums - 1.0) < 1e-9).all()


# ---------------------------------------------------------------- cost engine ---
def test_turnover_charged_slippage():
    """Building the position from cash costs slippage on the first bar's turnover."""
    frames = {"SPY": _const_growth(10, 0.0)}  # flat price, isolate slippage
    idx = frames["SPY"].index
    eq = static_mix_returns(frames, {"SPY": 1.0}, idx, slippage_pct=0.01, rf_daily=0.0)
    # bar 0: prev_w 0 -> target 1.0, turnover 1.0 -> 1% haircut, then held (no more turnover)
    assert eq.iloc[-1] == pytest.approx(100000.0 * 0.99, rel=1e-9)
