"""Tests for the OOS validation harness slice logic.

The harness reuses the (already leakage-tested) walk-forward backtester; the only
new bug surface is slicing a full-history result into sub-windows with
**base-independent** metrics. These tests pin that math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult
from backtest.oos_validation import slice_metrics


def _make_result(n: int = 600, start: str = "2018-01-01") -> tuple[BacktestResult, pd.Series]:
    """Synthetic full-history result: known returns, derived equity + close."""
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    strat_ret = pd.Series(rng.normal(0.0003, 0.01, n), index=idx, name="return")
    asset_ret = pd.Series(rng.normal(0.0004, 0.012, n), index=idx, name="asset_return")
    equity = pd.Series(100000.0 * (1.0 + strat_ret).cumprod(), index=idx, name="equity")
    close = pd.Series(50.0 * (1.0 + asset_ret).cumprod(), index=idx, name="close")
    hist = pd.DataFrame(
        {"risk_state": ["normal"] * n, "port_return": strat_ret.values}, index=idx
    )
    res = BacktestResult(
        equity_curve=equity, returns=strat_ret, asset_returns=asset_ret,
        regime_history=hist, symbol="TEST", initial_capital=100000.0,
    )
    return res, close


def test_slice_total_return_is_base_independent():
    """strat_return over a sub-window = compounded slice returns, NOT eq/initial_capital."""
    res, close = _make_result()
    # mid window, well clear of the initial-capital base
    sm = slice_metrics(res, close, "mid", "2018-06-01", "2018-12-31")
    assert sm is not None
    mask = (res.returns.index >= pd.Timestamp("2018-06-01")) & (
        res.returns.index <= pd.Timestamp("2018-12-31")
    )
    expected = float((1.0 + res.returns[mask]).prod() - 1.0)
    assert abs(sm.strat_return - expected) < 1e-12
    # base-independent: must differ from the from-inception equity ratio over the slice
    from_incept = float(res.equity_curve[mask].iloc[-1] / res.initial_capital - 1.0)
    assert abs(sm.strat_return - from_incept) > 1e-6


def test_buy_hold_slice_compounds_asset_returns():
    res, close = _make_result()
    sm = slice_metrics(res, close, "mid", "2018-06-01", "2018-12-31")
    mask = (res.asset_returns.index >= pd.Timestamp("2018-06-01")) & (
        res.asset_returns.index <= pd.Timestamp("2018-12-31")
    )
    expected = float((1.0 + res.asset_returns[mask]).prod() - 1.0)
    assert abs(sm.buy_hold_return - expected) < 1e-12


def test_window_outside_span_returns_none():
    res, close = _make_result(n=300, start="2020-01-01")
    # crisis entirely before the data → < 2 bars in window
    assert slice_metrics(res, close, "gfc", "2008-01-01", "2009-01-01") is None


def test_full_window_uses_whole_span():
    res, close = _make_result(n=400)
    sm = slice_metrics(res, close, "full")
    assert sm.n_bars == 400
    expected = float((1.0 + res.returns).prod() - 1.0)
    assert abs(sm.strat_return - expected) < 1e-12


def test_pct_halted_counts_halted_bars():
    res, close = _make_result(n=200)
    res.regime_history["risk_state"] = ["halted"] * 50 + ["normal"] * 150
    sm = slice_metrics(res, close, "full")
    assert abs(sm.pct_halted - 0.25) < 1e-9
