"""Tests for the vol-aware slippage estimator (T5.1).

Estimates the backtester's slippage_vol_coeff from realized paper fills: regress
realized slippage on the bar's ATR% through the origin, so the backtest charges
more slippage in turbulent regimes. Affects RESEARCH backtests only — gates are
judged on real fills, and gate-evidence backtests are never re-run. Pure + tested.
"""

from __future__ import annotations

import pytest

from core import slippage as sl


def test_realized_slippage_buy_adverse_positive():
    # bought above the decision price -> adverse slippage, positive bps
    bps = sl.realized_slippage_bps(decision_price=100.0, fill_price=100.5, side="buy")
    assert bps == pytest.approx(50.0)                  # 0.5% = 50 bps


def test_realized_slippage_sell_adverse_positive():
    # sold below the decision price -> adverse, positive bps
    bps = sl.realized_slippage_bps(decision_price=100.0, fill_price=99.5, side="sell")
    assert bps == pytest.approx(50.0)


def test_realized_slippage_favorable_negative():
    bps = sl.realized_slippage_bps(decision_price=100.0, fill_price=99.5, side="buy")
    assert bps == pytest.approx(-50.0)                 # bought cheaper than decision


def test_estimate_coeff_through_origin():
    # slippage(frac) = coeff * atr_pct  ->  coeff recoverable
    samples = [{"slippage_bps": 20.0, "atr_pct": 0.02},   # 0.002 = coeff*0.02 -> 0.1
               {"slippage_bps": 40.0, "atr_pct": 0.04},
               {"slippage_bps": 10.0, "atr_pct": 0.01}]
    coeff = sl.estimate_vol_coeff(samples, min_samples=3)
    assert coeff == pytest.approx(0.1, rel=1e-6)


def test_estimate_coeff_insufficient_samples_returns_none():
    assert sl.estimate_vol_coeff([{"slippage_bps": 20.0, "atr_pct": 0.02}],
                                 min_samples=30) is None


def test_estimate_coeff_ignores_nonpositive_atr():
    samples = [{"slippage_bps": 20.0, "atr_pct": 0.0},
               {"slippage_bps": 20.0, "atr_pct": 0.02}]
    coeff = sl.estimate_vol_coeff(samples, min_samples=1)
    assert coeff == pytest.approx(0.1, rel=1e-6)        # the atr=0 row is dropped
