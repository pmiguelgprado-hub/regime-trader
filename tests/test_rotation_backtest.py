"""Tests for the cross-asset rotation backtest (Backtester.run_rotation).

Deterministic / network-free: the synthetic regime-switching OHLCV drives the
proxy regime; the basket tickers reuse the same series (structure, not realism).
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from backtest.backtester import Backtester, BacktestConfig
from core.asset_rotation import RotationConfig
from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import StrategyConfig
from core.risk_manager import RiskConfig, RiskManager
from data.feature_engineering import FeatureEngineer

for _n in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)


def _make_backtester(credit_cash: bool = True) -> Backtester:
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    return Backtester(
        BacktestConfig(step_size=126, credit_cash_rf=credit_cash),
        hmm,
        StrategyConfig(),
        RiskManager(RiskConfig()),
        FeatureEngineer(),
    )


@pytest.fixture(scope="module")
def _ohlcv():
    from conftest import make_synthetic_ohlcv

    return make_synthetic_ohlcv()


@pytest.fixture(scope="module")
def _frames(_ohlcv):
    return {s: _ohlcv for s in ("SPY", "QQQ", "TLT", "GLD")}


def test_run_exposes_vol_rank(_ohlcv):
    """run() now records the clean volatility tier per bar."""
    res = _make_backtester().run({"SPY": _ohlcv})
    assert "vol_rank" in res.regime_history.columns
    vr = res.regime_history["vol_rank"]
    assert ((vr >= 0.0) & (vr <= 1.0)).all()


def test_run_rotation_produces_aligned_equity(_frames):
    eq = _make_backtester().run_rotation(_frames, RotationConfig())
    assert isinstance(eq, pd.Series)
    assert len(eq) > 0
    assert eq.notna().all()
    assert (eq > 0).all()


def test_run_rotation_is_deterministic(_frames):
    a = _make_backtester().run_rotation(_frames, RotationConfig())
    b = _make_backtester().run_rotation(_frames, RotationConfig())
    pd.testing.assert_series_equal(a, b)


def test_risk_off_bars_hold_no_equities(_frames):
    """When the proxy tier is risk-off (vol_rank>=0.67), equity weights are 0."""
    bt = _make_backtester()
    eq, w = bt.run_rotation(_frames, RotationConfig(), return_weights=True)
    base = bt.run({"SPY": _frames["SPY"]})
    vr = base.regime_history["vol_rank"].reindex(w.index)
    risk_off = vr >= 0.67
    if risk_off.any():
        assert (w.loc[risk_off, "SPY"].abs() < 1e-12).all()
        assert (w.loc[risk_off, "QQQ"].abs() < 1e-12).all()


def test_risk_on_bars_hold_no_defensive(_frames):
    """When the proxy tier is risk-on (vol_rank<=0.33), defensive weights are 0."""
    bt = _make_backtester()
    _, w = bt.run_rotation(_frames, RotationConfig(), return_weights=True)
    base = bt.run({"SPY": _frames["SPY"]})
    vr = base.regime_history["vol_rank"].reindex(w.index)
    risk_on = vr <= 0.33
    if risk_on.any():
        assert (w.loc[risk_on, "TLT"].abs() < 1e-12).all()
        assert (w.loc[risk_on, "GLD"].abs() < 1e-12).all()


def test_gross_never_exceeds_cap(_frames):
    """No leverage: summed risky weights never exceed the gross cap (1.0)."""
    _, w = _make_backtester().run_rotation(_frames, RotationConfig(), return_weights=True)
    gross = w.sum(axis=1)
    assert (gross <= 1.0 + 1e-9).all()


def test_vr_transform_none_is_identity(_frames):
    """No transform == passing an identity transform (control-test plumbing)."""
    bt = _make_backtester()
    a = bt.run_rotation(_frames, RotationConfig())
    b = bt.run_rotation(_frames, RotationConfig(), vr_transform=lambda x: x)
    pd.testing.assert_series_equal(a, b)


def test_vr_transform_can_force_risk_off(_frames):
    """A transform pinning vol_rank high forces the risk-off (no-equity) tier."""
    _, w = _make_backtester().run_rotation(
        _frames, RotationConfig(), return_weights=True, vr_transform=lambda x: 1.0
    )
    assert (w["SPY"].abs() < 1e-12).all()
    assert (w["QQQ"].abs() < 1e-12).all()


def test_missing_symbol_frame_raises(_ohlcv):
    frames = {"SPY": _ohlcv, "QQQ": _ohlcv, "TLT": _ohlcv}  # GLD missing
    with pytest.raises(ValueError, match="GLD"):
        _make_backtester().run_rotation(frames, RotationConfig())
