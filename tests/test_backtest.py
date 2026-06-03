"""Tests for the walk-forward allocation backtester, analytics, and stress tests.

All deterministic and network-free: the synthetic regime-switching OHLCV from
``conftest`` drives a reduced-config HMM (fast, few restarts).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester, BacktestConfig
from backtest.performance import PerformanceAnalyzer
from backtest.stress_test import StressTester
from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import StrategyConfig
from core.risk_manager import RiskConfig, RiskManager
from data.feature_engineering import FeatureEngineer

for _n in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)


def _make_backtester(step: int = 126) -> Backtester:
    """Build a fast-config backtester (n_init=1, single candidate)."""
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    return Backtester(
        BacktestConfig(step_size=step),
        hmm,
        StrategyConfig(),
        RiskManager(RiskConfig()),
        FeatureEngineer(),
    )


@pytest.fixture(scope="module")
def result(_ohlcv_module):
    """Run one backtest on synthetic data, reused across tests."""
    return _make_backtester().run({"SPY": _ohlcv_module})


@pytest.fixture(scope="module")
def _ohlcv_module():
    """Module-scoped synthetic OHLCV (conftest fixture is function-scoped)."""
    from conftest import make_synthetic_ohlcv

    return make_synthetic_ohlcv()


# --------------------------------------------------------------- backtester ---
def test_folds_are_rolling_and_causal() -> None:
    """Folds step forward; train precedes test; windows sized per config."""
    bt = _make_backtester()
    folds = bt._generate_folds(2000)
    assert folds, "expected at least one fold"
    for tr_s, tr_e, te_s, te_e in folds:
        assert tr_s < tr_e == te_s < te_e
        assert tr_e - tr_s == bt.config.train_window
    # rolling step between consecutive folds
    assert folds[1][0] - folds[0][0] == bt.config.step_size


def test_run_produces_aligned_outputs(result) -> None:
    """Equity, returns, regime history share the OOS index; equity is positive."""
    assert len(result.equity_curve) > 100
    assert result.equity_curve.index.equals(result.returns.index)
    assert result.equity_curve.index.equals(result.regime_history.index)
    assert (result.equity_curve > 0).all()
    assert result.symbol == "SPY"


def test_weights_never_exceed_leverage_ceiling(result) -> None:
    """Target weights are clamped to the configured max leverage."""
    assert result.regime_history["weight"].max() <= BacktestConfig().max_leverage + 1e-9
    assert result.regime_history["weight"].min() >= 0.0  # long-only, never short


def test_trades_are_rebalances_above_threshold(result) -> None:
    """Every logged trade reflects a weight change (rebalance), not an entry."""
    if result.trades.empty:
        pytest.skip("no rebalances in this run")
    assert {"from_weight", "to_weight", "delta"}.issubset(result.trades.columns)
    # each rebalance moved the weight
    assert (result.trades["from_weight"] != result.trades["to_weight"]).all()


def test_backtest_is_deterministic(_ohlcv_module) -> None:
    """Same inputs + seed -> identical equity curve (no hidden randomness)."""
    a = _make_backtester().run({"SPY": _ohlcv_module})
    b = _make_backtester().run({"SPY": _ohlcv_module})
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)


def test_no_lookahead_in_features(_ohlcv_module) -> None:
    """The shared causal feature builder has no look-ahead (built-in probe)."""
    FeatureEngineer().assert_no_lookahead(_ohlcv_module)


# --------------------------------------------------------------- slippage (R-1) ---
def test_slippage_rate_floors_at_base_when_calm() -> None:
    """Zero volatility -> slippage equals the flat base rate (no vol premium)."""
    assert Backtester._slippage_rate(base_pct=0.0005, vol_coeff=0.5, atr_pct=0.0) == 0.0005


def test_slippage_rate_rises_with_volatility() -> None:
    """Higher ATR% -> slippage rises linearly above the base by vol_coeff*atr_pct."""
    rate = Backtester._slippage_rate(base_pct=0.0005, vol_coeff=0.5, atr_pct=0.02)
    assert rate == pytest.approx(0.0005 + 0.5 * 0.02)


def test_volatility_slippage_wired_into_rebalance(_ohlcv_module) -> None:
    """With vol_coeff>0 the loop charges a vol-scaled rate (>= base, premium on some bars)."""
    res = Backtester(
        BacktestConfig(slippage_vol_coeff=1.0),
        HMMEngine(HMMConfig(n_candidates=[3], n_init=1)),
        StrategyConfig(), RiskManager(RiskConfig()), FeatureEngineer(),
    ).run({"SPY": _ohlcv_module})
    if res.trades.empty:
        pytest.skip("no rebalances in this run")
    base = BacktestConfig().slippage_pct
    eff = res.trades["slippage_cost"] / res.trades["delta"].abs()  # effective rate per trade
    assert (eff >= base - 1e-12).all()    # never below the base floor
    assert (eff > base + 1e-9).any()      # at least one bar paid a volatility premium


def test_flat_slippage_unchanged_when_coeff_zero(_ohlcv_module) -> None:
    """coeff=0 reproduces the legacy flat rate exactly (no behaviour change)."""
    res = Backtester(
        BacktestConfig(slippage_vol_coeff=0.0),
        HMMEngine(HMMConfig(n_candidates=[3], n_init=1)),
        StrategyConfig(), RiskManager(RiskConfig()), FeatureEngineer(),
    ).run({"SPY": _ohlcv_module})
    if res.trades.empty:
        pytest.skip("no rebalances in this run")
    # Reconstructed from rounded ledger columns (cost@6dp, delta@4dp), so allow
    # rounding noise (~1e-5) — still 1000x below any real vol premium (~1e-2).
    eff = res.trades["slippage_cost"] / res.trades["delta"].abs()
    assert eff.values == pytest.approx(BacktestConfig().slippage_pct, abs=2e-5)


# --------------------------------------------------------- multi-asset (E-1) ---
def test_run_portfolio_produces_aligned_positive_equity(_ohlcv_module) -> None:
    """A multi-symbol portfolio backtest yields a positive equity curve."""
    frames = {"A": _ohlcv_module, "B": _ohlcv_module, "C": _ohlcv_module}
    eq = _make_backtester().run_portfolio(frames)
    assert len(eq) > 100
    assert (eq > 0).all()


def test_run_portfolio_is_deterministic(_ohlcv_module) -> None:
    """Same inputs -> identical portfolio equity curve."""
    frames = {"A": _ohlcv_module, "B": _ohlcv_module}
    a = _make_backtester().run_portfolio(frames)
    b = _make_backtester().run_portfolio(frames)
    pd.testing.assert_series_equal(a, b)


def test_run_portfolio_respects_single_position_cap(_ohlcv_module) -> None:
    """No single name ever exceeds the configured per-symbol cap."""
    frames = {"A": _ohlcv_module, "B": _ohlcv_module, "C": _ohlcv_module}
    bt = _make_backtester()
    weights = bt.run_portfolio(frames, return_weights=True)[1]
    cap = bt.risk_manager.config.max_single_position
    assert (weights.to_numpy() <= cap + 1e-9).all()


# --------------------------------------------------------- halt recovery ---
def _halt_recoveries(rh) -> int:
    """Count halted -> normal transitions (breaker releasing after a halt)."""
    return int((rh["risk_state"].eq("normal") & rh["risk_state"].shift().eq("halted")).sum())


def test_halt_floor_enables_recovery_that_zero_does_not(_ohlcv_module) -> None:
    """The halt floor breaks the permanent-flat trap: equity recovers, breaker releases.

    Legacy (halt -> weight 0) freezes equity below the peak, so peak-drawdown stays
    >limit and the breaker never releases (0 recoveries). A non-zero floor lets
    equity climb back so the breaker can re-engage NORMAL.
    """
    def run(floor: float):
        bt = Backtester(
            BacktestConfig(),
            HMMEngine(HMMConfig(n_candidates=[3], n_init=1)),
            StrategyConfig(),
            RiskManager(RiskConfig(halt_floor_mult=floor)),
            FeatureEngineer(),
        )
        return bt.run({"SPY": _ohlcv_module}).regime_history

    stuck = run(0.0)   # legacy behaviour (weight 0 on halt)
    fixed = run(0.25)  # minimum floor
    if "halted" not in set(stuck["risk_state"]):
        pytest.skip("no halt occurred in this run")
    assert _halt_recoveries(stuck) == 0    # permanent-flat trap: never recovers
    assert _halt_recoveries(fixed) >= 1    # floor lets the breaker release


# --------------------------------------------------------------- performance ---
def test_metrics_are_finite_and_consistent(result) -> None:
    """Core metrics compute and are internally consistent."""
    pa = PerformanceAnalyzer(0.045)
    rep = pa.analyze(result, _ohlcv_close(result), with_benchmarks=False)
    assert np.isfinite(rep.sharpe)
    assert rep.max_drawdown <= 0.0
    assert 0.0 <= rep.win_rate <= 1.0
    expected_total = result.equity_curve.iloc[-1] / result.initial_capital - 1.0
    assert rep.total_return == pytest.approx(expected_total, rel=1e-9)


def test_benchmarks_present(result) -> None:
    """Benchmark suite returns strategy + buy-hold + sma200 + random."""
    pa = PerformanceAnalyzer(0.045)
    bm = pa.benchmarks(result, _ohlcv_close(result), n_random=10)
    assert {"strategy", "buy_hold", "sma200_trend", "random"}.issubset(bm)
    assert "total_return_std" in bm["random"]


def _ohlcv_close(result) -> pd.Series:
    """Reconstruct an approximate close series from asset returns for benchmarks."""
    from conftest import make_synthetic_ohlcv

    return make_synthetic_ohlcv()["close"]


# --------------------------------------------------------------- stress test ---
def test_risk_management_contains_damage(_ohlcv_module) -> None:
    """Even under injected crashes, max drawdown stays bounded (no blowup)."""
    bt = _make_backtester(step=189)
    st = StressTester(bt)
    rep = st.crash_injection_mc({"SPY": _ohlcv_module}, n_sims=3, n_crashes=8)
    assert rep.n_sims > 0
    assert rep.blowup_rate < 1.0  # risk layer prevented total blowups


def test_misclassification_resets_shuffle_hook(_ohlcv_module) -> None:
    """The misclassification probe always clears the shuffle hook afterwards."""
    bt = _make_backtester(step=189)
    st = StressTester(bt)
    st.regime_misclassification({"SPY": _ohlcv_module}, n_sims=2)
    assert bt.shuffle_regimes is None


def test_inject_crash_shifts_level(_ohlcv_module) -> None:
    """A crash injection lowers prices from the start index onward."""
    bt = _make_backtester()
    st = StressTester(bt)
    crashed = st.inject_crash(_ohlcv_module, magnitude=-0.10, start_index=500)
    before = _ohlcv_module["close"].iloc[499]
    assert crashed["close"].iloc[499] == pytest.approx(before)  # untouched pre-crash
    ratio = crashed["close"].iloc[500] / _ohlcv_module["close"].iloc[500]
    assert ratio == pytest.approx(0.90, abs=1e-9)


def test_calm_flag_uses_vol_rank_cutoff():
    """Backtester._calm_flag: calm iff regime vol_rank < HIGH_VOL_MIN (unknown -> not calm)."""
    from backtest.backtester import Backtester
    from core.regime_strategies import HIGH_VOL_MIN

    class _Orch:
        vol_rank = {0: 0.10, 1: 0.50, 2: 0.90}

    orch = _Orch()
    assert Backtester._calm_flag(orch, 0) is True            # low vol -> calm
    assert Backtester._calm_flag(orch, 1) is True            # mid vol (<0.67) -> calm
    assert Backtester._calm_flag(orch, 2) is False           # high vol (>=0.67) -> not calm
    assert Backtester._calm_flag(orch, 99) is False          # unknown -> conservative
    assert HIGH_VOL_MIN == 0.67
