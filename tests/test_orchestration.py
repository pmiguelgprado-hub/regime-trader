"""Tests for the TradingSystem orchestration core (no network, no broker)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from core.risk_manager import RiskConfig, RiskManager, RiskState
from data.feature_engineering import FeatureEngineer
from main import TradingSystem, _needs_retrain, parse_args

for _n in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

CONFIG = {"broker": {"symbols": ["SPY"]}, "backtest": {"initial_capital": 100000}}


@pytest.fixture(scope="module")
def fitted_system(_ohlcv):
    """A TradingSystem with a reduced HMM fitted on synthetic data."""
    fe = FeatureEngineer()
    feats = fe.build_features(_ohlcv)
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    hmm.fit(feats)
    orch = StrategyOrchestrator(StrategyConfig(), hmm.regime_info)
    sys_ = TradingSystem(CONFIG, hmm, orch, RiskManager(RiskConfig()), fe, dry_run=True)
    sys_.buffers["SPY"] = _ohlcv
    return sys_


@pytest.fixture(scope="module")
def _ohlcv():
    from conftest import make_synthetic_ohlcv

    return make_synthetic_ohlcv()


def test_process_symbol_approves_with_nonzero_shares(fitted_system) -> None:
    """The full pipeline yields a risk-approved signal with real share count."""
    results = fitted_system.process_symbol("SPY")
    assert results, "expected at least one signal"
    approved = [d for _, d in results if d.approved]
    assert approved, "expected an approved decision"
    # field-flow: approved_shares originates in the RiskDecision, not the raw Signal
    assert approved[0].modified_signal.metadata["approved_shares"] > 0


def test_process_symbol_dry_run_sends_no_orders(fitted_system) -> None:
    """Dry-run never calls an executor (there is none wired)."""
    assert fitted_system.executor is None
    fitted_system.process_symbol("SPY")  # must not raise despite no executor
    assert fitted_system.recent_signals  # but it did record decisions


def test_empty_buffer_is_safe() -> None:
    """Processing an unknown/empty symbol returns no signals."""
    fe = FeatureEngineer()
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    sys_ = TradingSystem(CONFIG, hmm, None, RiskManager(RiskConfig()), fe, dry_run=True)
    assert sys_.process_symbol("SPY") == []


# ------------------------------------------------------------- breaker rollover ---
def test_daily_rollover_resets_counters() -> None:
    """A date rollover clears the daily trade counter."""
    risk = RiskManager(RiskConfig())
    sys_ = TradingSystem(CONFIG, None, None, risk, None, dry_run=True)
    sys_._maybe_rollover(pd.Timestamp("2024-01-01"))
    risk._daily_trades = 5
    sys_._maybe_rollover(pd.Timestamp("2024-01-02"))   # new day
    assert risk._daily_trades == 0


def test_weekly_rollover_invokes_breaker_weekly_reset() -> None:
    """A week rollover calls the breaker's weekly reset (latch semantics are
    covered directly in test_risk_validate.py)."""
    risk = RiskManager(RiskConfig())
    sys_ = TradingSystem(CONFIG, None, None, risk, None, dry_run=True)
    calls = {"n": 0}
    risk.breaker.reset_weekly = lambda: calls.__setitem__("n", calls["n"] + 1)  # type: ignore
    sys_._maybe_rollover(pd.Timestamp("2024-01-01"))   # week 1
    sys_._maybe_rollover(pd.Timestamp("2024-01-02"))   # same week -> no weekly reset
    assert calls["n"] == 0
    sys_._maybe_rollover(pd.Timestamp("2024-01-08"))   # next week -> reset
    assert calls["n"] == 1


# ------------------------------------------------------------- state + retrain ---
def test_state_snapshot_round_trip(tmp_path) -> None:
    """save_state then load_state restores the equity peak."""
    risk = RiskManager(RiskConfig())
    risk._equity_peak = 123_456.0
    sys_ = TradingSystem(CONFIG, None, None, risk, None, dry_run=True)
    path = tmp_path / "snap.json"
    sys_.save_state(str(path))

    risk2 = RiskManager(RiskConfig())
    sys2 = TradingSystem(CONFIG, None, None, risk2, None, dry_run=True)
    state = sys2.load_state(str(path))
    assert state is not None
    assert risk2._equity_peak == pytest.approx(123_456.0)


def test_needs_retrain_logic(tmp_path) -> None:
    """Retrain when the model is missing or older than the age limit."""
    import os
    import time

    missing = tmp_path / "nope.pkl"
    assert _needs_retrain(missing) is True

    fresh = tmp_path / "hmm.pkl"
    fresh.write_text("x")
    assert _needs_retrain(fresh, max_age_days=7) is False

    old = time.time() - 8 * 86400
    os.utime(fresh, (old, old))
    assert _needs_retrain(fresh, max_age_days=7) is True


# ------------------------------------------------------------------- CLI ---
def test_cli_flat_mode_flags() -> None:
    """Mode flags parse independently."""
    assert parse_args(["--backtest", "--symbols", "SPY"]).backtest is True
    assert parse_args(["--dry-run"]).dry_run is True
    assert parse_args(["--train-only"]).train_only is True
    assert parse_args([]).backtest is False  # default -> live


def test_cli_modes_are_mutually_exclusive() -> None:
    """Two mode flags at once is a parse error."""
    with pytest.raises(SystemExit):
        parse_args(["--backtest", "--dry-run"])
