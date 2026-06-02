"""Tests for the once-per-day decision cycle (daily-timeframe live path).

A daily strategy must NOT drive off the minute-bar websocket. Instead it runs
one cycle per day on the freshly-closed daily bars: refresh buffers from
history, process each symbol once, update risk posture, persist state. This is
the orchestration `run_once` uses; the broker wiring around it stays
pragma-no-cover.
"""

from __future__ import annotations

import logging

import pandas as pd

from conftest import make_synthetic_ohlcv
from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from core.risk_manager import RiskConfig, RiskManager
from data.feature_engineering import FeatureEngineer
from main import TradingSystem

for _n in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

CONFIG = {"broker": {"symbols": ["SPY"], "timeframe": "1Day"},
          "hmm": {"min_train_bars": 504}, "backtest": {"initial_capital": 100000}}


class FakeMarketData:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def get_history(self, symbol, timeframe, lookback_bars):
        self.calls.append((symbol, timeframe, lookback_bars))
        return self.frames[symbol].tail(lookback_bars)


def _fitted_dry_system() -> TradingSystem:
    ohlcv = make_synthetic_ohlcv()
    fe = FeatureEngineer()
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    hmm.fit(fe.build_features(ohlcv))
    orch = StrategyOrchestrator(StrategyConfig(), hmm.regime_info)
    return TradingSystem(CONFIG, hmm, orch, RiskManager(RiskConfig()), fe, dry_run=True)


def test_run_cycle_seeds_processes_and_returns_signals(tmp_path) -> None:
    """One cycle backfills from history and produces a decision the same call."""
    sys_ = _fitted_dry_system()
    fake_md = FakeMarketData({"SPY": make_synthetic_ohlcv()})

    results = sys_.run_cycle(fake_md, state_path=str(tmp_path / "snap.json"))

    assert fake_md.calls, "cycle must refresh buffers from history"
    assert "SPY" in sys_.buffers and not sys_.buffers["SPY"].empty
    assert results, "expected a decision on the daily bar"


def test_run_cycle_persists_state(tmp_path) -> None:
    """A cycle writes the state snapshot the dashboard reads."""
    sys_ = _fitted_dry_system()
    fake_md = FakeMarketData({"SPY": make_synthetic_ohlcv()})
    snap = tmp_path / "snap.json"

    sys_.run_cycle(fake_md, state_path=str(snap))

    assert snap.exists()


def test_snapshot_has_dashboard_fields(tmp_path) -> None:
    """The snapshot carries the rich regime/risk/regime-table the dashboard needs."""
    import json

    sys_ = _fitted_dry_system()
    fake_md = FakeMarketData({"SPY": make_synthetic_ohlcv()})
    snap = tmp_path / "snap.json"
    sys_.run_cycle(fake_md, state_path=str(snap))
    s = json.loads(snap.read_text())

    assert "regime" in s
    r = s["regime"]
    assert {"name", "confidence", "stability_bars", "confirmed", "vol_rank",
            "runner_ups"} <= set(r)
    assert 0.0 <= r["confidence"] <= 1.0
    assert isinstance(r["runner_ups"], dict) and r["runner_ups"]

    assert "risk" in s
    assert {"state", "daily_dd", "peak_dd", "peak_dd_limit", "leverage_limit",
            "breakers_clear"} <= set(s["risk"])

    assert s["regime_table"]
    assert {"id", "name", "exp_return", "exp_vol", "strategy",
            "max_leverage"} <= set(s["regime_table"][0])


def test_run_cycle_skips_symbols_without_history(tmp_path) -> None:
    """A symbol with no history is skipped, not fatal."""
    sys_ = _fitted_dry_system()
    fake_md = FakeMarketData({"SPY": make_synthetic_ohlcv()})  # no data for a 2nd symbol
    sys_.symbols = ["SPY", "NOPE"]

    # NOPE raises KeyError in the fake; the cycle must tolerate a bad symbol
    results = sys_.run_cycle(fake_md, state_path=str(tmp_path / "snap.json"))
    assert results  # SPY still processed
