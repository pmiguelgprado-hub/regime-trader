"""Tests for live-loop buffer backfill (C1).

The live loop must seed its rolling buffers with history *before* the first
stream bar; otherwise ``build_features`` returns empty for ~450 warmup bars and
the bot is a silent no-op at startup. These tests pin that contract without a
broker or network (a fake MarketData stands in for the Alpaca history fetch).
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
          "backtest": {"initial_capital": 100000}}


class FakeMarketData:
    """Stand-in for data.market_data.MarketData.get_history (no network)."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, str, int]] = []

    def get_history(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        self.calls.append((symbol, timeframe, lookback_bars))
        return self.frames[symbol].tail(lookback_bars)


def _fitted_system() -> TradingSystem:
    """A TradingSystem with a fitted HMM and EMPTY buffers (cold start)."""
    ohlcv = make_synthetic_ohlcv()
    fe = FeatureEngineer()
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    hmm.fit(fe.build_features(ohlcv))
    orch = StrategyOrchestrator(StrategyConfig(), hmm.regime_info)
    return TradingSystem(CONFIG, hmm, orch, RiskManager(RiskConfig()), fe, dry_run=True)


def test_cold_buffer_emits_nothing() -> None:
    """The C1 bug symptom: with no backfill, the first bar produces no signal."""
    sys_ = _fitted_system()
    assert sys_.process_symbol("SPY") == []


def test_seed_buffers_makes_first_bar_emit() -> None:
    """After seeding from history, process_symbol emits on bar 1 (C1 fixed)."""
    sys_ = _fitted_system()
    fake_md = FakeMarketData({"SPY": make_synthetic_ohlcv()})

    sys_.seed_buffers(fake_md)

    assert "SPY" in sys_.buffers and not sys_.buffers["SPY"].empty
    assert sys_.process_symbol("SPY"), "expected a signal on the first bar after backfill"


def test_seed_buffers_uses_config_timeframe_and_warmup_margin() -> None:
    """Backfill fetches enough warmup (>= min_train_bars + z-score/SMA margin)
    at the configured timeframe."""
    sys_ = _fitted_system()
    fake_md = FakeMarketData({"SPY": make_synthetic_ohlcv()})

    sys_.seed_buffers(fake_md)

    assert len(fake_md.calls) == 1
    symbol, timeframe, lookback = fake_md.calls[0]
    assert symbol == "SPY"
    assert timeframe == "1Day"
    # default min_train_bars (504) + warmup margin for z-score(252)+SMA200
    assert lookback >= 504 + 200
