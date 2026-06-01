"""Test that the live rolling buffer stays bounded (H3).

Without a cap, ingest_bar's pd.concat grows the buffer (and the per-bar
build_features cost) without limit over a long session. The buffer must be
trimmed to a fixed rolling window after each ingest.
"""

from __future__ import annotations

import pandas as pd

from core.risk_manager import RiskConfig, RiskManager
from main import TradingSystem

CONFIG = {"broker": {"symbols": ["SPY"]}, "hmm": {"min_train_bars": 504},
          "backtest": {"initial_capital": 100000}}


def _bar(i: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6}],
        index=[pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)],
    )


def test_ingest_bar_caps_the_rolling_buffer() -> None:
    """Ingesting far more bars than the cap leaves the buffer at the cap."""
    sys_ = TradingSystem(CONFIG, None, None, RiskManager(RiskConfig()), None, dry_run=True)
    cap = sys_._buffer_cap
    for i in range(cap + 500):
        sys_.ingest_bar("SPY", _bar(i))
    assert len(sys_.buffers["SPY"]) == cap


def test_cap_exceeds_feature_warmup() -> None:
    """The cap must stay well above the ~450-bar feature warmup so features
    never go empty after trimming."""
    sys_ = TradingSystem(CONFIG, None, None, RiskManager(RiskConfig()), None, dry_run=True)
    assert sys_._buffer_cap >= 700
