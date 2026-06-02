"""Tests for stream reconnect + staleness watchdog primitives (S-2)."""

from __future__ import annotations

import pandas as pd
import pytest

from broker.stream_supervisor import reconnect_delay, run_with_reconnect, stream_is_stale
from main import TradingSystem

_CFG = {"broker": {"symbols": ["SPY"], "max_bar_gap_sec": 300},
        "hmm": {"min_train_bars": 504}, "backtest": {"initial_capital": 100000}}


# ----------------------------------------------------------- backoff ---
def test_reconnect_delay_grows_exponentially_and_caps() -> None:
    assert reconnect_delay(0, base=1.0, cap=60.0) == 1.0
    assert reconnect_delay(1, base=1.0, cap=60.0) == 2.0
    assert reconnect_delay(3, base=1.0, cap=60.0) == 8.0
    assert reconnect_delay(10, base=1.0, cap=60.0) == 60.0   # capped


# --------------------------------------------------------- staleness ---
def test_stream_stale_only_when_market_open_and_gap_exceeded() -> None:
    assert stream_is_stale(last_bar_age_sec=400, max_gap_sec=300, market_open=True) is True
    assert stream_is_stale(last_bar_age_sec=400, max_gap_sec=300, market_open=False) is False
    assert stream_is_stale(last_bar_age_sec=100, max_gap_sec=300, market_open=True) is False


# --------------------------------------------------------- reconnect ---
def test_reconnect_retries_then_succeeds() -> None:
    """Transient failures are retried with backoff until the stream runs."""
    calls = {"n": 0}
    slept: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("dropped")
        return "connected"

    result = run_with_reconnect(flaky, max_retries=5, base=1.0, cap=60.0, sleep=slept.append)

    assert result == "connected"
    assert calls["n"] == 3
    assert slept == [1.0, 2.0]   # two backoffs before the 3rd (successful) attempt


def test_reconnect_gives_up_after_max_retries() -> None:
    """Persistent failure re-raises after exhausting retries."""
    slept: list[float] = []

    def always_fails():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        run_with_reconnect(always_fails, max_retries=3, base=1.0, cap=60.0, sleep=slept.append)
    assert len(slept) == 3   # slept before each retry, then gave up


# ----------------------------------------------------- watchdog wiring ---
def _bare_system() -> TradingSystem:
    return TradingSystem(_CFG, hmm=object(), orchestrator=None,
                         risk_manager=None, feature_engineer=None, dry_run=True)


def test_check_stream_health_flags_stale_only_when_open() -> None:
    sys_ = _bare_system()
    sys_._last_bar_ts = pd.Timestamp("2024-01-02 10:00:00")
    later = pd.Timestamp("2024-01-02 10:10:00")   # 600s gap > 300
    assert sys_.check_stream_health(later, market_open=True) is True
    assert sys_.check_stream_health(later, market_open=False) is False


def test_check_stream_health_false_before_first_bar() -> None:
    sys_ = _bare_system()   # _last_bar_ts is None
    assert sys_.check_stream_health(pd.Timestamp("2024-01-02 10:10:00"), market_open=True) is False
