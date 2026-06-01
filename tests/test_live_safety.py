"""Tests for the live safety slice (C2 + C5 + C6).

The dominant drawdown for a long-only, hold-through strategy is *unrealized*
mark-to-market loss — price falling while holding, with no fills. So the
circuit breaker must be fed from broker equity **once per bar**, not only from
realized P&L on the (rare) rebalance sells. These tests drive a declining
per-bar equity series and assert NORMAL -> REDUCED -> HALTED, liquidation (C6),
and that fills do NOT separately move the breaker (no double counting).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from broker.position_tracker import FillEvent, PositionTracker
from core.regime_strategies import Direction, Signal
from core.risk_manager import PortfolioState, RiskConfig, RiskManager, RiskState
from main import TradingSystem

for _n in ("core.risk_manager", "broker.position_tracker"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

CONFIG = {"broker": {"symbols": ["SPY"]}, "backtest": {"initial_capital": 100000}}


class FakeExecutor:
    """Records circuit-breaker liquidation calls."""

    def __init__(self) -> None:
        self.close_all_calls = 0

    def close_all_positions(self):
        self.close_all_calls += 1
        return []


def _system() -> tuple[TradingSystem, RiskManager, FakeExecutor]:
    risk = RiskManager(RiskConfig())              # lock_file None -> no repo lock
    tracker = PositionTracker(SimpleNamespace())
    execu = FakeExecutor()
    sys_ = TradingSystem(CONFIG, None, None, risk, None, execu, tracker, dry_run=False)
    return sys_, risk, execu


def test_mtm_equity_decline_latches_reduced_then_halted_and_liquidates() -> None:
    """A falling equity series (no fills) walks the breaker and liquidates."""
    sys_, risk, execu = _system()

    sys_._update_risk_posture(100_000)               # set peak
    assert risk.breaker.state is RiskState.NORMAL

    sys_._update_risk_posture(98_000)                # -2% bar -> reduce
    assert risk.breaker.state is RiskState.REDUCED

    halted = sys_._update_risk_posture(96_000)       # compounded < -3% -> halt
    assert risk.breaker.state is RiskState.HALTED
    assert risk.state is RiskState.HALTED            # synced into sizing/veto
    assert halted is True
    assert execu.close_all_calls >= 1                # C6 liquidation


def test_slow_bleed_trips_peak_halt_even_without_a_daily_breach() -> None:
    """A slow -10%-from-peak drift halts via the PEAK breaker even when each
    day's loss stays under the daily threshold (the buy-and-hold MtM case)."""
    sys_, risk, execu = _system()
    sys_._update_risk_posture(100_000)               # peak
    equity = 100_000
    for _ in range(12):
        equity *= 0.99                               # -1%/bar, under the 2% daily reduce
        risk.reset_daily()                           # simulate a daily rollover each bar
        sys_._update_risk_posture(equity)
    assert risk.breaker.state is RiskState.HALTED     # peak DD >= 10% -> halt
    assert execu.close_all_calls >= 1


def test_halted_breaker_rejects_new_entries() -> None:
    """Once HALTED, validate_signal vetoes any new long."""
    sys_, risk, execu = _system()
    sys_._update_risk_posture(100_000)
    sys_._update_risk_posture(95_000)                # -5% bar -> halt
    assert risk.breaker.state is RiskState.HALTED

    sig = Signal(symbol="SPY", direction=Direction.LONG, entry_price=100.0,
                 stop_loss=95.0, position_size_pct=0.6, leverage=1.0,
                 metadata={"approved_shares": 100})
    ps = PortfolioState(equity=95_000, positions=[], circuit_breaker_status=risk.state)
    assert not risk.validate_signal(sig, ps).approved


def test_stable_equity_stays_normal() -> None:
    """Flat equity never trips the breaker or liquidates."""
    sys_, risk, execu = _system()
    sys_._update_risk_posture(100_000)
    sys_._update_risk_posture(100_500)               # up bar
    assert risk.breaker.state is RiskState.NORMAL
    assert execu.close_all_calls == 0


def test_on_fill_updates_tracker_without_moving_the_breaker() -> None:
    """Fills feed position/P&L tracking only; the breaker is driven by equity,
    so a losing sell fill must not separately latch it (no double counting)."""
    sys_, risk, execu = _system()
    sys_.tracker.on_fill(FillEvent(symbol="SPY", qty=1000, price=100.0, side="buy"))

    sys_.on_fill(FillEvent(symbol="SPY", qty=1000, price=80.0, side="sell"))  # big realized loss

    assert risk.breaker.state is RiskState.NORMAL    # breaker untouched by the fill
    assert execu.close_all_calls == 0
    assert sys_.tracker.get_position("SPY") is None   # but the position closed
