"""Tests for the live safety slice (C2 + C5 + C6).

Fills must feed the circuit breaker (C2), so a run of realized losses latches
NORMAL -> REDUCED -> HALTED; on HALT the loop must liquidate (C6) and the risk
layer must reject new entries. C5 (the WebSocket fill subscription) is thin
plumbing routed into the tested ``on_fill`` handler below.
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


def _system_long_4000() -> tuple[TradingSystem, RiskManager, FakeExecutor]:
    """A live system holding 4000 SPY @ 100 (equity 100k), breaker armed."""
    risk = RiskManager(RiskConfig())              # lock_file None -> no repo lock
    tracker = PositionTracker(SimpleNamespace())  # client unused (equity passed in)
    execu = FakeExecutor()
    sys_ = TradingSystem(CONFIG, None, None, risk, None, execu, tracker, dry_run=False)
    tracker.on_fill(FillEvent(symbol="SPY", qty=4000, price=100.0, side="buy"))
    return sys_, risk, execu


def test_losing_fills_latch_reduced_then_halted_and_liquidate() -> None:
    """Realized losses walk the breaker NORMAL->REDUCED->HALTED and trigger liquidation."""
    sys_, risk, execu = _system_long_4000()

    sys_.on_fill(FillEvent(symbol="SPY", qty=1000, price=97.5, side="sell"), equity=100_000)
    assert risk.breaker.state is RiskState.REDUCED   # -2.5% day -> reduce

    sys_.on_fill(FillEvent(symbol="SPY", qty=1000, price=98.0, side="sell"), equity=100_000)
    assert risk.breaker.state is RiskState.HALTED     # compounded < -3% -> halt
    assert risk.state is RiskState.HALTED             # synced into sizing/veto
    assert execu.close_all_calls >= 1                 # C6: liquidation fired


def test_halted_breaker_rejects_new_entries() -> None:
    """Once HALTED, validate_signal vetoes any new long."""
    sys_, risk, execu = _system_long_4000()
    sys_.on_fill(FillEvent(symbol="SPY", qty=3000, price=90.0, side="sell"), equity=100_000)
    assert risk.breaker.state is RiskState.HALTED

    sig = Signal(symbol="SPY", direction=Direction.LONG, entry_price=100.0,
                 stop_loss=95.0, position_size_pct=0.6, leverage=1.0,
                 metadata={"approved_shares": 100})
    ps = PortfolioState(equity=100_000, positions=[],
                        circuit_breaker_status=risk.state)
    decision = risk.validate_signal(sig, ps)
    assert not decision.approved


def test_winning_fills_keep_breaker_normal_no_liquidation() -> None:
    """A profitable sell does not trip the breaker or liquidate."""
    sys_, risk, execu = _system_long_4000()
    sys_.on_fill(FillEvent(symbol="SPY", qty=1000, price=105.0, side="sell"), equity=100_000)
    assert risk.breaker.state is RiskState.NORMAL
    assert execu.close_all_calls == 0
