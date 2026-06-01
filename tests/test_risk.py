"""Tests for the risk manager."""

from __future__ import annotations

import pytest

from core.risk_manager import RiskConfig, RiskManager, RiskState


@pytest.fixture
def risk() -> RiskManager:
    """Default risk manager for tests."""
    return RiskManager(RiskConfig())


def test_position_size_respects_max_risk_per_trade(risk: RiskManager) -> None:
    """Sizing should cap loss at max_risk_per_trade of equity."""
    equity = 100_000.0
    price = 100.0
    stop_distance = 10.0  # wide stop -> risk cap binds before concentration
    res = risk.position_size(equity, price, stop_distance, target_weight=0.99)
    assert res.approved
    # max loss if stopped out must not exceed 1% of equity
    max_loss = res.shares * stop_distance
    assert max_loss <= risk.config.max_risk_per_trade * equity + price


def test_position_size_respects_single_position_cap(risk: RiskManager) -> None:
    """Sizing should not exceed max_single_position weight."""
    equity = 100_000.0
    price = 100.0
    stop_distance = 0.5  # tiny stop -> concentration cap binds
    res = risk.position_size(equity, price, stop_distance, target_weight=0.99)
    assert res.approved
    assert res.notional <= risk.config.max_single_position * equity + price


def test_exposure_cap_enforced(risk: RiskManager) -> None:
    """check_exposure should reject trades breaching the gross-exposure cap."""
    equity = 100_000.0
    cap = risk.config.max_exposure * risk.config.max_leverage * equity
    assert risk.check_exposure(0.0, cap * 0.9, equity) is True
    assert risk.check_exposure(0.0, cap * 1.1, equity) is False


def test_leverage_cap_enforced(risk: RiskManager) -> None:
    """Leverage must never exceed max_leverage."""
    equity = 100_000.0
    over_lev = (risk.config.max_leverage + 0.5) * equity
    assert risk.check_exposure(0.0, over_lev, equity) is False


def test_concurrent_position_limit(risk: RiskManager) -> None:
    """check_concurrent should block beyond max_concurrent."""
    assert risk.check_concurrent(risk.config.max_concurrent - 1) is True
    assert risk.check_concurrent(risk.config.max_concurrent) is False


def test_daily_trade_limit(risk: RiskManager) -> None:
    """check_daily_trade_limit should block beyond max_daily_trades."""
    for _ in range(risk.config.max_daily_trades):
        assert risk.check_daily_trade_limit() is True
        risk.record_trade()
    assert risk.check_daily_trade_limit() is False
    risk.reset_daily()
    assert risk.check_daily_trade_limit() is True


def test_daily_drawdown_reduce_then_halt(risk: RiskManager) -> None:
    """Daily drawdown should move state REDUCED then HALTED."""
    c = risk.config
    assert risk.update_drawdown_state(99_000, -c.daily_dd_reduce, 0.0) is RiskState.REDUCED
    assert risk.update_drawdown_state(98_000, -c.daily_dd_halt, 0.0) is RiskState.HALTED


def test_weekly_drawdown_reduce_then_halt(risk: RiskManager) -> None:
    """Weekly drawdown should move state REDUCED then HALTED."""
    c = risk.config
    assert risk.update_drawdown_state(99_000, 0.0, -c.weekly_dd_reduce) is RiskState.REDUCED
    assert risk.update_drawdown_state(95_000, 0.0, -c.weekly_dd_halt) is RiskState.HALTED


def test_max_drawdown_from_peak_halts(risk: RiskManager) -> None:
    """Drawdown from peak beyond max_dd_from_peak should halt."""
    risk.update_drawdown_state(100_000, 0.0, 0.0)  # set peak
    dd = risk.config.max_dd_from_peak
    halted = risk.update_drawdown_state(100_000 * (1 - dd - 0.005), 0.0, 0.0)
    assert halted is RiskState.HALTED


def test_halted_blocks_sizing(risk: RiskManager) -> None:
    """No new exposure may be sized while HALTED."""
    risk.state = RiskState.HALTED
    res = risk.position_size(100_000, 100, 5, 0.5)
    assert res.approved is False
    assert res.shares == 0
