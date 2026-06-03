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


def test_exposure_cap_is_max_leverage(risk: RiskManager) -> None:
    """The gross-exposure ceiling is max_leverage (1.25x): reachable, not exceeded (H4)."""
    equity = 100_000.0
    ceiling = risk.config.max_leverage * equity          # 1.25x
    assert risk.check_exposure(0.0, ceiling * 0.99, equity) is True    # 1.24x ok
    assert risk.check_exposure(0.0, ceiling * 1.01, equity) is False   # > 1.25x rejected


def test_low_vol_125x_leverage_is_reachable(risk: RiskManager) -> None:
    """The configured 1.25x low-vol leverage is live (H4: was dead-capped at 1.0x
    by the old max_exposure*max_leverage product)."""
    equity = 100_000.0
    assert risk.check_exposure(0.0, 1.20 * equity, equity) is True


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


def test_halt_uses_floor_multiplier_not_zero() -> None:
    """Halt sizing keeps a minimum floor so equity can recover (not 0 -> frozen)."""
    rm = RiskManager(RiskConfig(halt_floor_mult=0.25))
    rm.state = RiskState.HALTED
    assert rm.target_size_multiplier() == 0.25


def test_halt_floor_default_is_nonzero() -> None:
    """Default config no longer freezes capital to 0 on halt."""
    rm = RiskManager(RiskConfig())
    rm.state = RiskState.HALTED
    assert rm.target_size_multiplier() > 0.0


def test_reentry_config_default_disabled_and_streak_resets() -> None:
    """Re-entry knob defaults to disabled (legacy); reset clears the calm streak."""
    cfg = RiskConfig()
    assert cfg.peak_reentry_calm_bars == 0          # default = legacy/disabled
    rm = RiskManager(cfg)
    assert rm._calm_streak == 0
    rm._calm_streak = 4
    rm.reset()
    assert rm._calm_streak == 0
