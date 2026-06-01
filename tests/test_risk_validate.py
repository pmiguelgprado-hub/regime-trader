"""Tests for the Phase-5 risk layer: validate_signal + CircuitBreaker."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import Direction, Signal
from core.risk_manager import (
    CircuitBreaker,
    Position,
    PortfolioState,
    RiskConfig,
    RiskManager,
    RiskState,
)

for _n in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)


def make_signal(**kw) -> Signal:
    """Build a valid baseline long signal, overridable by keyword."""
    base = dict(
        symbol="SPY", direction=Direction.LONG, entry_price=100.0, stop_loss=95.0,
        position_size_pct=0.95, leverage=1.0, regime_probability=0.90,
        regime_name="bull", metadata={},
    )
    base.update(kw)
    return Signal(**base)


def make_state(**kw) -> PortfolioState:
    """Build a healthy portfolio snapshot, overridable by keyword."""
    base = dict(equity=100_000.0, cash=100_000.0, buying_power=200_000.0)
    base.update(kw)
    return PortfolioState(**base)


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(RiskConfig())


# --------------------------------------------------------- validate_signal ---
def test_missing_stop_is_rejected(rm: RiskManager) -> None:
    """A signal without a stop loss must be refused outright."""
    d = rm.validate_signal(make_signal(stop_loss=0.0), make_state())
    assert not d.approved and "stop" in d.rejection_reason.lower()


def test_valid_signal_sized_by_one_percent_risk(rm: RiskManager) -> None:
    """Size respects 1% risk, 15% concentration, and overnight 3x-gap budget."""
    d = rm.validate_signal(make_signal(), make_state())
    assert d.approved
    shares = d.modified_signal.metadata["approved_shares"]
    notional = d.modified_signal.metadata["approved_notional"]
    # conc cap = 0.15*100k/100 = 150 sh; overnight 3x*5*sh <= 2% equity -> 133 sh binds
    assert shares == 133
    assert 3 * 5.0 * shares <= 0.02 * 100_000 + 15  # 3x-gap loss within 2% budget
    assert notional <= 0.15 * 100_000 + 1e-6        # within 15% portfolio cap
    assert any("overnight" in m for m in d.modifications)


def test_regime_cap_applied_before_portfolio_cap(rm: RiskManager) -> None:
    """A small regime allocation caps size below the 15% portfolio cap."""
    d = rm.validate_signal(make_signal(position_size_pct=0.05), make_state())
    assert d.approved
    # 0.05*100k/100 = 50 sh (regime), well below the 150 portfolio-cap shares
    assert d.modified_signal.metadata["approved_shares"] == 50


def test_minimum_position_floor(rm: RiskManager) -> None:
    """Positions below the $100 minimum are rejected."""
    sig = make_signal(entry_price=50.0, stop_loss=49.0, position_size_pct=0.0009)
    d = rm.validate_signal(sig, make_state())
    assert not d.approved and "minimum" in d.rejection_reason.lower()


@pytest.mark.parametrize("override", [
    dict(metadata={"uncertainty": True}),
    dict(regime_probability=0.10),
])
def test_leverage_forced_by_signal_conditions(rm: RiskManager, override) -> None:
    """Uncertainty / low confidence force leverage down to 1.0x."""
    d = rm.validate_signal(make_signal(leverage=1.25, **override), make_state())
    assert d.approved
    assert d.modified_signal.leverage == 1.0
    assert any("leverage forced" in m for m in d.modifications)


def test_leverage_forced_by_open_positions(rm: RiskManager) -> None:
    """3+ open positions force leverage to 1.0x."""
    state = make_state(positions=[Position(f"X{i}", 5000.0) for i in range(3)])
    d = rm.validate_signal(make_signal(leverage=1.25), state)
    assert d.approved and d.modified_signal.leverage == 1.0


def test_leverage_forced_by_flicker(rm: RiskManager) -> None:
    """High flicker rate forces leverage to 1.0x."""
    d = rm.validate_signal(make_signal(leverage=1.25), make_state(flicker_rate=5))
    assert d.approved and d.modified_signal.leverage == 1.0


def test_low_vol_keeps_leverage_when_clean(rm: RiskManager) -> None:
    """With no force conditions, 1.25x leverage is preserved."""
    d = rm.validate_signal(make_signal(leverage=1.25), make_state())
    assert d.approved and d.modified_signal.leverage == 1.25


def test_duplicate_order_blocked(rm: RiskManager) -> None:
    """Same symbol+direction within the dedupe window is rejected."""
    import time
    state = make_state(recent_orders=[("SPY", "long", time.time())])
    d = rm.validate_signal(make_signal(), state)
    assert not d.approved and "duplicate" in d.rejection_reason.lower()


def test_wide_spread_blocked(rm: RiskManager) -> None:
    """A bid-ask spread over 0.5% is rejected."""
    d = rm.validate_signal(make_signal(metadata={"bid": 100.0, "ask": 101.0}), make_state())
    assert not d.approved and "spread" in d.rejection_reason.lower()


def test_correlation_above_reject_threshold_rejects(rm: RiskManager) -> None:
    """Correlation > 0.85 with a held position rejects the trade."""
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 0.01, 80))
    near = base + pd.Series(np.random.default_rng(1).normal(0, 0.0005, 80))  # ~0.99
    state = make_state(positions=[Position("QQQ", 5000.0)],
                       price_history={"SPY": base, "QQQ": near})
    d = rm.validate_signal(make_signal(), state)
    assert not d.approved and "correlation" in d.rejection_reason.lower()


def test_correlation_in_reduce_band_halves_size(rm: RiskManager) -> None:
    """Correlation in (0.70, 0.85] halves the position and is recorded."""
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 0.01, 80))
    mid = 0.85 * base + pd.Series(np.random.default_rng(42).normal(0, 0.007, 80))
    corr = float(pd.concat([base, mid], axis=1).dropna().corr().iloc[0, 1])
    assert 0.70 < corr <= 0.85, f"fixture corr {corr:.3f} not in reduce band"

    state = make_state(positions=[Position("QQQ", 5000.0)],
                       price_history={"SPY": base, "QQQ": mid})
    full = rm.validate_signal(make_signal(), make_state())  # uncorrelated baseline
    reduced = rm.validate_signal(make_signal(), state)
    assert reduced.approved
    assert any("halved" in m for m in reduced.modifications)
    assert (reduced.modified_signal.metadata["approved_shares"]
            == full.modified_signal.metadata["approved_shares"] // 2)


def test_halted_state_vetoes(rm: RiskManager) -> None:
    """A HALTED circuit-breaker status vetoes every signal."""
    d = rm.validate_signal(make_signal(), make_state(circuit_breaker_status=RiskState.HALTED))
    assert not d.approved


# ----------------------------------------------------------- CircuitBreaker ---
def test_lock_file_written_on_peak_dd(tmp_path) -> None:
    """Peak DD beyond the limit writes a halt lock file and blocks trading."""
    lock = tmp_path / "trading_halted.lock"
    cb = CircuitBreaker(RiskConfig(), lock_path=str(lock))
    cb.update(pnl=0.0, equity=100_000)           # set peak
    cb.update(pnl=-0.11, equity=89_000)          # -11% from peak > 10%
    assert cb.state is RiskState.HALTED
    assert lock.exists()
    assert cb.check() is False


def test_validate_signal_rejects_when_lock_present(tmp_path) -> None:
    """validate_signal refuses while the lock file exists."""
    lock = tmp_path / "halt.lock"
    lock.write_text("halted")
    rm = RiskManager(RiskConfig(lock_file=str(lock)))
    d = rm.validate_signal(make_signal(), make_state())
    assert not d.approved and "lock" in d.rejection_reason.lower()


def test_daily_latch_and_reset() -> None:
    """Daily breaker latches the worst posture until reset_daily clears it."""
    cb = CircuitBreaker(RiskConfig())
    cb.update(pnl=-0.025, equity=97_500)         # -2.5% -> REDUCED
    assert cb.state is RiskState.REDUCED
    cb.update(pnl=-0.01, equity=96_500)          # compound past 3% -> HALTED
    assert cb.state is RiskState.HALTED
    cb.update(pnl=+0.05, equity=101_000)         # recovery does NOT unlatch
    assert cb.state is RiskState.HALTED
    cb.reset_daily()
    assert cb.state is RiskState.NORMAL
    assert len(cb.get_history()) >= 2


def test_weekly_reset_independent_of_daily() -> None:
    """reset_daily clears the daily latch but leaves the weekly latch."""
    cb = CircuitBreaker(RiskConfig())
    cb.update(pnl=-0.055, equity=94_500)         # -5.5% -> weekly REDUCED (and daily HALT)
    cb.reset_daily()                             # clears daily; weekly remains
    assert cb.state is RiskState.REDUCED
    cb.reset_weekly()
    assert cb.state is RiskState.NORMAL
