"""Tests for live delta/rebalance gate (C4).

The live loop must trade the *delta* between the target position and what is
already held — buy to increase, sell to reduce, skip small drift — instead of
buying the full target every bar (which over-accumulates). The decision is a
pure function of target/held shares and weights, tested directly here.
"""

from __future__ import annotations

from main import TradingSystem

THRESHOLD = 0.10


def test_open_from_flat_buys_full_target() -> None:
    """Flat -> target: buy the whole target."""
    order = TradingSystem._rebalance_order(
        target_shares=100, held_shares=0,
        target_weight=0.60, current_weight=0.0, threshold=THRESHOLD,
    )
    assert order == ("buy", 100)


def test_at_target_does_not_rebuy() -> None:
    """Already at target on a repeated same-signal bar: no order (the C4 bug)."""
    order = TradingSystem._rebalance_order(
        target_shares=100, held_shares=100,
        target_weight=0.60, current_weight=0.60, threshold=THRESHOLD,
    )
    assert order is None


def test_lower_target_sells_the_delta() -> None:
    """Target drops below held: sell exactly the delta."""
    order = TradingSystem._rebalance_order(
        target_shares=50, held_shares=100,
        target_weight=0.30, current_weight=0.60, threshold=THRESHOLD,
    )
    assert order == ("sell", 50)


def test_small_drift_below_threshold_is_skipped() -> None:
    """Weight drift under the rebalance threshold: no churn."""
    order = TradingSystem._rebalance_order(
        target_shares=105, held_shares=100,
        target_weight=0.63, current_weight=0.60, threshold=THRESHOLD,
    )
    assert order is None


def test_must_exit_sells_all_even_below_threshold() -> None:
    """Target collapses to zero: liquidate regardless of the threshold."""
    order = TradingSystem._rebalance_order(
        target_shares=0, held_shares=100,
        target_weight=0.0, current_weight=0.60, threshold=THRESHOLD,
    )
    assert order == ("sell", 100)


def test_increase_above_threshold_buys_the_delta() -> None:
    """Target rises well above held: buy the delta."""
    order = TradingSystem._rebalance_order(
        target_shares=200, held_shares=100,
        target_weight=0.90, current_weight=0.45, threshold=THRESHOLD,
    )
    assert order == ("buy", 100)
