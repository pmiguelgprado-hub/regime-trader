"""Tests for multi-asset portfolio allocation (E-1)."""

from __future__ import annotations

import pytest

from core.portfolio import portfolio_target_weights


def test_equal_split_capped_by_single_position() -> None:
    """A budget split across names is capped per name by max_single."""
    w = portfolio_target_weights(0.60, ["A", "B", "C"], max_single=0.15, max_concurrent=5)
    assert w == {"A": 0.15, "B": 0.15, "C": 0.15}   # 0.20 each -> capped to 0.15


def test_equal_split_below_cap() -> None:
    """Below the cap, the budget splits equally."""
    w = portfolio_target_weights(0.30, ["A", "B", "C"], max_single=0.15, max_concurrent=5)
    assert w == pytest.approx({"A": 0.10, "B": 0.10, "C": 0.10})


def test_max_concurrent_limits_holdings() -> None:
    """No more than max_concurrent names receive weight."""
    w = portfolio_target_weights(0.90, list("ABCDEFGHIJ"), max_single=0.15, max_concurrent=3)
    assert len([s for s, x in w.items() if x > 0]) == 3


def test_single_symbol_gets_capped_budget() -> None:
    """One symbol: the whole budget, capped at max_single."""
    w = portfolio_target_weights(0.95, ["SPY"], max_single=0.15, max_concurrent=5)
    assert w == {"SPY": 0.15}


def test_empty_universe_or_zero_budget() -> None:
    """No symbols or no budget -> no allocation."""
    assert portfolio_target_weights(0.6, [], max_single=0.15, max_concurrent=5) == {}
    assert portfolio_target_weights(0.0, ["A"], max_single=0.15, max_concurrent=5) == {}
