"""Tests for the intraday risk monitor (core/risk_monitor.py).

The monitor is RISK-ONLY: it never adds exposure, never trades intraday alpha
(timeframe is LOCKED daily — M3). It watches the book's intraday drawdown vs the
prior close and walks an escalation ladder: ok -> alert -> derisk -> flatten.
Within a session the ladder only escalates (no flip-flop re-entry intraday); a
new session resets it — the halt-latch catastrophe taught us latches must
auto-release.
"""
from __future__ import annotations

from core.risk_monitor import (
    RiskThresholds,
    assess_intraday_risk,
    escalate,
    plan_derisk_orders,
    plan_flatten_orders,
)

T = RiskThresholds(alert_dd=0.02, derisk_dd=0.04, flatten_dd=0.08, derisk_scale=0.5)


# ------------------------------------------------------------- assessment ---
def test_flat_day_is_ok() -> None:
    assert assess_intraday_risk(100_000, 100_000, T) == "ok"


def test_gain_is_ok() -> None:
    assert assess_intraday_risk(103_000, 100_000, T) == "ok"


def test_small_loss_below_alert_is_ok() -> None:
    assert assess_intraday_risk(98_100, 100_000, T) == "ok"


def test_alert_threshold() -> None:
    assert assess_intraday_risk(98_000, 100_000, T) == "alert"


def test_derisk_threshold() -> None:
    assert assess_intraday_risk(96_000, 100_000, T) == "derisk"


def test_flatten_threshold() -> None:
    assert assess_intraday_risk(91_000, 100_000, T) == "flatten"


def test_zero_last_equity_is_ok_not_crash() -> None:
    # Broker glitch returning 0 must not trigger a panic flatten.
    assert assess_intraday_risk(100_000, 0.0, T) == "ok"


# -------------------------------------------------------------- escalation ---
def test_escalate_only_up() -> None:
    assert escalate("derisk", "alert") == "derisk"
    assert escalate("alert", "flatten") == "flatten"
    assert escalate("ok", "ok") == "ok"


def test_escalate_never_deescalates_intraday() -> None:
    assert escalate("flatten", "ok") == "flatten"


# ------------------------------------------------------------------ orders ---
def test_derisk_sells_half_of_each_position() -> None:
    held = {"AAPL": 10, "MSFT": 5}
    orders = plan_derisk_orders(held, scale=0.5)
    assert {"symbol": "AAPL", "side": "sell", "qty": 5} in orders
    assert {"symbol": "MSFT", "side": "sell", "qty": 2} in orders  # floor of 2.5


def test_derisk_skips_zero_qty_sells() -> None:
    held = {"TINY": 1}
    # scale 0.5 of 1 share floors to 0 -> no order
    assert plan_derisk_orders(held, scale=0.5) == []


def test_derisk_never_buys() -> None:
    held = {"AAPL": 10, "SHORTED": -4}
    orders = plan_derisk_orders(held, scale=0.5)
    assert all(o["side"] == "sell" and o["qty"] > 0 for o in orders)


def test_flatten_sells_everything() -> None:
    held = {"AAPL": 10, "MSFT": 5}
    orders = plan_flatten_orders(held)
    assert {"symbol": "AAPL", "side": "sell", "qty": 10} in orders
    assert {"symbol": "MSFT", "side": "sell", "qty": 5} in orders
    assert len(orders) == 2
