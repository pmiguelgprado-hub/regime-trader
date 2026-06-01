"""Tests for the broker layer: client safety/backoff, executor, tracker."""

from __future__ import annotations

import os

import pytest

from broker.alpaca_client import LIVE_CONFIRM_PHRASE, AlpacaClient, AlpacaConfig
from broker.order_executor import OrderExecutor, OrderStatus, OrderType
from broker.position_tracker import FillEvent, PositionTracker
from core.risk_manager import CircuitBreaker, PortfolioState, RiskConfig, RiskState


# --------------------------------------------------------------- client ---
def test_paper_is_the_default() -> None:
    """A bare config defaults to paper trading."""
    assert AlpacaConfig("k", "s").paper is True


def test_from_env_requires_keys(monkeypatch) -> None:
    """from_env raises when credentials are absent."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValueError):
        AlpacaConfig.from_env()


def test_live_requires_exact_confirmation_phrase() -> None:
    """Live trading is gated behind the exact typed phrase."""
    wrong = AlpacaClient(AlpacaConfig("k", "s", paper=False), confirm_fn=lambda p: "yes")
    with pytest.raises(PermissionError):
        wrong._require_live_confirmation()
    # exact phrase passes (no SDK touched)
    ok = AlpacaClient(AlpacaConfig("k", "s", paper=False),
                      confirm_fn=lambda p: LIVE_CONFIRM_PHRASE)
    ok._require_live_confirmation()  # should not raise


def test_retry_succeeds_after_transient_failures() -> None:
    """_retry retries with backoff and returns once the call succeeds."""
    client = AlpacaClient(AlpacaConfig("k", "s"), max_retries=5, backoff_base=0.0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert client._retry(flaky, "test") == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_exhausting_attempts() -> None:
    """_retry raises ConnectionError after all attempts fail."""
    client = AlpacaClient(AlpacaConfig("k", "s"), max_retries=3, backoff_base=0.0)
    with pytest.raises(ConnectionError):
        client._retry(lambda: (_ for _ in ()).throw(RuntimeError("down")), "test")


# -------------------------------------------------------------- executor ---
def test_marketable_limit_offsets_by_ten_bps() -> None:
    """Marketable limit is +0.1% for buys, -0.1% for sells."""
    assert OrderExecutor._marketable_limit(100.0, "buy") == pytest.approx(100.1)
    assert OrderExecutor._marketable_limit(100.0, "sell") == pytest.approx(99.9)


def test_modify_stop_tighten_only() -> None:
    """A stop may tighten (move up) but never widen (move down)."""
    ex = OrderExecutor(AlpacaClient(AlpacaConfig("k", "s")))
    widen = ex.modify_stop("SPY", new_stop=90.0, current_stop=95.0)
    assert widen.status is OrderStatus.REJECTED and "widen" in widen.message
    # tightening with no order id is a different (benign) rejection, not a widen block
    tighten = ex.modify_stop("SPY", new_stop=98.0, current_stop=95.0)
    assert "widen" not in tighten.message


# --------------------------------------------------------------- tracker ---
class FakeClient:
    """Stand-in exposing only get_account / get_positions for the tracker."""

    def __init__(self, positions, equity=100_000.0):
        self._positions = positions
        self._equity = equity

    def get_account(self):
        return {"equity": self._equity, "cash": self._equity}

    def get_positions(self):
        return self._positions


def test_reconcile_detects_quantity_mismatch() -> None:
    """reconcile flags a tracked-vs-broker qty mismatch and adopts the broker."""
    broker_pos = [dict(symbol="SPY", qty=50.0, avg_entry_price=100.0,
                       current_price=101.0, market_value=5050.0, unrealized_pl=50.0)]
    tracker = PositionTracker(FakeClient(broker_pos))
    tracker.on_fill(FillEvent("SPY", 100, 100.0, "buy"))  # tracked thinks 100
    diff = tracker.reconcile()
    assert "SPY" in diff
    assert tracker.get_position("SPY").qty == 50.0  # broker is truth


def test_fill_handler_updates_portfolio_and_breaker() -> None:
    """A losing close fill realizes P&L into the circuit breaker + portfolio."""
    tracker = PositionTracker(client=None)
    ps = PortfolioState(equity=100_000.0)
    cb = CircuitBreaker(RiskConfig())

    tracker.apply_fill_to_risk(FillEvent("SPY", 100, 100.0, "buy"), ps, cb)
    assert len(ps.positions) == 1
    # close at a loss large enough to trip the daily breaker (>3% of equity)
    tracker.apply_fill_to_risk(FillEvent("SPY", 100, 60.0, "sell"), ps, cb)
    assert len(ps.positions) == 0
    assert cb.state is RiskState.HALTED          # -$4000 = -4% daily -> HALT
    assert ps.circuit_breaker_status is RiskState.HALTED
    assert tracker.compute_pnl()["realized"] == pytest.approx(-4000.0)
