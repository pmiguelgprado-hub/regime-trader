"""Tests for the order executor (mocked Alpaca trading client)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from broker.alpaca_client import AlpacaClient, AlpacaConfig
from broker.order_executor import OrderExecutor, OrderStatus, OrderType
from core.regime_strategies import Direction, Signal


class MockTrading:
    """Minimal stand-in for alpaca-py TradingClient (records calls)."""

    def __init__(self) -> None:
        self.submitted: list = []
        self.replaced: list = []
        self.canceled: list = []
        self.fail_submit = False
        # limit orders return this status on submit + on poll (default: instant fill)
        self.limit_status = "filled"
        self.limit_filled_qty: float | None = None

    def submit_order(self, order_data):
        if self.fail_submit:
            raise RuntimeError("insufficient buying power")
        self.submitted.append(order_data)
        is_limit = getattr(order_data, "limit_price", None) is not None
        status = self.limit_status if is_limit else "filled"
        qty = getattr(order_data, "qty", 0)
        # bracket orders come back with child legs (take-profit + stop)
        legs = None
        if "bracket" in str(getattr(order_data, "order_class", "")).lower():
            legs = [SimpleNamespace(id="tp-1", order_type="limit"),
                    SimpleNamespace(id="stop-1", order_type="stop")]
        return SimpleNamespace(
            id="lim-1" if is_limit else "ord-1",
            client_order_id=getattr(order_data, "client_order_id", None),
            symbol=order_data.symbol, status=status,
            filled_qty=qty if status == "filled" else 0, filled_avg_price=100.0,
            legs=legs,
        )

    def cancel_order_by_id(self, order_id):
        self.canceled.append(order_id)

    def replace_order_by_id(self, order_id, order_data):
        self.replaced.append((order_id, order_data))
        return SimpleNamespace(id=order_id, client_order_id=None, symbol="SPY",
                               status="replaced", filled_qty=0, filled_avg_price=0.0)

    def get_order_by_id(self, order_id):
        qty = self.limit_filled_qty if self.limit_filled_qty is not None else 0
        return SimpleNamespace(id=order_id, symbol="SPY", status=self.limit_status,
                               filled_qty=qty, filled_avg_price=100.0, client_order_id=None)


def make_signal(**kw) -> Signal:
    base = dict(symbol="SPY", direction=Direction.LONG, entry_price=100.0,
                stop_loss=95.0, metadata={"approved_shares": 100})
    base.update(kw)
    return Signal(**base)


@pytest.fixture
def mock_trading() -> MockTrading:
    return MockTrading()


@pytest.fixture
def executor(mock_trading: MockTrading) -> OrderExecutor:
    """Order executor wired to a mock Alpaca client (no network)."""
    client = AlpacaClient(AlpacaConfig("k", "s", paper=True))
    client._trading_client = mock_trading
    return OrderExecutor(client, fill_timeout_sec=0.0, poll_interval_sec=0.0)


def test_execute_signal_submits_market_order(executor, mock_trading) -> None:
    """execute_signal should submit a market order for a long signal."""
    res = executor.execute_signal(make_signal(), OrderType.MARKET)
    assert res.status is OrderStatus.FILLED
    assert len(mock_trading.submitted) == 1
    assert getattr(mock_trading.submitted[0], "qty") == 100


def test_submit_limit_order_requires_limit_price(executor) -> None:
    """Limit orders without a price should be rejected."""
    res = executor.submit_order("SPY", 100, "buy", OrderType.LIMIT)
    assert res.status is OrderStatus.REJECTED and "limit_price" in res.message


def test_cancel_order_returns_cancelled_status(executor, mock_trading) -> None:
    """cancel_order should yield CANCELLED status."""
    res = executor.cancel_order("ord-1")
    assert res.status is OrderStatus.CANCELLED
    assert mock_trading.canceled == ["ord-1"]


def test_modify_order_replaces_quantity(executor, mock_trading) -> None:
    """modify_order should submit a replacement with the new qty."""
    executor.modify_order("ord-1", qty=50)
    assert mock_trading.replaced and mock_trading.replaced[0][0] == "ord-1"
    assert mock_trading.replaced[0][1].qty == 50


def test_rejected_order_surfaces_message(executor, mock_trading) -> None:
    """Broker rejection should populate OrderResult.message."""
    mock_trading.fail_submit = True
    res = executor.submit_order("SPY", 100, "buy", OrderType.MARKET)
    assert res.status is OrderStatus.REJECTED and "buying power" in res.message


def test_trade_id_propagates_as_client_order_id(executor, mock_trading) -> None:
    """The signal's trade_id is set as the broker client_order_id."""
    res = executor.submit_order("SPY", 100, "buy", OrderType.MARKET, trade_id="rt-abc123")
    assert res.trade_id == "rt-abc123"
    assert mock_trading.submitted[0].client_order_id == "rt-abc123"


def _market_requests(mock_trading) -> list:
    return [s for s in mock_trading.submitted if type(s).__name__ == "MarketOrderRequest"]


def test_submit_signal_times_out_then_retries_at_market(mock_trading) -> None:
    """An unfilled limit is cancelled and resubmitted at market for full qty."""
    client = AlpacaClient(AlpacaConfig("k", "s"))
    client._trading_client = mock_trading
    ex = OrderExecutor(client, fill_timeout_sec=0.01, poll_interval_sec=0.0)
    mock_trading.limit_status = "new"          # limit never fills
    mock_trading.limit_filled_qty = 0

    res = ex.submit_signal(make_signal(), retry_market=True)

    assert mock_trading.canceled == ["lim-1"]          # remainder cancelled
    market = _market_requests(mock_trading)
    assert len(market) == 1 and market[0].qty == 100   # full qty retried
    assert res.status is OrderStatus.FILLED


def test_submit_signal_partial_fill_nets_remaining_qty(mock_trading) -> None:
    """A partial limit fill at timeout only retries the UNFILLED remainder."""
    client = AlpacaClient(AlpacaConfig("k", "s"))
    client._trading_client = mock_trading
    ex = OrderExecutor(client, fill_timeout_sec=0.01, poll_interval_sec=0.0)
    mock_trading.limit_status = "partially_filled"
    mock_trading.limit_filled_qty = 40         # 40 of 100 filled

    ex.submit_signal(make_signal(), retry_market=True)

    market = _market_requests(mock_trading)
    assert len(market) == 1 and market[0].qty == 60    # 100 - 40, not 100


def test_submit_signal_no_retry_when_disabled(mock_trading) -> None:
    """With retry disabled, an unfilled limit is cancelled, not market-bought."""
    client = AlpacaClient(AlpacaConfig("k", "s"))
    client._trading_client = mock_trading
    ex = OrderExecutor(client, fill_timeout_sec=0.01, poll_interval_sec=0.0)
    mock_trading.limit_status = "new"
    mock_trading.limit_filled_qty = 0

    res = ex.submit_signal(make_signal(), retry_market=False)
    assert res.status is OrderStatus.CANCELLED
    assert _market_requests(mock_trading) == []


# ----------------------------------------------------- bracket stop (C3) ---
def test_bracket_entry_attaches_stop_at_correct_price(executor, mock_trading) -> None:
    """A bracket entry sends a BRACKET-class order with the signal's stop."""
    from alpaca.trading.enums import OrderClass

    executor.submit_bracket_order(make_signal(stop_loss=95.0))
    req = mock_trading.submitted[-1]
    assert req.order_class == OrderClass.BRACKET
    assert float(req.stop_loss.stop_price) == 95.0


def test_bracket_result_captures_stop_leg_id(executor, mock_trading) -> None:
    """The stop child-leg id is captured so trailing stops can modify it."""
    res = executor.submit_bracket_order(make_signal())
    assert res.stop_order_id == "stop-1"


def test_stop_leg_id_helper_finds_the_stop_leg() -> None:
    """_stop_leg_id picks the stop leg out of a bracket's children."""
    order = SimpleNamespace(legs=[
        SimpleNamespace(id="tp-9", order_type="limit"),
        SimpleNamespace(id="stop-9", order_type="stop"),
    ])
    assert OrderExecutor._stop_leg_id(order) == "stop-9"


def test_stop_leg_id_none_when_no_legs() -> None:
    """A plain (non-bracket) order has no stop leg."""
    assert OrderExecutor._stop_leg_id(SimpleNamespace(legs=None)) is None
