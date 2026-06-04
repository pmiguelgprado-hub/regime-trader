"""Order executor: placement, modification, cancellation.

Translates risk-approved :class:`~core.regime_strategies.Signal`s into Alpaca
orders. Defaults to marketable **limit** orders (±0.1% of reference price),
cancels unfilled orders after a timeout, and optionally retries at market.
Bracket (OCO) orders attach a stop and take-profit to the entry.

The order quantity is **not** recomputed here — it comes from the risk layer's
``signal.metadata["approved_shares"]`` (Phase 5). Each order carries a unique
``trade_id`` set as the broker ``client_order_id``, linking
signal → risk_decision → order → fill for reconciliation.

alpaca-py request/enum classes are imported lazily so importing this module
never pulls in the trading SDK on the backtest path.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from broker.alpaca_client import AlpacaClient
from core.regime_strategies import Direction, Signal

logger = logging.getLogger(__name__)

LIMIT_OFFSET = 0.001  # ±0.1% marketable-limit offset


class OrderType(Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Lifecycle status of a submitted order."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Alpaca order-status string -> our OrderStatus.
_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED, "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING, "accepted_for_bidding": OrderStatus.SUBMITTED,
    "filled": OrderStatus.FILLED, "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "canceled": OrderStatus.CANCELLED, "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED, "done_for_day": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED, "suspended": OrderStatus.REJECTED,
}


@dataclass
class OrderResult:
    """Result of an order operation.

    Attributes:
        order_id: Broker-assigned order id.
        trade_id: Our client_order_id linking signal->order->fill.
        symbol: Ticker.
        status: Current order status.
        filled_qty: Quantity filled so far.
        avg_fill_price: Average fill price.
        message: Broker message / error detail.
        stop_order_id: Broker id of the bracket's stop child-leg, if any
            (needed to modify/trail the protective stop).
    """

    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    symbol: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    message: str = ""
    stop_order_id: Optional[str] = None


class OrderExecutor:
    """Places and manages orders via the Alpaca client."""

    def __init__(
        self,
        client: AlpacaClient,
        fill_timeout_sec: float = 30.0,
        poll_interval_sec: float = 1.0,
    ) -> None:
        """Initialize the executor.

        Args:
            client: Connected Alpaca client.
            fill_timeout_sec: Seconds to wait for a limit fill before cancel.
            poll_interval_sec: Poll cadence while waiting for a fill (0 in tests).
        """
        self.client = client
        self.fill_timeout_sec = fill_timeout_sec
        self.poll_interval_sec = poll_interval_sec

    # --------------------------------------------------------- signal entry ---
    def execute_signal(
        self, signal: Signal, order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Submit an order derived from a risk-approved trade signal.

        Args:
            signal: Risk-approved signal (qty from ``metadata['approved_shares']``).
            order_type: Order type to use (MARKET default).

        Returns:
            `OrderResult` for the submission.
        """
        qty = self._approved_qty(signal)
        if qty <= 0:
            return OrderResult(symbol=signal.symbol, status=OrderStatus.REJECTED,
                               message="signal has no approved_shares")
        side = "buy" if signal.direction is Direction.LONG else "sell"
        limit_price = None
        if order_type is OrderType.LIMIT:
            limit_price = self._marketable_limit(signal.entry_price, side)
        return self.submit_order(signal.symbol, qty, side, order_type,
                                 limit_price=limit_price, trade_id=self._new_trade_id())

    def submit_signal(
        self, signal: Signal, retry_market: bool = True
    ) -> OrderResult:
        """Submit a marketable-limit order for a signal, with timeout + retry.

        Places a limit at ±0.1% of the entry price, waits up to
        ``fill_timeout_sec`` for a fill, cancels if unfilled, and (optionally)
        retries at market.

        Args:
            signal: Risk-approved signal.
            retry_market: Whether to resubmit at market after a limit timeout.

        Returns:
            Final `OrderResult`.
        """
        qty = self._approved_qty(signal)
        if qty <= 0:
            return OrderResult(symbol=signal.symbol, status=OrderStatus.REJECTED,
                               message="signal has no approved_shares")
        side = "buy" if signal.direction is Direction.LONG else "sell"
        trade_id = self._new_trade_id()
        limit_price = self._marketable_limit(signal.entry_price, side)
        res = self.submit_order(signal.symbol, qty, side, OrderType.LIMIT,
                                limit_price=limit_price, trade_id=trade_id)
        if not res.order_id:
            return res

        res = self._await_fill(res)
        if res.status is OrderStatus.FILLED:
            return res

        # unfilled (or partially filled) after timeout -> cancel the remainder
        self.cancel_order(res.order_id)
        remaining = qty - int(res.filled_qty)
        if remaining <= 0:
            res.status = OrderStatus.FILLED
            return res
        if retry_market:
            logger.info("Limit %s filled %d/%d; retrying remaining %d at market",
                        signal.symbol, int(res.filled_qty), qty, remaining)
            return self.submit_order(signal.symbol, remaining, side, OrderType.MARKET,
                                     trade_id=self._new_trade_id())
        res.status = OrderStatus.CANCELLED
        res.message = f"limit timed out ({int(res.filled_qty)}/{qty} filled), no market retry"
        return res

    def submit_bracket_order(self, signal: Signal) -> OrderResult:
        """Submit an entry with attached stop and take-profit (OCO bracket).

        Args:
            signal: Risk-approved signal; ``stop_loss`` required, ``take_profit``
                defaults to a 2:1 reward/risk target if unset.

        Returns:
            `OrderResult` for the bracket entry.
        """
        qty = self._approved_qty(signal)
        if qty <= 0:
            return OrderResult(symbol=signal.symbol, status=OrderStatus.REJECTED,
                               message="signal has no approved_shares")
        if not signal.stop_loss or signal.stop_loss <= 0:
            return OrderResult(symbol=signal.symbol, status=OrderStatus.REJECTED,
                               message="bracket order requires a stop loss")
        take_profit = signal.take_profit or (
            signal.entry_price + 2.0 * (signal.entry_price - signal.stop_loss)
        )
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest, StopLossRequest, TakeProfitRequest,
        )

        trade_id = self._new_trade_id()
        req = MarketOrderRequest(
            symbol=signal.symbol, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(signal.stop_loss, 2)),
            client_order_id=trade_id,
        )
        try:
            order = self.client.trading.submit_order(order_data=req)
        except Exception as exc:  # noqa: BLE001
            return OrderResult(symbol=signal.symbol, trade_id=trade_id,
                               status=OrderStatus.REJECTED, message=str(exc))
        return self._parse_order(order, trade_id)

    # ----------------------------------------------------------- raw orders ---
    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trade_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit a raw order to the broker.

        Args:
            symbol: Ticker.
            qty: Share quantity.
            side: "buy" or "sell".
            order_type: Order type.
            limit_price: Limit price (required for LIMIT / STOP_LIMIT).
            stop_price: Stop price (required for STOP / STOP_LIMIT).
            trade_id: Optional client_order_id (generated if absent).

        Returns:
            `OrderResult`.
        """
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and not limit_price:
            return OrderResult(symbol=symbol, status=OrderStatus.REJECTED,
                               message="limit order requires a limit_price")
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and not stop_price:
            return OrderResult(symbol=symbol, status=OrderStatus.REJECTED,
                               message="stop order requires a stop_price")

        trade_id = trade_id or self._new_trade_id()
        try:
            req = self._build_request(symbol, qty, side, order_type,
                                      limit_price, stop_price, trade_id)
            order = self.client.trading.submit_order(order_data=req)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Order submit failed for %s: %s", symbol, exc)
            return OrderResult(symbol=symbol, trade_id=trade_id,
                               status=OrderStatus.REJECTED, message=str(exc))
        return self._parse_order(order, trade_id)

    def submit_market_orders(self, orders: list[dict]) -> list["OrderResult"]:
        """Submit a batch of plain market orders for a cross-sectional book rebalance.

        Each order is ``{symbol, side, qty}`` (see
        :func:`~core.cross_sectional_ranking.plan_rebalance_orders`, which emits sells
        before buys). Plain market orders, no bracket/stop — the book's risk is
        diversification + the HMM gross overlay, not per-name ATR stops. Submission is
        best-effort: a rejected name is recorded in its ``OrderResult`` and does not abort
        the batch.

        Args:
            orders: Ordered list of ``{symbol, side, qty}`` (sells first).

        Returns:
            One :class:`OrderResult` per non-empty order, in submission order.
        """
        results: list[OrderResult] = []
        for o in orders:
            qty = int(o.get("qty", 0))
            if qty <= 0:
                continue
            results.append(self.submit_order(
                o["symbol"], qty, o["side"], OrderType.MARKET,
                trade_id=self._new_trade_id(),
            ))
        return results

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order.

        Args:
            order_id: Broker order id.

        Returns:
            `OrderResult` reflecting cancellation.
        """
        try:
            self.client.trading.cancel_order_by_id(order_id)
        except Exception as exc:  # noqa: BLE001
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED, message=str(exc))
        return OrderResult(order_id=order_id, status=OrderStatus.CANCELLED)

    def modify_order(
        self,
        order_id: str,
        qty: Optional[int] = None,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        """Modify an open order (replace).

        Args:
            order_id: Broker order id.
            qty: New quantity.
            limit_price: New limit price.

        Returns:
            `OrderResult` for the replacement.
        """
        from alpaca.trading.requests import ReplaceOrderRequest

        req = ReplaceOrderRequest(
            qty=qty,
            limit_price=round(limit_price, 2) if limit_price is not None else None,
        )
        try:
            order = self.client.trading.replace_order_by_id(order_id, order_data=req)
        except Exception as exc:  # noqa: BLE001
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED, message=str(exc))
        return self._parse_order(order, None)

    def modify_stop(
        self, symbol: str, new_stop: float, current_stop: Optional[float] = None,
        stop_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Move a protective stop — **tighten only, never widen**.

        For a long position a tighter stop is a *higher* stop. A request to
        lower the stop (widen risk) is refused as a no-op.

        Args:
            symbol: Ticker.
            new_stop: Proposed stop price.
            current_stop: Existing stop price (for the tighten check).
            stop_order_id: Broker id of the open stop leg to replace.

        Returns:
            `OrderResult` (REJECTED no-op if the move would widen risk).
        """
        if current_stop is not None and new_stop < current_stop - 1e-9:
            logger.info("modify_stop refused for %s: %.2f would widen from %.2f",
                        symbol, new_stop, current_stop)
            return OrderResult(symbol=symbol, status=OrderStatus.REJECTED,
                               message="stop may only tighten (move up), never widen")
        if not stop_order_id:
            return OrderResult(symbol=symbol, status=OrderStatus.REJECTED,
                               message="no stop_order_id to modify")
        from alpaca.trading.requests import ReplaceOrderRequest

        try:
            order = self.client.trading.replace_order_by_id(
                stop_order_id, order_data=ReplaceOrderRequest(stop_price=round(new_stop, 2))
            )
        except Exception as exc:  # noqa: BLE001
            return OrderResult(symbol=symbol, status=OrderStatus.REJECTED, message=str(exc))
        return self._parse_order(order, None)

    def close_position(self, symbol: str) -> OrderResult:
        """Liquidate a single position at market.

        Args:
            symbol: Ticker.

        Returns:
            `OrderResult` for the close order.
        """
        try:
            order = self.client.trading.close_position(symbol)
        except Exception as exc:  # noqa: BLE001
            return OrderResult(symbol=symbol, status=OrderStatus.REJECTED, message=str(exc))
        return self._parse_order(order, None)

    def close_all_positions(self, cancel_orders: bool = True) -> list[OrderResult]:
        """Liquidate every position (circuit-breaker HALT response).

        Args:
            cancel_orders: Also cancel all open orders.

        Returns:
            List of `OrderResult` for each close.
        """
        try:
            orders = self.client.trading.close_all_positions(cancel_orders=cancel_orders)
        except Exception as exc:  # noqa: BLE001
            logger.error("close_all_positions failed: %s", exc)
            return [OrderResult(status=OrderStatus.REJECTED, message=str(exc))]
        results = []
        for o in orders or []:
            body = getattr(o, "body", o)
            results.append(self._parse_order(body, None))
        return results

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Fetch current status of an order.

        Args:
            order_id: Broker order id.

        Returns:
            `OrderStatus`.
        """
        order = self.client.trading.get_order_by_id(order_id)
        return _STATUS_MAP.get(str(order.status).lower().split(".")[-1], OrderStatus.PENDING)

    # ------------------------------------------------------------- helpers ---
    def _await_fill(self, res: OrderResult) -> OrderResult:
        """Poll an order until filled or the fill timeout elapses.

        Tracks ``filled_qty`` each poll so a partial fill at timeout can be
        netted out before any market retry.

        Args:
            res: The submitted order's initial result.

        Returns:
            Updated `OrderResult` (FILLED or last-seen status + filled_qty).
        """
        deadline = time.monotonic() + self.fill_timeout_sec
        while time.monotonic() < deadline:
            latest = self._fetch_order(res.order_id)
            res.status = latest.status
            res.filled_qty = latest.filled_qty
            res.avg_fill_price = latest.avg_fill_price
            if latest.status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
                return res
            if self.poll_interval_sec > 0:
                time.sleep(self.poll_interval_sec)
            else:
                break
        return res

    def _fetch_order(self, order_id: str) -> OrderResult:
        """Fetch and parse an order's current state.

        Args:
            order_id: Broker order id.

        Returns:
            Parsed `OrderResult`.
        """
        return self._parse_order(self.client.trading.get_order_by_id(order_id), None)

    def _build_request(self, symbol, qty, side, order_type, limit_price, stop_price, trade_id):
        """Construct the appropriate alpaca order request object."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest, StopOrderRequest,
        )

        oside = OrderSide.BUY if side == "buy" else OrderSide.SELL
        common = dict(symbol=symbol, qty=qty, side=oside,
                      time_in_force=TimeInForce.DAY, client_order_id=trade_id)
        if order_type is OrderType.MARKET:
            return MarketOrderRequest(**common)
        if order_type is OrderType.LIMIT:
            return LimitOrderRequest(limit_price=round(limit_price, 2), **common)
        if order_type is OrderType.STOP:
            return StopOrderRequest(stop_price=round(stop_price, 2), **common)
        return StopLimitOrderRequest(limit_price=round(limit_price, 2),
                                     stop_price=round(stop_price, 2), **common)

    @staticmethod
    def _marketable_limit(price: float, side: str) -> float:
        """Marketable limit price ±0.1% of reference (buys up, sells down)."""
        return price * (1.0 + LIMIT_OFFSET) if side == "buy" else price * (1.0 - LIMIT_OFFSET)

    @staticmethod
    def _new_trade_id() -> str:
        """Generate a unique trade id (used as client_order_id)."""
        return f"rt-{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _approved_qty(signal: Signal) -> int:
        """Risk-approved share quantity from the signal metadata."""
        return int(signal.metadata.get("approved_shares", 0))

    @staticmethod
    def _stop_leg_id(order: Any) -> Optional[str]:
        """Return the id of a bracket order's stop child-leg, if present.

        Args:
            order: An alpaca ``Order`` (or stand-in) that may carry ``legs``.

        Returns:
            The stop leg's id as a string, or None if there is no stop leg.
        """
        legs = (order.get("legs") if isinstance(order, dict)
                else getattr(order, "legs", None)) or []
        for leg in legs:
            if isinstance(leg, dict):
                leg_type = leg.get("order_type") or leg.get("type") or ""
                leg_id = leg.get("id")
            else:
                leg_type = getattr(leg, "order_type", None) or getattr(leg, "type", "")
                leg_id = getattr(leg, "id", None)
            if "stop" in str(leg_type).lower():
                return str(leg_id) if leg_id else None
        return None

    @staticmethod
    def _parse_order(order: Any, trade_id: Optional[str]) -> OrderResult:
        """Map an alpaca ``Order`` (or dict) into an :class:`OrderResult`."""
        def g(attr, default=None):
            if isinstance(order, dict):
                return order.get(attr, default)
            return getattr(order, attr, default)

        status_raw = str(g("status", "pending")).lower().split(".")[-1]
        filled_price = g("filled_avg_price")
        return OrderResult(
            order_id=str(g("id")) if g("id") else None,
            trade_id=g("client_order_id") or trade_id,
            symbol=g("symbol", "") or "",
            status=_STATUS_MAP.get(status_raw, OrderStatus.PENDING),
            filled_qty=float(g("filled_qty") or 0.0),
            avg_fill_price=float(filled_price) if filled_price else 0.0,
            message="",
            stop_order_id=OrderExecutor._stop_leg_id(order),
        )
