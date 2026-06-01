"""Position tracker: open positions, weights, P&L, and fill handling.

Maintains a live view of held positions and portfolio-level metrics for the
strategy and risk layers, updates on every fill, and reconciles tracked state
against the broker on startup.

Design (mirrors the risk layer's state-machine/actor split): the **fill
handler** (:meth:`PositionTracker.on_fill`) is a pure function of a fill event
plus current state — fully unit-testable. The **WebSocket plumbing**
(:meth:`PositionTracker.subscribe_fills`) just routes broker events into that
handler and is intentionally thin. alpaca-py is imported lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single open position with live and entry-time context.

    Attributes:
        symbol: Ticker.
        qty: Share quantity (signed: negative = short; system is long-only).
        avg_entry_price: Average entry price.
        current_price: Latest mark price.
        market_value: qty * current_price.
        unrealized_pnl: Mark-to-market P&L.
        weight: Position weight vs. account equity.
        entry_time: Timestamp of first entry fill.
        stop_level: Current protective stop price.
        stop_order_id: Broker id of the open stop leg (needed by modify_stop).
        regime_at_entry: Regime label when the position was opened.
        regime_current: Latest regime label.
        holding_period_bars: Bars held since entry.
    """

    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    weight: float = 0.0
    entry_time: Optional[datetime] = None
    stop_level: float = 0.0
    stop_order_id: Optional[str] = None
    regime_at_entry: Optional[str] = None
    regime_current: Optional[str] = None
    holding_period_bars: int = 0


@dataclass
class PortfolioSnapshot:
    """Aggregate portfolio state at a point in time.

    Attributes:
        equity: Total account equity.
        cash: Available cash.
        gross_exposure: Sum of absolute market values.
        net_exposure: Signed sum of market values.
        positions: Map of symbol -> `Position`.
        realized_pnl_day: Realized P&L since session open.
    """

    equity: float = 0.0
    cash: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl_day: float = 0.0


@dataclass
class FillEvent:
    """A normalized fill notification.

    Attributes:
        symbol: Ticker.
        qty: Filled quantity (always positive).
        price: Fill price.
        side: "buy" or "sell".
        timestamp: Fill time.
        regime: Regime label active at fill (for entry/exit attribution).
    """

    symbol: str
    qty: float
    price: float
    side: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    regime: Optional[str] = None


class PositionTracker:
    """Tracks live positions and portfolio-level metrics."""

    def __init__(self, client: AlpacaClient) -> None:
        """Initialize the tracker.

        Args:
            client: Connected Alpaca client.
        """
        self.client = client
        self._positions: dict[str, Position] = {}
        self.realized_pnl_day: float = 0.0

    # ------------------------------------------------------------ fills ---
    def on_fill(self, fill: FillEvent) -> float:
        """Apply a fill to tracked state — the pure, testable fill handler.

        Buys add to (or open) a position and update the average entry price;
        sells reduce (or close) it and realize P&L. Returns the realized P&L of
        this fill so callers can feed the circuit breaker.

        Args:
            fill: Normalized fill event.

        Returns:
            Realized P&L from this fill (0.0 for opening/adding fills).
        """
        pos = self._positions.get(fill.symbol)
        realized = 0.0

        if fill.side == "buy":
            if pos is None:
                pos = Position(
                    symbol=fill.symbol, qty=fill.qty, avg_entry_price=fill.price,
                    current_price=fill.price, entry_time=fill.timestamp,
                    regime_at_entry=fill.regime, regime_current=fill.regime,
                )
                self._positions[fill.symbol] = pos
            else:
                total_cost = pos.avg_entry_price * pos.qty + fill.price * fill.qty
                pos.qty += fill.qty
                pos.avg_entry_price = total_cost / pos.qty if pos.qty else fill.price
        else:  # sell -> reduce / close, realize P&L
            if pos is None:
                logger.warning("Sell fill for untracked %s; ignoring", fill.symbol)
                return 0.0
            closed = min(fill.qty, pos.qty)
            realized = (fill.price - pos.avg_entry_price) * closed
            self.realized_pnl_day += realized
            pos.qty -= closed
            if pos.qty <= 1e-9:
                del self._positions[fill.symbol]

        if fill.regime and fill.symbol in self._positions:
            self._positions[fill.symbol].regime_current = fill.regime
        self._mark(fill.symbol, fill.price)
        logger.info("Fill %s %s %.0f @ %.2f (realized %.2f)",
                    fill.side, fill.symbol, fill.qty, fill.price, realized)
        return realized

    def apply_fill_to_risk(
        self, fill: FillEvent, portfolio_state: Any, circuit_breaker: Any
    ) -> None:
        """Apply a fill, then propagate to PortfolioState and the CircuitBreaker.

        Args:
            fill: Normalized fill event.
            portfolio_state: ``core.risk_manager.PortfolioState`` to refresh.
            circuit_breaker: ``core.risk_manager.CircuitBreaker`` to update with
                realized P&L.
        """
        realized = self.on_fill(fill)
        equity = portfolio_state.equity if portfolio_state else 0.0
        if circuit_breaker is not None and equity > 0:
            circuit_breaker.update(pnl=realized / equity, equity=equity, regime=fill.regime)
        if portfolio_state is not None:
            portfolio_state.positions = self.to_risk_positions()
            portfolio_state.circuit_breaker_status = circuit_breaker.state if circuit_breaker else portfolio_state.circuit_breaker_status

    # ------------------------------------------------------- broker sync ---
    def refresh(self) -> PortfolioSnapshot:
        """Pull latest positions/account from the broker.

        Returns:
            Current `PortfolioSnapshot`.
        """
        account = self.client.get_account()
        equity = account["equity"]
        broker_positions = self.client.get_positions()
        for bp in broker_positions:
            pos = self._positions.get(bp["symbol"]) or Position(symbol=bp["symbol"])
            pos.qty = bp["qty"]
            pos.avg_entry_price = bp["avg_entry_price"]
            pos.current_price = bp["current_price"]
            pos.market_value = bp["market_value"]
            pos.unrealized_pnl = bp["unrealized_pl"]
            pos.weight = bp["market_value"] / equity if equity else 0.0
            self._positions[bp["symbol"]] = pos
        return self._snapshot(equity, account["cash"])

    def reconcile(self) -> dict[str, str]:
        """Reconcile tracked positions against the broker (broker is truth).

        Returns:
            Map of symbol -> discrepancy description for any mismatch found.
            Tracked state is overwritten to match the broker.
        """
        broker = {bp["symbol"]: bp for bp in self.client.get_positions()}
        discrepancies: dict[str, str] = {}

        for symbol in set(self._positions) | set(broker):
            tracked_qty = self._positions[symbol].qty if symbol in self._positions else 0.0
            broker_qty = broker[symbol]["qty"] if symbol in broker else 0.0
            if abs(tracked_qty - broker_qty) > 1e-6:
                discrepancies[symbol] = (
                    f"tracked qty {tracked_qty} != broker qty {broker_qty}"
                )
                logger.warning("Reconcile mismatch %s: %s", symbol, discrepancies[symbol])

        # adopt the broker as source of truth
        if discrepancies:
            self.refresh()
            for symbol in list(self._positions):
                if symbol not in broker:
                    del self._positions[symbol]
        return discrepancies

    def sync_on_startup(self) -> PortfolioSnapshot:
        """Refresh from the broker and reconcile (call once at startup).

        Returns:
            The post-sync `PortfolioSnapshot`.
        """
        snap = self.refresh()
        self.reconcile()
        return snap

    def subscribe_fills(self, regime_provider=None) -> None:  # pragma: no cover - live socket
        """Subscribe to the broker trade-update stream and route fills.

        Thin plumbing only: each broker trade-update is normalized into a
        :class:`FillEvent` and passed to :meth:`on_fill`. Not unit-tested (live
        WebSocket); the handler it calls is.

        Args:
            regime_provider: Optional zero-arg callable returning the current
                regime label to tag fills with.
        """
        from alpaca.trading.stream import TradingStream

        stream = TradingStream(self.client.config.api_key,
                               self.client.config.secret_key,
                               paper=self.client.config.paper)

        async def _handler(data):
            if str(getattr(data, "event", "")) not in ("fill", "partial_fill"):
                return
            order = data.order
            self.on_fill(FillEvent(
                symbol=order.symbol, qty=float(data.qty or order.filled_qty),
                price=float(data.price or order.filled_avg_price),
                side=str(order.side).split(".")[-1].lower(),
                regime=regime_provider() if regime_provider else None,
            ))

        stream.subscribe_trade_updates(_handler)
        stream.run()

    # --------------------------------------------------------- accessors ---
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return a single tracked position.

        Args:
            symbol: Ticker.

        Returns:
            `Position` or None if flat.
        """
        return self._positions.get(symbol)

    def get_weights(self) -> dict[str, float]:
        """Return current per-symbol portfolio weights.

        Returns:
            Map of symbol -> weight.
        """
        return {s: p.weight for s, p in self._positions.items()}

    def compute_pnl(self) -> dict[str, float]:
        """Compute realized + unrealized P&L.

        Returns:
            Dict with realized/unrealized/total P&L.
        """
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return {
            "realized": self.realized_pnl_day,
            "unrealized": unrealized,
            "total": self.realized_pnl_day + unrealized,
        }

    def to_risk_positions(self) -> list:
        """Bridge tracked positions into ``risk_manager.Position`` objects.

        Returns:
            List of ``core.risk_manager.Position`` for ``validate_signal``.
        """
        from core.risk_manager import Position as RiskPosition

        return [
            RiskPosition(symbol=p.symbol, market_value=p.market_value, side="long")
            for p in self._positions.values()
        ]

    def advance_bar(self) -> None:
        """Increment the holding-period counter for every open position."""
        for p in self._positions.values():
            p.holding_period_bars += 1

    # ----------------------------------------------------------- internal ---
    def _mark(self, symbol: str, price: float) -> None:
        """Mark a position to a new price and recompute value/P&L."""
        pos = self._positions.get(symbol)
        if pos is None:
            return
        pos.current_price = price
        pos.market_value = pos.qty * price
        pos.unrealized_pnl = (price - pos.avg_entry_price) * pos.qty

    def _snapshot(self, equity: float, cash: float) -> PortfolioSnapshot:
        """Build a `PortfolioSnapshot` from current tracked positions."""
        gross = sum(abs(p.market_value) for p in self._positions.values())
        net = sum(p.market_value for p in self._positions.values())
        return PortfolioSnapshot(
            equity=equity, cash=cash, gross_exposure=gross, net_exposure=net,
            positions=dict(self._positions), realized_pnl_day=self.realized_pnl_day,
        )
