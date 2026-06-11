"""Staggered (TWAP) execution: time-sliced marketable limits with escalation.

Slippage mitigation for regime-transition days, when the whole book re-sizes at
once. Each parent order is split into ``n_slices`` child slices; each slice is
submitted as a *marketable limit* (buy at ask + offset, sell at bid - offset:
immediate execution expected, but the limit caps the damage if the quote moves),
polled a bounded number of times, then cancelled; whatever remains unfilled
after the last slice is completed with a plain market order so the rebalance
always finishes (an unfinished rebalance is a bigger risk than slippage).

Alpaca supports no iceberg/hidden orders (triage memo 3b) — time slicing is the
implementable half of "execution antifragility". Ships OFF: the daily book of
S&P 500 large caps at paper size barely moves a spread; this exists for
transition days and is enabled per-run, never by default.

The broker adapter is injected (``submit_limit`` / ``submit_market`` /
``order_filled_qty`` / ``cancel``), so the module is testable without a network
and reusable over :class:`broker.order_executor.OrderExecutor`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class StaggeredConfig:
    """TWAP knobs (fixed defaults; not swept)."""

    n_slices: int = 4                 # child slices per parent order
    slice_wait_sec: float = 30.0      # wait between status checks
    status_checks: int = 4            # polls per slice before cancelling
    limit_offset_bp: float = 2.0      # marketable-limit give beyond the touch (bp)


def slice_qty(qty: int, n_slices: int) -> list[int]:
    """Split ``qty`` into near-equal positive slices, remainder front-loaded.

    Args:
        qty: Parent order size (shares).
        n_slices: Requested slice count.

    Returns:
        Slice sizes (sum == qty, no zeros; fewer slices when qty < n_slices).
    """
    if qty <= 0 or n_slices <= 0:
        return []
    n = min(n_slices, qty)
    base, rem = divmod(qty, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


class StaggeredExecutor:
    """Drives one parent order through the slice/poll/escalate cycle."""

    def __init__(
        self,
        broker: Any,
        cfg: StaggeredConfig | None = None,
        waiter: Callable[[float], None] = time.sleep,
        quote_fn: Optional[Callable[[str], Optional[dict]]] = None,
    ) -> None:
        """Wire the executor.

        Args:
            broker: Adapter exposing ``submit_limit(symbol, qty, side, limit_price)``,
                ``submit_market(symbol, qty, side)``, ``order_filled_qty(order_id)``
                and ``cancel(order_id)``.
            cfg: TWAP knobs.
            waiter: Injectable sleep (tests pass a no-op).
            quote_fn: ``symbol -> {bid, ask}`` or None when no NBBO is available.
        """
        self.broker = broker
        self.cfg = cfg or StaggeredConfig()
        self.waiter = waiter
        self.quote_fn = quote_fn or (lambda s: None)

    def _limit_price(self, quote: dict, side: str) -> float:
        """Marketable limit: cross the touch plus a bounded give."""
        give = self.cfg.limit_offset_bp / 10_000.0
        if side == "buy":
            return round(float(quote["ask"]) * (1.0 + give), 2)
        return round(float(quote["bid"]) * (1.0 - give), 2)

    def execute(self, symbol: str, qty: int, side: str) -> dict:
        """Run the staggered cycle for one parent order.

        Args:
            symbol: Ticker.
            qty: Parent size (shares, > 0).
            side: ``"buy"`` or ``"sell"``.

        Returns:
            ``{symbol, side, requested, filled, escalated}`` — ``escalated`` is
            the remainder completed via market order.
        """
        report = {"symbol": symbol, "side": side, "requested": qty,
                  "filled": 0, "escalated": 0}
        if qty <= 0:
            return report

        quote = self.quote_fn(symbol)
        if not quote or not quote.get("bid") or not quote.get("ask"):
            # no NBBO -> cannot price a limit; degrade to the legacy behaviour
            self.broker.submit_market(symbol, qty, side)
            report["filled"] = qty
            return report

        remaining = qty
        for child in slice_qty(qty, self.cfg.n_slices):
            quote = self.quote_fn(symbol) or quote          # refresh when possible
            res = self.broker.submit_limit(
                symbol, child, side, self._limit_price(quote, side))
            oid = getattr(res, "order_id", None)
            filled = 0
            for _ in range(self.cfg.status_checks):
                self.waiter(self.cfg.slice_wait_sec)
                filled = int(self.broker.order_filled_qty(oid))
                if filled >= child:
                    break
            if filled < child:
                self.broker.cancel(oid)
                logger.info("Slice %s %d/%d %s stuck at %d; cancelled",
                            symbol, child, qty, side, filled)
            report["filled"] += min(filled, child)
            remaining -= min(filled, child)

        if remaining > 0:
            # finish the rebalance: an incomplete book is worse than slippage
            self.broker.submit_market(symbol, remaining, side)
            report["filled"] += remaining
            report["escalated"] = remaining
            logger.info("Escalated %d %s %s to market", remaining, symbol, side)
        return report
