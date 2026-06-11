"""Tests for staggered (TWAP) execution — slippage mitigation on transition days.

Alpaca supports no hidden/iceberg orders (triage memo 3b), so the implementable
half is time-slicing: child slices as marketable limits, unfilled remainder
escalated to market at the end. Pure slice math + stub-client flow tests; no
sleeps (waiter injected).
"""

from __future__ import annotations

from types import SimpleNamespace

from broker.staggered import StaggeredConfig, StaggeredExecutor, slice_qty


# ---------------------------------------------------------------- slice math ---
def test_slice_qty_distributes_remainder_front_loaded() -> None:
    assert slice_qty(10, 3) == [4, 3, 3]
    assert slice_qty(9, 3) == [3, 3, 3]
    assert slice_qty(2, 3) == [1, 1]          # fewer slices than requested, no zeros
    assert slice_qty(0, 3) == []
    assert slice_qty(5, 1) == [5]


# ------------------------------------------------------------- executor flow ---
class _StubExec:
    """Records (symbol, qty, side, type, limit) submissions; scripted fills."""

    def __init__(self, fill_after: int = 1):
        self.submitted: list[dict] = []
        self._fill_after = fill_after
        self._checks: dict[str, int] = {}

    def submit_limit(self, symbol, qty, side, limit_price):
        oid = f"o{len(self.submitted)}"
        self.submitted.append(dict(symbol=symbol, qty=qty, side=side,
                                   type="limit", limit=limit_price, id=oid))
        return SimpleNamespace(order_id=oid, filled_qty=0)

    def submit_market(self, symbol, qty, side):
        self.submitted.append(dict(symbol=symbol, qty=qty, side=side,
                                   type="market", id=f"o{len(self.submitted)}"))
        return SimpleNamespace(order_id=f"m{len(self.submitted)}", filled_qty=qty)

    def order_filled_qty(self, order_id):
        """Each limit fills fully after `fill_after` status checks."""
        n = self._checks.get(order_id, 0) + 1
        self._checks[order_id] = n
        qty = next(o["qty"] for o in self.submitted if o.get("id") == order_id)
        return qty if n >= self._fill_after else 0

    def cancel(self, order_id):
        self.submitted.append(dict(type="cancel", id=order_id))


def _quote(bid=99.9, ask=100.1):
    return {"bid": bid, "ask": ask}


def test_slices_submitted_as_marketable_limits_and_fill() -> None:
    stub = _StubExec(fill_after=1)
    ex = StaggeredExecutor(stub, StaggeredConfig(n_slices=3), waiter=lambda s: None,
                           quote_fn=lambda sym: _quote())
    report = ex.execute("AAPL", qty=10, side="buy")
    limits = [o for o in stub.submitted if o["type"] == "limit"]
    assert [o["qty"] for o in limits] == [4, 3, 3]
    assert all(o["limit"] >= 100.1 for o in limits)      # buy: at/above ask
    assert report["filled"] == 10
    assert report["escalated"] == 0


def test_unfilled_remainder_escalates_to_market() -> None:
    """Limits that never fill are cancelled and the remainder goes market."""
    stub = _StubExec(fill_after=99)                       # never fills
    ex = StaggeredExecutor(stub, StaggeredConfig(n_slices=2, status_checks=2),
                           waiter=lambda s: None, quote_fn=lambda sym: _quote())
    report = ex.execute("AAPL", qty=10, side="sell")
    markets = [o for o in stub.submitted if o["type"] == "market"]
    cancels = [o for o in stub.submitted if o["type"] == "cancel"]
    assert len(cancels) == 2                              # both stuck limits cancelled
    assert sum(o["qty"] for o in markets) == 10           # full size completed
    assert report["escalated"] == 10
    sells = [o for o in stub.submitted if o["type"] == "limit"]
    assert all(o["limit"] <= 99.9 for o in sells)         # sell: at/below bid


def test_missing_quote_falls_back_to_single_market_order() -> None:
    """No NBBO -> no limit pricing possible -> degrade to today's behaviour."""
    stub = _StubExec()
    ex = StaggeredExecutor(stub, StaggeredConfig(n_slices=4), waiter=lambda s: None,
                           quote_fn=lambda sym: None)
    report = ex.execute("AAPL", qty=7, side="buy")
    assert [o["type"] for o in stub.submitted] == ["market"]
    assert report["filled"] == 7
