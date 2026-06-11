"""Intraday risk monitor — crash mitigation for the daily cross-sectional book.

RISK-ONLY, by design. The book's signals are daily and LOCKED (M3: the HMM, vol
regimes, and breakers are daily-calibrated; intraday alpha is unvalidated and the
falsification record says the HMM is a vol classifier, not a return predictor).
What an intraday loop CAN responsibly do is watch the book between the daily
rebalances and cut exposure when the day turns into a crash:

    ok -> alert -> derisk (sell `derisk_scale` of every name) -> flatten (sell all)

assessed on the account's intraday return (equity vs the prior session close).
Escalation is monotonic within a session — once de-risked, an afternoon bounce
does not re-buy (intraday re-entry is exactly the flip-flop that burned the old
halt latch). The next daily rebalance resets the ladder and restores targets,
so the latch auto-releases by construction.

Pure logic only; broker wiring lives in main.run_risk_check.
"""
from __future__ import annotations

from dataclasses import dataclass

# Severity order for the escalation ladder.
_LADDER = ("ok", "alert", "derisk", "flatten")


@dataclass(frozen=True)
class RiskThresholds:
    """Intraday drawdown thresholds (positive fractions) + de-risk scale."""

    alert_dd: float = 0.02      # alert only
    derisk_dd: float = 0.04     # sell (1 - derisk_scale) of every position
    flatten_dd: float = 0.08    # emergency exit: liquidate the book
    derisk_scale: float = 0.5   # fraction of each position KEPT after a derisk


def assess_intraday_risk(equity: float, last_equity: float,
                         thresholds: RiskThresholds) -> str:
    """Map the intraday return to a ladder action.

    Args:
        equity: Current account equity.
        last_equity: Equity at the prior session close.
        thresholds: Ladder thresholds.

    Returns:
        ``"ok" | "alert" | "derisk" | "flatten"``. A non-positive
        ``last_equity`` (broker glitch) returns ``"ok"`` — a bad divisor must
        not trigger a panic liquidation.
    """
    if last_equity <= 0:
        return "ok"
    ret = equity / last_equity - 1.0
    if ret <= -thresholds.flatten_dd:
        return "flatten"
    if ret <= -thresholds.derisk_dd:
        return "derisk"
    if ret <= -thresholds.alert_dd:
        return "alert"
    return "ok"


def escalate(prev: str, new: str) -> str:
    """Combine the session's previous action with the new assessment.

    Monotonic within a session: the ladder only climbs. De-escalation happens
    only via the next daily rebalance resetting the state.
    """
    return max(prev, new, key=_LADDER.index)


def plan_derisk_orders(held_shares: dict[str, int], scale: float) -> list[dict]:
    """Sell ``(1 - scale)`` of every long position (pure).

    Shorts (negative qty) are left alone — covering a short ADDS exposure risk
    on the buy side and the current books are long-only anyway. Quantities
    floor to whole shares; a 1-share name floors to 0 and is skipped.

    Returns:
        ``[{symbol, side: "sell", qty}, ...]`` sorted by symbol.
    """
    orders: list[dict] = []
    for sym in sorted(held_shares):
        qty = int(held_shares[sym])
        if qty <= 0:
            continue
        sell = int(qty * (1.0 - scale))
        if sell > 0:
            orders.append({"symbol": sym, "side": "sell", "qty": sell})
    return orders


def plan_flatten_orders(held_shares: dict[str, int]) -> list[dict]:
    """Liquidate every long position (pure)."""
    return [{"symbol": sym, "side": "sell", "qty": int(q)}
            for sym, q in sorted(held_shares.items()) if int(q) > 0]
