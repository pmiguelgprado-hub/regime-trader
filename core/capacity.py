"""Capacity / %ADV analysis (gap 3).

The cross-sectional book ranks the whole S&P 500 and can select small, thin names
in its tail. A target notional that is a large fraction of a name's average daily
dollar volume (ADV) is a fill that looks costless in a backtest but would move the
market in reality — a hidden capacity wall. This module measures each target's
%ADV and flags the offenders.

Log-only by design: it reports, it does NOT resize the book. Re-weighting live
positions to respect a capacity cap mid-gate would change the frozen strategy's
behavior; that is a book-renewal change (cf. T5.2). Pure + unit-tested.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def adv_dollar(frame: pd.DataFrame, window: int = 20) -> Optional[float]:
    """Average daily dollar volume over the trailing ``window`` bars (None if absent)."""
    if frame is None or "close" not in frame or "volume" not in frame or len(frame) == 0:
        return None
    tail = frame.tail(window)
    dollar = (tail["close"].astype(float) * tail["volume"].astype(float))
    return float(dollar.mean())


def pct_adv(notional: float, frame: pd.DataFrame, window: int = 20) -> Optional[float]:
    """Target notional as a fraction of the name's ADV$ (None if ADV unavailable)."""
    adv = adv_dollar(frame, window)
    if adv is None or adv <= 0:
        return None
    return float(notional) / adv


def capacity_report(targets: list[dict], frames: dict[str, pd.DataFrame],
                    max_pct_adv: float = 0.05, window: int = 20) -> list[dict]:
    """Per-target %ADV with a flag for those exceeding ``max_pct_adv``.

    Args:
        targets: ``[{symbol, notional, ...}, ...]`` (the rebalance plan).
        frames: ``{symbol: OHLCV}`` with close + volume.
        max_pct_adv: Flag threshold (0.05 = a target above 5% of ADV).
        window: Trailing bars for the ADV estimate.

    Returns:
        ``[{symbol, notional, pct_adv, flagged}, ...]``.
    """
    out = []
    for t in targets:
        sym = t.get("symbol")
        pa = pct_adv(float(t.get("notional", 0.0)), frames.get(sym), window)
        out.append({"symbol": sym, "notional": float(t.get("notional", 0.0)),
                    "pct_adv": pa, "flagged": bool(pa is not None and pa > max_pct_adv)})
    return out


def worst_offenders(report: list[dict], n: int = 5) -> list[dict]:
    """The ``n`` targets with the highest %ADV (ignoring rows with no estimate)."""
    rated = [r for r in report if r.get("pct_adv") is not None]
    return sorted(rated, key=lambda r: r["pct_adv"], reverse=True)[:n]


def sector_concentration(symbols: list[str], sector_map: dict[str, str]) -> dict[str, float]:
    """Fraction of the book in each GICS sector ({} for an empty book).

    Equal-name weighting (the book's convention); unmapped names fall to ``UNKNOWN``.
    """
    if not symbols:
        return {}
    counts: dict[str, int] = {}
    for s in symbols:
        sec = sector_map.get(s, "UNKNOWN")
        counts[sec] = counts.get(sec, 0) + 1
    n = len(symbols)
    return {sec: k / n for sec, k in counts.items()}


def sector_cap_breaches(symbols: list[str], sector_map: dict[str, str],
                        max_sector_frac: float = 0.30) -> dict[str, float]:
    """Sectors whose realized share exceeds the cap (log-only drift detector, T5.2).

    Selection already enforces the cap; a non-empty result here means the realized
    book drifted past it (a bug, or names with stale sector tags) — surfaced, not
    auto-corrected (re-weighting mid-gate would change the frozen book)."""
    return {sec: frac for sec, frac in sector_concentration(symbols, sector_map).items()
            if frac > max_sector_frac}
