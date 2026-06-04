"""Daily track-record recorder for the cross-sectional book (forward-gate plumbing).

The frozen pre-registration evaluates the book over a ≥12-month forward window against
**two** benchmarks (equal-weight S&P 500 and SPY), risk-adjusted and net of cost. That
needs a persistent **daily NAV series for all three** — but the live path only writes
``book_snapshot.json``, which is overwritten every run. Without an accumulating series the
gate has nothing to run on at month 12 and the entire wait is wasted.

This module is that recorder. Each trading day it appends one row — book equity, plus
chained EW-S&P500 and SPY index levels — to a CSV. The two benchmark indices are **seeded
equal to the book's day-1 equity**, so all three curves start at the same value and are
directly comparable (returns, Sharpe, maxDD, DSR all derive from the levels at eval time).

This is pure measurement: it touches no signal, no construction knob, and cannot affect
the gate's frozen parameters. The return math and the idempotent CSV append are unit-tested
here; the daily data fetch (broker equity, SPY + constituent closes) is live glue in
``main`` (injected, so this stays network-free and deterministic).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

COLUMNS = ["date", "book_nav", "spy_nav", "ew_nav"]


def simple_return(prev: float, cur: float) -> float:
    """One-period simple return ``cur/prev - 1`` (0.0 if ``prev`` is non-positive)."""
    if prev is None or prev <= 0.0:
        return 0.0
    return cur / prev - 1.0


def equal_weight_return(prev: dict[str, float], cur: dict[str, float]) -> float:
    """Equal-weight daily return across names present (with a valid prior) in both maps.

    The EW-S&P 500 benchmark return: the simple mean of each constituent's one-day return,
    over the names that have a usable previous close (``prev > 0``) and a current close.
    Names missing on either side, or with a non-positive prior, are skipped. Returns 0.0
    when no name qualifies (the index simply holds flat that day rather than crashing).

    Args:
        prev: ``{symbol: previous_close}``.
        cur: ``{symbol: current_close}``.

    Returns:
        The equal-weight cross-sectional mean return.
    """
    rets = [cur[s] / prev[s] - 1.0
            for s in prev.keys() & cur.keys()
            if prev[s] and prev[s] > 0.0]
    return sum(rets) / len(rets) if rets else 0.0


def load_track_record(path: str) -> pd.DataFrame:
    """Load the track-record CSV (empty, correctly-typed frame if the file is absent)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p)


def append_day(path: str, date: str, book_equity: float,
               spy_ret: float, ew_ret: float) -> None:
    """Append one day's NAV row, chaining the benchmark levels (idempotent on date).

    On the **first** row there is no prior level, so both benchmark indices are seeded at
    ``book_equity`` (the day's returns are not applied — there is nothing to compound from).
    On later rows the benchmark levels compound from the previous row by ``spy_ret`` /
    ``ew_ret``. If ``date`` already equals the last recorded date the call is a **no-op**
    (a re-run within the same day must not duplicate or overwrite the row).

    Args:
        path: Destination CSV.
        date: ISO date string for the row (the dedup key).
        book_equity: The book's account equity that day (the real, net NAV level).
        spy_ret: SPY one-day simple return (from :func:`simple_return`).
        ew_ret: EW-S&P 500 one-day return (from :func:`equal_weight_return`).
    """
    df = load_track_record(path)
    if not df.empty and str(df.iloc[-1]["date"]) == str(date):
        return  # idempotent: same-day re-run is a no-op
    if df.empty:
        spy_nav = ew_nav = book_equity            # seed all three equal on day 1
    else:
        last = df.iloc[-1]
        spy_nav = float(last["spy_nav"]) * (1.0 + spy_ret)
        ew_nav = float(last["ew_nav"]) * (1.0 + ew_ret)
    row = {"date": date, "book_nav": book_equity, "spy_nav": spy_nav, "ew_nav": ew_nav}
    out = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
