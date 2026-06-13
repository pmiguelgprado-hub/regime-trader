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

**Benchmark realization (matches the frozen gate's net-of-cost intent).** The two
benchmarks are recorded as **investable ETF buy-and-hold NAVs**: SPY for the cap-weight
index and **RSP** (Invesco S&P 500 Equal Weight) for the equal-weight control. Using RSP —
rather than a hand-rolled mean-of-daily-returns — avoids a subtle bias: a synthetic
daily-rebalanced EW would be recorded *gross* of its real turnover cost, while the book NAV
is *net* (real fills), so the comparison would be unfairly hard on the book. Two real ETF
price series (≈costless to hold, net of their tiny expense ratio) are buy-and-hold by
construction and directly comparable to the book's realized equity. The book equity path
itself is the one series not cleanly reconstructable after the fact, which is the core
reason this recorder must run daily.

Known residual approximation: ``book_nav`` is raw account equity and does not credit idle
cash at the risk-free rate (the gate's ``credit_cash_rf``); at ~98% gross invested this is
~0.1%/yr and is left uncaptured (documented, not silently dropped).

This is pure measurement: it touches no signal, no construction knob, and cannot affect the
gate's frozen parameters. The return math and the idempotent CSV append are unit-tested here;
the daily data fetch (broker equity, SPY + RSP closes) is live glue in ``main`` (injected, so
this stays network-free and deterministic).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

COLUMNS = ["date", "book_nav", "spy_nav", "ew_nav", "challenger_nav", "quality_nav",
           "code_sha"]


def simple_return(prev: float, cur: float) -> float:
    """One-period simple return ``cur/prev - 1`` (0.0 if ``prev`` is non-positive)."""
    if prev is None or prev <= 0.0:
        return 0.0
    return cur / prev - 1.0


def portfolio_return(weights: dict[str, float], rets: dict[str, float]) -> Optional[float]:
    """Weighted one-day portfolio return; the unallocated remainder is cash at 0.

    A symbol whose return is missing from ``rets`` contributes 0 (treated as cash —
    conservative and explicit rather than silently re-normalizing the book). Returns
    ``None`` for empty ``weights`` so the caller records a gap instead of a fake 0%.
    """
    if not weights:
        return None
    return sum(w * rets.get(sym, 0.0) for sym, w in weights.items())


def snapshot_weights(snapshot_path: str) -> dict[str, float]:
    """Target weights from a dry-run book snapshot ({} if absent).

    A dry-run sleeve (challenger, quality) has no broker account of its own — see
    deploy/com.regimetrader.{challenger,quality}.plist — so its NAV is synthesized by
    marking the snapshot's target weights to market each day (T0.1 / T2.1 gate feed).
    """
    p = Path(snapshot_path)
    if not p.exists():
        return {}
    snap = json.loads(p.read_text())
    return {t["symbol"]: float(t["weight"]) for t in snap.get("targets", [])}


# Back-compat alias (the challenger feed predates the generic name).
challenger_weights = snapshot_weights


def staleness_bdays(path: str, today: str) -> Optional[int]:
    """Business days between the last recorded row and ``today`` (heartbeat, T0.5).

    Weekend gaps are free (Friday row checked on Monday = 1). ``None`` when the
    file is absent or empty — a not-yet-started series is not a stale one.

    Args:
        path: Track-record CSV.
        today: ISO date to measure staleness against.
    """
    df = load_track_record(path)
    if df.empty:
        return None
    last = str(df.iloc[-1]["date"])
    if last >= today:
        return 0
    return max(0, len(pd.bdate_range(last, today)) - 1)


def load_track_record(path: str) -> pd.DataFrame:
    """Load the track-record CSV (empty, correctly-typed frame if the file is absent).

    Columns added after the series started (``challenger_nav``, ``code_sha``) are
    backfilled as NaN on legacy rows — additive schema, existing row values immutable.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def append_day(path: str, date: str, book_equity: float,
               spy_ret: float, ew_ret: float,
               challenger_ret: Optional[float] = None,
               quality_ret: Optional[float] = None,
               code_sha: Optional[str] = None) -> None:
    """Append one day's NAV row, chaining the benchmark levels (idempotent on date).

    On the **first** row there is no prior level, so both benchmark indices are seeded at
    ``book_equity`` (the day's returns are not applied — there is nothing to compound from).
    On later rows the benchmark levels compound from the previous row by ``spy_ret`` /
    ``ew_ret``. If ``date`` already equals the last recorded date the call is a **no-op**
    (a re-run within the same day must not duplicate or overwrite the row).

    ``challenger_nav`` is the synthetic NAV of the dry-run challenger book (T0.1 gate
    feed): seeded at ``book_equity`` on its first observation — including a re-seed after
    a gap, since chaining across NaN is impossible — then compounded by ``challenger_ret``.
    ``None`` records a gap (NaN), never a fake 0% day.

    Args:
        path: Destination CSV.
        date: ISO date string for the row (the dedup key).
        book_equity: The book's account equity that day (the real, net NAV level).
        spy_ret: SPY one-day simple return (from :func:`simple_return`).
        ew_ret: Equal-weight benchmark one-day return — RSP ETF (from :func:`simple_return`).
        challenger_ret: Challenger book one-day return (from :func:`portfolio_return`),
            or ``None`` when no challenger snapshot is available.
        quality_ret: Quality sleeve one-day return (same synthesis), or ``None``.
        code_sha: Short git SHA of the checked-out code that produced the row
            (T0.3 — launchd runs whatever is checked out; this makes drift auditable).
    """
    df = load_track_record(path)
    if not df.empty and str(df.iloc[-1]["date"]) == str(date):
        return  # idempotent: same-day re-run is a no-op
    if df.empty:
        spy_nav = ew_nav = book_equity            # seed all three equal on day 1
        last = None
    else:
        last = df.iloc[-1]
        spy_nav = float(last["spy_nav"]) * (1.0 + spy_ret)
        ew_nav = float(last["ew_nav"]) * (1.0 + ew_ret)

    def _synthetic_nav(ret: Optional[float], col: str) -> Optional[float]:
        # Dry-run sleeve NAV: gap (None) -> NaN; first obs or post-gap -> (re)seed at book
        # equity (can't chain from NaN); else compound the prior level by the day's return.
        prev = None if last is None else last[col]
        if ret is None:
            return None
        if prev is None or pd.isna(prev):
            return book_equity
        return float(prev) * (1.0 + ret)

    ch_nav = _synthetic_nav(challenger_ret, "challenger_nav")
    q_nav = _synthetic_nav(quality_ret, "quality_nav")
    row = {"date": date, "book_nav": book_equity, "spy_nav": spy_nav, "ew_nav": ew_nav,
           "challenger_nav": ch_nav, "quality_nav": q_nav, "code_sha": code_sha}
    out = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
