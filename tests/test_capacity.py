"""Tests for capacity / %ADV analysis (gap 3).

The cross-sectional book can pick small-cap tail names where a target notional is
a large fraction of average daily dollar volume — fills that look free in a
backtest but would move the market live. This computes each target's %ADV and
flags the offenders. Log-only (does not resize the book during a gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import capacity as cap


def _frame(close, vol):
    n = len(close)
    return pd.DataFrame({"close": close, "volume": vol},
                        index=pd.bdate_range("2026-01-01", periods=n))


def test_adv_dollar_is_close_times_volume_mean():
    f = _frame([10.0] * 30, [1_000_000] * 30)
    assert cap.adv_dollar(f, window=20) == 10.0 * 1_000_000


def test_pct_adv_of_target():
    f = _frame([10.0] * 30, [1_000_000] * 30)         # ADV$ = 10M
    # a $500k target is 5% of ADV
    assert cap.pct_adv(500_000, f) == 0.05


def test_capacity_report_flags_over_threshold():
    frames = {
        "BIG": _frame([100.0] * 30, [5_000_000] * 30),   # ADV$ 500M
        "SMALL": _frame([5.0] * 30, [50_000] * 30),       # ADV$ 250k
    }
    targets = [{"symbol": "BIG", "notional": 1_000_000},   # 0.2% ADV
               {"symbol": "SMALL", "notional": 50_000}]     # 20% ADV -> flag
    rep = cap.capacity_report(targets, frames, max_pct_adv=0.05)
    flagged = {r["symbol"] for r in rep if r["flagged"]}
    assert flagged == {"SMALL"}
    assert all("pct_adv" in r for r in rep)


def test_capacity_report_handles_missing_frame():
    rep = cap.capacity_report([{"symbol": "X", "notional": 1000}], {}, max_pct_adv=0.05)
    assert rep[0]["pct_adv"] is None and rep[0]["flagged"] is False


def test_worst_offenders_sorted():
    frames = {
        "A": _frame([10.0] * 30, [100_000] * 30),     # ADV$ 1M
        "B": _frame([10.0] * 30, [10_000] * 30),      # ADV$ 100k
    }
    targets = [{"symbol": "A", "notional": 50_000}, {"symbol": "B", "notional": 50_000}]
    rep = cap.capacity_report(targets, frames, max_pct_adv=0.05)
    worst = cap.worst_offenders(rep, n=1)
    assert worst[0]["symbol"] == "B"                  # 50% ADV > A's 5%


def test_sector_concentration_fractions():
    sm = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Tech"}
    conc = cap.sector_concentration(["A", "B", "C", "D"], sm)
    assert conc["Tech"] == 0.75 and conc["Energy"] == 0.25


def test_sector_cap_breaches_flags_over():
    sm = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy"}
    breaches = cap.sector_cap_breaches(["A", "B", "C", "D"], sm, max_sector_frac=0.30)
    assert "Tech" in breaches and breaches["Tech"] == 0.75
    assert "Energy" not in breaches


def test_sector_cap_no_breach_when_within():
    sm = {"A": "Tech", "B": "Energy", "C": "Health", "D": "Financials"}
    assert cap.sector_cap_breaches(["A", "B", "C", "D"], sm, max_sector_frac=0.30) == {}
