"""Tests for the daily data-quality sentinel (gap 5).

A single day of bad data during a 12-month gate = corrupt evidence. The sentinel
runs on each daily cycle and flags stale prices, anomalous returns (split/feed
errors), and high panel NaN rates BEFORE they silently poison the track record.
Pure checks, unit-tested; the alert dispatch is thin glue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import data_quality as dq


def _series(vals, end="2026-06-12"):
    idx = pd.bdate_range(end=end, periods=len(vals))
    return pd.DataFrame({"close": vals}, index=idx)


def test_clean_series_no_issues():
    s = _series([100, 101, 100.5, 102, 101.5])
    assert dq.check_price_series("SPY", s, asof="2026-06-12") == []


def test_stale_last_bar_flagged():
    s = _series([100, 101, 102], end="2026-06-01")
    issues = dq.check_price_series("SPY", s, asof="2026-06-12", max_stale_bdays=2)
    assert any(i.kind == "stale" for i in issues)
    assert all(i.symbol == "SPY" for i in issues)


def test_anomalous_return_flagged():
    s = _series([100, 101, 152, 103])          # +50% jump = likely split/feed error
    issues = dq.check_price_series("XYZ", s, asof="2026-06-12", max_abs_ret=0.40)
    assert any(i.kind == "jump" for i in issues)


def test_nonpositive_price_flagged():
    s = _series([100, 0.0, 101])
    issues = dq.check_price_series("XYZ", s, asof="2026-06-12")
    assert any(i.kind == "nonpositive" for i in issues)


def test_empty_series_flagged():
    issues = dq.check_price_series("XYZ", pd.DataFrame({"close": []}), asof="2026-06-12")
    assert any(i.kind == "empty" for i in issues)


def test_panel_nan_rate():
    panel = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, np.nan, np.nan]})
    rate = dq.panel_nan_rate(panel)
    assert rate == 2 / 6


def test_panel_nan_rate_flagged_over_threshold():
    panel = pd.DataFrame({"A": [1.0, np.nan], "B": [np.nan, np.nan]})
    issues = dq.check_panel(panel, max_nan_rate=0.30)
    assert any(i.kind == "nan_rate" for i in issues)


def test_summary_groups_by_kind():
    issues = [dq.Issue("SPY", "stale", "x"), dq.Issue("AAPL", "jump", "y"),
              dq.Issue("MSFT", "jump", "z")]
    s = dq.summary(issues)
    assert s["jump"] == 2 and s["stale"] == 1
