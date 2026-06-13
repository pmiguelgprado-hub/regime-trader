"""Tests for the gate-countdown data function (T0.6).

Surfaces, per forward gate, days elapsed/remaining of the 12-month window, the
rolling Deflated Sharpe over the accumulated NAV series, and the ledger n_trials
that deflates it. Pure data (the Streamlit panel is thin glue)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.dashboard_data import gate_status


def _track(n=300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-06-05", periods=n)
    out = {"date": [d.date().isoformat() for d in dates]}
    for col, mu in (("book_nav", 0.0006), ("challenger_nav", 0.0004), ("quality_nav", 0.0005)):
        rets = rng.normal(mu, 0.01, n)
        out[col] = list(100_000 * np.cumprod(1 + rets))
    return pd.DataFrame(out)


def test_gate_status_rows_per_known_gate():
    rows = {r["name"]: r for r in gate_status(_track(), n_trials_fn=lambda fam: 1,
                                              today="2026-07-05")}
    assert {"baseline", "challenger", "quality"} <= set(rows)


def test_days_elapsed_and_remaining():
    rows = {r["name"]: r for r in gate_status(_track(), n_trials_fn=lambda fam: 1,
                                              today="2026-07-05")}
    base = rows["baseline"]
    assert base["days_elapsed"] == 30                  # 2026-06-05 -> 2026-07-05
    assert base["days_remaining"] == 365 - 30
    assert base["window_days"] == 365


def test_dsr_present_with_enough_obs():
    rows = {r["name"]: r for r in gate_status(_track(n=300), n_trials_fn=lambda fam: 2,
                                              today="2026-07-05")}
    base = rows["baseline"]
    assert base["n_obs"] >= 250
    assert 0.0 <= base["dsr"] <= 1.0
    assert base["n_trials"] == 2                        # plumbed from the ledger fn


def test_dsr_none_when_too_few_obs():
    short = _track(n=5)
    rows = {r["name"]: r for r in gate_status(short, n_trials_fn=lambda fam: 1,
                                              today="2026-06-12", min_obs=30)}
    assert rows["baseline"]["dsr"] is None


def test_missing_sleeve_column_skipped_gracefully():
    df = _track().drop(columns=["quality_nav"])
    rows = {r["name"]: r for r in gate_status(df, n_trials_fn=lambda fam: 1,
                                              today="2026-07-05")}
    assert "quality" not in rows and "baseline" in rows


def test_all_nan_sleeve_column_skipped():
    df = _track()
    df["challenger_nav"] = np.nan
    rows = {r["name"]: r for r in gate_status(df, n_trials_fn=lambda fam: 1,
                                              today="2026-07-05")}
    assert "challenger" not in rows
