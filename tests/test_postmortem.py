"""Tests for the monthly postmortem report (T4.5 + gap 4 live regime attribution)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import postmortem as pm


def _track(n=40, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-06-01", periods=n)
    out = {"date": [d.date().isoformat() for d in dates]}
    out["book_nav"] = list(100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame(out)


def test_book_stats_basic():
    s = pm.book_stats(_track(), "book_nav")
    assert s["n_obs"] >= 30
    assert "total_return" in s and "max_drawdown" in s and "sharpe_ann" in s
    assert s["max_drawdown"] <= 0.0


def test_book_stats_empty_column():
    df = _track(); df["challenger_nav"] = np.nan
    assert pm.book_stats(df, "challenger_nav")["n_obs"] == 0


def test_regime_attribution_groups_returns():
    df = _track(30)
    regimes = ["calm"] * 15 + ["turbulent"] * 15
    attr = pm.regime_attribution(df["book_nav"], regimes)
    assert set(attr) <= {"calm", "turbulent"}
    assert all("mean_ret" in v and "n" in v for v in attr.values())


def test_monthly_postmortem_markdown_renders():
    md = pm.monthly_postmortem_markdown(
        month="2026-06", track_df=_track(),
        ledger_counts={"momentum": 3, "quality": 2},
        alert_counts={"data_quality": 1},
        shadow={"agreement_rate": 0.8, "hmm_switches": 4, "jm_switches": 2, "n_days": 20})
    assert "Postmortem" in md and "2026-06" in md
    assert "momentum" in md and "Gate" in md


def test_monthly_postmortem_handles_empty_track():
    md = pm.monthly_postmortem_markdown(
        month="2026-06", track_df=pd.DataFrame(columns=["date", "book_nav"]),
        ledger_counts={}, alert_counts={}, shadow={"n_days": 0})
    assert "Postmortem" in md
