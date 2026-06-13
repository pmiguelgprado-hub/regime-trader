"""Tests for the HMM-vs-JumpModel shadow log + monthly report (T1.1/T1.4)."""

from __future__ import annotations

import pandas as pd

from core import shadow_regime as sr


def test_make_row_records_both_engines():
    row = sr.make_row("2026-06-13", hmm_vol_rank=0.2, jm_vol_rank=0.25)
    assert row["date"] == "2026-06-13"
    assert row["hmm_vol_rank"] == 0.2 and row["jm_vol_rank"] == 0.25
    assert row["agree"] is True                        # both risk-on (<0.5)


def test_disagreement_when_opposite_halves():
    row = sr.make_row("2026-06-13", hmm_vol_rank=0.1, jm_vol_rank=0.9)
    assert row["agree"] is False


def test_append_idempotent_per_date(tmp_path):
    p = str(tmp_path / "shadow_regime.csv")
    sr.append_row(p, sr.make_row("2026-06-13", 0.2, 0.2))
    sr.append_row(p, sr.make_row("2026-06-13", 0.9, 0.1))   # same day -> no-op
    df = pd.read_csv(p)
    assert len(df) == 1 and df.iloc[0]["hmm_vol_rank"] == 0.2


def test_monthly_report_counts_agreement_and_flicker(tmp_path):
    p = str(tmp_path / "shadow_regime.csv")
    # 5 days: 3 agree, 2 disagree; JM flips once, HMM flips 3x
    seq = [(0.2, 0.2), (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    for i, (h, j) in enumerate(seq):
        sr.append_row(p, sr.make_row(f"2026-06-{10+i:02d}", h, j))
    rep = sr.monthly_report(p, month="2026-06")
    assert rep["n_days"] == 5
    assert 0.0 <= rep["agreement_rate"] <= 1.0
    assert rep["hmm_switches"] >= rep["jm_switches"]    # JM flickers less here
    md = sr.report_markdown(rep)
    assert "Shadow regime report" in md and "2026-06" in md


def test_monthly_report_empty_when_no_rows(tmp_path):
    rep = sr.monthly_report(str(tmp_path / "none.csv"), month="2026-06")
    assert rep["n_days"] == 0
