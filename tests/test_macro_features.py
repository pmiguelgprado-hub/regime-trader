"""Tests for macro risk-confirmation features (T1.3) — no network (injected fetch).

VIX/VIX3M term structure (backwardation precedes drawdowns) + FRED credit/curve
spreads. Framing is strict: **risk CONFIRMATION, not return timing** (timing is
falsified). These features NEVER enter the champion HMM's panel — they are shadow
only. Pure parsing/scoring is unit-tested; the fetch is injected.
"""

from __future__ import annotations

import pytest

from data import macro_features as mf


def test_term_structure_backwardation_flagged():
    ts = mf.term_structure(vix=28.0, vix3m=24.0)       # front > back = stress
    assert ts["ratio"] == pytest.approx(28.0 / 24.0)
    assert ts["backwardation"] is True


def test_term_structure_contango_not_flagged():
    ts = mf.term_structure(vix=18.0, vix3m=21.0)       # normal upward term structure
    assert ts["backwardation"] is False


def test_parse_fred_csv_latest_value():
    csv = "observation_date,BAMLH0A0HYM2\n2026-06-10,3.20\n2026-06-11,3.45\n"
    val, date = mf.parse_fred_latest(csv)
    assert val == pytest.approx(3.45) and date == "2026-06-11"


def test_parse_fred_skips_missing_dots():
    csv = "observation_date,X\n2026-06-10,3.20\n2026-06-11,.\n"   # FRED uses '.' for NA
    val, date = mf.parse_fred_latest(csv)
    assert val == pytest.approx(3.20) and date == "2026-06-10"    # last valid


def test_risk_confirmation_high_when_stressed():
    score = mf.risk_confirmation(backwardation=True, hy_oas=6.5, hy_oas_hi=5.0)
    assert score > 0.5


def test_risk_confirmation_low_when_calm():
    score = mf.risk_confirmation(backwardation=False, hy_oas=3.0, hy_oas_hi=5.0)
    assert score < 0.5


def test_risk_confirmation_in_unit_interval():
    for bw in (True, False):
        for oas in (1.0, 4.0, 9.0):
            s = mf.risk_confirmation(backwardation=bw, hy_oas=oas, hy_oas_hi=5.0)
            assert 0.0 <= s <= 1.0


def test_fetch_term_structure_uses_injected_loader():
    def fake(symbol, **kw):
        import pandas as pd
        val = {"^VIX": 30.0, "^VIX3M": 25.0}[symbol]
        return pd.DataFrame({"close": [val]}, index=pd.bdate_range("2026-06-11", periods=1))
    ts = mf.fetch_term_structure(loader=fake)
    assert ts["backwardation"] is True
