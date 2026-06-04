"""Tests for fundamental cross-sectional features (ML v2 Phase 1, no network)."""

from __future__ import annotations

from datetime import date

from core.fundamental_features import (
    compute_features,
    latest_pit_rows,
    quality_features,
)


def _block() -> dict:
    return {"ticker": "AAPL", "statements": [
        {"statement": "PL", "data": [
            {"Fiscal Year": 2022, "Publish Date": "2022-10-28",
             "Revenue": 100.0, "Gross Profit": 40.0, "Net Income": 20.0},
            {"Fiscal Year": 2023, "Publish Date": "2023-11-03",
             "Revenue": 120.0, "Gross Profit": 60.0, "Net Income": 30.0},
        ]},
        {"statement": "BS", "data": [
            {"Fiscal Year": 2023, "Publish Date": "2023-11-03",
             "Total Assets": 300.0, "Total Equity": 100.0, "Total Liabilities": 200.0},
        ]},
    ]}


def test_pit_excludes_rows_published_after_asof() -> None:
    # As-of mid-2023: the FY2023 PL (published 2023-11-03) is NOT yet public.
    rows = latest_pit_rows(_block(), asof=date(2023, 6, 30))
    assert rows["PL"]["Fiscal Year"] == 2022          # only the 2022 row is public
    assert "BS" not in rows                            # BS published later -> excluded


def test_pit_uses_latest_published_row() -> None:
    rows = latest_pit_rows(_block(), asof=date(2024, 1, 1))
    assert rows["PL"]["Fiscal Year"] == 2023          # latest public PL


def test_quality_features_ratios() -> None:
    rows = latest_pit_rows(_block(), asof=date(2024, 1, 1))
    f = quality_features(rows)
    assert f["gross_margin"] == 60.0 / 120.0          # 0.50
    assert f["roe"] == 30.0 / 100.0                    # 0.30
    assert f["roa"] == 30.0 / 300.0
    assert f["gross_profitability"] == 60.0 / 300.0   # Novy-Marx
    assert f["leverage"] == 200.0 / 300.0


def test_safe_div_missing_inputs_yield_none() -> None:
    f = quality_features({"PL": {"Gross Profit": 60.0}, "BS": {}})   # no assets
    assert f["gross_profitability"] is None
    assert f["gross_margin"] is None                  # no revenue


def test_compute_features_none_when_no_public_data() -> None:
    # As-of before any publish date -> nothing public -> None.
    assert compute_features(_block(), asof=date(2000, 1, 1)) is None
