"""Tests for the cross-sectional feature panel (no network — all data injected).

The panel is the Phase 2 deliverable: rows = universe names, cols = the v1 price
momentum (the alpha spine) + the 5 Phase 1 quality factors (point-in-time as-of).
It feeds the Phase 3 model. Pure: momentum reuses ``cross_sectional_ranking`` and
quality reuses ``fundamental_features`` — no re-derived math.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core import feature_panel


def _close(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": prices})


def _frames() -> dict[str, pd.DataFrame]:
    up = list(np.linspace(100.0, 200.0, 260))   # strong momentum, enough history
    flat = [150.0] * 260                          # zero momentum
    short = [100.0, 110.0]                         # too little history -> nan momentum
    return {"AAA": _close(up), "BBB": _close(flat), "CCC": _close(short)}


def _blocks() -> dict[str, dict]:
    aaa = {"statements": [
        {"statement": "PL", "data": [{"Publish Date": "2023-11-03",
                                      "Revenue": 400.0, "Gross Profit": 170.0, "Net Income": 100.0}]},
        {"statement": "BS", "data": [{"Publish Date": "2023-11-03",
                                      "Total Assets": 350.0, "Total Equity": 50.0,
                                      "Total Liabilities": 300.0}]},
    ]}
    # BBB has no fundamentals block -> quality should be NaN, row still present
    return {"AAA": aaa}


def test_panel_columns_and_index() -> None:
    panel = feature_panel.build_panel(_frames(), _blocks(), asof=date(2024, 1, 1))
    assert list(panel.columns) == feature_panel.PANEL_COLUMNS
    assert panel.index.name == "ticker"
    assert "AAA" in panel.index and "BBB" in panel.index


def test_momentum_spine_values() -> None:
    panel = feature_panel.build_panel(_frames(), _blocks(), asof=date(2024, 1, 1),
                                      lookback=252, skip=21)
    assert panel.loc["AAA", "momentum"] > 0.0     # rising series
    assert panel.loc["BBB", "momentum"] == 0.0    # flat series


def test_quality_joined_pointintime() -> None:
    panel = feature_panel.build_panel(_frames(), _blocks(), asof=date(2024, 1, 1))
    assert panel.loc["AAA", "roe"] == 100.0 / 50.0
    assert panel.loc["AAA", "gross_profitability"] == 170.0 / 350.0


def test_missing_fundamentals_are_nan_row_kept() -> None:
    panel = feature_panel.build_panel(_frames(), _blocks(), asof=date(2024, 1, 1))
    assert np.isnan(panel.loc["BBB", "roe"])      # no block -> NaN, not dropped
    assert "BBB" in panel.index


def test_drop_incomplete_filters_short_history_and_missing() -> None:
    panel = feature_panel.build_panel(_frames(), _blocks(), asof=date(2024, 1, 1),
                                      drop_incomplete=True)
    # CCC (nan momentum) and BBB (no quality) drop; only AAA is fully populated
    assert list(panel.index) == ["AAA"]


def test_coverage_counts() -> None:
    panel = feature_panel.build_panel(_frames(), _blocks(), asof=date(2024, 1, 1))
    cov = feature_panel.coverage(panel)
    assert cov["universe"] == 3
    assert cov["with_momentum"] == 2      # AAA, BBB (CCC too short)
    assert cov["complete"] == 1           # only AAA has momentum + all 5 quality
