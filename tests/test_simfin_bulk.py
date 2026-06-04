"""Tests for the SimFin bulk adapter (no network — frames are synthetic).

The adapter restructures the bulk income/balance DataFrames into the same
``company_block`` dict shape that Phase 1's ``fundamental_features`` already
consumes, so the (tested) ratio math is reused unchanged. Point-in-time history
must be preserved: ALL rows per ticker survive, not just the latest.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from core import fundamental_features as ff
from data import simfin_bulk


def _income() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            # AAPL: two years (history must be preserved)
            {"Ticker": "AAPL", "Report Date": "2022-09-24", "Publish Date": "2022-10-28",
             "Revenue": 394.0, "Gross Profit": 170.0, "Net Income": 99.0},
            {"Ticker": "AAPL", "Report Date": "2023-09-30", "Publish Date": "2023-11-03",
             "Revenue": 383.0, "Gross Profit": 169.0, "Net Income": 97.0},
            {"Ticker": "MSFT", "Report Date": "2023-06-30", "Publish Date": "2023-07-27",
             "Revenue": 211.0, "Gross Profit": 146.0, "Net Income": 72.0},
        ]
    )
    return df.set_index(["Ticker", "Report Date"])


def _balance() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"Ticker": "AAPL", "Report Date": "2022-09-24", "Publish Date": "2022-10-28",
             "Total Assets": 352.0, "Total Equity": 50.0, "Total Liabilities": 302.0},
            {"Ticker": "AAPL", "Report Date": "2023-09-30", "Publish Date": "2023-11-03",
             "Total Assets": 352.0, "Total Equity": 62.0, "Total Liabilities": 290.0},
            {"Ticker": "MSFT", "Report Date": "2023-06-30", "Publish Date": "2023-07-27",
             "Total Assets": 411.0, "Total Equity": 206.0, "Total Liabilities": 205.0},
        ]
    )
    return df.set_index(["Ticker", "Report Date"])


def test_to_company_blocks_shape_and_keys() -> None:
    blocks = simfin_bulk.to_company_blocks(_income(), _balance())
    assert set(blocks) == {"AAPL", "MSFT"}
    stmts = {b["statement"] for b in blocks["AAPL"]["statements"]}
    assert stmts == {"PL", "BS"}


def test_history_preserved_not_collapsed() -> None:
    blocks = simfin_bulk.to_company_blocks(_income(), _balance())
    pl = next(b for b in blocks["AAPL"]["statements"] if b["statement"] == "PL")
    assert len(pl["data"]) == 2  # both fiscal years, not just the latest
    assert {r["Publish Date"] for r in pl["data"]} == {"2022-10-28", "2023-11-03"}


def test_block_feeds_phase1_features() -> None:
    blocks = simfin_bulk.to_company_blocks(_income(), _balance())
    # as-of after the 2023 publish: latest_pit_rows must pick the 2023 row
    feats = ff.compute_features(blocks["AAPL"], asof=date(2024, 1, 1))
    assert feats is not None
    assert feats["roe"] == 97.0 / 62.0          # uses 2023 BS (latest PIT)
    assert feats["gross_margin"] == 169.0 / 383.0

    # as-of between the two publishes: must use the 2022 row (anti look-ahead)
    feats_early = ff.compute_features(blocks["AAPL"], asof=date(2023, 6, 1))
    assert feats_early["roe"] == 99.0 / 50.0


def test_missing_statement_still_builds_block() -> None:
    # ticker only in income -> block with PL only, no crash
    blocks = simfin_bulk.to_company_blocks(_income(), pd.DataFrame())
    assert "AAPL" in blocks
    stmts = {b["statement"] for b in blocks["AAPL"]["statements"]}
    assert stmts == {"PL"}


def test_load_bulk_injectable_fetch() -> None:
    sentinel = (_income(), _balance())
    inc, bal = simfin_bulk.load_bulk(fetch=lambda: sentinel)
    assert inc is sentinel[0] and bal is sentinel[1]


def test_merge_segments_unions_and_dedups() -> None:
    # general (AAPL/MSFT) + a "banks" segment (JPM, plus a duplicate AAPL row)
    banks = pd.DataFrame(
        [
            {"Ticker": "JPM", "Report Date": "2023-12-31", "Publish Date": "2024-02-01",
             "Revenue": 158.0, "Net Income": 49.0},                 # no Gross Profit (banks)
            {"Ticker": "AAPL", "Report Date": "2023-09-30", "Publish Date": "2099-01-01",
             "Revenue": -1.0, "Net Income": -1.0},                  # duplicate -> must be dropped
        ]
    ).set_index(["Ticker", "Report Date"])
    merged = simfin_bulk._merge_segments([_income(), banks])
    tickers = set(merged.index.get_level_values("Ticker"))
    assert tickers == {"AAPL", "MSFT", "JPM"}                       # financial recovered
    # general AAPL row wins the collision (Revenue 383, not the -1 sentinel)
    aapl_2023 = merged.loc[("AAPL", "2023-09-30")]
    assert aapl_2023["Revenue"] == 383.0
    # bank with no Gross Profit column -> NaN there, but block still usable for roe/roa
    blocks = simfin_bulk.to_company_blocks(merged, pd.DataFrame())
    assert "JPM" in blocks


def test_merge_segments_skips_empty() -> None:
    merged = simfin_bulk._merge_segments([_income(), pd.DataFrame(), None])
    assert set(merged.index.get_level_values("Ticker")) == {"AAPL", "MSFT"}
