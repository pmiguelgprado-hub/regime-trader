"""Tests for the SEC EDGAR fundamentals adapter (T2.1) — no network (injected fetch).

The adapter must reproduce, from free SEC companyfacts JSON, the exact
``company_block`` shape that ``core.fundamental_features.compute_features``
already consumes (SimFin stub replaced, nothing downstream changes). The two
things that must be right or the whole sleeve is corrupt:

* **PIT = filing date.** A fact is usable from ``filed`` (when it became public),
  never its fiscal period ``end`` (which leaks 1-3 months).
* **First-filed-only.** When a later 10-K restates a prior period, the value that
  *existed at the time* is the originally-filed one, not the restatement.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.fundamental_features import compute_features
from data import edgar_data as ed


# --- synthetic companyfacts -------------------------------------------------------


def _fact(end, val, filed, fy, fp="FY", form="10-K", accn="a-1"):
    return {"end": end, "val": val, "filed": filed, "fy": fy, "fp": fp,
            "form": form, "accn": accn}


def _facts_json():
    """Two fiscal years; FY2023 is RESTATED by a later filing (later `filed`)."""
    def us_gaap(concept, series):
        return {concept: {"units": {"USD": series}}}

    facts = {"facts": {"us-gaap": {}}}
    g = facts["facts"]["us-gaap"]
    # Revenues: FY2022 + FY2023 original + FY2023 restated (filed later, must be ignored)
    g.update(us_gaap("Revenues", [
        _fact("2022-12-31", 1000, "2023-02-15", 2022, accn="fy22"),
        _fact("2023-12-31", 1200, "2024-02-15", 2023, accn="fy23"),
        _fact("2023-12-31", 9999, "2025-02-15", 2023, accn="fy24-restate"),
    ]))
    g.update(us_gaap("GrossProfit", [
        _fact("2022-12-31", 400, "2023-02-15", 2022, accn="fy22"),
        _fact("2023-12-31", 500, "2024-02-15", 2023, accn="fy23"),
    ]))
    g.update(us_gaap("NetIncomeLoss", [
        _fact("2022-12-31", 200, "2023-02-15", 2022, accn="fy22"),
        _fact("2023-12-31", 250, "2024-02-15", 2023, accn="fy23"),
    ]))
    g.update(us_gaap("Assets", [
        _fact("2022-12-31", 5000, "2023-02-15", 2022, accn="fy22"),
        _fact("2023-12-31", 5500, "2024-02-15", 2023, accn="fy23"),
    ]))
    g.update(us_gaap("StockholdersEquity", [
        _fact("2022-12-31", 2000, "2023-02-15", 2022, accn="fy22"),
        _fact("2023-12-31", 2200, "2024-02-15", 2023, accn="fy23"),
    ]))
    g.update(us_gaap("Liabilities", [
        _fact("2022-12-31", 3000, "2023-02-15", 2022, accn="fy22"),
        _fact("2023-12-31", 3300, "2024-02-15", 2023, accn="fy23"),
    ]))
    return facts


# --- concept extraction + fallback chain ------------------------------------------


def test_concept_uses_first_tag_in_fallback_chain():
    facts = {"facts": {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": [_fact("2023-12-31", 1200, "2024-02-15", 2023)]}},
    }}}
    series = ed._concept_series(facts, ed.REVENUE_TAGS)
    assert series and series[0]["val"] == 1200          # fell through Revenues -> SalesRevenueNet


def test_concept_missing_returns_empty():
    assert ed._concept_series({"facts": {"us-gaap": {}}}, ed.REVENUE_TAGS) == []


def test_concept_merges_across_tag_migration():
    """Real AAPL bug: revenue tag changes across years; the union must cover all.

    ``Revenues`` is present but only for an old year; the recent years live under
    ``RevenueFromContractWithCustomerExcludingAssessedTax``. Picking the first
    non-empty tag would drop every recent period."""
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [_fact("2018-09-30", 265, "2018-11-05", 2018)]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            _fact("2024-09-30", 391, "2024-11-01", 2024)]}},
    }}}
    block = ed.to_company_block(facts)
    pl = {b["statement"]: b for b in block["statements"]}["PL"]
    years = {r["_fy"]: r["Revenue"] for r in pl["data"]}
    assert years == {2018: 265, 2024: 391}          # both periods present


# --- first-filed-only (anti-restatement) ------------------------------------------


def test_first_filed_only_keeps_original_not_restatement():
    series = ed._concept_series(_facts_json(), ed.REVENUE_TAGS)
    pit = ed._first_filed_only(series)
    fy23 = [r for r in pit if r["fy"] == 2023][0]
    assert fy23["val"] == 1200                          # original, NOT the 9999 restatement
    assert fy23["filed"] == "2023-12-31" or fy23["filed"] == "2024-02-15"


# --- company_block shape + PIT wiring ---------------------------------------------


def test_to_company_block_matches_simfin_shape():
    block = ed.to_company_block(_facts_json())
    stmts = {b["statement"]: b for b in block["statements"]}
    assert set(stmts) == {"PL", "BS"}
    pl_row = stmts["PL"]["data"][0]
    assert "Publish Date" in pl_row and "Revenue" in pl_row
    assert "Total Assets" in stmts["BS"]["data"][0]


def test_publish_date_is_filing_date_not_period_end():
    block = ed.to_company_block(_facts_json())
    pl = {b["statement"]: b for b in block["statements"]}["PL"]
    fy23 = [r for r in pl["data"] if r.get("_fy") == 2023][0]
    assert fy23["Publish Date"] == "2024-02-15"          # filed, not 2023-12-31


def test_pit_asof_excludes_unpublished_period():
    """As of just after FY2022's filing, FY2023 (filed 2024) must be invisible."""
    block = ed.to_company_block(_facts_json())
    feats = compute_features(block, asof=date(2023, 6, 1))
    # FY2022 numbers: gross_profitability = 400/5000 = 0.08
    assert feats is not None
    assert feats["gross_profitability"] == pytest.approx(400 / 5000)
    assert feats["roe"] == pytest.approx(200 / 2000)


def test_pit_asof_after_fy23_uses_original_restated_value():
    block = ed.to_company_block(_facts_json())
    feats = compute_features(block, asof=date(2024, 6, 1))
    # FY2023 ORIGINAL revenue 1200 (not the 9999 restatement): gross_margin = 500/1200
    assert feats["gross_margin"] == pytest.approx(500 / 1200)
    assert feats["gross_profitability"] == pytest.approx(500 / 5500)


def test_annual_only_excludes_quarterly():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _fact("2023-03-31", 300, "2023-04-20", 2023, fp="Q1", form="10-Q"),
            _fact("2023-12-31", 1200, "2024-02-15", 2023, fp="FY", form="10-K"),
        ]}},
    }}}
    block = ed.to_company_block(facts)               # annual_only default
    pl = {b["statement"]: b for b in block["statements"]}["PL"]
    assert all(r["_fp"] == "FY" for r in pl["data"])


# --- CIK mapping ------------------------------------------------------------------


def test_ticker_to_cik_zero_pads():
    body = '{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}'
    fetch = lambda url: (200, body)
    assert ed.ticker_to_cik("aapl", fetch) == "0000320193"


def test_ticker_to_cik_unknown_raises():
    fetch = lambda url: (200, '{"0": {"cik_str": 1, "ticker": "FOO", "title": "Foo"}}')
    with pytest.raises(KeyError):
        ed.ticker_to_cik("NOPE", fetch)


# --- universe loader: on-disk cache + resilience ----------------------------------


def test_load_blocks_caches_and_skips_refetch(tmp_path):
    calls = {"n": 0}
    tickers = '{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}'

    def fetch(url):
        calls["n"] += 1
        if "company_tickers" in url:
            return 200, tickers
        return 200, __import__("json").dumps(_facts_json())

    blocks = ed.load_blocks(["AAPL"], fetch=fetch, cache_dir=str(tmp_path))
    assert "AAPL" in blocks and blocks["AAPL"]["statements"]
    first = calls["n"]
    # second call hits the on-disk cache (no companyfacts refetch; tickers may re-resolve)
    ed.load_blocks(["AAPL"], fetch=fetch, cache_dir=str(tmp_path))
    assert calls["n"] - first <= 1


def test_load_blocks_skips_unresolvable_ticker(tmp_path):
    def fetch(url):
        if "company_tickers" in url:
            return 200, '{"0": {"cik_str": 1, "ticker": "AAPL", "title": "Apple"}}'
        return 200, __import__("json").dumps(_facts_json())

    # ZZZZ not in the mapping -> skipped, not fatal
    blocks = ed.load_blocks(["AAPL", "ZZZZ"], fetch=fetch, cache_dir=str(tmp_path))
    assert set(blocks) == {"AAPL"}
