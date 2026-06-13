"""SEC EDGAR fundamentals adapter — free, point-in-time (T2.1).

Replaces the paid SimFin stub (``data.simfin_data``) as the source of
quality/profitability fundamentals, with **zero downstream change**: it emits the
same ``company_block`` shape that ``core.fundamental_features.compute_features``
already consumes (``{statements: [{statement, data:[rows]}]}``, each row carrying a
``Publish Date``). SEC's companyfacts API is free, needs no key (just a User-Agent),
allows ~10 req/s, and — crucially — every XBRL fact carries its **filing date**, so
point-in-time is real (not approximated, as it would be with a current-snapshot
vendor feed).

Two correctness pillars, both unit-tested with synthetic JSON (network injected):

* **PIT = filing date.** A fact is usable from its ``filed`` date (when it became
  public), never its fiscal ``end`` date — using ``end`` leaks 1-3 months of future
  knowledge into the cross-section.
* **First-filed-only (anti-restatement).** A later 10-K restates prior periods; the
  value that *existed at time t* is the one originally filed for that period, so for
  each (fiscal-year, fiscal-period) we keep the earliest-``filed`` fact and drop
  restatements.

XBRL tag fallback chains absorb cross-company reporting variation (e.g. ``Revenues``
vs ``SalesRevenueNet`` vs the ASC-606 contract-revenue tag). Annual periods only by
default — mixing quarterly flows (3-month Net Income) with annual would break
cross-sectional ratio comparability (Novy-Marx gross profitability is an annual factor).

Survivorship caveat unchanged from the price data: a historical cross-section built
from *today's* constituents is biased; the EDGAR feed is PIT-honest **going forward**,
which is why the quality sleeve is judged by a forward pre-registered gate
(docs/analysis/2026-0X-quality-edgar-prereg.md), not a historical-backtest claim.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable, Optional

# (status_code, body_text) <- url
Fetcher = Callable[[str], "tuple[int, str]"]

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# XBRL us-gaap tag fallback chains (first present wins). Documented, extend as needed.
REVENUE_TAGS = ["Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerExcludingAssessedTax"]
GROSS_PROFIT_TAGS = ["GrossProfit"]
NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
ASSETS_TAGS = ["Assets"]
EQUITY_TAGS = ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
LIABILITIES_TAGS = ["Liabilities"]

# SimFin-shaped field name -> (statement, tag chain)
_PL_FIELDS = {"Revenue": REVENUE_TAGS, "Gross Profit": GROSS_PROFIT_TAGS,
              "Net Income": NET_INCOME_TAGS}
_BS_FIELDS = {"Total Assets": ASSETS_TAGS, "Total Equity": EQUITY_TAGS,
              "Total Liabilities": LIABILITIES_TAGS}


def _user_agent() -> str:
    """Contact UA string SEC requires (configurable via env; sane default)."""
    return os.environ.get("SEC_USER_AGENT", "regime-trader research pmiguelgprado@gmail.com")


def _default_fetch(url: str) -> "tuple[int, str]":  # pragma: no cover - network
    """Real HTTP GET against SEC with the required User-Agent (stdlib only)."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent(),
                                               "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            import gzip
            body = gzip.decompress(body)
        return r.status, body.decode("utf-8", "replace")


def ticker_to_cik(ticker: str, fetch: Fetcher = _default_fetch) -> str:
    """Resolve a ticker to its zero-padded 10-digit CIK via SEC's mapping file.

    Args:
        ticker: Stock ticker (case-insensitive).
        fetch: Injected HTTP getter (url -> (status, body)).

    Returns:
        10-digit zero-padded CIK string (companyfacts URL form).

    Raises:
        KeyError: If the ticker is not in SEC's list.
        RuntimeError: On a non-200 response.
    """
    status, body = fetch(TICKERS_URL)
    if status != 200:
        raise RuntimeError(f"SEC tickers fetch failed: HTTP {status}")
    table = json.loads(body)
    want = ticker.strip().upper()
    for row in table.values():
        if str(row.get("ticker", "")).upper() == want:
            return f"{int(row['cik_str']):010d}"
    raise KeyError(f"ticker not found in SEC mapping: {ticker}")


def company_facts(cik: str, fetch: Fetcher = _default_fetch) -> dict:
    """Fetch + parse the companyfacts JSON for a 10-digit CIK.

    Raises:
        RuntimeError: On a non-200 response.
    """
    status, body = fetch(FACTS_URL.format(cik=cik))
    if status != 200:
        raise RuntimeError(f"SEC companyfacts fetch failed for {cik}: HTTP {status}")
    return json.loads(body)


def _concept_series(facts: dict, tags: list[str]) -> list[dict]:
    """Merged USD fact list across the whole fallback chain (chain order; [] if none).

    A single company migrates tags over time (e.g. Apple: ``SalesRevenueNet`` for
    FY2014-17, ``RevenueFromContractWithCustomerExcludingAssessedTax`` from FY2018
    under ASC 606), so the *union* of the chain is needed — picking only the first
    tag that has *any* data would silently miss every period reported under a later
    tag. Per-period selection (earliest-filed across tags) happens in
    :func:`_first_filed_only`.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: list[dict] = []
    for tag in tags:
        node = gaap.get(tag)
        if node:
            merged.extend(node.get("units", {}).get("USD", []))
    return merged


def _first_filed_only(series: list[dict]) -> list[dict]:
    """Anti-restatement: per (fiscal-year, fiscal-period) keep the earliest-filed fact.

    A later filing restates prior periods; the value that existed *at the time* is the
    originally-filed one. Facts missing ``fy``/``filed`` are dropped (can't be placed
    point-in-time safely).
    """
    best: dict[tuple, dict] = {}
    for f in series:
        if f.get("fy") is None or not f.get("filed"):
            continue
        key = (f["fy"], f.get("fp", "FY"))
        cur = best.get(key)
        if cur is None or str(f["filed"]) < str(cur["filed"]):
            best[key] = f
    return list(best.values())


def to_company_block(facts: dict, annual_only: bool = True) -> dict[str, Any]:
    """Build the SimFin-shaped ``company_block`` from raw companyfacts JSON.

    Rows are aligned by fiscal period (fy, fp); each row's ``Publish Date`` is the
    latest filing date among its line items (same filing in practice). Internal
    ``_fy``/``_fp`` keys are carried for auditing and ignored downstream.

    Args:
        facts: Parsed companyfacts JSON.
        annual_only: Keep only fiscal-year (``fp == "FY"``) periods — quarterly flows
            are not ratio-comparable to annual ones across the cross-section.

    Returns:
        ``{"statements": [{"statement": "PL", "data": [rows]}, {"statement": "BS", ...}]}``
    """
    statements = []
    for stmt, fields in (("PL", _PL_FIELDS), ("BS", _BS_FIELDS)):
        periods: dict[tuple, dict] = {}
        for field_name, tags in fields.items():
            for f in _first_filed_only(_concept_series(facts, tags)):
                if annual_only and f.get("fp", "FY") != "FY":
                    continue
                key = (f["fy"], f.get("fp", "FY"))
                row = periods.setdefault(key, {"_fy": f["fy"], "_fp": f.get("fp", "FY"),
                                               "Publish Date": ""})
                row[field_name] = f["val"]
                # Publish Date = latest filing among this period's line items.
                if str(f["filed"]) > str(row["Publish Date"]):
                    row["Publish Date"] = str(f["filed"])
        rows = sorted(periods.values(), key=lambda r: r["Publish Date"])
        statements.append({"statement": stmt, "data": rows})
    return {"statements": statements}
