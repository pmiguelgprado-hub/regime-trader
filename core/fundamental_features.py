"""Fundamental cross-sectional features (ML v2 — Phase 1: features).

Turns SimFin statements into a small set of **quality / profitability** factors per
company, to complement the v1 price momentum signal. These are among the most robust
documented cross-sectional predictors and — crucially — need only the income statement
(PL) and balance sheet (BS), no market cap, so there is no price-data dependency here:

- ``gross_profitability`` = Gross Profit / Total Assets  (Novy-Marx 2013 — strong factor)
- ``roe``                 = Net Income / Total Equity
- ``roa``                 = Net Income / Total Assets
- ``gross_margin``        = Gross Profit / Revenue
- ``leverage``            = Total Liabilities / Total Assets   (lower is better)

**Point-in-time (anti look-ahead):** for an as-of date, each statement is the most recent
period whose **Publish Date <= as-of** — i.e. data the market actually had. Using the
fiscal Report Date instead would leak ~1-3 months of future knowledge.

Pure + unit-tested (no network); the SimFin fetch lives in ``data.simfin_data``. Phase 2
(the model) consumes a cross-sectional panel of these features + momentum.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def _pub_date(row: dict[str, Any]) -> Optional[date]:
    """Parse a row's Publish Date (the point-in-time gate); None if absent/bad."""
    raw = row.get("Publish Date")
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def latest_pit_rows(company_block: dict[str, Any],
                    asof: Optional[date] = None) -> dict[str, dict]:
    """Pick the latest PL/BS rows published on or before ``asof`` (point-in-time).

    Args:
        company_block: One company entry from :func:`data.simfin_data.statements`
            (has a ``statements`` list of ``{statement, data:[rows]}``).
        asof: As-of date (defaults to today). Rows published after it are excluded.

    Returns:
        ``{"PL": row, "BS": row}`` for the statements present; a statement with no
        publish-eligible row is omitted.
    """
    asof = asof or date.today()
    out: dict[str, dict] = {}
    for blk in company_block.get("statements", []):
        eligible = [r for r in blk.get("data", [])
                    if (_pub_date(r) is not None and _pub_date(r) <= asof)]
        if eligible:
            out[blk["statement"]] = max(eligible, key=lambda r: _pub_date(r))
    return out


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Divide, returning None on missing values or a zero/None denominator."""
    if num is None or den in (None, 0):
        return None
    return num / den


def quality_features(rows: dict[str, dict]) -> dict[str, Optional[float]]:
    """Compute the quality/profitability ratios from point-in-time PL/BS rows.

    Args:
        rows: ``{"PL": row, "BS": row}`` from :func:`latest_pit_rows`.

    Returns:
        ``{factor: value_or_None}``. A factor is None when its inputs are missing
        (the ranker drops names with missing features, like momentum does).
    """
    pl, bs = rows.get("PL", {}), rows.get("BS", {})
    revenue = pl.get("Revenue")
    gross = pl.get("Gross Profit")
    net = pl.get("Net Income")
    assets = bs.get("Total Assets")
    equity = bs.get("Total Equity")
    liabilities = bs.get("Total Liabilities")
    return {
        "gross_profitability": _safe_div(gross, assets),
        "roe": _safe_div(net, equity),
        "roa": _safe_div(net, assets),
        "gross_margin": _safe_div(gross, revenue),
        "leverage": _safe_div(liabilities, assets),
    }


def compute_features(company_block: dict[str, Any],
                     asof: Optional[date] = None) -> Optional[dict[str, float]]:
    """End-to-end: SimFin company block -> point-in-time quality features.

    Args:
        company_block: One entry from :func:`data.simfin_data.statements`.
        asof: As-of date (defaults to today).

    Returns:
        ``{factor: value}`` (None values kept) or ``None`` if no PL/BS is available
        as of the date.
    """
    rows = latest_pit_rows(company_block, asof)
    if "PL" not in rows and "BS" not in rows:
        return None
    return quality_features(rows)
