"""SimFin **bulk** adapter (ML v2 Phase 2 — universe-scale data foundation).

Phase 1's REST loader (``data.simfin_data``) is per-ticker and the free tier 429s
after ~10 calls — useless for a 503-name cross-sectional panel. SimFin's *bulk*
datasets ship the whole US universe in a single cached download (no per-ticker
rate limit), via the ``simfin`` package. This module wraps that bulk fetch and
**re-shapes** its income / balance DataFrames into the exact ``company_block``
dict that Phase 1's :mod:`core.fundamental_features` already consumes — so the
tested point-in-time + ratio math is reused unchanged, not re-implemented.

Schema note (verified against the free annual US datasets, 2026-06-04): the bulk
columns ``Revenue / Gross Profit / Net Income`` (income) and ``Total Assets /
Total Equity / Total Liabilities`` (balance) match the field names
:func:`core.fundamental_features.quality_features` reads **exactly**, and every
row carries a ``Publish Date`` — the locked anti-look-ahead invariant. No column
mapping is needed.

Survivorship honesty (unchanged from Phase 1): the free history is *current*
filings, so a model trained on it is survivorship-biased and is judged on the
**forward** pre-registered gate, never a backtest edge. All historical rows per
ticker are preserved here because Phase 4 walk-forward needs as-of panels at past
dates; :func:`core.fundamental_features.latest_pit_rows` does the as-of selection.

The bulk network call sits behind one injectable ``fetch`` so the re-shaping logic
unit-tests on synthetic DataFrames with no network (same pattern as
``data.simfin_data``). The REST loader is kept for single-ticker spot checks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# () -> (income_df, balance_df)
BulkFetcher = Callable[[], "tuple[pd.DataFrame, pd.DataFrame]"]

_DATA_DIR = Path.home() / "AIOS" / "tmp" / "simfin_data"


def _default_bulk_fetch(market: str = "us", variant: str = "annual") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real bulk download via the ``simfin`` package (cached to :data:`_DATA_DIR`).

    Reads ``SIMFIN_API_KEY`` from the environment (loaded from ``.env``); falls back
    to the anonymous ``"free"`` key. Never logs the key.
    """
    import simfin as sf  # local import: keep the package optional for unit tests

    key = os.environ.get("SIMFIN_API_KEY", "").strip() or "free"
    sf.set_api_key(key)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    sf.set_data_dir(str(_DATA_DIR))
    income = sf.load_income(variant=variant, market=market)
    balance = sf.load_balance(variant=variant, market=market)
    return income, balance


def load_bulk(market: str = "us", variant: str = "annual",
              fetch: Optional[BulkFetcher] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the bulk ``(income_df, balance_df)`` frames.

    Args:
        market, variant: SimFin dataset selectors (default US annual).
        fetch: Injected ``() -> (income, balance)`` for tests; defaults to the real
            bulk download.

    Returns:
        ``(income_df, balance_df)`` MultiIndexed by ``(Ticker, Report Date)``.
    """
    if fetch is not None:
        return fetch()
    return _default_bulk_fetch(market=market, variant=variant)


def _rows_by_ticker(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Group a bulk statement frame into ``{ticker: [row dicts]}`` (all rows kept).

    ``Ticker`` / ``Report Date`` are restored from the index so each row dict carries
    them alongside ``Publish Date`` and the financial fields.
    """
    if df is None or df.empty:
        return {}
    flat = df.reset_index()
    out: dict[str, list[dict[str, Any]]] = {}
    for ticker, grp in flat.groupby("Ticker"):
        out[str(ticker)] = grp.to_dict("records")
    return out


def to_company_blocks(income_df: pd.DataFrame,
                      balance_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Re-shape bulk frames into ``{ticker: company_block}`` for Phase 1 features.

    Each ``company_block`` is ``{"statements": [{"statement": "PL"|"BS", "data": [rows]}]}``
    — the shape :func:`core.fundamental_features.compute_features` consumes. A ticker
    present in only one frame yields a block with just that statement (the feature
    code already tolerates a missing PL or BS). All historical rows are preserved.

    Args:
        income_df: Bulk income statement frame (the ``PL`` source).
        balance_df: Bulk balance sheet frame (the ``BS`` source).

    Returns:
        ``{ticker: company_block}`` over the union of tickers in either frame.
    """
    pl = _rows_by_ticker(income_df)
    bs = _rows_by_ticker(balance_df)
    out: dict[str, dict[str, Any]] = {}
    for ticker in set(pl) | set(bs):
        statements: list[dict[str, Any]] = []
        if ticker in pl:
            statements.append({"statement": "PL", "data": pl[ticker]})
        if ticker in bs:
            statements.append({"statement": "BS", "data": bs[ticker]})
        out[ticker] = {"statements": statements}
    return out
