"""Cross-sectional feature panel (ML v2 — Phase 2 deliverable).

Assembles the model's input table: one **row per universe name**, columns =
the v1 price **momentum** (the alpha spine) plus the 5 Phase 1 **quality** factors,
evaluated **point-in-time** at an as-of date. This panel is what the Phase 3 model
(gradient-boosted trees, per the locked plan) will learn the cross-sectional return
ranking from — replacing the rules-only v1 signal.

No re-derived math: momentum reuses :func:`core.cross_sectional_ranking.momentum_score`
and the quality factors reuse :func:`core.fundamental_features.compute_features`. The
panel just joins them on the ticker. Value (earnings/book yield) is deliberately **not**
included yet — it needs shares-outstanding and expands degrees of freedom before the
model exists, and overfit is the stated project risk; it is a later explicit decision.

Survivorship/forward-only framing is inherited from the data layer (see
:mod:`data.simfin_bulk`): this panel trains a forward-deployed model judged by the
pre-registered gate, never a backtest edge claim.

Pure + unit-tested: prices and fundamental blocks are injected, no network.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import pandas as pd

from core.cross_sectional_ranking import DEFAULT_LOOKBACK, DEFAULT_SKIP, momentum_score
from core.fundamental_features import compute_features

# The 5 Phase 1 quality factors, in panel order.
QUALITY_COLUMNS = ["gross_profitability", "roe", "roa", "gross_margin", "leverage"]
PANEL_COLUMNS = ["momentum"] + QUALITY_COLUMNS


def build_panel(
    close_frames: dict[str, pd.DataFrame],
    company_blocks: dict[str, dict[str, Any]],
    asof: Optional[date] = None,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    drop_incomplete: bool = False,
) -> pd.DataFrame:
    """Build the cross-sectional feature panel.

    The momentum spine drives the row set: a name needs a price history to be in the
    book at all (it is the v1 alpha). Quality factors are joined by ticker; a name with
    no fundamentals block — or a factor with missing inputs — is ``NaN`` (the model /
    ranker handles missing the same way v1 drops un-scorable names).

    Args:
        close_frames: ``{ticker: OHLCV}`` with a ``close`` column, each ending at the
            decision bar (the momentum source; e.g. from ``data.constituents.load_many``).
        company_blocks: ``{ticker: company_block}`` from
            :func:`data.simfin_bulk.to_company_blocks` (the quality source).
        asof: Point-in-time date for the fundamentals (defaults to today). Only filings
            published on or before this date are used.
        lookback, skip: Momentum window (12-1 by default).
        drop_incomplete: If True, keep only rows with a non-NaN momentum **and** all 5
            quality factors present (a fully populated training row).

    Returns:
        DataFrame indexed by ``ticker`` (name ``"ticker"``), columns
        :data:`PANEL_COLUMNS`, sorted by ticker for determinism.
    """
    rows: dict[str, dict[str, float]] = {}
    for ticker, df in close_frames.items():
        mom = momentum_score(df["close"], lookback=lookback, skip=skip)
        block = company_blocks.get(ticker)
        feats = compute_features(block, asof) if block else None
        row: dict[str, Any] = {"momentum": mom}
        for col in QUALITY_COLUMNS:
            row[col] = (feats or {}).get(col)
        rows[ticker] = row

    panel = pd.DataFrame.from_dict(rows, orient="index")
    if panel.empty:
        panel = pd.DataFrame(columns=PANEL_COLUMNS)
    else:
        panel = panel.reindex(columns=PANEL_COLUMNS)
    panel.index.name = "ticker"
    panel = panel.sort_index()

    if drop_incomplete:
        panel = panel.dropna(subset=PANEL_COLUMNS)
    return panel


def coverage(panel: pd.DataFrame) -> dict[str, int]:
    """Report panel completeness — the real-build sanity check (unit-green != panel-works).

    Args:
        panel: A panel from :func:`build_panel` (NaNs intact, i.e. not yet filtered).

    Returns:
        ``{"universe": rows, "with_momentum": rows with non-NaN momentum,
        "complete": rows with momentum + all 5 quality factors}``.
    """
    if panel.empty:
        return {"universe": 0, "with_momentum": 0, "complete": 0}
    with_mom = int(panel["momentum"].notna().sum())
    complete = int(panel[PANEL_COLUMNS].notna().all(axis=1).sum())
    return {"universe": int(len(panel)), "with_momentum": with_mom, "complete": complete}
