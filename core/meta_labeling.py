"""Triple-barrier meta-labeling pipeline (T3.1, Lopez de Prado 2018, ch. 3).

Meta-labeling trains a *secondary* model to decide whether to act on (and how
large to size) the *primary* model's signals — trading recall for precision,
which lifts Sharpe/F1 without touching the primary alpha. The secondary model
needs labels, and labels need closed trades: at the current cadence ~200
round-trips is 12-18 months out. So we build the LABELING now, accumulate during
the gates, and train the model later under its own pre-registration (sizing-only,
roadmap §0).

Each entry is labeled by the first barrier its forward path touches:

* ``+1`` — the upper (profit-take) barrier at ``+pt``,
* ``-1`` — the lower (stop-loss) barrier at ``-sl``,
* ``0``  — neither within ``max_hold`` bars (the vertical/time barrier).

Pure + deterministic; the accumulation glue (entry detection from fills) is thin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

COLUMNS = ["entry_date", "symbol", "label", "barrier", "ret", "holding_days"]


def triple_barrier_label(prices: pd.Series, entry_i: int, pt: float, sl: float,
                         max_hold: int) -> dict:
    """Label one entry by the first of {profit-take, stop-loss, time} barriers hit.

    Args:
        prices: Price series (ascending time index).
        entry_i: Integer position of the entry bar in ``prices``.
        pt: Profit-take threshold (e.g. 0.05 = +5%); upper barrier at ``entry*(1+pt)``.
        sl: Stop-loss threshold (e.g. 0.05 = -5%); lower barrier at ``entry*(1-sl)``.
        max_hold: Vertical (time) barrier in bars.

    Returns:
        ``{label, barrier, ret, holding_days}`` for the realized path. The path is
        scanned bar by bar from ``entry_i+1``; ``pt`` is checked before ``sl`` on a
        bar that breaches both (long book — the up move is the realized exit first
        only when it occurs first; on a single bar breaching both, pt is assumed
        touched first, the standard long convention).
    """
    entry = float(prices.iloc[entry_i])
    up, dn = entry * (1.0 + pt), entry * (1.0 - sl)
    end = min(entry_i + max_hold, len(prices) - 1)
    for j in range(entry_i + 1, end + 1):
        p = float(prices.iloc[j])
        if p >= up:
            return {"label": 1, "barrier": "pt", "ret": p / entry - 1.0,
                    "holding_days": j - entry_i}
        if p <= dn:
            return {"label": -1, "barrier": "sl", "ret": p / entry - 1.0,
                    "holding_days": j - entry_i}
    p = float(prices.iloc[end])
    return {"label": 0, "barrier": "time", "ret": p / entry - 1.0,
            "holding_days": end - entry_i}


def label_events(prices: pd.Series, event_positions: list[int], pt: float, sl: float,
                 max_hold: int, symbol: str = "") -> pd.DataFrame:
    """Triple-barrier label a set of entry positions -> one row per event."""
    rows = []
    for i in event_positions:
        lab = triple_barrier_label(prices, i, pt, sl, max_hold)
        rows.append({"entry_date": str(prices.index[i])[:10], "symbol": symbol,
                     **lab})
    return pd.DataFrame(rows, columns=COLUMNS)


def append_labels(path: str, rows: list[dict]) -> int:
    """Append label rows, skipping any whose (entry_date, symbol) already exists.

    Dedup makes the monthly labeling job idempotent — re-running it cannot
    double-count a cohort. Returns the number of NEW rows written.
    """
    if not rows:
        return 0
    p = Path(path)
    existing = pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else pd.DataFrame(columns=COLUMNS)
    seen = {(str(r["entry_date"]), str(r["symbol"]))
            for _, r in existing.iterrows()} if not existing.empty else set()
    fresh = [r for r in rows if (str(r["entry_date"]), str(r.get("symbol", ""))) not in seen]
    if not fresh:
        return 0
    out = pd.concat([existing, pd.DataFrame(fresh, columns=COLUMNS)], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return len(fresh)


def label_count(path: str) -> int:
    """Number of accumulated labels (0 if the store is absent)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    return int(len(pd.read_csv(p)))


def ready_to_train(path: str, min_labels: int = 200) -> bool:
    """Whether enough round-trips have accumulated to fit the secondary model.

    The threshold is deliberately high: a meta-model fit on a handful of trades
    overfits. Until this is True, accumulate — do not train (roadmap §T3.1).
    """
    return label_count(path) >= min_labels
