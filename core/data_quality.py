"""Daily data-quality sentinel (gap 5).

A single day of corrupt market data during a 12-month forward gate poisons the
evidence irreversibly — a bad split, a stale feed, a NaN-riddled panel. This
module is the daily check that catches it *before* it lands in the track record:
pure, injectable functions returning structured :class:`Issue` records, plus a
``summary`` for one-line alerting. The alert dispatch is thin glue in ``main``.

Checks are deliberately conservative (flag, don't auto-correct) — the point is to
make a bad-data day loud, so the operator can quarantine that row, not to silently
patch it (which would itself contaminate the gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Issue:
    """One data-quality finding."""

    symbol: str
    kind: str          # empty | stale | jump | nonpositive | nan_rate
    detail: str


def check_price_series(symbol: str, df: pd.DataFrame, asof: str,
                       max_stale_bdays: int = 3, max_abs_ret: float = 0.40) -> list[Issue]:
    """Flag staleness, anomalous one-day returns, and non-positive prices.

    Args:
        symbol: Ticker (for the issue records).
        df: OHLCV frame with a ``close`` column, datetime-indexed ascending.
        asof: ISO date the data is being used as-of (staleness reference).
        max_stale_bdays: Business days the last bar may lag ``asof`` before stale.
        max_abs_ret: One-day |return| above which a bar is a likely split/feed error.

    Returns:
        List of :class:`Issue` (empty when clean).
    """
    issues: list[Issue] = []
    if df is None or len(df) == 0 or "close" not in df:
        return [Issue(symbol, "empty", "no price data")]
    close = df["close"].astype(float)

    last_date = str(df.index[-1])[:10]
    if last_date < asof:
        lag = max(0, len(pd.bdate_range(last_date, asof)) - 1)
        if lag > max_stale_bdays:
            issues.append(Issue(symbol, "stale",
                                f"last bar {last_date} is {lag} bdays before {asof}"))

    if (close <= 0).any():
        issues.append(Issue(symbol, "nonpositive",
                            f"{int((close <= 0).sum())} non-positive close(s)"))

    rets = close.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    big = rets[rets.abs() > max_abs_ret]
    if not big.empty:
        worst = float(big.iloc[big.abs().to_numpy().argmax()])
        issues.append(Issue(symbol, "jump",
                            f"{len(big)} bar(s) |ret|>{max_abs_ret:.0%}; worst {worst:+.1%}"))
    return issues


def panel_nan_rate(panel: pd.DataFrame) -> float:
    """Fraction of NaN cells in a feature/price panel (0.0 for an empty panel)."""
    if panel is None or panel.size == 0:
        return 0.0
    return float(panel.isna().to_numpy().sum()) / float(panel.size)


def check_panel(panel: pd.DataFrame, max_nan_rate: float = 0.30) -> list[Issue]:
    """Flag a panel whose NaN rate exceeds ``max_nan_rate``."""
    rate = panel_nan_rate(panel)
    if rate > max_nan_rate:
        return [Issue("<panel>", "nan_rate", f"NaN rate {rate:.1%} > {max_nan_rate:.0%}")]
    return []


def summary(issues: list[Issue]) -> dict[str, int]:
    """Count issues by kind (for a one-line alert message)."""
    out: dict[str, int] = {}
    for i in issues:
        out[i.kind] = out.get(i.kind, 0) + 1
    return out
