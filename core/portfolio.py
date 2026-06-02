"""Multi-asset portfolio allocation (E-1).

Turns a market-wide gross-exposure *budget* (set by the volatility regime) into
per-symbol target weights, under the existing risk caps. v1 is equal-weight; the
``weighting`` hook reserves room for vol-parity / trend-weighting later.

Design (see docs/analysis/2026-06-02-optimization-and-roadmap.md §2):
the HMM regime sets HOW MUCH gross exposure the whole book runs; this function
decides HOW it is spread across names. The per-name ``max_single`` and the
``max_concurrent`` count are the multi-asset risk caps (M4/M5) made concrete.
"""

from __future__ import annotations


def portfolio_target_weights(
    gross_budget: float,
    symbols: list[str],
    max_single: float,
    max_concurrent: int,
    weighting: str = "equal",
) -> dict[str, float]:
    """Per-symbol target weights for a gross-exposure budget.

    Args:
        gross_budget: Total gross exposure to allocate (e.g. 0.95 * leverage).
        symbols: Candidate universe (order = priority for the concurrency cap).
        max_single: Max weight any single symbol may hold.
        max_concurrent: Max number of simultaneous holdings.
        weighting: Allocation scheme (``"equal"`` only in v1).

    Returns:
        ``{symbol: weight}`` for the selected names (empty if no budget/universe).
    """
    if gross_budget <= 0 or not symbols or max_concurrent <= 0:
        return {}
    selected = symbols[:max_concurrent]
    if weighting != "equal":  # pragma: no cover - reserved for vol-parity/trend
        raise ValueError(f"unsupported weighting: {weighting}")
    share = gross_budget / len(selected)
    weight = min(share, max_single)
    return {s: weight for s in selected}
