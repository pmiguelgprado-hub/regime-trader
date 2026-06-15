"""Split-conformal prediction (T3.2 study core).

Distribution-free prediction intervals with a finite-sample coverage guarantee
(Vovk; Angelopoulos-Bates). Calibrate the (1-alpha) quantile of nonconformity
scores on a held-out set; then any prediction interval is ``point ± q`` and covers
the truth with probability ≥ 1-alpha regardless of the data distribution.

Intended use is a **study, not a deployment**: does the conformal interval *width*
(an uncertainty measure) add anything over the meta_overlay predictive entropy the
book already computes, as a signal for scaling gross down when the regime call is
uncertain? The honest test is to falsify — the two may capture the same thing. The
study plan lives in docs/analysis/2026-06-14-conformal-study.md; this is the pure,
tested machinery it will use once the shadow logs accumulate.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def split_conformal_quantile(calibration_scores: Sequence[float], alpha: float = 0.1) -> float:
    """The conformal quantile of nonconformity scores (finite-sample corrected).

    Uses the ``ceil((n+1)(1-alpha))/n`` order statistic, which gives the marginal
    coverage guarantee ≥ 1-alpha. Empty calibration -> ``inf`` (no finite interval).

    Args:
        calibration_scores: Held-out nonconformity scores (e.g. |residuals|).
        alpha: Miscoverage level (0.1 -> 90% intervals).

    Returns:
        The quantile ``q`` (interval half-width for symmetric scores).
    """
    s = np.sort(np.asarray(list(calibration_scores), dtype=float))
    n = s.size
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank >= n:                                      # not enough data for the guarantee
        return float(s[-1])
    return float(s[rank - 1])


def conformal_interval(point: float, q: float) -> "tuple[float, float]":
    """Symmetric prediction interval ``[point - q, point + q]``."""
    return (point - q, point + q)


def coverage(intervals: Sequence["tuple[float, float]"], actuals: Sequence[float]) -> float:
    """Empirical coverage: fraction of actuals inside their interval (0.0 if empty)."""
    if not len(actuals):
        return 0.0
    hits = sum(lo <= a <= hi for (lo, hi), a in zip(intervals, actuals))
    return hits / len(actuals)
