"""Drift-detection primitives for continuous learning (A-3).

Pure functions used to decide when the live HMM has drifted enough to warrant a
retrain, *between* the age-based checks. Two signals:

* **Population Stability Index (PSI)** — how far the live feature distribution
  has moved from the training distribution. Rule of thumb: ``< 0.10`` stable,
  ``0.10–0.25`` moderate shift, ``> 0.25`` significant shift.
* **Normalized posterior entropy** — how uncertain the regime classifier is,
  scaled to ``[0, 1]`` (0 = a single state is certain, 1 = uniform/maximally
  uncertain). Rising entropy means the model no longer fits the data cleanly.

The thin :func:`drift_triggers_retrain` predicate combines the two so callers
(the live loop, A-1) stay declarative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """Population Stability Index between two 1-D samples.

    Bins are derived from quantiles of ``expected`` (so the reference defines
    the partition), then both samples' proportions are compared. A small
    epsilon floors empty bins to keep the log finite.

    Args:
        expected: Reference sample (e.g. training-window feature values).
        actual: Current sample (e.g. recent live feature values).
        bins: Number of quantile bins.

    Returns:
        PSI (>= 0). Larger means more distributional shift.
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()

    # quantile edges from the reference; widen the outer edges to catch tails
    edges = np.quantile(expected, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:  # degenerate (constant) reference
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_prop = e_counts / max(e_counts.sum(), 1) + _EPS
    a_prop = a_counts / max(a_counts.sum(), 1) + _EPS

    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def max_feature_psi(
    expected: pd.DataFrame, actual: pd.DataFrame, bins: int = 10
) -> float:
    """Worst-case PSI across the feature columns shared by two frames.

    Computes PSI per shared column and returns the maximum, so a single
    strongly-drifted feature is enough to flag drift.

    Args:
        expected: Reference feature frame (training window).
        actual: Current feature frame (recent live window).
        bins: Quantile bins per column.

    Returns:
        Max per-column PSI, or ``0.0`` if no columns are shared.
    """
    cols = [c for c in expected.columns if c in actual.columns]
    if not cols:
        return 0.0
    return max(
        population_stability_index(expected[c].to_numpy(), actual[c].to_numpy(), bins)
        for c in cols
    )


def normalized_entropy(proba: np.ndarray) -> float:
    """Shannon entropy of a posterior distribution, scaled to ``[0, 1]``.

    Args:
        proba: Probability vector over regimes (need not be exactly normalized;
            it is renormalized defensively). Length >= 1.

    Returns:
        ``0.0`` when all mass is on one state, ``1.0`` for a uniform
        distribution. A length-1 distribution returns ``0.0``.
    """
    p = np.asarray(proba, dtype=float).ravel()
    p = p / max(p.sum(), _EPS)
    n = p.size
    if n <= 1:
        return 0.0
    nz = p[p > 0.0]
    entropy = -np.sum(nz * np.log(nz))
    return float(entropy / np.log(n))


def drift_triggers_retrain(
    psi: float, entropy: float, psi_threshold: float, entropy_threshold: float
) -> bool:
    """Whether observed drift warrants an out-of-cycle retrain.

    Args:
        psi: Latest feature-PSI value.
        entropy: Latest normalized posterior entropy.
        psi_threshold: PSI above which a retrain is requested.
        entropy_threshold: Entropy above which a retrain is requested.

    Returns:
        True if either signal breaches its threshold.
    """
    return psi > psi_threshold or entropy > entropy_threshold


def recent_vs_prior_psi(panel, window: int = 126, bins: int = 5):
    """Max feature PSI of the last ``window`` bars vs the ``window`` before them.

    A self-contained drift gauge for the live loop: no stored training snapshot
    needed — compare the recent window against the immediately prior one. Returns
    ``0.0`` when there are fewer than ``2*window`` rows (not enough history to judge).

    Defaults (window 126 ≈ 6 months, 5 bins) keep the small-sample noise floor low:
    empirically max PSI ≈ 0.21 between two windows from the same distribution, so a
    drift threshold of ~0.30+ sits clear of noise. Finer bins on short windows
    inflate PSI through empty-bin terms.

    Args:
        panel: Feature DataFrame (datetime-indexed, model feature columns).
        window: Bars in each comparison window.
        bins: Quantile bins per feature.

    Returns:
        Max per-column PSI between the two windows.
    """
    if panel is None or len(panel) < 2 * window:
        return 0.0
    prior = panel.iloc[-2 * window:-window]
    recent = panel.iloc[-window:]
    return max_feature_psi(prior, recent, bins)
