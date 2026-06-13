"""Combinatorial Purged Cross-Validation (T4.3, Lopez de Prado 2018, ch. 12).

The project's walk-forward (504/126/126) is a *single* train/test path, so its
Sharpe verdict has high variance — one lucky or unlucky test window can flip it.
CPCV instead splits the sample into ``n_groups`` contiguous groups, holds out
every ``k_test``-group combination as the test set (``C(n_groups, k_test)`` of
them), and reports the **distribution** of the test metric across all those paths.
Two leakage controls, as in LdP:

* **Purge** — a train observation is dropped if it falls inside any held-out test
  group (no overlap between train and test).
* **Embargo** — additionally drop the ``embargo`` train bars immediately *after*
  each test block, where serial correlation would otherwise leak test information
  into training.

Pure + deterministic; complements :func:`backtest.performance.pbo_cscv` (which
estimates the probability of backtest overfitting). Operates on a realized
per-period return series — the strategy's own returns — so it is cheap to run on
the accumulating track record or any candidate's backtest output.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np


def combinatorial_splits(n_groups: int, k_test: int) -> list[tuple[int, ...]]:
    """All ``C(n_groups, k_test)`` choices of test-group indices."""
    return list(combinations(range(n_groups), k_test))


def _group_bounds(n_obs: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal [start, end) index bounds for each group."""
    edges = np.linspace(0, n_obs, n_groups + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def purged_train_test(n_obs: int, n_groups: int, test_groups: "tuple[int, ...]",
                      embargo: int = 0) -> "tuple[np.ndarray, np.ndarray]":
    """Train/test index arrays for one CPCV path, purged + embargoed.

    Args:
        n_obs: Number of observations.
        n_groups: Number of contiguous groups to split into.
        test_groups: Group indices held out as test.
        embargo: Train bars to drop immediately after each test block.

    Returns:
        ``(train_idx, test_idx)`` sorted int arrays (disjoint).
    """
    bounds = _group_bounds(n_obs, n_groups)
    test_mask = np.zeros(n_obs, dtype=bool)
    embargo_mask = np.zeros(n_obs, dtype=bool)
    for g in test_groups:
        s, e = bounds[g]
        test_mask[s:e] = True
        if embargo > 0:
            embargo_mask[e:min(n_obs, e + embargo)] = True
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask & ~embargo_mask)[0]
    return train_idx, test_idx


def _sharpe(rets: np.ndarray) -> float:
    """Per-bar Sharpe of a return slice (0.0 if degenerate)."""
    if rets.size < 2:
        return 0.0
    sd = rets.std(ddof=1)
    return float(rets.mean() / sd) if sd > 0 else 0.0


def cpcv_sharpe_paths(returns, n_groups: int = 8, k_test: int = 2,
                      embargo: int = 1) -> np.ndarray:
    """Test-set per-bar Sharpe across every combinatorial path (the distribution).

    Returns an empty array when there are too few observations to form the groups.
    """
    rets = np.asarray(returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size < n_groups:
        return np.array([])
    paths = []
    for test_groups in combinatorial_splits(n_groups, k_test):
        _, test_idx = purged_train_test(rets.size, n_groups, test_groups, embargo)
        if test_idx.size >= 2:
            paths.append(_sharpe(rets[test_idx]))
    return np.array(paths)


def cpcv_summary(returns, n_groups: int = 8, k_test: int = 2,
                 embargo: int = 1) -> dict:
    """Distribution summary of the CPCV test Sharpe paths.

    Returns:
        ``{n_paths, mean_sharpe, std_sharpe, p05, p95, prob_negative}`` (zeros when
        no path could be formed). ``prob_negative`` — the share of paths with a
        negative test Sharpe — is the headline robustness number.
    """
    paths = cpcv_sharpe_paths(returns, n_groups, k_test, embargo)
    if paths.size == 0:
        return {"n_paths": 0, "mean_sharpe": 0.0, "std_sharpe": 0.0,
                "p05": 0.0, "p95": 0.0, "prob_negative": 0.0}
    return {
        "n_paths": int(paths.size),
        "mean_sharpe": float(paths.mean()),
        "std_sharpe": float(paths.std(ddof=1)) if paths.size > 1 else 0.0,
        "p05": float(np.percentile(paths, 5)),
        "p95": float(np.percentile(paths, 95)),
        "prob_negative": float((paths < 0).mean()),
    }
