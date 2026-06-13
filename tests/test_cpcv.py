"""Tests for Combinatorial Purged Cross-Validation (T4.3, Lopez de Prado).

The walk-forward 504/126/126 is a SINGLE path -> high-variance verdicts. CPCV
splits the sample into G groups, holds out every k-group combination as test
(C(G,k) test sets, many backtest paths), purges train observations that overlap
test, and embargoes the bars just after each test block. The output is a
DISTRIBUTION of the metric, not a point estimate. Complements pbo_cscv.
"""

from __future__ import annotations

import numpy as np

from backtest import cpcv


def test_combinatorial_splits_count():
    # C(6, 2) = 15 test combinations
    splits = cpcv.combinatorial_splits(n_groups=6, k_test=2)
    assert len(splits) == 15
    assert all(len(s) == 2 for s in splits)
    assert all(0 <= g < 6 for s in splits for g in s)


def test_purged_train_test_partitions_without_leak():
    train, test = cpcv.purged_train_test(n_obs=100, n_groups=10, test_groups=(2, 5),
                                         embargo=0)
    train, test = set(train.tolist()), set(test.tolist())
    assert train.isdisjoint(test)                      # no observation in both
    assert len(test) == 20                             # 2 groups x 10 obs


def test_embargo_drops_bars_after_test_block():
    no_emb, _ = cpcv.purged_train_test(100, 10, (5,), embargo=0)
    emb, _ = cpcv.purged_train_test(100, 10, (5,), embargo=3)
    # embargo removes additional train bars immediately after the test block
    assert len(emb) < len(no_emb)


def test_sharpe_paths_distribution_shape():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, 260)
    paths = cpcv.cpcv_sharpe_paths(rets, n_groups=8, k_test=2, embargo=1)
    from math import comb
    assert len(paths) == comb(8, 2)
    assert np.isfinite(paths).all()


def test_summary_reports_mean_and_prob_negative():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.004, 0.008, 400)               # strong positive drift (Sharpe ~0.5/bar)
    s = cpcv.cpcv_summary(rets, n_groups=8, k_test=2)
    assert "mean_sharpe" in s and "prob_negative" in s and "n_paths" in s
    assert 0.0 <= s["prob_negative"] <= 1.0
    assert s["mean_sharpe"] > 0                         # positive-drift series
    assert s["prob_negative"] < 0.5                     # most paths positive


def test_too_few_obs_returns_empty():
    assert cpcv.cpcv_sharpe_paths(np.array([0.01, 0.02]), n_groups=8, k_test=2).size == 0
