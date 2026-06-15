"""Tests for split-conformal prediction (T3.2 study core).

Distribution-free prediction intervals with finite-sample coverage: calibrate a
nonconformity quantile on held-out data, then every interval is point ± q. The
study (separate doc) asks whether conformal interval WIDTH adds anything over the
existing meta_overlay predictive entropy as a gross-scaling uncertainty signal —
falsify before building. This is the reusable core; pure + tested.
"""

from __future__ import annotations

import numpy as np

from core import conformal as cf


def test_quantile_finite_sample_correction():
    scores = list(range(1, 101))                       # 1..100
    q = cf.split_conformal_quantile(scores, alpha=0.1)
    # ceil((100+1)*0.9)/100 -> the ~91st order statistic
    assert 90 <= q <= 92


def test_quantile_alpha_monotone():
    rng = np.random.default_rng(0)
    scores = list(np.abs(rng.normal(0, 1, 500)))
    q90 = cf.split_conformal_quantile(scores, alpha=0.1)
    q80 = cf.split_conformal_quantile(scores, alpha=0.2)
    assert q90 >= q80                                  # higher coverage -> wider


def test_interval_brackets_point():
    lo, hi = cf.conformal_interval(10.0, q=2.0)
    assert lo == 8.0 and hi == 12.0


def test_empirical_coverage_near_target():
    rng = np.random.default_rng(1)
    cal = list(np.abs(rng.normal(0, 1, 1000)))         # |residual| calibration scores
    q = cf.split_conformal_quantile(cal, alpha=0.1)
    test_resid = rng.normal(0, 1, 2000)
    covered = cf.coverage([cf.conformal_interval(0.0, q)] * len(test_resid), test_resid)
    assert 0.85 <= covered <= 0.95                     # ~90% target

def test_quantile_empty_is_inf():
    assert cf.split_conformal_quantile([], alpha=0.1) == float("inf")


def test_normalized_interval_width():
    # width scales with the local uncertainty estimate
    narrow = cf.conformal_interval(0.0, q=1.0)
    wide = cf.conformal_interval(0.0, q=3.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
