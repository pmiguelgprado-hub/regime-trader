"""Tests for Bayesian Online Changepoint Detection (T1.2, Adams-MacKay).

Model-free corroborator of the meta_overlay hazard (which is model-internal to the
HMM). Maintains a run-length posterior with a constant hazard and a Normal-inverse-
Gamma (Student-t) predictive; the changepoint score is the probability mass on a
fresh run. Shadow only. Tests pin: a mean shift spikes the score; a constant series
does not; outputs are valid probabilities.
"""

from __future__ import annotations

import numpy as np

from core.changepoint import bocpd, changepoint_score


def test_output_length_and_range():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    cp = bocpd(x, hazard_lambda=50.0)
    assert len(cp) == len(x)
    assert np.all((cp >= 0.0) & (cp <= 1.0))


def test_mean_shift_spikes_changepoint_prob():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0.0, 0.5, 80), rng.normal(5.0, 0.5, 80)])
    cp = bocpd(x, hazard_lambda=100.0)
    # the largest changepoint probability should land near the true break (t=80)
    peak = int(np.argmax(cp[1:]) + 1)
    assert 78 <= peak <= 90
    assert cp[peak] > cp[40]                            # spike vs a stable point


def test_constant_series_low_changepoint_prob():
    x = np.zeros(60)
    cp = bocpd(x, hazard_lambda=100.0)
    assert np.max(cp[5:]) < 0.5                         # no spurious changepoints


def test_changepoint_score_is_latest_value():
    rng = np.random.default_rng(2)
    # a single large outlier as the FINAL bar — the latest-bar score must spike,
    # since the changepoint probability peaks AT the break, not bars after it.
    stable = rng.normal(0, 0.5, 50)
    jumped = np.concatenate([stable[:-1], [8.0]])
    s_jump = changepoint_score(jumped, hazard_lambda=100.0)
    s_stable = changepoint_score(stable, hazard_lambda=100.0)
    assert 0.0 <= s_jump <= 1.0
    assert s_jump > s_stable


def test_too_short_series_returns_zeros():
    cp = bocpd(np.array([1.0]), hazard_lambda=100.0)
    assert len(cp) == 1 and cp[0] == 0.0
