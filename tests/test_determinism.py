"""Determinism regression tests (T0.4, R-4 audit).

R-4: identical runs in *separate processes* produced Sharpe in a 0.37-0.49 band.
Per-process seeding was already deterministic (``random_state + i`` per restart),
so the variance must enter through the environment: BLAS/OMP threading, data
end-date = "now", or near-tie restart selection. These tests pin the in-process
guarantee — double fit on identical data must be **bit-identical**, not merely
close — so any future nondeterminism (library upgrade, threading change, code
edit) fails loudly instead of silently widening the band. The cross-process
environment is pinned operationally: ``OMP_NUM_THREADS=1`` /
``OPENBLAS_NUM_THREADS=1`` in every launchd plist (same T0.4 change).

``transition_hash`` is the audit primitive: a short content hash of the fitted
transition matrix, logged on every live run, so champion drift is detectable
from logs alone (and the daily champion-unchanged assert has something to compare).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.hmm_engine import HMMConfig, HMMEngine
from data.feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def features():
    from conftest import make_synthetic_ohlcv

    return FeatureEngineer().build_features(make_synthetic_ohlcv())


def _fit(features) -> HMMEngine:
    eng = HMMEngine(HMMConfig(n_candidates=[3], n_init=3, min_train_bars=400))
    eng.fit(features)
    return eng


@pytest.fixture(scope="module")
def twin_engines(features):
    """Two engines fitted independently on identical data and config."""
    return _fit(features), _fit(features)


def test_double_fit_identical_parameters(twin_engines) -> None:
    a, b = twin_engines
    assert a.n_regimes == b.n_regimes
    np.testing.assert_array_equal(a.model.transmat_, b.model.transmat_)
    np.testing.assert_array_equal(a.model.means_, b.model.means_)
    np.testing.assert_array_equal(a.model.startprob_, b.model.startprob_)


def test_double_fit_identical_posteriors(twin_engines, features) -> None:
    a, b = twin_engines
    pa = a.predict_regime_proba(features)
    pb = b.predict_regime_proba(features)
    np.testing.assert_array_equal(pa.to_numpy(), pb.to_numpy())


def test_double_fit_identical_labels(twin_engines) -> None:
    a, b = twin_engines
    assert a.labels == b.labels
    assert a.metadata.bic == b.metadata.bic


def test_transition_hash_stable_across_refits(twin_engines) -> None:
    a, b = twin_engines
    assert a.transition_hash() == b.transition_hash()
    assert len(a.transition_hash()) == 16          # short digest, log-friendly


def test_transition_hash_detects_drift(twin_engines) -> None:
    a, _ = twin_engines
    before = a.transition_hash()
    a.model.transmat_ = a.model.transmat_ + 1e-9   # any drift, however small
    try:
        assert a.transition_hash() != before
    finally:
        a.model.transmat_ = a.model.transmat_ - 1e-9


def test_transition_hash_requires_fitted_model() -> None:
    with pytest.raises(RuntimeError):
        HMMEngine(HMMConfig()).transition_hash()
