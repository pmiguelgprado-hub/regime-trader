"""Tests for the HMM regime-detection engine."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.hmm_engine import LABEL_SCHEMES, HMMConfig, HMMEngine, Regime
from data.feature_engineering import RVOL_20_IDX, FeatureEngineer

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def features():
    """Standardized features from module-scoped synthetic OHLCV."""
    from conftest import make_synthetic_ohlcv

    return FeatureEngineer().build_features(make_synthetic_ohlcv())


@pytest.fixture(scope="module")
def engine(features):
    """HMM fitted once over candidates [3, 4]."""
    eng = HMMEngine(HMMConfig(n_candidates=[3, 4], n_init=4, min_train_bars=400))
    eng.fit(features)
    return eng


def test_fit_requires_min_train_bars(features) -> None:
    """fit should reject data shorter than min_train_bars."""
    eng = HMMEngine(HMMConfig(min_train_bars=10_000))
    with pytest.raises(ValueError):
        eng.fit(features)


def test_model_selection_picks_lowest_bic(engine) -> None:
    """Selected regime count must have the minimum BIC among candidates."""
    md = engine.metadata
    assert md is not None
    assert md.all_bic, "candidate BIC scores must be logged"
    best = min(md.all_bic, key=md.all_bic.get)
    assert engine.n_regimes == best
    assert md.bic == pytest.approx(md.all_bic[best])


def test_bic_formula() -> None:
    """BIC = -2*ll + p*log(N); free-param count matches full-cov formula."""
    n, d, N, ll = 3, 14, 800, -1000.0
    p = (n - 1) + n * (n - 1) + n * d + n * d * (d + 1) // 2
    expected = -2 * ll + p * np.log(N)
    assert HMMEngine._free_params(n, d) == p
    assert HMMEngine._bic(ll, n, d, N) == pytest.approx(expected)


def test_labels_sorted_by_ascending_return(engine) -> None:
    """Regime labels must follow the scheme ordered by ascending mean return."""
    scheme = LABEL_SCHEMES[engine.n_regimes]
    # order states by expected_return, map to labels, compare to scheme
    by_ret = sorted(engine.regime_info.values(), key=lambda r: r.expected_return)
    assert [r.regime_name for r in by_ret] == scheme


def test_regime_info_exposes_volatility(engine) -> None:
    """Each regime must carry expected_volatility (strategy sorts by it)."""
    for info in engine.regime_info.values():
        assert isinstance(info.expected_volatility, float)
        assert info.max_leverage_allowed >= 0.0
        assert 0.0 <= info.max_position_size_pct <= 1.0


def test_filtered_proba_is_distribution(engine, features) -> None:
    """predict_regime_proba rows are valid probability distributions."""
    proba = engine.predict_regime_proba(features)
    arr = proba.to_numpy()
    assert (arr >= -1e-9).all() and (arr <= 1 + 1e-9).all()
    np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-8)


def test_transition_matrix_rows_sum_to_one(engine) -> None:
    """Learned transition matrix rows must sum to 1."""
    tm = engine.get_transition_matrix()
    assert tm.shape == (engine.n_regimes, engine.n_regimes)
    np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-8)


def test_stability_filter_requires_consecutive_bars(engine, features) -> None:
    """A regime is confirmed only after persisting stability_bars bars."""
    states = engine.predict_regime_filtered(features)
    for s in states:
        if s.is_confirmed:
            assert s.consecutive_bars >= engine.config.stability_bars
        # first bar of any run is never confirmed (unless stability_bars==1)
    # consecutive_bars resets to 1 on a raw change
    for prev, cur in zip(states[:-1], states[1:]):
        if cur.state_id != prev.state_id:
            assert cur.consecutive_bars == 1


def test_flicker_rate_and_flag(engine, features) -> None:
    """Flicker rate counts changes in the trailing window; flag respects it."""
    engine.predict_regime_filtered(features)
    rate = engine.get_regime_flicker_rate()
    assert 0 <= rate <= engine.config.flicker_window
    assert engine.is_flickering() == (rate > engine.config.flicker_threshold)


def test_get_regime_stability_matches_last_state(engine, features) -> None:
    """get_regime_stability returns the latest run length."""
    states = engine.predict_regime_filtered(features)
    assert engine.get_regime_stability() == states[-1].consecutive_bars


def test_save_load_roundtrip(engine, features, tmp_path) -> None:
    """A reloaded engine reproduces identical filtered posteriors."""
    path = tmp_path / "model.pkl"
    engine.save(path)
    reloaded = HMMEngine.load(path)
    assert reloaded.n_regimes == engine.n_regimes
    assert reloaded.feature_columns == engine.feature_columns
    p1 = engine.predict_regime_proba(features).to_numpy()
    p2 = reloaded.predict_regime_proba(features).to_numpy()
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_predict_before_fit_raises() -> None:
    """Inference before fitting must raise."""
    with pytest.raises(RuntimeError):
        HMMEngine().get_transition_matrix()
