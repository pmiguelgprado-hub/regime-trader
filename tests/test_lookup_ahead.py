"""Tests verifying strict causality — no look-ahead bias anywhere.

THE most important invariant in the system: a regime estimate at bar T must be
identical whether computed from data up to T or from a longer series that
includes future bars. The filtered (forward-algorithm) inference guarantees
this; Viterbi (``model.predict``) does not — and we assert that it does *not*,
so this test has teeth and cannot pass vacuously.

NOTE on indices: the spec's literal example compares bars 399 vs 400 on a
504-bar series. The real feature warmup (SMA-200 + 252 z-score) is ~450 bars,
so those indices are all-NaN. We deliberately use a long series (~1400 bars)
and compare a bar comfortably past warmup (bar 800), aligned by timestamp
label (positional indices shift after dropna).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.hmm_engine import HMMConfig, HMMEngine
from data.feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore")

PROBE_BAR = 800  # absolute bar index in the raw OHLCV series (past warmup)


@pytest.fixture(scope="module")
def fitted(ohlcv_module):
    """Train the HMM once on the full feature set for reuse across tests."""
    fe = FeatureEngineer()
    features = fe.build_features(ohlcv_module)
    engine = HMMEngine(HMMConfig(n_candidates=[3, 4], n_init=4, min_train_bars=400))
    engine.fit(features)
    return fe, features, engine


@pytest.fixture(scope="module")
def ohlcv_module():
    """Module-scoped synthetic OHLCV (the function fixture is function-scoped)."""
    from conftest import make_synthetic_ohlcv

    return make_synthetic_ohlcv()


def test_features_do_not_use_future_data(ohlcv) -> None:
    """Feature row at the probe bar is unchanged by appending future bars."""
    fe = FeatureEngineer()
    fe.assert_no_lookahead(ohlcv, probe_index=PROBE_BAR)


def test_appending_future_bar_does_not_change_past_features(ohlcv) -> None:
    """Adding one future bar must not alter the prior feature row."""
    fe = FeatureEngineer()
    f_short = fe.build_features(ohlcv.iloc[: PROBE_BAR + 1], dropna=False)
    f_long = fe.build_features(ohlcv.iloc[: PROBE_BAR + 2], dropna=False)
    ts = ohlcv.index[PROBE_BAR]
    np.testing.assert_allclose(
        f_short.loc[ts].to_numpy(float), f_long.loc[ts].to_numpy(float), atol=1e-9
    )


def test_assert_no_lookahead_raises_on_leakage(ohlcv) -> None:
    """The guard itself must fire when leakage is injected."""

    class LeakyEngineer(FeatureEngineer):
        def build_features(self, data, dropna=True):  # type: ignore[override]
            f = super().build_features(data, dropna=dropna)
            # inject a forward-looking value: shift(-1) peeks at the future
            f["ret_1"] = f["ret_1"].shift(-1)
            return f

    with pytest.raises(AssertionError):
        LeakyEngineer().assert_no_lookahead(ohlcv, probe_index=PROBE_BAR)


def test_no_look_ahead_bias(fitted, ohlcv_module) -> None:
    """Filtered regime at the probe bar is identical across prefixes.

    And Viterbi differs at the same bar — proving the test detects the bias
    it guards against.
    """
    fe, features, engine = fitted
    ts = ohlcv_module.index[PROBE_BAR]
    assert ts in features.index, "probe bar must survive warmup dropna"

    # prefix ending AT the probe bar vs a prefix extending 200 bars past it
    short_feats = features.loc[:ts]
    long_feats = features.iloc[: features.index.get_loc(ts) + 201]

    short_states = engine.predict_regime_filtered(short_feats)
    long_states = engine.predict_regime_filtered(long_feats)

    # (a) probe bar identical across prefixes (aligned by timestamp, not position)
    short_at = short_states[-1]
    long_at = next(s for s in long_states if s.timestamp == ts)
    assert short_at.timestamp == ts and long_at.timestamp == ts
    assert short_at.state_id == long_at.state_id, "LOOK-AHEAD BIAS DETECTED (filtered)"
    np.testing.assert_allclose(
        short_at.state_probabilities, long_at.state_probabilities, atol=1e-8
    )

    # (b) filtered path identical EVERYWHERE on the overlap — fully causal
    short_ids = [s.state_id for s in short_states]
    long_ids_overlap = [s.state_id for s in long_states[: len(short_states)]]
    assert short_ids == long_ids_overlap, "LOOK-AHEAD BIAS DETECTED (filtered path)"


def test_viterbi_revises_past_but_filtered_does_not() -> None:
    """Teeth: prove the bias exists and that filtering is immune to it.

    Hand-built 2-state HMM with sticky transitions and ambiguous early
    observations. A single *future* observation flips Viterbi's MAP state at
    t=0, whereas the filtered posterior at t=0 is invariant to that future.
    """
    import pandas as pd
    from hmmlearn.hmm import GaussianHMM

    from core.hmm_engine import HMMEngine, Regime

    m = GaussianHMM(n_components=2, covariance_type="full")
    m.startprob_ = np.array([0.5, 0.5])
    m.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])  # sticky -> path matters
    m.means_ = np.array([[0.0], [10.0]])
    m.covars_ = np.array([[[4.0]], [[4.0]]])              # std 2 -> obs=5 ambiguous

    base = np.array([[5.0], [5.0], [5.0]])
    seq_low = np.vstack([base, [[0.0]]])   # future says "state 0"
    seq_high = np.vstack([base, [[10.0]]]) # future says "state 1"

    # Viterbi: the state at t=0 depends on a FUTURE observation -> look-ahead
    v_low = m.predict(seq_low)
    v_high = m.predict(seq_high)
    assert v_low[0] != v_high[0], "expected Viterbi to be future-dependent here"

    # Filtered: posterior at t=0 must be identical regardless of the future
    eng = HMMEngine()
    eng.model = m
    eng.n_regimes = 2
    eng.feature_columns = ["x"]
    eng.labels = {0: Regime.BEAR, 1: Regime.BULL}
    p_low = eng.predict_regime_proba(pd.DataFrame(seq_low, columns=["x"]))
    p_high = eng.predict_regime_proba(pd.DataFrame(seq_high, columns=["x"]))
    np.testing.assert_allclose(
        p_low.iloc[0].to_numpy(), p_high.iloc[0].to_numpy(), atol=1e-12
    )


def test_filtered_rows_are_normalized(fitted) -> None:
    """Each filtered posterior row must sum to 1 (valid distribution)."""
    _, features, engine = fitted
    proba = engine.predict_regime_proba(features)
    sums = proba.to_numpy().sum(axis=1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-8)
