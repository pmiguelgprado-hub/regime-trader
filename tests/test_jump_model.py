"""Tests for the Statistical Jump Model regime engine (T1.1).

Shu/Mulvey/Nystrup: k-means-like clustering of feature rows with a jump penalty
λ that charges state changes over time, yielding MORE PERSISTENT regimes (less
flicker) than the HMM — flicker being our known pain. Shadow-only: never drives
orders. Tests pin the defining behaviours: persistence rises with λ, and the
model separates a synthetic low-vol/high-vol regime split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.jump_model import JumpModel


def _two_regime_panel(seed=0):
    """120 low-vol bars then 120 high-vol bars.

    Features mimic the real panel (e.g. rvol_20): a calm regime sits at a low level
    with little spread, a turbulent regime at a higher level with more spread — so
    the two are separable by centroid *and* by within-state dispersion."""
    rng = np.random.default_rng(seed)
    lo = rng.normal(0.5, 0.3, (120, 2))
    hi = rng.normal(3.0, 1.2, (120, 2))
    X = np.vstack([lo, hi])
    return pd.DataFrame(X, columns=["f1", "f2"],
                        index=pd.bdate_range("2025-01-01", periods=240))


def _n_switches(states) -> int:
    s = np.asarray(states)
    return int((s[1:] != s[:-1]).sum())


def test_fit_returns_one_label_per_row():
    jm = JumpModel(n_states=2, jump_penalty=10.0, random_state=0).fit(_two_regime_panel())
    assert len(jm.states_) == 240
    assert set(np.unique(jm.states_)) <= {0, 1}


def test_separates_two_regimes():
    panel = _two_regime_panel()
    jm = JumpModel(n_states=2, jump_penalty=50.0, random_state=0).fit(panel)
    # the first 120 rows should be (almost) one state, the last 120 the other
    first, second = jm.states_[:120], jm.states_[120:]
    pure_first = max(np.mean(first == 0), np.mean(first == 1))
    pure_second = max(np.mean(second == 0), np.mean(second == 1))
    assert pure_first > 0.9 and pure_second > 0.9
    assert first[60] != second[60]                      # different dominant state


def test_higher_penalty_fewer_switches():
    panel = _two_regime_panel(seed=3)
    lo = JumpModel(n_states=3, jump_penalty=0.0, random_state=0).fit(panel)
    hi = JumpModel(n_states=3, jump_penalty=200.0, random_state=0).fit(panel)
    assert _n_switches(hi.states_) <= _n_switches(lo.states_)


def test_deterministic_with_seed():
    panel = _two_regime_panel()
    a = JumpModel(n_states=2, jump_penalty=20.0, random_state=7).fit(panel)
    b = JumpModel(n_states=2, jump_penalty=20.0, random_state=7).fit(panel)
    np.testing.assert_array_equal(a.states_, b.states_)


def test_predict_assigns_states_to_new_rows():
    panel = _two_regime_panel()
    jm = JumpModel(n_states=2, jump_penalty=20.0, random_state=0).fit(panel)
    pred = jm.predict(panel.iloc[-30:])
    assert len(pred) == 30
    assert set(np.unique(pred)) <= {0, 1}


def test_regime_labels_ordered_by_volatility():
    """regime_labels ranks states 0..K-1 by ascending within-state dispersion,
    so label 0 = calmest (comparable to the HMM's vol_rank ordering)."""
    panel = _two_regime_panel()
    jm = JumpModel(n_states=2, jump_penalty=50.0, random_state=0).fit(panel)
    labels = jm.regime_labels()
    assert set(labels.values()) == {0, 1}
    # the state dominating the high-vol block must map to the higher rank
    hi_state = int(np.bincount(jm.states_[120:]).argmax())
    assert labels[hi_state] == 1


def test_vol_rank_for_latest_bar_in_unit_interval():
    panel = _two_regime_panel()
    jm = JumpModel(n_states=3, jump_penalty=30.0, random_state=0).fit(panel)
    vr = jm.vol_rank()
    assert 0.0 <= vr <= 1.0
