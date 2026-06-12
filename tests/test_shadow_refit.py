"""Tests for the pin-champion dual-log (T0.4 operative amendment).

The champion is pinned (no age-based refit). While the old weekly-refit rule
would have swapped models, we instead fit a *shadow* engine and log how much it
disagrees with the pinned champion — two weeks of rows demonstrate (or refute)
refit/pinned equivalence without ever touching what drives orders.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_synthetic_ohlcv
from core.hmm_engine import HMMConfig, HMMEngine
from core.shadow_refit import append_row, compare_engines
from data.feature_engineering import FeatureEngineer


@pytest.fixture(scope="module")
def features():
    return FeatureEngineer().build_features(make_synthetic_ohlcv())


@pytest.fixture(scope="module")
def engine(features):
    eng = HMMEngine(HMMConfig(n_candidates=[3], n_init=2, min_train_bars=400))
    eng.fit(features)
    return eng


def test_compare_identical_engines_agree(engine, features) -> None:
    row = compare_engines(engine, engine, features)
    assert row["agree"] is True
    assert row["champion_hash"] == row["shadow_hash"] == engine.transition_hash()
    assert row["champion_regime"] == row["shadow_regime"]
    assert 0.0 <= row["champion_conf"] <= 1.0
    assert row["date"] == str(features.index[-1])[:10]


def test_compare_detects_disagreement(engine, features) -> None:
    """A shadow whose labels are scrambled must show up as disagreement."""
    import copy

    shadow = copy.deepcopy(engine)
    # rotate the label map so today's argmax state maps to a different regime name
    n = shadow.n_regimes
    shadow.labels = {i: engine.labels[(i + 1) % n] for i in range(n)}
    row = compare_engines(engine, shadow, features)
    assert row["champion_regime"] != row["shadow_regime"]
    assert row["agree"] is False


def test_append_row_idempotent_per_date(tmp_path, engine, features) -> None:
    p = tmp_path / "shadow_refit.csv"
    row = compare_engines(engine, engine, features)
    append_row(str(p), row)
    append_row(str(p), row)                      # same-day re-run: no duplicate
    df = pd.read_csv(p)
    assert len(df) == 1
    assert bool(df.iloc[0]["agree"]) is True
