"""Tests for the drift-detection primitives (A-3).

Pure, deterministic, network-free. PSI and posterior entropy feed the retrain
trigger; here we pin their mathematical behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.drift import (
    drift_triggers_retrain,
    max_feature_psi,
    normalized_entropy,
    population_stability_index,
)


# ------------------------------------------------------------------- PSI ---
def test_psi_zero_for_identical_distributions() -> None:
    """Same sample compared to itself -> no population shift (PSI ~ 0)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x.copy()) < 1e-6


def test_psi_large_for_shifted_distribution() -> None:
    """A 3-sigma mean shift -> PSI well past the 0.25 'significant' threshold."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 5000)
    shifted = rng.normal(3.0, 1.0, 5000)
    assert population_stability_index(base, shifted) > 0.25


def test_psi_is_nonnegative() -> None:
    """PSI is a divergence: never negative."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 2000)
    b = rng.normal(0.5, 1.0, 2000)
    assert population_stability_index(a, b) >= 0.0


# --------------------------------------------------------------- entropy ---
def test_entropy_zero_for_certain_posterior() -> None:
    """A degenerate posterior (all mass on one state) has zero uncertainty."""
    assert normalized_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_entropy_one_for_uniform_posterior() -> None:
    """A uniform posterior is maximally uncertain -> normalized entropy 1.0."""
    assert normalized_entropy(np.array([1 / 3, 1 / 3, 1 / 3])) == pytest.approx(1.0)


# ----------------------------------------------------- multi-feature PSI ---
def test_max_feature_psi_reports_worst_column() -> None:
    """Across feature columns, the aggregate PSI is driven by the drifted one."""
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)})
    live = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(4, 1, 3000)})
    assert max_feature_psi(train, live) > 0.25


def test_max_feature_psi_zero_for_same_frame() -> None:
    """Identical frames -> no drift on any column."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    assert max_feature_psi(df, df.copy()) < 1e-6


# --------------------------------------------------------------- trigger ---
def test_trigger_fires_when_psi_exceeds_threshold() -> None:
    """High feature drift alone is enough to request a retrain."""
    assert drift_triggers_retrain(psi=0.30, entropy=0.2, psi_threshold=0.25, entropy_threshold=0.9)


def test_trigger_silent_when_both_within_limits() -> None:
    """Calm features and a confident posterior -> no retrain requested."""
    assert not drift_triggers_retrain(psi=0.05, entropy=0.2, psi_threshold=0.25, entropy_threshold=0.9)


def test_recent_vs_prior_psi_below_noise_floor_when_stable() -> None:
    """Two adjacent windows from the same distribution stay below the noise floor."""
    import numpy as np
    import pandas as pd

    from core.drift import recent_vs_prior_psi
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f": rng.normal(0, 1, 400)},
                      index=pd.bdate_range("2025-01-01", periods=400))
    assert recent_vs_prior_psi(df) < 0.25


def test_recent_vs_prior_psi_high_on_shift() -> None:
    import numpy as np
    import pandas as pd

    from core.drift import recent_vs_prior_psi
    rng = np.random.default_rng(1)
    vals = np.concatenate([rng.normal(0, 1, 260), rng.normal(6, 1, 126)])
    df = pd.DataFrame({"f": vals}, index=pd.bdate_range("2025-01-01", periods=len(vals)))
    assert recent_vs_prior_psi(df) > 0.5


def test_recent_vs_prior_psi_short_history_zero() -> None:
    import pandas as pd

    from core.drift import recent_vs_prior_psi
    df = pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2025-01-01", periods=3))
    assert recent_vs_prior_psi(df) == 0.0
