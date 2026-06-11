"""Tests for the inter-sleeve risk allocator (library only — NOT wired to live).

ERC (equal risk contribution) on a shrunk, optionally regime-mixed covariance is
the defensible default when NO sleeve has a validated edge: it budgets risk, not
expected return. Kelly (fractional, capped) exists but is OFF by default and
only ever tilts sleeves explicitly marked forward-validated — Kelly on unvalidated
mu-hat estimates is leverage on estimation noise (triage memo §1, 3a).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.allocator import (
    AllocatorConfig,
    allocate,
    erc_weights,
    kelly_weights,
    regime_mixed_covariance,
)


def _returns(cov: np.ndarray, n: int = 4000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)
    return pd.DataFrame(X, columns=[f"s{i}" for i in range(cov.shape[0])])


# ---------------------------------------------------------------------- ERC ---
def test_erc_equal_vol_uncorrelated_is_equal_weight() -> None:
    w = erc_weights(np.diag([0.04, 0.04]))
    assert w == pytest.approx([0.5, 0.5], abs=1e-6)


def test_erc_inverse_vol_when_uncorrelated() -> None:
    """Uncorrelated ERC = inverse-vol: vols (1, 2) -> weights (2/3, 1/3)."""
    w = erc_weights(np.diag([1.0, 4.0]))
    assert w == pytest.approx([2 / 3, 1 / 3], abs=1e-6)


def test_erc_risk_contributions_equalized_with_correlation() -> None:
    """Defining property: w_i (Sigma w)_i equal across assets, weights sum to 1."""
    cov = np.array([
        [0.040, 0.018, 0.002],
        [0.018, 0.090, 0.002],
        [0.002, 0.002, 0.010],
    ])
    w = np.asarray(erc_weights(cov))
    rc = w * (cov @ w)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w > 0).all()
    assert rc.max() - rc.min() == pytest.approx(0.0, abs=1e-8)


# -------------------------------------------------------------------- Kelly ---
def test_kelly_single_asset_closed_form() -> None:
    """f* = mu / sigma^2, scaled by the fraction."""
    w = kelly_weights(np.array([0.08]), np.array([[0.04]]), fraction=0.25)
    assert w == pytest.approx([0.25 * 0.08 / 0.04])     # 0.5


def test_kelly_uncorrelated_assets_independent_sizing() -> None:
    mu = np.array([0.08, 0.02])
    cov = np.diag([0.04, 0.04])
    w = kelly_weights(mu, cov, fraction=1.0)
    assert w == pytest.approx([2.0, 0.5])


def test_kelly_gross_cap_scales_down_proportionally() -> None:
    mu = np.array([0.08, 0.02])
    cov = np.diag([0.04, 0.04])
    w = np.asarray(kelly_weights(mu, cov, fraction=1.0, cap_gross=1.0))
    assert np.abs(w).sum() == pytest.approx(1.0)
    assert w[0] / w[1] == pytest.approx(4.0)            # relative sizing preserved


# ------------------------------------------------------- regime-mixed Sigma ---
def test_regime_mixed_covariance_interpolates_tiers() -> None:
    """Sigma_t = sum_tier P(tier) * Sigma_tier; pure tiers recover each block."""
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0, 0.01, size=(500, 2))
    panic = rng.normal(0.0, 0.05, size=(500, 2))
    rets = pd.DataFrame(np.vstack([calm, panic]), columns=["a", "b"])
    tiers = np.array(["low"] * 500 + ["high"] * 500)

    cov_low = regime_mixed_covariance(rets, tiers, {"low": 1.0, "mid": 0.0, "high": 0.0})
    cov_high = regime_mixed_covariance(rets, tiers, {"low": 0.0, "mid": 0.0, "high": 1.0})
    cov_mix = regime_mixed_covariance(rets, tiers, {"low": 0.5, "mid": 0.0, "high": 0.5})

    assert cov_high[0, 0] > 10 * cov_low[0, 0]          # panic var >> calm var
    np.testing.assert_allclose(cov_mix, 0.5 * cov_low + 0.5 * cov_high, rtol=1e-12)


def test_regime_mixed_covariance_falls_back_to_pooled_for_thin_tier() -> None:
    """A tier with too few bars uses the pooled estimate instead of garbage."""
    rets = _returns(np.diag([0.01, 0.01]), n=300)
    tiers = np.array(["low"] * 297 + ["high"] * 3)      # 3 bars cannot estimate Sigma
    cov = regime_mixed_covariance(rets, tiers, {"low": 0.0, "mid": 0.0, "high": 1.0},
                                  min_obs=30)
    pooled = np.cov(rets.to_numpy(), rowvar=False)
    # variances must match the pooled scale (off-diagonals hover near 0, where
    # relative comparisons are meaningless), proving the 3-bar tier was not used
    np.testing.assert_allclose(np.diag(cov), np.diag(pooled), rtol=0.5)


# ----------------------------------------------------------------- allocate ---
def test_allocate_defaults_to_erc_risk_budget() -> None:
    rets = _returns(np.diag([0.01, 0.04]))
    out = allocate(rets, AllocatorConfig())
    assert set(out) == {"s0", "s1"}
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
    assert out["s0"] > out["s1"]                        # lower vol -> bigger budget


def test_allocate_kelly_requires_validated_sleeves() -> None:
    """Kelly enabled but sleeves unvalidated -> pure ERC (no mu-hat leverage)."""
    rets = _returns(np.diag([0.01, 0.04]))
    cfg = AllocatorConfig(kelly_enabled=True)
    erc_out = allocate(rets, AllocatorConfig())
    kelly_out = allocate(rets, cfg, validated={"s0": False, "s1": False})
    assert kelly_out == pytest.approx(erc_out)


def test_allocate_kelly_tilts_only_validated_sleeve() -> None:
    """With one validated sleeve, its weight moves from the ERC budget; the
    output stays a fully-invested simplex (risk budget, not leverage)."""
    cov = np.diag([0.0004, 0.0004])
    rng = np.random.default_rng(5)
    X = rng.multivariate_normal([0.0008, 0.0], cov, size=6000)  # s0 has real drift
    rets = pd.DataFrame(X, columns=["s0", "s1"])
    cfg = AllocatorConfig(kelly_enabled=True, kelly_fraction=0.25)
    out = allocate(rets, cfg, validated={"s0": True, "s1": False})
    erc_out = allocate(rets, AllocatorConfig())
    assert out["s0"] > erc_out["s0"]                    # validated sleeve tilted up
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
    assert min(out.values()) >= 0.0
