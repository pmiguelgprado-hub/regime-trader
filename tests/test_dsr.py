"""Tests for the Deflated / Probabilistic Sharpe Ratio (Bailey & López de Prado)."""

import math

import pytest

from backtest.performance import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


# ----------------------------------------------------------------------- PSR ---
def test_psr_equals_half_at_benchmark():
    """SR exactly at the benchmark -> 50% probability."""
    assert probabilistic_sharpe_ratio(0.1, 0.1, n_obs=100) == pytest.approx(0.5, abs=1e-9)


def test_psr_known_normal_case():
    """Hand-computed value for normal returns (skew=0, kurt=3)."""
    from scipy.stats import norm

    sr, n = 0.1, 101
    denom = math.sqrt(1.0 + (3 - 1) / 4 * sr * sr)
    expected = float(norm.cdf((sr - 0.0) * math.sqrt(n - 1) / denom))
    assert probabilistic_sharpe_ratio(sr, 0.0, n) == pytest.approx(expected, rel=1e-12)


def test_psr_increases_with_n_obs():
    """More observations of the same positive SR -> more confidence."""
    lo = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=50)
    hi = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=500)
    assert hi > lo > 0.5


def test_psr_below_benchmark_under_half():
    assert probabilistic_sharpe_ratio(0.02, 0.1, n_obs=200) < 0.5


def test_fat_tails_lower_psr():
    """Higher kurtosis -> lower confidence in the same Sharpe."""
    normal = probabilistic_sharpe_ratio(0.1, 0.0, 200, skew=0.0, kurt=3.0)
    fat = probabilistic_sharpe_ratio(0.1, 0.0, 200, skew=0.0, kurt=9.0)
    assert fat < normal


# -------------------------------------------------------- expected_max_sharpe ---
def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(n_trials=10, trials_sr_std=0.05)
    b = expected_max_sharpe(n_trials=1000, trials_sr_std=0.05)
    assert b > a > 0


def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe(n_trials=1, trials_sr_std=0.05) == 0.0


# ----------------------------------------------------------------------- DSR ---
def test_dsr_single_trial_is_psr_vs_zero():
    """n_trials=1 (frozen knobs) -> DSR == PSR against 0."""
    d = deflated_sharpe_ratio(0.1, n_obs=200, n_trials=1)
    p = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=200)
    assert d == pytest.approx(p, rel=1e-12)


def test_dsr_deflates_under_multiple_testing():
    """Same SR, but counting many trials -> lower DSR (harder bar)."""
    single = deflated_sharpe_ratio(0.1, n_obs=500, n_trials=1)
    swept = deflated_sharpe_ratio(0.1, n_obs=500, n_trials=100, trials_sr_std=0.05)
    assert swept < single
