"""Tests for cross-asset regime rotation (vía B).

The rotation map is theory-driven and frozen in
docs/analysis/2026-06-04-rotation-prereg.md (zero fitted params). These tests pin
the frozen knobs so a later edit that silently changes the economic priors fails
loudly.
"""

import math

import numpy as np
import pytest

from core.asset_rotation import RotationConfig, rotation_weights, vol_target_scale


@pytest.fixture
def cfg() -> RotationConfig:
    return RotationConfig()  # frozen defaults = the pre-registered knobs


# ------------------------------------------------------------- rotation_weights ---
def test_risk_on_is_all_equities(cfg):
    """vol_rank <= 0.33 (low vol) -> 100% equities, no defensive, no cash."""
    w = rotation_weights(0.0, cfg)
    assert pytest.approx(w["SPY"], abs=1e-9) == 0.5
    assert pytest.approx(w["QQQ"], abs=1e-9) == 0.5
    assert w.get("TLT", 0.0) == 0.0
    assert w.get("GLD", 0.0) == 0.0
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0  # fully risk-on, cash=0


def test_risk_off_has_no_equities_and_holds_cash(cfg):
    """vol_rank >= 0.67 (high vol) -> 0% equities, 60% defensive, 40% cash."""
    w = rotation_weights(1.0, cfg)
    assert w.get("SPY", 0.0) == 0.0
    assert w.get("QQQ", 0.0) == 0.0
    assert pytest.approx(w["TLT"], abs=1e-9) == 0.3
    assert pytest.approx(w["GLD"], abs=1e-9) == 0.3
    # risky sum is 0.6; the remaining 0.4 is the implicit cash sleeve
    assert pytest.approx(sum(w.values()), abs=1e-9) == 0.6


def test_mid_is_balanced(cfg):
    """0.33 < vol_rank < 0.67 -> 60% equities / 40% defensive, cash=0."""
    w = rotation_weights(0.5, cfg)
    assert pytest.approx(w["SPY"], abs=1e-9) == 0.3
    assert pytest.approx(w["QQQ"], abs=1e-9) == 0.3
    assert pytest.approx(w["TLT"], abs=1e-9) == 0.2
    assert pytest.approx(w["GLD"], abs=1e-9) == 0.2
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0


def test_tier_cutpoints_match_orchestrator(cfg):
    """Boundaries reuse the orchestrator terciles (0.33 / 0.67)."""
    # exactly at low cut -> still risk-on tier (<=)
    assert rotation_weights(0.33, cfg).get("TLT", 0.0) == 0.0
    # exactly at high cut -> risk-off tier (>=)
    assert rotation_weights(0.67, cfg).get("SPY", 0.0) == 0.0


# ------------------------------------------------------------ vol_target_scale ---
def test_high_vol_scales_down():
    """Realized vol above target -> scale < 1."""
    rng = np.random.default_rng(0)
    rets = list(rng.normal(0, 0.03, 20))  # ~48% annual vol >> 10% target
    k = vol_target_scale(rets, target_vol=0.10, cap=1.0)
    assert 0.0 < k < 1.0


def test_low_vol_capped():
    """Realized vol below target -> scale hits the cap (no leverage)."""
    rets = [0.0001] * 20  # near-zero vol
    k = vol_target_scale(rets, target_vol=0.10, cap=1.0)
    assert k == 1.0


def test_zero_vol_returns_cap():
    """Degenerate zero-vol window -> cap, never divide-by-zero."""
    k = vol_target_scale([0.0] * 20, target_vol=0.10, cap=1.0)
    assert k == 1.0


def test_scale_matches_realized_vol_formula():
    rets = ([0.05, -0.05] * 10)  # mean 0, sample std = 0.05 exactly
    k = vol_target_scale(rets, target_vol=0.10, cap=1.0, floor=0.0)
    realized = float(np.std(rets, ddof=1)) * math.sqrt(252)
    assert pytest.approx(k, rel=1e-9) == min(1.0, 0.10 / realized)
    assert 0.0 < k < 1.0  # 0.05/bar is way above a 10% annual target


def test_short_window_returns_cap():
    """Too few points to estimate vol -> cap (conservative, no spurious sizing)."""
    assert vol_target_scale([0.01, 0.02], target_vol=0.10, cap=1.0, min_obs=5) == 1.0
