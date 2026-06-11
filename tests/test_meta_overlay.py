"""Tests for the fuzzy regime layer (meta-overlay, 2026-06-11).

The layer consumes the HMM's *filtered* posterior (already causal) instead of the
argmax state: a probability-weighted vol rank removes the argmax cliff from the
gross overlay, and a transition hazard scores the risk of being in the high-vol
tier next bar. Zero new fitted parameters — nothing to sweep, nothing to overfit
(docs/analysis/2026-06-11-meta-overlay-triage.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.meta_overlay import (
    high_tier_hazard,
    predictive_entropy_norm,
    prob_weighted_vol_rank,
    vol_rank_for_overlay,
)

# ------------------------------------------------- prob_weighted_vol_rank ---
def test_one_hot_posterior_recovers_state_rank() -> None:
    """Degenerate (certain) posterior must equal that state's static rank."""
    vr = {0: 0.0, 1: 0.5, 2: 1.0}
    assert prob_weighted_vol_rank(np.array([0.0, 1.0, 0.0]), vr) == pytest.approx(0.5)


def test_uniform_posterior_is_mean_rank() -> None:
    """Uniform uncertainty over {0, 0.5, 1} ranks -> expected rank 0.5."""
    vr = {0: 0.0, 1: 0.5, 2: 1.0}
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    assert prob_weighted_vol_rank(p, vr) == pytest.approx(0.5)


def test_partial_posterior_interpolates_continuously() -> None:
    """70/30 split between low and high ranks -> 0.3 (the cliff is gone)."""
    vr = {0: 0.0, 1: 1.0}
    assert prob_weighted_vol_rank(np.array([0.7, 0.3]), vr) == pytest.approx(0.3)


def test_missing_state_rank_defaults_conservative() -> None:
    """A state id absent from the rank map counts as high-vol (rank 1.0)."""
    vr = {0: 0.0}                       # state 1 missing
    out = prob_weighted_vol_rank(np.array([0.5, 0.5]), vr)
    assert out == pytest.approx(0.5 * 0.0 + 0.5 * 1.0)


def test_empty_posterior_is_conservative() -> None:
    """No posterior (unfitted/edge) -> conservative high-vol rank."""
    assert prob_weighted_vol_rank(np.empty(0), {0: 0.0}) == pytest.approx(1.0)


# ------------------------------------------------------- high_tier_hazard ---
def test_hazard_zero_when_locked_in_low_state() -> None:
    """Identity transitions from a certain low-vol state -> no hazard."""
    A = np.eye(2)
    vr = {0: 0.0, 1: 1.0}
    assert high_tier_hazard(np.array([1.0, 0.0]), A, vr) == pytest.approx(0.0)


def test_hazard_one_when_locked_in_high_state() -> None:
    """Mass already in the high tier stays there under identity transitions."""
    A = np.eye(2)
    vr = {0: 0.0, 1: 1.0}
    assert high_tier_hazard(np.array([0.0, 1.0]), A, vr) == pytest.approx(1.0)


def test_hazard_matches_hand_computed_chain() -> None:
    """pi=[1,0], A=[[.9,.1],[.05,.95]], high={1} -> hazard = 0.1."""
    A = np.array([[0.9, 0.1], [0.05, 0.95]])
    vr = {0: 0.0, 1: 1.0}
    assert high_tier_hazard(np.array([1.0, 0.0]), A, vr) == pytest.approx(0.1)


def test_hazard_mixes_posterior_mass() -> None:
    """pi=[.5,.5] under the same chain -> .5*.1 + .5*.95 = 0.525."""
    A = np.array([[0.9, 0.1], [0.05, 0.95]])
    vr = {0: 0.0, 1: 1.0}
    assert high_tier_hazard(np.array([0.5, 0.5]), A, vr) == pytest.approx(0.525)


def test_hazard_uses_high_vol_min_tier_cut() -> None:
    """Only states with rank >= HIGH_VOL_MIN count as the hazardous tier."""
    from core.regime_strategies import HIGH_VOL_MIN

    # three states, ranks 0 / 0.5 / 1.0 — only rank 1.0 is >= HIGH_VOL_MIN (0.67)
    A = np.full((3, 3), 1 / 3)
    vr = {0: 0.0, 1: 0.5, 2: 1.0}
    out = high_tier_hazard(np.array([1.0, 0.0, 0.0]), A, vr)
    assert out == pytest.approx(1 / 3)
    assert 0.5 < HIGH_VOL_MIN  # guard: mid tier must not be counted


def test_hazard_empty_inputs_conservative() -> None:
    """Missing posterior/transmat -> conservative hazard 1.0."""
    assert high_tier_hazard(np.empty(0), np.eye(2), {0: 0.0, 1: 1.0}) == 1.0


# ------------------------------------------------ predictive_entropy_norm ---
def test_entropy_zero_for_deterministic_prediction() -> None:
    A = np.eye(2)
    assert predictive_entropy_norm(np.array([1.0, 0.0]), A) == pytest.approx(0.0)


def test_entropy_one_for_uniform_prediction() -> None:
    A = np.full((2, 2), 0.5)
    assert predictive_entropy_norm(np.array([1.0, 0.0]), A) == pytest.approx(1.0)


def test_entropy_single_state_is_zero() -> None:
    """A 1-state chain has no uncertainty (and must not divide by log(1)=0)."""
    assert predictive_entropy_norm(np.array([1.0]), np.eye(1)) == 0.0


# ------------------------------------------------------ vol_rank_for_overlay ---
def test_overlay_dispatch_prob_mode_uses_posterior() -> None:
    """hmm_prob mode -> probability-weighted rank, not the argmax rank."""
    vr = {0: 0.0, 1: 1.0}
    out = vol_rank_for_overlay(
        "hmm_prob", state_probabilities=np.array([0.7, 0.3]), state_id=0, vol_rank_map=vr
    )
    assert out == pytest.approx(0.3)


def test_overlay_dispatch_legacy_modes_use_argmax_rank() -> None:
    """Every other mode keeps the deployed behaviour byte-for-byte (argmax rank)."""
    vr = {0: 0.0, 1: 1.0}
    for mode in ("none", "hmm", "crash_only", "vol_target", "both"):
        out = vol_rank_for_overlay(
            mode, state_probabilities=np.array([0.7, 0.3]), state_id=0, vol_rank_map=vr
        )
        assert out == pytest.approx(0.0), mode


def test_overlay_dispatch_unknown_state_defaults_mid() -> None:
    """Argmax path mirrors main.py's `.get(state_id, 0.5)` neutral default."""
    out = vol_rank_for_overlay(
        "hmm", state_probabilities=np.empty(0), state_id=9, vol_rank_map={0: 0.0}
    )
    assert out == pytest.approx(0.5)


# ------------------------------------------------- _overlay_gross plumbing ---
def test_overlay_gross_hmm_prob_scales_like_hmm() -> None:
    """`hmm_prob` maps rank -> gross exactly like `hmm` (the caller supplies the
    posterior-weighted rank); today it must NOT fall through to 1.0."""
    from core.asset_rotation import regime_gross_scale
    from core.cross_sectional_ranking import _overlay_gross

    for rank in (0.0, 0.45, 0.85):
        got = _overlay_gross(
            top=[], frames={}, ts=None, vol_rank=rank, overlay="hmm_prob",
            risk_on_gross=1.0, risk_off_gross=0.5, target_vol=0.12,
            vol_window=126, gross_cap=1.0, gross_floor=0.0,
        )
        assert got == pytest.approx(regime_gross_scale(rank, 1.0, 0.5)), rank


# ----------------------------------------------------- backtester plumbing ---
def _fast_backtester():
    from backtest.backtester import Backtester, BacktestConfig
    from core.hmm_engine import HMMConfig, HMMEngine
    from core.regime_strategies import StrategyConfig
    from core.risk_manager import RiskConfig, RiskManager
    from data.feature_engineering import FeatureEngineer

    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    return Backtester(
        BacktestConfig(step_size=252), hmm, StrategyConfig(),
        RiskManager(RiskConfig()), FeatureEngineer(),
    )


@pytest.fixture(scope="module")
def bt_result():
    from conftest import make_synthetic_ohlcv

    return _fast_backtester().run({"SPY": make_synthetic_ohlcv()})


def test_regime_history_carries_posterior_columns(bt_result) -> None:
    """Each OOS bar exposes the posterior-weighted rank and the transition hazard."""
    hist = bt_result.regime_history
    for col in ("vol_rank_prob", "regime_hazard"):
        assert col in hist.columns, col
        assert hist[col].notna().all(), col
        assert ((hist[col] >= 0.0) & (hist[col] <= 1.0)).all(), col


def test_run_portfolio_vol_rank_col_feeds_weight_fn(bt_result) -> None:
    """`vol_rank_col="vol_rank_prob"` must hand weight_fn the posterior column."""
    from conftest import make_synthetic_ohlcv

    frames = {"SPY": make_synthetic_ohlcv()}
    seen: list[float] = []

    def spy_fn(ts, vol_rank):
        seen.append(float(vol_rank))
        return {"SPY": 0.5}

    bt = _fast_backtester()
    bt.run_portfolio(frames, weight_fn=spy_fn, vol_rank_col="vol_rank_prob")
    hist = bt_result.regime_history  # same engine config + data -> same regimes
    assert len(seen) == len(hist)
    np.testing.assert_allclose(np.array(seen), hist["vol_rank_prob"].to_numpy(),
                               rtol=0, atol=1e-12)
