"""Tests for the challenger book: residual (idiosyncratic) momentum + vol-target overlay.

The challenger swaps the frozen baseline's two parts: alpha becomes residual momentum
(market-beta stripped, Blitz-Huij-Martens/Chaves iMOM) and the risk overlay can use
constant-vol targeting (Daniel-Moskowitz/Barroso-Santa-Clara). Plus the CSCV PBO
overfitting control. All pure, causal, no network.

Note on semantics: residual *momentum* scores RECENT firm-specific outperformance against
beta+alpha fit on a longer estimation window — a name with steady idiosyncratic drift over
its whole history has no momentum (the drift is absorbed by alpha); the signal fires when
the recent residual run exceeds the name's own historical average.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.performance import pbo_cscv
from core.cross_sectional_ranking import (
    _overlay_gross,
    _weight_names,
    book_targets_fixed_selection,
    compute_book_targets_challenger,
    make_book_weights_challenger,
    rank_universe_residual,
    residual_momentum_score,
)

BARS = 700
RECENT = 252


def _prices_from_returns(rets: np.ndarray, start: float = 100.0) -> pd.Series:
    idx = pd.date_range("2018-01-01", periods=len(rets) + 1, freq="B")
    closes = start * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    return pd.Series(closes, index=idx, name="close")


def _name(rng, mkt, beta=1.0, recent_drift=0.0, noise=0.002) -> pd.Series:
    """A name = beta·market + idiosyncratic noise, with a recent idiosyncratic drift."""
    idio = rng.normal(0.0, noise, len(mkt))
    idio[-RECENT:] += recent_drift
    return _prices_from_returns(beta * mkt + idio)


# ----------------------------------------------------- residual momentum ---
def test_residual_momentum_pure_market_name_scores_near_zero() -> None:
    """A name that is *only* beta (no idiosyncratic drift) has residuals ≈ 0."""
    rng = np.random.default_rng(0)
    mkt = rng.normal(0.0005, 0.01, BARS)
    name = _prices_from_returns(1.2 * mkt + rng.normal(0.0, 1e-5, BARS))
    score = residual_momentum_score(name, _prices_from_returns(mkt))
    assert abs(score) < 3.0


def test_residual_momentum_positive_for_recent_idiosyncratic_uptrend() -> None:
    """Recent firm-specific outperformance (above the name's own history) -> positive."""
    rng = np.random.default_rng(1)
    mkt = rng.normal(0.0, 0.01, BARS)
    name = _name(rng, mkt, beta=1.0, recent_drift=0.0015)
    score = residual_momentum_score(name, _prices_from_returns(mkt))
    assert score > 0.3


def test_residual_momentum_insufficient_history_is_nan() -> None:
    rng = np.random.default_rng(2)
    short = rng.normal(0, 0.01, 30)
    assert math.isnan(
        residual_momentum_score(_prices_from_returns(short), _prices_from_returns(short))
    )


def test_rank_universe_residual_orders_by_recent_idiosyncratic_alpha() -> None:
    rng = np.random.default_rng(3)
    mkt = rng.normal(0.0, 0.01, BARS)
    frames = {
        "HI": pd.DataFrame({"close": _name(rng, mkt, recent_drift=0.0018)}),
        "MID": pd.DataFrame({"close": _name(rng, mkt, recent_drift=0.0008)}),
        "LO": pd.DataFrame({"close": _name(rng, mkt, recent_drift=-0.0012)}),
    }
    ranked = rank_universe_residual(frames, _prices_from_returns(mkt))
    assert ranked == ["HI", "MID", "LO"]


# --------------------------------------------------- challenger weight_fn ---
def _toy_universe(seed: int = 7, n: int = 6):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0003, 0.01, BARS)
    frames = {
        f"S{i}": pd.DataFrame({"close": _name(rng, mkt, recent_drift=0.0015 - i * 0.0006)})
        for i in range(n)
    }
    return frames, _prices_from_returns(mkt)


def test_challenger_overlay_none_pins_gross_to_one() -> None:
    frames, mkt = _toy_universe()
    wf = make_book_weights_challenger(
        frames, mkt, frac=0.5, max_concurrent=3, max_single=1.0, overlay="none",
    )
    ts = list(frames.values())[0].index[-1]
    w = wf(ts, vol_rank=0.5)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert len(w) == 3                                   # top-3 by residual momentum


def test_challenger_vol_target_derisks_a_volatile_book() -> None:
    """A high realized book vol vs a low target -> vol-target overlay scales gross < 1."""
    frames, mkt = _toy_universe()
    wf = make_book_weights_challenger(
        frames, mkt, frac=0.5, max_concurrent=3, max_single=1.0,
        overlay="vol_target", target_vol=0.02, vol_window=120,
    )
    ts = list(frames.values())[0].index[-1]
    w = wf(ts, vol_rank=0.5)
    assert sum(w.values()) < 1.0


def test_challenger_hmm_overlay_matches_baseline_scale() -> None:
    """overlay='hmm' reproduces the baseline regime_gross_scale tier behaviour."""
    from core.asset_rotation import regime_gross_scale

    frames, mkt = _toy_universe()
    wf = make_book_weights_challenger(
        frames, mkt, frac=0.5, max_concurrent=3, max_single=1.0,
        overlay="hmm", risk_on_gross=1.0, risk_off_gross=0.5,
    )
    ts = list(frames.values())[0].index[-1]
    w = wf(ts, vol_rank=0.9)                             # high-vol tier -> risk_off
    assert sum(w.values()) == pytest.approx(regime_gross_scale(0.9, 1.0, 0.5), abs=1e-9)


def test_compute_book_targets_challenger_overlay_modes() -> None:
    """The live one-shot picks top names by residual momentum and applies the overlay."""
    frames, mkt = _toy_universe()
    t_none = compute_book_targets_challenger(
        frames, mkt, vol_rank=0.1, frac=0.5, max_concurrent=3, max_single=1.0, overlay="none",
    )
    assert len(t_none) == 3
    assert sum(t_none.values()) == pytest.approx(1.0)
    t_hmm = compute_book_targets_challenger(
        frames, mkt, vol_rank=0.9, frac=0.5, max_concurrent=3, max_single=1.0,
        overlay="hmm", risk_off_gross=0.5,
    )
    assert sum(t_hmm.values()) == pytest.approx(0.5)     # high-vol tier de-risks to 0.5


def test_fixed_selection_keeps_names_and_rescales_gross() -> None:
    """Daily re-scale: keep the month's names, overlay sets gross; drop unpriced names."""
    frames, _ = _toy_universe()
    sel = ["S0", "S1", "S2", "GONE"]                 # GONE not in frames -> dropped
    none = book_targets_fixed_selection(frames, sel, vol_rank=0.5, overlay="none",
                                        max_single=1.0, max_concurrent=10)
    assert set(none) == {"S0", "S1", "S2"}
    assert sum(none.values()) == pytest.approx(1.0)
    derisked = book_targets_fixed_selection(frames, sel, vol_rank=0.5, overlay="vol_target",
                                            target_vol=0.02, vol_window=120,
                                            max_single=1.0, max_concurrent=10)
    assert sum(derisked.values()) < 1.0              # vol-target cuts gross, same names
    assert set(derisked) <= {"S0", "S1", "S2"}


def test_crash_only_overlay_full_until_panic_tier() -> None:
    """crash_only stays full in low/mid vol, de-risks only in the top (panic) tier."""
    args = (0.12, 126, 1.0, 0.0)
    assert _overlay_gross([], {}, None, 0.30, "crash_only", 1.0, 0.4, *args) == 1.0
    assert _overlay_gross([], {}, None, 0.60, "crash_only", 1.0, 0.4, *args) == 1.0
    assert _overlay_gross([], {}, None, 0.90, "crash_only", 1.0, 0.4, *args) == 0.4


def test_inv_vol_weighting_underweights_the_wilder_name() -> None:
    """Risk-parity-lite: the higher-vol name gets a smaller slice than the calm one."""
    rng = np.random.default_rng(5)
    frames = {
        "CALM": pd.DataFrame({"close": _prices_from_returns(rng.normal(0, 0.005, 300))}),
        "WILD": pd.DataFrame({"close": _prices_from_returns(rng.normal(0, 0.03, 300))}),
    }
    w = _weight_names(1.0, ["CALM", "WILD"], frames, None, "inv_vol",
                      max_single=1.0, max_concurrent=10)
    assert w["CALM"] > w["WILD"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------- PBO ---
def test_pbo_low_for_a_genuinely_dominant_config() -> None:
    """One config dominates every block -> selecting on IS never backfires -> PBO≈0."""
    rng = np.random.default_rng(11)
    T, N = 400, 5
    mat = rng.normal(0, 0.01, (T, N)) + np.linspace(0.0, 0.02, N)   # config N-1 dominates
    assert pbo_cscv(mat, n_splits=8) < 0.2


def test_pbo_elevated_for_pure_noise() -> None:
    """I.i.d. noise configs -> the IS winner is a coin flip OOS -> PBO well above 0."""
    rng = np.random.default_rng(12)
    p = pbo_cscv(rng.normal(0, 0.01, (400, 8)), n_splits=8)
    assert 0.2 < p < 0.95


def test_pbo_nan_for_single_config() -> None:
    assert math.isnan(pbo_cscv(np.random.default_rng(0).normal(0, 1, (100, 1))))
