"""Fuzzy regime layer: posterior-weighted vol rank + transition hazard.

The HMM already emits a *filtered* (causal) posterior over hidden states
(:meth:`core.hmm_engine.HMMEngine.predict_regime_proba`), but the gross-exposure
overlay only consumed the argmax state's static vol rank — a step function that
flips a whole tier the bar the MAP state changes (the "transition lag" cliff).

This module replaces that argmax read with expectations under the posterior:

* :func:`prob_weighted_vol_rank` — ``E[vol_rank | posterior]``, the Bayes point
  estimate of the volatility tier. Continuous in [0, 1], so
  :func:`core.asset_rotation.regime_gross_scale` produces a continuous gross.
* :func:`high_tier_hazard` — ``P(high-vol tier at t+1 | obs_1:t)`` under the
  learned transition matrix: the trigger for risk actions (tail hedge, alerts).
* :func:`predictive_entropy_norm` — normalized uncertainty of the one-step-ahead
  regime distribution (dashboard / uncertainty-mode metric).

Deliberately **zero new fitted parameters** (no smoothing half-life, no bands):
everything derives from quantities the HMM already learned, so there is nothing
to sweep and nothing to overfit. This is NOT a return predictor — it removes a
discretization artifact from the already-validated *risk overlay* role
(docs/analysis/2026-06-11-meta-overlay-triage.md).
"""

from __future__ import annotations

import numpy as np

from core.regime_strategies import HIGH_VOL_MIN

# Conservative fallbacks: an unknown/absent rank is treated as HIGH vol (1.0) by
# the posterior-weighted path, while the legacy argmax path keeps main.py's
# neutral 0.5 default so deployed behaviour is unchanged byte-for-byte.
_CONSERVATIVE_RANK = 1.0
_LEGACY_DEFAULT_RANK = 0.5


def prob_weighted_vol_rank(
    state_probabilities: np.ndarray,
    vol_rank_map: dict[int, float],
) -> float:
    """Expected volatility rank under the filtered posterior.

    ``E[vr | pi_t] = sum_s pi_t(s) * vr(s)`` — the Bayes-optimal point estimate
    of the vol tier under squared loss. Where the argmax rank jumps between the
    per-state values, this moves continuously as posterior mass shifts, so the
    overlay de-risks *as* the high-vol regime becomes likely instead of one
    cliff-edge bar after it is confirmed.

    Args:
        state_probabilities: Filtered posterior ``pi_t`` (length n_states).
        vol_rank_map: ``state_id -> static vol rank`` from
            :class:`~core.regime_strategies.StrategyOrchestrator`. Missing ids
            count as rank 1.0 (conservative: unknown = risky).

    Returns:
        Expected rank in [0, 1]; 1.0 (conservative) when the posterior is empty.
    """
    p = np.asarray(state_probabilities, dtype=float)
    if p.size == 0:
        return _CONSERVATIVE_RANK
    ranks = np.array([vol_rank_map.get(s, _CONSERVATIVE_RANK) for s in range(p.size)])
    # clip: dot of a simplex vector with ranks in [0,1] can exceed 1 by float eps
    return float(np.clip(np.dot(p, ranks), 0.0, 1.0))


def _high_tier_mask(vol_rank_map: dict[int, float], n_states: int) -> np.ndarray:
    """Indicator over states whose static rank sits in the high-vol tier."""
    return np.array(
        [vol_rank_map.get(s, _CONSERVATIVE_RANK) >= HIGH_VOL_MIN for s in range(n_states)],
        dtype=float,
    )


def high_tier_hazard(
    state_probabilities: np.ndarray,
    transmat: np.ndarray,
    vol_rank_map: dict[int, float],
) -> float:
    """One-step-ahead probability of the high-vol tier.

    ``h_t = (pi_t @ A) . 1_high`` where ``1_high`` marks states whose static vol
    rank is ``>= HIGH_VOL_MIN``. Mass already in the high tier counts (being in
    panic = maximal hazard), so this reads as "P(tomorrow is risk-off)" — the
    trigger for risk actions, never a direction forecast.

    Args:
        state_probabilities: Filtered posterior ``pi_t`` (length n_states).
        transmat: Learned transition matrix ``A`` (n_states x n_states).
        vol_rank_map: ``state_id -> static vol rank``.

    Returns:
        Hazard in [0, 1]; 1.0 (conservative) on empty/mismatched inputs.
    """
    p = np.asarray(state_probabilities, dtype=float)
    A = np.asarray(transmat, dtype=float)
    if p.size == 0 or A.ndim != 2 or A.shape[0] != p.size or A.shape[1] != p.size:
        return _CONSERVATIVE_RANK
    predictive = p @ A
    return float(np.clip(np.dot(predictive, _high_tier_mask(vol_rank_map, p.size)), 0.0, 1.0))


def predictive_entropy_norm(
    state_probabilities: np.ndarray,
    transmat: np.ndarray,
) -> float:
    """Normalized Shannon entropy of the one-step-ahead regime distribution.

    ``H(pi_t @ A) / log(n_states)`` in [0, 1]: 0 = tomorrow's regime is certain,
    1 = maximally uncertain. A 1-state chain returns 0.0 (no uncertainty, and no
    division by ``log(1) = 0``).

    Args:
        state_probabilities: Filtered posterior ``pi_t``.
        transmat: Learned transition matrix ``A``.

    Returns:
        Normalized entropy in [0, 1]; 1.0 (conservative) on bad inputs.
    """
    p = np.asarray(state_probabilities, dtype=float)
    A = np.asarray(transmat, dtype=float)
    if p.size == 0 or A.ndim != 2 or A.shape[0] != p.size or A.shape[1] != p.size:
        return 1.0
    if p.size == 1:
        return 0.0
    predictive = p @ A
    nz = predictive[predictive > 0.0]
    entropy = float(-(nz * np.log(nz)).sum())
    return entropy / float(np.log(p.size))


def vol_rank_for_overlay(
    overlay: str,
    state_probabilities: np.ndarray,
    state_id: int,
    vol_rank_map: dict[int, float],
) -> float:
    """Vol-rank input for the gross overlay, dispatched by overlay mode.

    ``"hmm_prob"`` reads the posterior-weighted rank; every other mode keeps the
    deployed argmax behaviour byte-for-byte (``vol_rank_map.get(state_id, 0.5)``,
    mirroring ``main.run_rebalance``), so adding the new mode cannot perturb the
    frozen books.

    Args:
        overlay: Overlay mode string from settings (``hmm_prob`` is the new one).
        state_probabilities: Filtered posterior at the decision bar.
        state_id: Argmax (MAP) state id at the decision bar.
        vol_rank_map: ``state_id -> static vol rank``.

    Returns:
        The vol rank the overlay should consume.
    """
    if overlay == "hmm_prob":
        return prob_weighted_vol_rank(state_probabilities, vol_rank_map)
    return float(vol_rank_map.get(state_id, _LEGACY_DEFAULT_RANK))
