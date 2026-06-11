"""Inter-sleeve risk allocator: ERC + shrinkage + regime-mixed covariance.

Library ONLY — deliberately not wired to any live book. Re-allocating capital
between the deployed baseline and the challenger mid-gate would change two
pre-registered strategies (docs/analysis/2026-06-11-meta-overlay-triage.md §1,
row 3a); this module exists so the allocation logic is built, tested and ready
for the day a sleeve passes its forward gate.

Design, in order of what the evidence supports:

* **ERC (equal risk contribution)** is the default: with *no* validated edges,
  expected-return estimates are noise, so the only honest allocation is a risk
  budget — every sleeve contributes the same portfolio risk (Maillard, Roncalli
  & Teiletche 2010; solved via the Spinu (2013) log-barrier convex program).
* **Ledoit-Wolf shrinkage** stabilizes the covariance (sklearn implementation).
* **Regime-mixed covariance** addresses the mandate's "hidden correlation":
  correlations are regime-dependent (they spike in panics), so per-vol-tier
  covariances are mixed by the *current* tier probabilities from the HMM
  posterior — the only validated thing the HMM emits.
* **Fractional Kelly** (``f = fraction * Sigma^{-1} mu``; full-covariance form,
  which already penalizes correlated bets) is implemented but OFF by default
  and only ever tilts sleeves explicitly marked forward-validated. Kelly on
  unvalidated mu-hat is leverage on estimation error (Michaud 1989).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIERS = ("low", "mid", "high")


@dataclass
class AllocatorConfig:
    """Allocator knobs (fixed defaults; Kelly gated)."""

    shrinkage: bool = True             # Ledoit-Wolf the sleeve covariance
    kelly_enabled: bool = False        # OFF: requires forward-validated sleeves
    kelly_fraction: float = 0.25       # quarter-Kelly when it does apply
    kelly_tilt_cap: float = 0.5        # max relative tilt away from the ERC budget
    min_obs: int = 60                  # min bars to estimate anything
    tier_probs: dict[str, float] | None = None   # current HMM tier probabilities
    tier_labels: np.ndarray | None = field(default=None, repr=False)  # per-bar tiers


def _shrunk_cov(returns: pd.DataFrame, shrinkage: bool) -> np.ndarray:
    """Sample or Ledoit-Wolf-shrunk covariance of sleeve returns."""
    X = returns.to_numpy(dtype=float)
    if not shrinkage or X.shape[0] < 2 * X.shape[1]:
        return np.cov(X, rowvar=False)
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(X).covariance_


def regime_mixed_covariance(
    returns: pd.DataFrame,
    tier_labels: np.ndarray,
    tier_probs: dict[str, float],
    min_obs: int = 60,
    shrinkage: bool = True,
) -> np.ndarray:
    """Posterior-mixed covariance: ``Sigma_t = sum_tier P(tier) Sigma_tier``.

    Each vol tier's covariance is estimated only on bars the HMM labelled with
    that tier, then mixed by the *current* tier probabilities. This is where
    "hidden correlation" lives: equity/defensive correlations estimated in calm
    markets understate panic co-movement, so a rising high-tier probability
    drags the allocation toward the panic-era covariance before the panic
    fully arrives.

    Args:
        returns: Sleeve return matrix (T x N), aligned with ``tier_labels``.
        tier_labels: Per-bar tier strings (``low|mid|high``), length T.
        tier_probs: Current tier probabilities (need not sum to 1; renormalized).
        min_obs: Tiers with fewer bars fall back to the pooled covariance.
        shrinkage: Ledoit-Wolf each per-tier estimate.

    Returns:
        Mixed covariance (N x N).
    """
    X = returns.to_numpy(dtype=float)
    pooled = _shrunk_cov(returns, shrinkage)
    labels = np.asarray(tier_labels)
    total = sum(max(0.0, float(tier_probs.get(t, 0.0))) for t in TIERS)
    if total <= 0.0 or len(labels) != X.shape[0]:
        return pooled

    mixed = np.zeros_like(pooled)
    for tier in TIERS:
        p = max(0.0, float(tier_probs.get(tier, 0.0))) / total
        if p == 0.0:
            continue
        mask = labels == tier
        if mask.sum() < min_obs:
            cov_t = pooled                       # thin tier -> pooled, not noise
        else:
            cov_t = _shrunk_cov(returns.loc[mask], shrinkage)
        mixed += p * cov_t
    return mixed


def erc_weights(cov: np.ndarray) -> list[float]:
    """Equal-risk-contribution weights (long-only, fully invested).

    Solves the Spinu (2013) convex program ``min 1/2 w'Sigma w - (1/n) sum log w``
    whose optimum has equal risk contributions ``w_i (Sigma w)_i``, then
    normalizes onto the simplex.

    Args:
        cov: Covariance matrix (N x N), positive definite.

    Returns:
        Weights summing to 1, all positive.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 1:
        return [1.0]

    # Cyclical coordinate descent (Spinu 2013): each coordinate's first-order
    # condition  cov_ii w_i^2 + (sum_{j!=i} cov_ij w_j) w_i - 1/n = 0  has a
    # positive root in closed form; cycling converges to machine precision.
    w = 1.0 / np.sqrt(np.diag(cov))          # inverse-vol start (exact if diagonal)
    for _ in range(10_000):
        w_prev = w.copy()
        for i in range(n):
            b = float(cov[i] @ w) - cov[i, i] * w[i]   # cross term sum_{j!=i}
            w[i] = (-b + np.sqrt(b * b + 4.0 * cov[i, i] / n)) / (2.0 * cov[i, i])
        if np.max(np.abs(w - w_prev)) < 1e-14:
            break
    w = w / w.sum()
    return [float(v) for v in w]


def kelly_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    fraction: float = 0.25,
    cap_gross: float | None = None,
) -> list[float]:
    """Fractional multi-asset Kelly: ``f = fraction * Sigma^{-1} mu``.

    The full-covariance form already shrinks correlated bets relative to naive
    per-asset ``mu/sigma^2`` sizing — that *is* the correlation penalty. The
    fraction guards against estimation error; the optional gross cap rescales
    proportionally (relative sizing preserved).

    Args:
        mu: Expected per-period excess returns (N,).
        cov: Covariance matrix (N x N).
        fraction: Kelly fraction (0.25 = quarter-Kelly).
        cap_gross: Max gross ``sum |f_i|`` (None = uncapped).

    Returns:
        Kelly fractions per asset.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    f = fraction * np.linalg.solve(cov, mu)
    if cap_gross is not None:
        gross = float(np.abs(f).sum())
        if gross > cap_gross > 0.0:
            f = f * (cap_gross / gross)
    return [float(v) for v in f]


def allocate(
    sleeve_returns: pd.DataFrame,
    config: AllocatorConfig,
    validated: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Risk-budget weights across sleeves (the library's one entry point).

    Default: ERC on the (shrunk, optionally regime-mixed) covariance — a pure
    risk budget. When Kelly is enabled, sleeves marked forward-validated get a
    bounded multiplicative tilt from their Kelly fraction; everything is then
    renormalized back onto the simplex (this allocates *budget*, not leverage).

    Args:
        sleeve_returns: Per-sleeve return columns (aligned daily bars).
        config: Allocator knobs.
        validated: ``{sleeve: passed_forward_gate}``; missing = not validated.

    Returns:
        ``{sleeve: weight}`` summing to 1.
    """
    names = list(sleeve_returns.columns)
    if len(sleeve_returns) < config.min_obs:
        return {s: 1.0 / len(names) for s in names}      # too little data: EW

    if config.tier_probs is not None and config.tier_labels is not None:
        cov = regime_mixed_covariance(
            sleeve_returns, config.tier_labels, config.tier_probs,
            min_obs=config.min_obs, shrinkage=config.shrinkage,
        )
    else:
        cov = _shrunk_cov(sleeve_returns, config.shrinkage)

    w = np.asarray(erc_weights(cov))

    use_kelly = (
        config.kelly_enabled
        and validated is not None
        and any(validated.get(s, False) for s in names)
    )
    if use_kelly:
        mu = sleeve_returns.mean().to_numpy()
        f = np.asarray(kelly_weights(mu, cov, fraction=config.kelly_fraction))
        f = np.clip(f, 0.0, None)                        # risk budget: no shorts
        scale = np.ones(len(names))
        ref = float(np.median(f[f > 0])) if (f > 0).any() else 1.0
        for i, s in enumerate(names):
            if validated.get(s, False) and ref > 0:
                # bounded multiplicative tilt around the ERC budget
                tilt = np.clip(f[i] / ref, 1.0 - config.kelly_tilt_cap,
                               1.0 + config.kelly_tilt_cap)
                scale[i] = tilt
        w = w * scale
        w = w / w.sum()

    return {s: float(v) for s, v in zip(names, w)}
