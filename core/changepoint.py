"""Bayesian Online Changepoint Detection (T1.2, Adams & MacKay 2007).

A **model-free** corroborator of the transition hazard in ``core.meta_overlay``
(which is internal to the HMM). BOCPD maintains the posterior over the current
*run length* (bars since the last changepoint) under a constant hazard, with a
Normal-inverse-Gamma conjugate predictive (Student-t marginal) that learns the
mean and variance online per run. The changepoint score is the posterior mass
that the run just reset.

Shadow only: a possible future use is corroborating the hedge trigger, but only
via a pre-registered amendment after the current gate closes (roadmap §T1.2).
Pure NumPy, deterministic. Inputs are a 1-D series (e.g. proxy daily returns).
"""

from __future__ import annotations

import numpy as np


def _student_t_pdf(x: np.ndarray, mu: np.ndarray, kappa: np.ndarray,
                   alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Posterior predictive (Student-t) of a Normal-inverse-Gamma, vectorized over runs."""
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    scale = np.sqrt(scale2)
    z = (x - mu) / scale
    from math import lgamma
    # gamma ratio per run-length (alpha varies) — compute elementwise
    coef = np.exp(np.array([lgamma(a + 0.5) - lgamma(a) for a in alpha]))
    norm = coef / (np.sqrt(nu * np.pi) * scale)
    return norm * (1.0 + (z * z) / nu) ** (-(nu + 1.0) / 2.0)


def bocpd(data, hazard_lambda: float = 100.0,
          mu0: float = 0.0, kappa0: float = 1.0,
          alpha0: float = 1.0, beta0: float = 1.0) -> np.ndarray:
    """Per-step changepoint probability (mass on run-length 0 after the update).

    Args:
        data: 1-D observation series.
        hazard_lambda: Expected run length (constant hazard H = 1/lambda).
        mu0, kappa0, alpha0, beta0: Normal-inverse-Gamma prior hyperparameters.

    Returns:
        Array (len == len(data)) of changepoint probabilities in [0, 1]
        (``[0]`` is 0.0 — no prior bar to break from).
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if T < 2:
        return np.zeros(T)
    H = 1.0 / float(hazard_lambda)

    # run-length posterior (grows by one each step); NIG params per run length
    R = np.array([1.0])
    mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])
    cp = np.zeros(T)

    for t in range(T):
        pred = _student_t_pdf(x[t], mu, kappa, alpha, beta)      # predictive per run
        growth = R * pred * (1.0 - H)                            # run continues
        cp_mass = float(np.sum(R * pred * H))                    # run resets
        new_R = np.empty(R.size + 1)
        new_R[0] = cp_mass
        new_R[1:] = growth
        s = new_R.sum()
        if s > 0:
            new_R /= s
        R = new_R
        cp[t] = R[0]

        # update NIG params: prepend the prior (fresh run), bump existing runs
        new_mu = np.empty(mu.size + 1); new_kappa = np.empty(mu.size + 1)
        new_alpha = np.empty(mu.size + 1); new_beta = np.empty(mu.size + 1)
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0
        new_kappa[1:] = kappa + 1.0
        new_alpha[1:] = alpha + 0.5
        new_mu[1:] = (kappa * mu + x[t]) / (kappa + 1.0)
        new_beta[1:] = beta + (kappa * (x[t] - mu) ** 2) / (2.0 * (kappa + 1.0))
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        # cap the run-length vector to keep it bounded (drop negligible tail)
        if R.size > 300:
            R, mu, kappa, alpha, beta = (a[:300] for a in (R, mu, kappa, alpha, beta))
            R = R / R.sum()
    return cp


def changepoint_score(data, hazard_lambda: float = 100.0, **prior) -> float:
    """Latest-bar changepoint probability (0.0 for a too-short series)."""
    cp = bocpd(data, hazard_lambda=hazard_lambda, **prior)
    return float(cp[-1]) if cp.size else 0.0
