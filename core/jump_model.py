"""Statistical Jump Model — regime engine challenger (T1.1, shadow-only).

Shu, Mulvey & Kolm (arXiv 2402.05272) and Nystrup et al.: cluster feature rows
like k-means, but add a **jump penalty** ``λ`` that charges every change of state
along the time axis. The penalty makes the inferred regime sequence persistent,
so it flickers far less than an HMM's argmax path — and flicker is exactly the
known pain of the deployed HMM (``monitoring/alerts.flicker_exceeded``). In the
papers the more-persistent labels translate into better downside-risk timing.

This is a **shadow** engine: it never drives orders. It runs alongside the
champion HMM, its disagreement logged for the monthly shadow report (T1.4).
Promotion would require a NEW pre-registered book (roadmap §0), never a hot-swap.

Optimisation (coordinate descent, to a local optimum per restart; best of
``n_init`` kept):

1. **Assign** — given centroids, find the state sequence minimizing
   ``Σ ‖x_t − μ_{s_t}‖² + λ·Σ 1[s_t ≠ s_{t-1}]`` by dynamic programming (a Viterbi
   over a uniform jump cost λ).
2. **Update** — set each centroid to the mean of its assigned rows.

Pure NumPy, deterministic given ``random_state`` (k-means++ seeding with a fixed
RNG). Standardizes features internally (z-score) so the Euclidean metric is scale-free.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class JumpModel:
    """Statistical jump model: persistent regime clustering with a jump penalty."""

    def __init__(self, n_states: int = 3, jump_penalty: float = 20.0,
                 n_init: int = 5, max_iter: int = 50, random_state: int = 42) -> None:
        """Args:
            n_states: Number of regimes K.
            jump_penalty: λ — cost charged per state change (higher = more persistent).
            n_init: Random restarts (best total loss kept).
            max_iter: Max coordinate-descent iterations per restart.
            random_state: RNG seed (determinism).
        """
        self.n_states = int(n_states)
        self.jump_penalty = float(jump_penalty)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.centroids_: Optional[np.ndarray] = None
        self.states_: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._sd: Optional[np.ndarray] = None
        self._dispersion: Optional[dict[int, float]] = None

    # --- internals ----------------------------------------------------------------
    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0)
            self._sd[self._sd == 0] = 1.0
        return (X - self._mu) / self._sd

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> "tuple[np.ndarray, float]":
        """DP state assignment minimizing distortion + λ·(state changes)."""
        T = X.shape[0]
        # squared euclidean distance to each centroid: (T, K)
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        K = centroids.shape[0]
        V = np.empty((T, K))
        back = np.zeros((T, K), dtype=int)
        V[0] = d[0]
        for t in range(1, T):
            for k in range(K):
                trans = V[t - 1] + self.jump_penalty * (np.arange(K) != k)
                j = int(np.argmin(trans))
                back[t, k] = j
                V[t, k] = d[t, k] + trans[j]
        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmin(V[-1]))
        for t in range(T - 1, 0, -1):
            states[t - 1] = back[t, states[t]]
        return states, float(V[-1, states[-1]])

    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        T = X.shape[0]
        idx = [int(rng.integers(T))]
        for _ in range(1, self.n_states):
            d2 = np.min(((X[:, None, :] - X[np.array(idx)][None, :, :]) ** 2).sum(axis=2), axis=1)
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(T, 1.0 / T)
            idx.append(int(rng.choice(T, p=probs)))
        return X[np.array(idx)].copy()

    # --- public API ---------------------------------------------------------------
    def fit(self, features: pd.DataFrame) -> "JumpModel":
        """Fit the model; populates ``states_`` (per-row labels) and ``centroids_``."""
        self.feature_columns = list(features.columns)
        X = self._standardize(features.to_numpy(dtype=float), fit=True)
        best_loss = float("inf")
        best_states = best_cent = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.random_state + i)
            cent = self._kmeanspp_init(X, rng)
            states = np.zeros(X.shape[0], dtype=int)
            for _ in range(self.max_iter):
                states, loss = self._assign(X, cent)
                new = cent.copy()
                for k in range(self.n_states):
                    rows = X[states == k]
                    if len(rows):
                        new[k] = rows.mean(axis=0)
                if np.allclose(new, cent):
                    cent = new
                    break
                cent = new
            states, loss = self._assign(X, cent)
            if loss < best_loss:
                best_loss, best_states, best_cent = loss, states, cent
        self.states_, self.centroids_ = best_states, best_cent
        # Vol proxy = within-state dispersion (RMS distance of assigned rows to their
        # centroid, in z-space). Direct volatility measure — works whether regimes
        # differ in mean (level features like rvol_20) or in variance.
        self._dispersion = {}
        for k in range(self.n_states):
            rows = X[best_states == k]
            self._dispersion[k] = (float(np.sqrt(((rows - best_cent[k]) ** 2).sum(axis=1).mean()))
                                   if len(rows) else 0.0)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """State sequence for new rows (DP assignment against fitted centroids)."""
        if self.centroids_ is None:
            raise RuntimeError("JumpModel is not fitted")
        X = self._standardize(features[self.feature_columns].to_numpy(dtype=float), fit=False)
        states, _ = self._assign(X, self.centroids_)
        return states

    def regime_labels(self) -> dict[int, int]:
        """Map raw state id -> vol-tier rank 0..K-1 (0 = calmest), by within-state dispersion.

        Comparable to the HMM's ascending-vol ordering, so the two engines' labels
        can be compared directly in the shadow report.
        """
        if self._dispersion is None:
            raise RuntimeError("JumpModel is not fitted")
        order = sorted(self._dispersion, key=lambda k: self._dispersion[k])
        return {state: rank for rank, state in enumerate(order)}

    def vol_rank(self) -> float:
        """Latest bar's vol tier scaled to [0, 1] (label / (K-1))."""
        if self.states_ is None:
            raise RuntimeError("JumpModel is not fitted")
        labels = self.regime_labels()
        denom = max(1, self.n_states - 1)
        return labels[int(self.states_[-1])] / denom
