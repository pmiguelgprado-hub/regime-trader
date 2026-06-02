"""HMM regime-detection engine — a *volatility/return classifier*.

The HMM detects which statistical regime the market is in. It does **not**
predict price direction. Regimes are labelled by mean return (ascending) for
human readability; the strategy layer independently sorts by *volatility*
(see :attr:`RegimeInfo.expected_volatility`).

Design pillars
--------------
* **Automatic model selection** over candidate state counts via BIC, with
  multiple random restarts per candidate (hmmlearn has no built-in ``n_init``,
  so restarts are looped here and the best log-likelihood is kept).
* **No look-ahead bias.** Online inference uses the **forward algorithm**
  (filtered: ``P(state_t | obs_1:t)``) in log space — never ``model.predict()``
  (Viterbi), which revises past states using future data.
* **Regime stability**: a regime change is only *confirmed* after persisting
  ``stability_bars`` bars; rapid switching ("flicker") is detected and flagged.

Viterbi (``model.predict``) is used **only** at training time to characterize
each hidden state — never for online/backtest inference.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from data.feature_engineering import RET_1_IDX, RVOL_20_IDX

logger = logging.getLogger(__name__)


# ===========================================================================
# Regime labels
# ===========================================================================
class Regime(Enum):
    """Human-readable regime labels, ordered by mean return (ascending)."""

    CRASH = "crash"
    STRONG_BEAR = "strong_bear"
    BEAR = "bear"
    WEAK_BEAR = "weak_bear"
    NEUTRAL = "neutral"
    WEAK_BULL = "weak_bull"
    BULL = "bull"
    STRONG_BULL = "strong_bull"
    EUPHORIA = "euphoria"
    UNKNOWN = "unknown"


# Label schemes keyed by regime count, ordered low-return -> high-return.
LABEL_SCHEMES: dict[int, list[Regime]] = {
    3: [Regime.BEAR, Regime.NEUTRAL, Regime.BULL],
    4: [Regime.CRASH, Regime.BEAR, Regime.BULL, Regime.EUPHORIA],
    5: [Regime.CRASH, Regime.BEAR, Regime.NEUTRAL, Regime.BULL, Regime.EUPHORIA],
    6: [
        Regime.CRASH,
        Regime.STRONG_BEAR,
        Regime.WEAK_BEAR,
        Regime.WEAK_BULL,
        Regime.STRONG_BULL,
        Regime.EUPHORIA,
    ],
    7: [
        Regime.CRASH,
        Regime.STRONG_BEAR,
        Regime.WEAK_BEAR,
        Regime.NEUTRAL,
        Regime.WEAK_BULL,
        Regime.STRONG_BULL,
        Regime.EUPHORIA,
    ],
}

# Default strategy hints per label (overridable by the strategy layer, which
# is authoritative). These exist so each regime carries actionable metadata.
_STRATEGY_HINTS: dict[Regime, dict] = {
    Regime.CRASH: dict(strat="defensive", lev=0.0, size=0.05, conf=0.65),
    Regime.STRONG_BEAR: dict(strat="defensive", lev=0.0, size=0.10, conf=0.60),
    Regime.BEAR: dict(strat="reduced", lev=0.0, size=0.15, conf=0.58),
    Regime.WEAK_BEAR: dict(strat="reduced", lev=0.0, size=0.20, conf=0.55),
    Regime.NEUTRAL: dict(strat="balanced", lev=1.0, size=0.25, conf=0.55),
    Regime.WEAK_BULL: dict(strat="trend", lev=1.0, size=0.30, conf=0.55),
    Regime.BULL: dict(strat="trend", lev=1.0, size=0.35, conf=0.55),
    Regime.STRONG_BULL: dict(strat="trend", lev=1.25, size=0.40, conf=0.55),
    Regime.EUPHORIA: dict(strat="momentum", lev=1.25, size=0.40, conf=0.60),
    Regime.UNKNOWN: dict(strat="defensive", lev=0.0, size=0.0, conf=1.0),
}


# ===========================================================================
# Dataclasses
# ===========================================================================
@dataclass
class RegimeInfo:
    """Static characterization of one hidden state / regime.

    Attributes:
        regime_id: Hidden-state index (HMM internal).
        regime_name: Human-readable :class:`Regime` label.
        expected_return: Mean of the ``ret_1`` feature for this state
            (standardized units; ordering is what matters for labelling).
        expected_volatility: Mean of the ``rvol_20`` feature for this state.
            The strategy layer sorts regimes by this value.
        recommended_strategy_type: Hint for the strategy layer.
        max_leverage_allowed: Suggested leverage ceiling for this regime.
        max_position_size_pct: Suggested per-position size cap.
        min_confidence_to_act: Min filtered probability to act on this regime.
    """

    regime_id: int
    regime_name: Regime
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Filtered regime estimate at a single bar.

    Attributes:
        label: Most-probable regime label at this bar (raw argmax).
        state_id: Most-probable hidden-state index.
        probability: Filtered probability of ``state_id`` (max posterior).
        state_probabilities: Full filtered distribution over states.
        timestamp: Bar timestamp.
        is_confirmed: True once the regime has persisted ``stability_bars``.
        consecutive_bars: Consecutive bars the raw regime has held.
    """

    label: Regime = Regime.UNKNOWN
    state_id: int = -1
    probability: float = 0.0
    state_probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))
    timestamp: Optional[pd.Timestamp] = None
    is_confirmed: bool = False
    consecutive_bars: int = 0


@dataclass
class HMMConfig:
    """Configuration for the HMM engine (mirrors the ``hmm`` settings block)."""

    n_candidates: list[int] = field(default_factory=lambda: [3, 4, 5])
    n_init: int = 10
    covariance_type: str = "full"
    min_train_bars: int = 504          # usable rows AFTER feature warmup (~2y)
    n_iter: int = 100
    min_covar: float = 1e-3
    covars_prior: float = 1e-2
    random_state: int = 42
    stability_bars: int = 3
    flicker_window: int = 20
    flicker_threshold: int = 4
    min_confidence: float = 0.55
    transition_size_reduction: float = 0.25


@dataclass
class HMMMetadata:
    """Persisted model metadata."""

    n_regimes: int
    bic: float
    log_likelihood: float
    training_date: str
    converged: bool
    n_iter: int
    feature_columns: list[str]
    labels: list[str]
    all_bic: dict[int, float]


# ===========================================================================
# Engine
# ===========================================================================
class HMMEngine:
    """Fits and serves a Gaussian HMM for market-regime detection."""

    def __init__(self, config: HMMConfig | None = None) -> None:
        """Initialize the engine.

        Args:
            config: HMM hyperparameters and gating thresholds.
        """
        self.config = config or HMMConfig()
        self.model: Optional[GaussianHMM] = None
        self.n_regimes: int = 0
        self.feature_columns: list[str] = []
        # state_id -> Regime label (assigned by ascending mean return)
        self.labels: dict[int, Regime] = {}
        self.regime_info: dict[int, RegimeInfo] = {}
        self.metadata: Optional[HMMMetadata] = None
        # rolling inference state (set by predict_regime_filtered)
        self._states: list[RegimeState] = []

    # ---------------------------------------------------------------- BIC ---
    @staticmethod
    def _free_params(n: int, d: int) -> int:
        """Number of free parameters for a full-covariance Gaussian HMM.

        Args:
            n: Number of hidden states.
            d: Feature dimensionality.

        Returns:
            Free-parameter count: startprob + transmat + means + full covars.
        """
        startprob = n - 1
        transmat = n * (n - 1)
        means = n * d
        covars = n * d * (d + 1) // 2
        return startprob + transmat + means + covars

    @classmethod
    def _bic(cls, log_likelihood: float, n: int, d: int, n_samples: int) -> float:
        """Bayesian Information Criterion (lower is better).

        ``BIC = -2 * log_likelihood + n_params * log(n_samples)``

        Args:
            log_likelihood: Total log-likelihood of the data under the model.
            n: Number of hidden states.
            d: Feature dimensionality.
            n_samples: Number of observations.

        Returns:
            BIC score.
        """
        return -2.0 * log_likelihood + cls._free_params(n, d) * np.log(n_samples)

    # ------------------------------------------------------------- fitting ---
    def _fit_single(self, X: np.ndarray, n: int, seed: int) -> tuple[GaussianHMM, float]:
        """Fit one HMM with a given state count and seed.

        Args:
            X: Feature matrix (n_samples, n_features).
            n: Number of hidden states.
            seed: Random seed for this restart.

        Returns:
            Tuple of (fitted model, total log-likelihood). Log-likelihood is
            ``-inf`` if the fit failed (e.g. singular covariance).
        """
        model = GaussianHMM(
            n_components=n,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            min_covar=self.config.min_covar,
            covars_prior=self.config.covars_prior,
            random_state=seed,
        )
        try:
            model.fit(X)
            ll = float(model.score(X))
            if not np.isfinite(ll):
                ll = float("-inf")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HMM fit failed (n=%d seed=%d): %s", n, seed, exc)
            ll = float("-inf")
        return model, ll

    def _select_and_fit(self, X: np.ndarray) -> tuple[GaussianHMM, int, dict[int, float]]:
        """Select the best state count by BIC, with restarts per candidate.

        For each candidate ``n``: run ``n_init`` restarts, keep the highest
        log-likelihood model, score its BIC. Choose the candidate with the
        lowest BIC. Logs every candidate's BIC and the selection.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Tuple of (best model, best n, {n: bic} for all candidates).
        """
        n_samples, d = X.shape
        best_per_n: dict[int, tuple[GaussianHMM, float]] = {}
        all_bic: dict[int, float] = {}

        for n in self.config.n_candidates:
            best_model: Optional[GaussianHMM] = None
            best_ll = float("-inf")
            for i in range(self.config.n_init):
                seed = self.config.random_state + i
                model, ll = self._fit_single(X, n, seed)
                if ll > best_ll:
                    best_ll, best_model = ll, model
            if best_model is None or not np.isfinite(best_ll):
                logger.warning("All restarts failed for n=%d; skipping", n)
                continue
            bic = self._bic(best_ll, n, d, n_samples)
            best_per_n[n] = (best_model, best_ll)
            all_bic[n] = bic
            logger.info(
                "HMM candidate n=%d: log_likelihood=%.2f BIC=%.2f converged=%s iter=%d",
                n, best_ll, bic, best_model.monitor_.converged, best_model.monitor_.iter,
            )

        if not all_bic:
            raise RuntimeError("No HMM candidate converged on the given data")

        best_n = min(all_bic, key=all_bic.get)
        logger.info(
            "HMM model selection: chose n=%d (BIC=%.2f) from candidates %s",
            best_n, all_bic[best_n], {k: round(v, 1) for k, v in all_bic.items()},
        )
        return best_per_n[best_n][0], best_n, all_bic

    def fit(self, features: pd.DataFrame) -> None:
        """Fit the HMM on standardized features and label the regimes.

        Args:
            features: Standardized, NaN-free feature matrix (rows >=
                ``min_train_bars``), columns in canonical order.

        Raises:
            ValueError: If too few rows are supplied.
        """
        if len(features) < self.config.min_train_bars:
            raise ValueError(
                f"need >= {self.config.min_train_bars} usable bars, got {len(features)}"
            )
        X = features.to_numpy(dtype=float)
        self.feature_columns = list(features.columns)

        model, n, all_bic = self._select_and_fit(X)
        self.model = model
        self.n_regimes = n
        self._assign_labels(X)

        ll = float(model.score(X))
        self.metadata = HMMMetadata(
            n_regimes=n,
            bic=all_bic[n],
            log_likelihood=ll,
            training_date=datetime.now(timezone.utc).isoformat(),
            converged=bool(model.monitor_.converged),
            n_iter=int(model.monitor_.iter),
            feature_columns=self.feature_columns,
            labels=[self.labels[i].value for i in range(n)],
            all_bic=all_bic,
        )

    def _assign_labels(self, X: np.ndarray) -> None:
        """Label states by ascending mean return; build :class:`RegimeInfo`.

        Uses the per-state means of the ``ret_1`` and ``rvol_20`` features
        (standardized; ordering is preserved by standardization). Viterbi is
        **not** needed here — the fitted means characterize each state.

        Args:
            X: Feature matrix used for fitting.
        """
        assert self.model is not None
        means = self.model.means_                       # (n, d)
        exp_return = means[:, RET_1_IDX]
        exp_vol = means[:, RVOL_20_IDX]

        # ascending mean return -> label order
        order = np.argsort(exp_return)
        scheme = LABEL_SCHEMES[self.n_regimes]
        self.labels = {int(state_id): scheme[rank] for rank, state_id in enumerate(order)}

        self.regime_info = {}
        for state_id in range(self.n_regimes):
            label = self.labels[state_id]
            hint = _STRATEGY_HINTS[label]
            self.regime_info[state_id] = RegimeInfo(
                regime_id=state_id,
                regime_name=label,
                expected_return=float(exp_return[state_id]),
                expected_volatility=float(exp_vol[state_id]),
                recommended_strategy_type=hint["strat"],
                max_leverage_allowed=hint["lev"],
                max_position_size_pct=hint["size"],
                min_confidence_to_act=hint["conf"],
            )

    # ------------------------------------------------------- emissions ---
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Per-state Gaussian log-emission probabilities.

        Computed manually from the fitted ``means_``/``covars_`` (full
        covariance) for version-independence and explicitness.

        Args:
            X: Feature matrix (T, d).

        Returns:
            Log-emission matrix (T, n_states).
        """
        assert self.model is not None
        n = self.n_regimes
        out = np.empty((X.shape[0], n))
        for k in range(n):
            out[:, k] = multivariate_normal.logpdf(
                X, mean=self.model.means_[k], cov=self.model.covars_[k],
                allow_singular=True,
            )
        return out

    # ------------------------------------------- filtered inference ---
    def _forward_filter(self, X: np.ndarray) -> np.ndarray:
        """Forward algorithm: filtered log-posteriors ``log P(s_t | o_1:t)``.

        Scaled (normalized each step) forward pass in log space. Strictly
        causal: row ``t`` depends only on observations ``0..t``.

        Args:
            X: Feature matrix (T, d).

        Returns:
            Filtered log-posterior matrix (T, n_states); each row
            log-sums to 0 (i.e. ``exp`` rows sum to 1).
        """
        assert self.model is not None
        # zeros -> -inf is intended (impossible transitions); logsumexp handles it
        with np.errstate(divide="ignore"):
            log_startprob = np.log(self.model.startprob_)
            log_transmat = np.log(self.model.transmat_)
        emis = self._log_emission(X)
        T, n = emis.shape
        log_post = np.empty((T, n))

        # t = 0
        a = log_startprob + emis[0]
        log_post[0] = a - logsumexp(a)
        # t = 1..T-1
        for t in range(1, T):
            # predict: log P(s_t=j | o_1:t-1) = logsumexp_i(post_{t-1,i} + A_ij)
            pred = logsumexp(log_post[t - 1][:, None] + log_transmat, axis=0)
            joint = pred + emis[t]
            log_post[t] = joint - logsumexp(joint)
        return log_post

    def predict_regime_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Filtered posterior distribution over regimes for each bar.

        Args:
            features: Standardized feature matrix (same columns as training).

        Returns:
            DataFrame (rows = bars, columns = state ids) of filtered
            probabilities, indexed by timestamp.
        """
        self._check_fitted()
        X = features[self.feature_columns].to_numpy(dtype=float)
        post = np.exp(self._forward_filter(X))
        return pd.DataFrame(post, index=features.index, columns=range(self.n_regimes))

    def predict_regime_filtered(self, features: pd.DataFrame) -> list[RegimeState]:
        """Filtered regime estimate per bar (forward algorithm, no look-ahead).

        Applies the stability filter sequentially (causally) to mark
        confirmation and count consecutive bars. The raw argmax regime and
        full distribution are always exposed.

        Args:
            features: Standardized feature matrix (same columns as training).

        Returns:
            List of :class:`RegimeState`, one per bar, in time order. Also
            cached on the engine for the stability/flicker getters.
        """
        self._check_fitted()
        X = features[self.feature_columns].to_numpy(dtype=float)
        log_post = self._forward_filter(X)
        post = np.exp(log_post)
        argmax = post.argmax(axis=1)

        states: list[RegimeState] = []
        confirmed_label: Optional[Regime] = None
        prev_raw: Optional[int] = None
        run_len = 0

        for t in range(len(X)):
            sid = int(argmax[t])
            label = self.labels[sid]
            prob = float(post[t, sid])

            if sid == prev_raw:
                run_len += 1
            else:
                # raw regime changed
                if prev_raw is not None:
                    logger.warning(
                        "Regime change (unconfirmed) %s -> %s @ %s (p=%.2f)",
                        self.labels.get(prev_raw), label, features.index[t], prob,
                    )
                run_len = 1
                prev_raw = sid

            is_confirmed = run_len >= self.config.stability_bars
            if is_confirmed and label != confirmed_label:
                logger.info(
                    "Regime confirmed: %s @ %s (held %d bars, p=%.2f)",
                    label, features.index[t], run_len, prob,
                )
                confirmed_label = label

            states.append(
                RegimeState(
                    label=label,
                    state_id=sid,
                    probability=prob,
                    state_probabilities=post[t].copy(),
                    timestamp=features.index[t],
                    is_confirmed=is_confirmed,
                    consecutive_bars=run_len,
                )
            )

        self._states = states
        return states

    # ------------------------------------------------- stability/flicker ---
    def get_regime_stability(self) -> int:
        """Consecutive bars the current (latest) raw regime has held.

        Returns:
            Run length of the latest regime (0 if no inference run yet).
        """
        return self._states[-1].consecutive_bars if self._states else 0

    def get_transition_matrix(self) -> np.ndarray:
        """Learned state-transition probability matrix.

        Returns:
            ``transmat_`` of shape (n_states, n_states).
        """
        self._check_fitted()
        return self.model.transmat_  # type: ignore[union-attr]

    def detect_regime_change(self) -> bool:
        """Whether the latest bar is a *confirmed* regime change.

        Returns:
            True only if the latest bar just reached confirmation and differs
            from the previously confirmed regime.
        """
        if len(self._states) < 2:
            return False
        last = self._states[-1]
        if not last.is_confirmed:
            return False
        # most recent confirmed label before `last`
        for s in reversed(self._states[:-1]):
            if s.is_confirmed:
                return s.label != last.label
        return True

    def get_regime_flicker_rate(self) -> int:
        """Number of raw regime changes within the trailing flicker window.

        Returns:
            Count of label changes over the last ``flicker_window`` bars.
        """
        window = self._states[-self.config.flicker_window :]
        if len(window) < 2:
            return 0
        return sum(
            1 for a, b in zip(window[:-1], window[1:]) if a.state_id != b.state_id
        )

    def is_flickering(self) -> bool:
        """Whether the regime is switching too rapidly to be trustworthy.

        Returns:
            True if the flicker rate exceeds ``flicker_threshold``.
        """
        return self.get_regime_flicker_rate() > self.config.flicker_threshold

    # ---------------------------------------------------------- persistence ---
    def save(self, path: str | Path) -> None:
        """Pickle the model, labels, regime info, and metadata.

        Args:
            path: Destination ``.pkl`` path.
        """
        self._check_fitted()
        payload = {
            "model": self.model,
            "n_regimes": self.n_regimes,
            "feature_columns": self.feature_columns,
            "labels": {k: v.value for k, v in self.labels.items()},
            "regime_info": self.regime_info,
            "metadata": self.metadata,
            "config": self.config,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Saved HMM model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HMMEngine":
        """Load a pickled engine.

        Security: only load model files produced by this application's
        :meth:`save` (trusted, self-generated artifacts) — pickle executes
        arbitrary code on load, so never point this at an untrusted file.

        Args:
            path: Source ``.pkl`` path.

        Returns:
            Restored :class:`HMMEngine`.
        """
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        engine = cls(payload["config"])
        engine.model = payload["model"]
        engine.n_regimes = payload["n_regimes"]
        engine.feature_columns = payload["feature_columns"]
        engine.labels = {k: Regime(v) for k, v in payload["labels"].items()}
        engine.regime_info = payload["regime_info"]
        engine.metadata = payload["metadata"]
        return engine

    def mean_log_likelihood(self, features: pd.DataFrame) -> float:
        """Per-bar log-likelihood of ``features`` under the fitted model.

        The champion-challenger promotion score (A-4): a model that explains a
        holdout better has a higher value. Normalized per bar so windows of
        different lengths compare directly.

        Args:
            features: Standardized feature matrix (same columns as training).

        Returns:
            ``model.score(X) / n_bars`` (``-inf`` if empty).
        """
        self._check_fitted()
        X = features[self.feature_columns].to_numpy(dtype=float)
        if len(X) == 0:
            return float("-inf")
        return float(self.model.score(X)) / len(X)

    def get_regime_info(self, state_id: int) -> RegimeInfo:
        """Return the :class:`RegimeInfo` for a hidden-state id.

        Args:
            state_id: Hidden-state index.

        Returns:
            The state's :class:`RegimeInfo`.
        """
        return self.regime_info[state_id]

    def _check_fitted(self) -> None:
        """Raise if the engine has not been fitted.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self.model is None:
            raise RuntimeError("HMMEngine is not fitted; call fit() first")
