"""Tests for in-loop retrain + propagation to the live orchestrator (A-1).

The live model used to refresh only on process startup (file-age check). A
long-running daily loop never refreshed it. And when it does refit, the new
engine MUST be pushed into the orchestrator via ``update_regime_infos`` or the
vol-rank map keeps pointing at the old model's states. These tests pin both.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import core.hmm_engine as hmm_engine
from conftest import make_synthetic_ohlcv
from core.hmm_engine import HMMConfig, HMMEngine, Regime, RegimeInfo
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from core.risk_manager import RiskConfig, RiskManager
from data.feature_engineering import FeatureEngineer
from main import TradingSystem

for _n in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

CONFIG = {"broker": {"symbols": ["SPY"], "timeframe": "1Day"},
          "hmm": {"min_train_bars": 504, "max_age_days": 7},
          "backtest": {"initial_capital": 100000}}


def _config(**hmm_overrides) -> dict:
    """Deep copy of CONFIG with hmm overrides (avoids cross-test mutation)."""
    cfg = copy.deepcopy(CONFIG)
    cfg["hmm"].update(hmm_overrides)
    return cfg


def _ri(rid: int, vol: float) -> RegimeInfo:
    """Minimal RegimeInfo with a controllable expected_volatility."""
    return RegimeInfo(
        regime_id=rid, regime_name=Regime.NEUTRAL, expected_return=0.0,
        expected_volatility=vol, recommended_strategy_type="balanced",
        max_leverage_allowed=1.0, max_position_size_pct=0.25, min_confidence_to_act=0.55,
    )


def _fitted_dry_system(config: dict | None = None) -> TradingSystem:
    ohlcv = make_synthetic_ohlcv()
    fe = FeatureEngineer()
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    hmm.fit(fe.build_features(ohlcv))
    orch = StrategyOrchestrator(StrategyConfig(), hmm.regime_info)
    return TradingSystem(config or CONFIG, hmm, orch, RiskManager(RiskConfig()), fe, dry_run=True)


# ---------------------------------------------------------- propagation ---
def test_install_model_swaps_engine_and_propagates_to_orchestrator() -> None:
    """install_model replaces the engine AND rebuilds the orchestrator's vol map."""
    orch = StrategyOrchestrator(StrategyConfig(), {0: _ri(0, vol=0.1), 1: _ri(1, vol=0.9)})
    sys_ = TradingSystem(CONFIG, hmm=object(), orchestrator=orch,
                         risk_manager=None, feature_engineer=None, dry_run=True)
    # new model with the volatilities swapped between states
    new = SimpleNamespace(regime_info={0: _ri(0, vol=0.9), 1: _ri(1, vol=0.1)})

    sys_.install_model(new)

    assert sys_.hmm is new
    assert orch.regime_infos == new.regime_info               # propagated
    assert orch.vol_rank[1] < orch.vol_rank[0]                # map reflects NEW vols


# ------------------------------------------------------------- retrain ---
def test_retrain_from_buffer_refits_and_propagates() -> None:
    """A retrain off the live buffer installs a fresh engine and rewires the map."""
    sys_ = _fitted_dry_system()
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    old = sys_.hmm

    assert sys_.retrain_from_buffer("SPY") is True
    assert sys_.hmm is not old
    assert sys_.orchestrator.regime_infos == sys_.hmm.regime_info


def test_retrain_from_buffer_skips_when_insufficient_data() -> None:
    """Too little history -> no retrain, engine left untouched (no crash)."""
    sys_ = _fitted_dry_system()
    old = sys_.hmm
    sys_.buffers["SPY"] = make_synthetic_ohlcv().head(50)

    assert sys_.retrain_from_buffer("SPY") is False
    assert sys_.hmm is old


# ------------------------------------------------------- age trigger ---
def _age_model(sys_: TradingSystem, days: float) -> None:
    sys_.hmm.metadata.training_date = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()


def test_maybe_retrain_fires_when_stale_and_enabled() -> None:
    """With auto_retrain on, a model older than max_age_days retrains in-loop."""
    sys_ = _fitted_dry_system(_config(auto_retrain=True))
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    old = sys_.hmm
    _age_model(sys_, 30)

    assert sys_.maybe_retrain("SPY") is True
    assert sys_.hmm is not old


def test_maybe_retrain_disabled_by_default() -> None:
    """Even a stale model is NOT auto-retrained unless auto_retrain is enabled."""
    sys_ = _fitted_dry_system()  # CONFIG has no auto_retrain -> default off
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    old = sys_.hmm
    _age_model(sys_, 30)

    assert sys_.maybe_retrain("SPY") is False
    assert sys_.hmm is old


def test_maybe_retrain_skips_when_fresh_even_if_enabled() -> None:
    """A freshly-trained model is not retrained, even with auto_retrain on."""
    sys_ = _fitted_dry_system(_config(auto_retrain=True))
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    old = sys_.hmm
    _age_model(sys_, 0)

    assert sys_.maybe_retrain("SPY") is False
    assert sys_.hmm is old


# --------------------------------------------------- promotion gate ---
def test_retrain_rejects_nonconverged_fit(monkeypatch) -> None:
    """A challenger that did not converge is not promoted (floor gate before A-4)."""

    class _NonConverged:
        def __init__(self, cfg):
            self.config = cfg
            self.regime_info = {0: _ri(0, vol=0.5)}

        def fit(self, feats):
            self.metadata = SimpleNamespace(converged=False)

    sys_ = _fitted_dry_system()
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    old = sys_.hmm
    monkeypatch.setattr(hmm_engine, "HMMEngine", _NonConverged)

    assert sys_.retrain_from_buffer("SPY") is False
    assert sys_.hmm is old   # bad fit not installed


def test_successful_retrain_persists_and_promotes_in_registry(tmp_path) -> None:
    """A promoted challenger is saved and marked champion in the registry (A-4 rollback)."""
    from core.model_registry import ModelRegistry

    reg = ModelRegistry(tmp_path)
    sys_ = _fitted_dry_system()
    sys_.registry = reg
    sys_.buffers["SPY"] = make_synthetic_ohlcv()

    assert sys_.retrain_from_buffer("SPY") is True
    assert reg.champion_version("SPY") is not None
    assert reg.load_champion("SPY") is not None


def test_challenger_rejected_when_worse_than_champion(monkeypatch) -> None:
    """Champion-challenger gate (A-4): a challenger that explains the holdout
    worse than the current champion is NOT promoted."""

    class _WorseFit:
        def __init__(self, cfg):
            self.config = cfg
            self.regime_info = {0: _ri(0, vol=0.5)}

        def fit(self, feats):
            self.metadata = SimpleNamespace(converged=True)

        def mean_log_likelihood(self, feats):
            return -1e9  # far worse than the real champion

    sys_ = _fitted_dry_system()
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    champ = sys_.hmm
    monkeypatch.setattr(hmm_engine, "HMMEngine", _WorseFit)

    assert sys_.retrain_from_buffer("SPY") is False
    assert sys_.hmm is champ   # champion kept
