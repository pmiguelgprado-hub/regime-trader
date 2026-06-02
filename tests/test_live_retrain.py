"""Tests for in-loop retrain + propagation to the live orchestrator (A-1).

The live model used to refresh only on process startup (file-age check). A
long-running daily loop never refreshed it. And when it does refit, the new
engine MUST be pushed into the orchestrator via ``update_regime_infos`` or the
vol-rank map keeps pointing at the old model's states. These tests pin both.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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


def _ri(rid: int, vol: float) -> RegimeInfo:
    """Minimal RegimeInfo with a controllable expected_volatility."""
    return RegimeInfo(
        regime_id=rid, regime_name=Regime.NEUTRAL, expected_return=0.0,
        expected_volatility=vol, recommended_strategy_type="balanced",
        max_leverage_allowed=1.0, max_position_size_pct=0.25, min_confidence_to_act=0.55,
    )


def _fitted_dry_system() -> TradingSystem:
    ohlcv = make_synthetic_ohlcv()
    fe = FeatureEngineer()
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    hmm.fit(fe.build_features(ohlcv))
    orch = StrategyOrchestrator(StrategyConfig(), hmm.regime_info)
    return TradingSystem(CONFIG, hmm, orch, RiskManager(RiskConfig()), fe, dry_run=True)


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
def test_maybe_retrain_fires_when_model_is_stale() -> None:
    """A model older than max_age_days triggers an in-loop retrain."""
    sys_ = _fitted_dry_system()
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    old = sys_.hmm
    sys_.hmm.metadata.training_date = (
        datetime.now(timezone.utc) - timedelta(days=30)
    ).isoformat()

    assert sys_.maybe_retrain("SPY") is True
    assert sys_.hmm is not old


def test_maybe_retrain_skips_when_model_is_fresh() -> None:
    """A freshly-trained model is not retrained."""
    sys_ = _fitted_dry_system()
    sys_.buffers["SPY"] = make_synthetic_ohlcv()
    old = sys_.hmm
    sys_.hmm.metadata.training_date = datetime.now(timezone.utc).isoformat()

    assert sys_.maybe_retrain("SPY") is False
    assert sys_.hmm is old
