"""Tests for volatility-based allocation strategies and the orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import Regime, RegimeInfo, RegimeState
from core.regime_strategies import (
    LABEL_TO_STRATEGY,
    BullTrendStrategy,
    CrashDefensiveStrategy,
    Direction,
    EuphoriaCautiousStrategy,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MeanReversionStrategy,
    MidVolCautiousStrategy,
    StrategyConfig,
    StrategyOrchestrator,
    atr_series,
    ema_series,
)


# --------------------------------------------------------------------------- helpers
def make_bars(n: int = 80, trend: str = "up", seed: int = 1) -> pd.DataFrame:
    """Build OHLCV bars with an up or down drift (enough for EMA50/ATR14)."""
    rng = np.random.default_rng(seed)
    slope = 0.002 if trend == "up" else -0.002
    close = 100.0 * np.exp(np.cumsum(rng.normal(slope, 0.004, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    vol = rng.lognormal(15, 0.3, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def regime_state(state_id: int, label: Regime, prob: float = 0.9) -> RegimeState:
    return RegimeState(
        label=label, state_id=state_id, probability=prob,
        state_probabilities=np.array([prob]), timestamp=pd.Timestamp("2022-04-01"),
        is_confirmed=True, consecutive_bars=5,
    )


@pytest.fixture
def config() -> StrategyConfig:
    return StrategyConfig()


# --------------------------------------------------------------------------- strategy unit tests
def test_low_vol_allocation_and_leverage(config) -> None:
    """Low-vol: 95% allocation, 1.25x leverage, LONG, correct stop formula."""
    bars = make_bars(trend="up")
    sig = LowVolBullStrategy(config).generate_signal("SPY", bars, regime_state(0, Regime.BULL))
    assert sig is not None
    assert sig.direction is Direction.LONG
    assert sig.position_size_pct == pytest.approx(0.95)
    assert sig.leverage == pytest.approx(1.25)
    price = bars["close"].iloc[-1]
    atr = atr_series(bars, 14).iloc[-1]
    ema = ema_series(bars["close"], 50).iloc[-1]
    expected_stop = max(price - 3 * atr, ema - 0.5 * atr)
    assert sig.stop_loss == pytest.approx(expected_stop)


def test_mid_vol_trend_intact_vs_broken(config) -> None:
    """Mid-vol: 95% when price>EMA50, 60% when price<EMA50; always 1.0x."""
    up = make_bars(trend="up", seed=2)
    down = make_bars(trend="down", seed=3)
    s_up = MidVolCautiousStrategy(config).generate_signal("SPY", up, regime_state(1, Regime.NEUTRAL))
    s_dn = MidVolCautiousStrategy(config).generate_signal("SPY", down, regime_state(1, Regime.NEUTRAL))
    assert up["close"].iloc[-1] > ema_series(up["close"], 50).iloc[-1]
    assert down["close"].iloc[-1] < ema_series(down["close"], 50).iloc[-1]
    assert s_up.position_size_pct == pytest.approx(0.95)
    assert s_dn.position_size_pct == pytest.approx(0.60)
    assert s_up.leverage == 1.0 and s_dn.leverage == 1.0


def test_high_vol_defensive_is_long_not_short(config) -> None:
    """High-vol: 60% @ 1.0x, LONG (never short), wider stop EMA50-1.0*ATR."""
    bars = make_bars(trend="down", seed=4)
    sig = HighVolDefensiveStrategy(config).generate_signal("SPY", bars, regime_state(2, Regime.CRASH))
    assert sig.direction is Direction.LONG
    assert sig.position_size_pct == pytest.approx(0.60)
    assert sig.leverage == 1.0
    atr = atr_series(bars, 14).iloc[-1]
    ema = ema_series(bars["close"], 50).iloc[-1]
    assert sig.stop_loss == pytest.approx(ema - 1.0 * atr)


def test_insufficient_history_returns_none(config) -> None:
    """Too few bars for EMA50/ATR14 -> no signal."""
    assert LowVolBullStrategy(config).generate_signal(
        "SPY", make_bars(n=20), regime_state(0, Regime.BULL)
    ) is None


def test_no_strategy_ever_shorts(config) -> None:
    """All three strategies emit LONG only across trend regimes."""
    for strat in (LowVolBullStrategy(config), MidVolCautiousStrategy(config),
                  HighVolDefensiveStrategy(config)):
        for trend in ("up", "down"):
            sig = strat.generate_signal("SPY", make_bars(trend=trend), regime_state(0, Regime.BEAR))
            assert sig.direction is Direction.LONG


# --------------------------------------------------------------------------- orchestrator tests
def make_infos(vols: dict[int, float], labels: dict[int, Regime]) -> dict[int, RegimeInfo]:
    return {
        rid: RegimeInfo(
            regime_id=rid, regime_name=labels[rid], expected_return=0.0,
            expected_volatility=vols[rid], recommended_strategy_type="x",
            max_leverage_allowed=1.25, max_position_size_pct=0.95, min_confidence_to_act=0.55,
        )
        for rid in vols
    }


def test_orchestrator_maps_by_vol_not_label(config) -> None:
    """Vol rank drives strategy choice; labels are ignored.

    Deliberately mislabel: the LOWEST-vol regime carries a 'CRASH' label and
    the HIGHEST-vol regime carries a 'BULL' label. Mapping must follow vol.
    """
    vols = {0: 0.50, 1: 0.10, 2: 0.90}   # rid1 lowest vol, rid2 highest
    labels = {0: Regime.NEUTRAL, 1: Regime.CRASH, 2: Regime.BULL}
    orch = StrategyOrchestrator(config, make_infos(vols, labels))
    assert isinstance(orch.get_strategy_for(1), LowVolBullStrategy)    # lowest vol
    assert isinstance(orch.get_strategy_for(0), MidVolCautiousStrategy)  # middle
    assert isinstance(orch.get_strategy_for(2), HighVolDefensiveStrategy)  # highest vol


def test_vol_rank_positions(config) -> None:
    """3-regime vol positions are 0.0, 0.5, 1.0 by ascending vol."""
    vols = {0: 0.3, 1: 0.1, 2: 0.2}
    labels = {i: Regime.NEUTRAL for i in vols}
    orch = StrategyOrchestrator(config, make_infos(vols, labels))
    assert orch.vol_rank[1] == pytest.approx(0.0)
    assert orch.vol_rank[2] == pytest.approx(0.5)
    assert orch.vol_rank[0] == pytest.approx(1.0)


def test_uncertainty_halves_size_and_forces_leverage(config) -> None:
    """Low confidence -> halve size, force 1.0x, annotate reasoning."""
    vols = {0: 0.1, 1: 0.5, 2: 0.9}
    labels = {i: Regime.NEUTRAL for i in vols}
    orch = StrategyOrchestrator(config, make_infos(vols, labels))
    bars = {"SPY": make_bars(trend="up")}
    # low-vol regime id 0 -> base 0.95 @ 1.25x; prob below threshold
    rs = regime_state(0, Regime.BULL, prob=0.40)
    sigs = orch.generate_signals(["SPY"], bars, rs, is_flickering=False)
    assert len(sigs) == 1
    s = sigs[0]
    assert s.position_size_pct == pytest.approx(0.95 * 0.50)
    assert s.leverage == 1.0
    assert "[UNCERTAINTY — size halved]" in s.reasoning
    assert s.metadata["uncertainty"] is True


def test_flicker_triggers_uncertainty_even_with_high_prob(config) -> None:
    """is_flickering=True forces uncertainty regardless of probability."""
    vols = {0: 0.1, 1: 0.5, 2: 0.9}
    labels = {i: Regime.NEUTRAL for i in vols}
    orch = StrategyOrchestrator(config, make_infos(vols, labels))
    bars = {"SPY": make_bars(trend="up")}
    rs = regime_state(0, Regime.BULL, prob=0.99)
    sigs = orch.generate_signals(["SPY"], bars, rs, is_flickering=True)
    assert sigs[0].leverage == 1.0
    assert sigs[0].position_size_pct == pytest.approx(0.95 * 0.50)
    assert sigs[0].metadata["uncertainty_flicker"] is True


def test_update_regime_infos_rebuilds_mapping(config) -> None:
    """After retrain, a regime's strategy follows its new vol rank."""
    labels = {0: Regime.NEUTRAL, 1: Regime.NEUTRAL, 2: Regime.NEUTRAL}
    orch = StrategyOrchestrator(config, make_infos({0: 0.1, 1: 0.5, 2: 0.9}, labels))
    assert isinstance(orch.get_strategy_for(0), LowVolBullStrategy)
    # retrain: regime 0 is now the HIGHEST vol
    orch.update_regime_infos(make_infos({0: 0.9, 1: 0.5, 2: 0.1}, labels))
    assert isinstance(orch.get_strategy_for(0), HighVolDefensiveStrategy)


# --------------------------------------------------------------------------- aliases / mapping
def test_backward_compatible_aliases() -> None:
    """Themed aliases resolve to the correct vol-based classes."""
    assert BullTrendStrategy is LowVolBullStrategy
    assert EuphoriaCautiousStrategy is LowVolBullStrategy
    assert MeanReversionStrategy is MidVolCautiousStrategy
    assert CrashDefensiveStrategy is HighVolDefensiveStrategy


def test_label_to_strategy_covers_all_labels() -> None:
    """Every Regime label has a fallback strategy mapping."""
    for label in Regime:
        assert label in LABEL_TO_STRATEGY
