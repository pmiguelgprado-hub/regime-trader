"""Volatility-based allocation strategies.

The HMM detects the *volatility environment*; this layer turns that into a
long-only target allocation. Core thesis: the edge comes from **avoiding big
drawdowns** via vol-based sizing, not from predicting direction.

    Low vol  -> fully invested + modest leverage (calm markets trend up)
    Mid vol  -> stay invested if trend intact, reduce if broken
    High vol -> reduce but stay partially invested (catch V-shaped rebounds)

**ALWAYS LONG. NEVER SHORT.** Shorting consistently destroyed returns in
walk-forward testing: markets drift up, V-recoveries are fast, and the HMM is
2-3 days late detecting them — so shorts get run over on the rebound. The
correct response to high volatility is *less allocation*, not reversing.

Strategy selection is driven by each regime's **volatility rank**, computed by
the orchestrator by sorting ``RegimeInfo.expected_volatility`` ascending. This
is independent of the HMM's *labels* (which sort by return): a regime labelled
``BULL`` is not necessarily low-vol, and the orchestrator ignores labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
from ta.volatility import AverageTrueRange

from core.hmm_engine import Regime, RegimeInfo, RegimeState

# Volatility-rank cut points (position in [0,1] across regimes by vol).
LOW_VOL_MAX = 0.33   # position <= 0.33  -> LowVolBullStrategy
HIGH_VOL_MIN = 0.67  # position >= 0.67  -> HighVolDefensiveStrategy


class Direction(Enum):
    """Trade direction. Shorting is intentionally not supported."""

    LONG = "long"
    FLAT = "flat"


@dataclass
class Signal:
    """A long-only allocation instruction for one symbol.

    Attributes:
        symbol: Ticker.
        direction: ``LONG`` or ``FLAT`` (never ``SHORT``).
        confidence: Regime probability backing the signal (0..1).
        entry_price: Reference price (latest close).
        stop_loss: Protective stop price (live trading; informational in BT).
        take_profit: Optional take-profit price.
        position_size_pct: Target gross allocation (0.60..0.95, pre-leverage).
        leverage: Leverage multiplier (1.0 or 1.25).
        regime_id: HMM hidden-state id driving this signal.
        regime_name: Human-readable regime label.
        regime_probability: Filtered probability of the regime.
        timestamp: Signal timestamp.
        reasoning: Human-readable explanation for audit/logging.
        strategy_name: Name of the strategy that produced the signal.
        metadata: Free-form extra fields (atr, ema, uncertainty flag, ...).
    """

    symbol: str
    direction: Direction = Direction.LONG
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: Optional[float] = None
    position_size_pct: float = 0.0
    leverage: float = 1.0
    regime_id: int = -1
    regime_name: str = ""
    regime_probability: float = 0.0
    timestamp: Optional[pd.Timestamp] = None
    reasoning: str = ""
    strategy_name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Configuration for the allocation strategies (mirrors `strategy`)."""

    low_vol_allocation: float = 0.95
    low_vol_leverage: float = 1.25
    mid_vol_allocation_trend: float = 0.95
    mid_vol_allocation_no_trend: float = 0.60
    high_vol_allocation: float = 0.60
    rebalance_threshold: float = 0.10
    min_confidence: float = 0.55
    uncertainty_size_mult: float = 0.50
    ema_period: int = 50
    atr_period: int = 14
    atr_mult_low: float = 3.0          # LowVol: price - 3*ATR floor
    ema_stop_atr_low: float = 0.5      # LowVol: EMA - 0.5*ATR
    ema_stop_atr_mid: float = 0.5      # MidVol: EMA - 0.5*ATR
    ema_stop_atr_high: float = 1.0     # HighVol: EMA - 1.0*ATR


# ===========================================================================
# Indicator helpers (causal — trailing only)
# ===========================================================================
def ema_series(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average of close.

    Args:
        close: Close-price series.
        period: EMA span.

    Returns:
        EMA series.
    """
    return close.ewm(span=period, adjust=False).mean()


def atr_series(bars: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range from OHLC bars.

    Args:
        bars: OHLCV DataFrame (needs high, low, close).
        period: ATR window.

    Returns:
        ATR series.
    """
    return AverageTrueRange(
        high=bars["high"], low=bars["low"], close=bars["close"], window=period
    ).average_true_range()


# ===========================================================================
# Strategy classes
# ===========================================================================
class BaseStrategy(ABC):
    """Abstract base for volatility-regime allocation strategies."""

    name: str = "base"

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize the strategy.

        Args:
            config: Allocation/leverage/stop parameters.
        """
        self.config = config

    @abstractmethod
    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        """Produce a long-only allocation signal for ``symbol``.

        Args:
            symbol: Ticker.
            bars: OHLCV history for the symbol (most recent bar last).
            regime_state: Current filtered regime estimate (market-wide).

        Returns:
            A :class:`Signal`, or ``None`` if there is insufficient history.
        """
        raise NotImplementedError

    # -- shared helpers -----------------------------------------------------
    def _indicators(self, bars: pd.DataFrame) -> Optional[tuple[float, float, float]]:
        """Compute (price, ema, atr) for the latest bar, or None if unusable.

        Args:
            bars: OHLCV history.

        Returns:
            Tuple ``(price, ema, atr)`` or ``None`` when history is too short
            or indicators are NaN.
        """
        if len(bars) < max(self.config.ema_period, self.config.atr_period) + 1:
            return None
        price = float(bars["close"].iloc[-1])
        ema = float(ema_series(bars["close"], self.config.ema_period).iloc[-1])
        atr = float(atr_series(bars, self.config.atr_period).iloc[-1])
        if not (price > 0 and atr == atr and ema == ema):  # NaN check
            return None
        return price, ema, atr

    def _make_signal(
        self,
        symbol: str,
        regime_state: RegimeState,
        price: float,
        allocation: float,
        leverage: float,
        stop_loss: float,
        reasoning: str,
        metadata: dict,
    ) -> Signal:
        """Assemble a :class:`Signal` with the common fields filled.

        Args:
            symbol: Ticker.
            regime_state: Backing regime state.
            price: Entry/reference price.
            allocation: Target gross allocation (pre-leverage).
            leverage: Leverage multiplier.
            stop_loss: Stop price.
            reasoning: Explanation string.
            metadata: Extra fields.

        Returns:
            The populated :class:`Signal`.
        """
        label = regime_state.label
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=allocation,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=label.value if isinstance(label, Regime) else str(label),
            regime_probability=regime_state.probability,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            strategy_name=self.name,
            metadata={"ema": metadata.get("ema"), "atr": metadata.get("atr"), **metadata},
        )


class LowVolBullStrategy(BaseStrategy):
    """Lowest-vol regimes: fully invested with modest leverage.

    Where most returns are generated — calm markets trend up, and modest
    leverage compounds. Stop: ``max(price - 3*ATR, EMA50 - 0.5*ATR)``.
    """

    name = "LowVolBull"

    def generate_signal(self, symbol, bars, regime_state):  # noqa: D102
        ind = self._indicators(bars)
        if ind is None:
            return None
        price, ema, atr = ind
        c = self.config
        stop = max(price - c.atr_mult_low * atr, ema - c.ema_stop_atr_low * atr)
        return self._make_signal(
            symbol, regime_state, price,
            allocation=c.low_vol_allocation, leverage=c.low_vol_leverage,
            stop_loss=stop,
            reasoning=(
                f"Low-vol regime (rank low): fully invested "
                f"{c.low_vol_allocation:.0%} @ {c.low_vol_leverage:.2f}x leverage"
            ),
            metadata={"ema": ema, "atr": atr},
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Middle-vol regimes: stay invested if trend intact, reduce if broken.

    If ``price > EMA50``: 95% @ 1.0x. Else: 60% @ 1.0x.
    Stop: ``EMA50 - 0.5*ATR``.
    """

    name = "MidVolCautious"

    def generate_signal(self, symbol, bars, regime_state):  # noqa: D102
        ind = self._indicators(bars)
        if ind is None:
            return None
        price, ema, atr = ind
        c = self.config
        if price > ema:
            alloc, why = c.mid_vol_allocation_trend, "trend intact (price>EMA50): stay invested"
        else:
            alloc, why = c.mid_vol_allocation_no_trend, "trend broken (price<EMA50): reduce"
        stop = ema - c.ema_stop_atr_mid * atr
        return self._make_signal(
            symbol, regime_state, price,
            allocation=alloc, leverage=1.0, stop_loss=stop,
            reasoning=f"Mid-vol regime: {why} -> {alloc:.0%} @ 1.0x",
            metadata={"ema": ema, "atr": atr, "above_ema": price > ema},
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """Top-vol regimes: reduce but stay partially invested (LONG, not short).

    60% @ 1.0x to catch sharp post-selloff rebounds. Stop: ``EMA50 - 1.0*ATR``
    (wider for volatile conditions).
    """

    name = "HighVolDefensive"

    def generate_signal(self, symbol, bars, regime_state):  # noqa: D102
        ind = self._indicators(bars)
        if ind is None:
            return None
        price, ema, atr = ind
        c = self.config
        stop = ema - c.ema_stop_atr_high * atr
        return self._make_signal(
            symbol, regime_state, price,
            allocation=c.high_vol_allocation, leverage=1.0, stop_loss=stop,
            reasoning=(
                f"High-vol regime: defensive but LONG "
                f"{c.high_vol_allocation:.0%} @ 1.0x (catch rebounds, never short)"
            ),
            metadata={"ema": ema, "atr": atr},
        )


# ---------------------------------------------------------------------------
# Backward-compatible aliases (label-themed names map to vol-based classes).
# The orchestrator selects by VOLATILITY RANK, not by these names — kept only
# so older references / label-driven code resolve to the right behaviour.
# ---------------------------------------------------------------------------
CrashDefensiveStrategy = HighVolDefensiveStrategy
StrongBearDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
WeakBearStrategy = MidVolCautiousStrategy
NeutralStrategy = MidVolCautiousStrategy
MeanReversionStrategy = MidVolCautiousStrategy
WeakBullStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
StrongBullStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

# Fallback label -> strategy mapping for every possible regime label. NOTE:
# this is a convenience only; labels do NOT determine volatility. The
# orchestrator ignores it and uses each regime's expected_volatility rank.
LABEL_TO_STRATEGY: dict[Regime, type[BaseStrategy]] = {
    Regime.CRASH: HighVolDefensiveStrategy,
    Regime.STRONG_BEAR: HighVolDefensiveStrategy,
    Regime.BEAR: MidVolCautiousStrategy,
    Regime.WEAK_BEAR: MidVolCautiousStrategy,
    Regime.NEUTRAL: MidVolCautiousStrategy,
    Regime.WEAK_BULL: MidVolCautiousStrategy,
    Regime.BULL: LowVolBullStrategy,
    Regime.STRONG_BULL: LowVolBullStrategy,
    Regime.EUPHORIA: LowVolBullStrategy,
    Regime.UNKNOWN: HighVolDefensiveStrategy,
}


# ===========================================================================
# Orchestrator
# ===========================================================================
class StrategyOrchestrator:
    """Maps regimes to strategies by volatility rank and emits signals."""

    def __init__(
        self, config: StrategyConfig, regime_infos: dict[int, RegimeInfo]
    ) -> None:
        """Initialize the orchestrator and build the regime->strategy map.

        Args:
            config: Strategy configuration.
            regime_infos: Map of ``regime_id -> RegimeInfo`` from the HMM.
        """
        self.config = config
        self._low = LowVolBullStrategy(config)
        self._mid = MidVolCautiousStrategy(config)
        self._high = HighVolDefensiveStrategy(config)
        self.regime_infos: dict[int, RegimeInfo] = {}
        self.regime_to_strategy: dict[int, BaseStrategy] = {}
        self.vol_rank: dict[int, float] = {}
        self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        """Rebuild the regime->strategy mapping (call after HMM retrain).

        Sorts regimes by ``expected_volatility`` ascending, computes each
        regime's normalized vol position ``rank / (n - 1)``, and assigns a
        strategy by the low/mid/high cut points. Independent of labels.

        Args:
            regime_infos: Map of ``regime_id -> RegimeInfo``.
        """
        self.regime_infos = dict(regime_infos)
        order = sorted(regime_infos, key=lambda rid: regime_infos[rid].expected_volatility)
        n = len(order)
        self.regime_to_strategy = {}
        self.vol_rank = {}
        for rank, rid in enumerate(order):
            position = rank / (n - 1) if n > 1 else 0.0
            self.vol_rank[rid] = position
            if position <= LOW_VOL_MAX:
                self.regime_to_strategy[rid] = self._low
            elif position >= HIGH_VOL_MIN:
                self.regime_to_strategy[rid] = self._high
            else:
                self.regime_to_strategy[rid] = self._mid

    def generate_signals(
        self,
        symbols: list[str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
    ) -> list[Signal]:
        """Emit long-only signals for ``symbols`` under the current regime.

        Applies uncertainty handling: when the regime probability is below
        ``min_confidence`` or the regime is flickering, halve every position
        size, force leverage to 1.0x, and annotate the reasoning.

        Args:
            symbols: Tickers to generate signals for.
            bars: Map of ``symbol -> OHLCV DataFrame``.
            regime_state: Current filtered regime estimate (market-wide).
            is_flickering: Whether the HMM regime is flickering.

        Returns:
            List of :class:`Signal` (one per symbol with sufficient history).
        """
        strategy = self.regime_to_strategy.get(regime_state.state_id)
        if strategy is None:
            return []
        uncertain = (
            regime_state.probability < self.config.min_confidence or is_flickering
        )
        signals: list[Signal] = []
        for symbol in symbols:
            sym_bars = bars.get(symbol)
            if sym_bars is None or sym_bars.empty:
                continue
            sig = strategy.generate_signal(symbol, sym_bars, regime_state)
            if sig is None:
                continue
            if uncertain:
                self._apply_uncertainty(sig, is_flickering)
            signals.append(sig)
        return signals

    def _apply_uncertainty(self, sig: Signal, is_flickering: bool) -> None:
        """Halve size and force 1.0x leverage under uncertainty.

        Args:
            sig: Signal to modify in place.
            is_flickering: Whether flickering contributed to uncertainty.
        """
        sig.position_size_pct *= self.config.uncertainty_size_mult
        sig.leverage = 1.0
        sig.reasoning += " [UNCERTAINTY — size halved]"
        sig.metadata["uncertainty"] = True
        sig.metadata["uncertainty_flicker"] = bool(is_flickering)

    def get_strategy_for(self, regime_id: int) -> Optional[BaseStrategy]:
        """Return the strategy mapped to a regime id.

        Args:
            regime_id: HMM hidden-state id.

        Returns:
            The mapped :class:`BaseStrategy`, or ``None`` if unknown.
        """
        return self.regime_to_strategy.get(regime_id)
