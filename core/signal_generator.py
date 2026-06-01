"""Signal generator: combines HMM regime + strategy + risk into orders.

Orchestrates the decision pipeline: detect regime -> compute allocation ->
size under risk limits -> emit concrete trade signals per symbol.

NOTE: Skeleton only — no logic implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import Signal, StrategyOrchestrator
from core.risk_manager import RiskManager


class SignalSide(Enum):
    """Direction of a trade signal."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeSignal:
    """A concrete, risk-checked instruction for one symbol.

    Attributes:
        symbol: Ticker.
        side: Buy/sell/hold.
        target_weight: Desired portfolio weight.
        shares: Risk-approved share delta.
        confidence: Regime confidence backing the signal.
        rationale: Human-readable explanation for audit/logging.
    """

    symbol: str
    side: SignalSide = SignalSide.HOLD
    target_weight: float = 0.0
    shares: int = 0
    confidence: float = 0.0
    rationale: str = ""


class SignalGenerator:
    """Combines regime detection, strategy, and risk into trade signals."""

    def __init__(
        self,
        hmm_engine: HMMEngine,
        orchestrator: StrategyOrchestrator,
        risk_manager: RiskManager,
    ) -> None:
        """Initialize the signal generator.

        Args:
            hmm_engine: Fitted regime-detection engine.
            orchestrator: Vol-rank strategy orchestrator.
            risk_manager: Sizing and drawdown gatekeeper.
        """
        self.hmm_engine = hmm_engine
        self.orchestrator = orchestrator
        self.risk_manager = risk_manager

    def generate(
        self,
        features: pd.DataFrame,
        prices: dict[str, float],
        equity: float,
        current_weights: dict[str, float],
        trend_flags: dict[str, bool],
    ) -> list[TradeSignal]:
        """Run the full pipeline and emit signals for the universe.

        Args:
            features: Feature matrix for regime detection.
            prices: Latest price per symbol.
            equity: Current account equity.
            current_weights: Live per-symbol weights.
            trend_flags: Per-symbol trend-present flags.

        Returns:
            List of `TradeSignal`, one per actionable symbol.
        """
        raise NotImplementedError

    def _detect_regime(self, features: pd.DataFrame) -> RegimeState:
        """Run regime detection for the latest bar.

        Args:
            features: Feature matrix.

        Returns:
            Latest `RegimeState`.
        """
        raise NotImplementedError

    def _to_signals(
        self,
        strategy_signals: list[Signal],
        prices: dict[str, float],
        equity: float,
        current_weights: dict[str, float],
        regime_state: RegimeState,
    ) -> list[TradeSignal]:
        """Convert strategy signals into risk-checked trade signals.

        Args:
            strategy_signals: Allocation signals from the orchestrator.
            prices: Latest prices.
            equity: Current equity.
            current_weights: Live weights.
            regime_state: Backing regime state.

        Returns:
            Risk-approved trade signals.
        """
        raise NotImplementedError
