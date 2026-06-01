"""Risk manager: position sizing, leverage caps, drawdown circuit breakers.

The risk layer is **defense in depth** and operates *independently of the HMM*.
Even if regime detection fails completely, the circuit breakers catch drawdowns
from *actual realized P&L*, and :meth:`RiskManager.validate_signal` holds
**absolute veto power** over any signal.

Two consumers, two breaker semantics
------------------------------------
* The **backtester** drives :meth:`RiskManager.update_drawdown_state` once per
  daily bar. It recomputes posture from scratch each bar (no latching) so a
  single bad bar does not freeze the whole walk-forward; the monotonic equity
  peak still provides a soft latch via the peak-drawdown breaker.
* The **live loop** drives :class:`CircuitBreaker`, which *latches* the worst
  posture for the rest of the day / week ("reduce rest of day", "halt rest of
  week") and is cleared only by :meth:`CircuitBreaker.reset_daily` /
  :meth:`CircuitBreaker.reset_weekly`. A peak drawdown beyond the limit writes a
  ``trading_halted.lock`` file requiring manual deletion to resume.

These are deliberately separate; do not unify them.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

from core.regime_strategies import Direction, Signal

logger = logging.getLogger(__name__)


class RiskState(Enum):
    """Operational risk posture derived from drawdown breakers."""

    NORMAL = "normal"
    REDUCED = "reduced"        # sizing scaled down after soft drawdown
    HALTED = "halted"          # trading suspended after hard drawdown


class BreakerType(Enum):
    """Which circuit breaker fired."""

    DAILY_REDUCE = "daily_reduce"
    DAILY_HALT = "daily_halt"
    WEEKLY_REDUCE = "weekly_reduce"
    WEEKLY_HALT = "weekly_halt"
    PEAK_HALT = "peak_halt"


# Allocation multiplier applied to target weights for each posture.
STATE_SIZE_MULT: dict[RiskState, float] = {
    RiskState.NORMAL: 1.0,
    RiskState.REDUCED: 0.5,
    RiskState.HALTED: 0.0,
}

_ORDER = {RiskState.NORMAL: 0, RiskState.REDUCED: 1, RiskState.HALTED: 2}


@dataclass
class RiskConfig:
    """Configuration for the risk manager (mirrors `risk` settings)."""

    max_risk_per_trade: float = 0.01
    max_exposure: float = 0.80
    max_leverage: float = 1.25
    max_single_position: float = 0.15
    max_concurrent: int = 5
    max_daily_trades: int = 20
    daily_dd_reduce: float = 0.02
    daily_dd_halt: float = 0.03
    weekly_dd_reduce: float = 0.05
    weekly_dd_halt: float = 0.07
    max_dd_from_peak: float = 0.10
    # position-level + order validation (Phase 5)
    max_sector_exposure: float = 0.30
    corr_reduce: float = 0.70
    corr_reject: float = 0.85
    corr_window: int = 60
    max_spread_pct: float = 0.005
    duplicate_window_sec: int = 60
    min_position_usd: float = 100.0
    overnight_gap_mult: float = 3.0
    overnight_gap_budget: float = 0.02
    reduce_size_mult: float = 0.50
    lock_file: Optional[str] = None


@dataclass
class SizingResult:
    """Outcome of a position-sizing request.

    Attributes:
        approved: Whether the trade is allowed under current limits.
        shares: Approved share quantity (0 if rejected).
        notional: Approved notional value.
        reason: Explanation when reduced or rejected.
    """

    approved: bool = False
    shares: int = 0
    notional: float = 0.0
    reason: str = ""


@dataclass
class Position:
    """A held position (live portfolio).

    Attributes:
        symbol: Ticker.
        market_value: Signed notional (positive long).
        side: ``"long"`` (the system is long-only).
        sector: Optional sector tag for correlated-exposure checks.
    """

    symbol: str
    market_value: float
    side: str = "long"
    sector: Optional[str] = None


@dataclass
class PortfolioState:
    """Snapshot of the live portfolio for signal validation.

    Attributes:
        equity: Current account equity.
        cash: Available cash.
        buying_power: Broker buying power.
        positions: Currently held positions.
        daily_pnl: Realized+unrealized return since session open (fraction).
        weekly_pnl: Return over the trailing week (fraction).
        peak_equity: Running equity peak.
        drawdown: Current drawdown from peak (negative fraction).
        circuit_breaker_status: Current breaker posture.
        flicker_rate: HMM flicker rate (regime changes in window).
        price_history: Optional per-symbol return series for correlation.
        recent_orders: Recent (symbol, direction, epoch_seconds) for dup checks.
        sector_map: Optional symbol -> sector mapping.
    """

    equity: float
    cash: float = 0.0
    buying_power: float = 0.0
    positions: list[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0
    circuit_breaker_status: RiskState = RiskState.NORMAL
    flicker_rate: int = 0
    price_history: dict[str, pd.Series] = field(default_factory=dict)
    recent_orders: list[tuple[str, str, float]] = field(default_factory=list)
    sector_map: dict[str, str] = field(default_factory=dict)


@dataclass
class RiskDecision:
    """Result of validating a signal against the risk layer.

    Attributes:
        approved: Whether the (possibly modified) signal may be sent.
        modified_signal: The signal after risk modifications (sized, deleveraged,
            correlation-trimmed); ``None`` when rejected.
        rejection_reason: Why the signal was rejected (empty when approved).
        modifications: Human-readable list of every modification applied.
    """

    approved: bool
    modified_signal: Optional[Signal] = None
    rejection_reason: str = ""
    modifications: list[str] = field(default_factory=list)


@dataclass
class BreakerEvent:
    """One circuit-breaker state transition, for the audit log.

    Attributes:
        timestamp: ISO timestamp of the transition.
        breaker_type: Which breaker fired.
        from_state: Posture before.
        to_state: Posture after.
        drawdown: Realized drawdown that triggered it.
        equity: Equity at the time.
        regime: HMM regime active at the time (to flag wrong-HMM events).
    """

    timestamp: str
    breaker_type: Optional[BreakerType]
    from_state: RiskState
    to_state: RiskState
    drawdown: float
    equity: float
    regime: Optional[str] = None


# ===========================================================================
# Circuit breaker (latching — live loop)
# ===========================================================================
class CircuitBreaker:
    """Latching drawdown circuit breaker for the live loop.

    Tracks three independent latched sub-postures (daily, weekly, peak); the
    overall posture is the most severe. Daily/weekly latches clear on their
    resets; the peak latch persists (and writes a lock file) until manually
    cleared.
    """

    def __init__(self, config: RiskConfig, lock_path: Optional[str | Path] = None) -> None:
        """Initialize the breaker.

        Args:
            config: Risk thresholds.
            lock_path: Path for the peak-DD halt lock file. ``None`` disables
                the persistent lock (in-memory HALT still applies) — the default
                for backtests and tests; the live loop supplies a path.
        """
        self.config = config
        self.lock_path = Path(lock_path) if lock_path else None
        self._daily_state = RiskState.NORMAL
        self._weekly_state = RiskState.NORMAL
        self._peak_state = RiskState.NORMAL
        self._daily_ret = 0.0
        self._weekly_ret = 0.0
        self._peak_equity = 0.0
        self._history: list[BreakerEvent] = []

    @property
    def state(self) -> RiskState:
        """Most severe latched posture across all breakers."""
        worst = max(
            (self._daily_state, self._weekly_state, self._peak_state),
            key=lambda s: _ORDER[s],
        )
        return worst

    def update(
        self, pnl: float, equity: float, regime: Optional[str] = None
    ) -> RiskState:
        """Fold one period's realized P&L into the latched posture.

        Args:
            pnl: Period return fraction (signed; a loss is negative).
            equity: Current equity (updates the running peak).
            regime: HMM regime active this period (audit only).

        Returns:
            Updated overall :class:`RiskState`.
        """
        c = self.config
        prev = self.state

        self._daily_ret = (1.0 + self._daily_ret) * (1.0 + pnl) - 1.0
        self._weekly_ret = (1.0 + self._weekly_ret) * (1.0 + pnl) - 1.0
        if equity > self._peak_equity:
            self._peak_equity = equity
        peak_dd = (
            (self._peak_equity - equity) / self._peak_equity
            if self._peak_equity > 0 else 0.0
        )

        daily_loss = -min(self._daily_ret, 0.0)
        weekly_loss = -min(self._weekly_ret, 0.0)
        fired: Optional[BreakerType] = None

        # daily (latch worst)
        if daily_loss >= c.daily_dd_halt:
            self._daily_state = RiskState.HALTED; fired = BreakerType.DAILY_HALT
        elif daily_loss >= c.daily_dd_reduce and _ORDER[self._daily_state] < _ORDER[RiskState.REDUCED]:
            self._daily_state = RiskState.REDUCED; fired = BreakerType.DAILY_REDUCE
        # weekly (latch worst)
        if weekly_loss >= c.weekly_dd_halt:
            self._weekly_state = RiskState.HALTED; fired = BreakerType.WEEKLY_HALT
        elif weekly_loss >= c.weekly_dd_reduce and _ORDER[self._weekly_state] < _ORDER[RiskState.REDUCED]:
            self._weekly_state = RiskState.REDUCED; fired = fired or BreakerType.WEEKLY_REDUCE
        # peak (latch, persistent + lock file)
        if peak_dd >= c.max_dd_from_peak:
            if self._peak_state is not RiskState.HALTED:
                self._peak_state = RiskState.HALTED; fired = BreakerType.PEAK_HALT
                self._write_lock(peak_dd, equity, regime)

        new = self.state
        if new is not prev:
            ev = BreakerEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                breaker_type=fired, from_state=prev, to_state=new,
                drawdown=max(daily_loss, weekly_loss, peak_dd), equity=equity, regime=regime,
            )
            self._history.append(ev)
            logger.warning(
                "CIRCUIT BREAKER %s: %s -> %s (DD %.2f%%, equity %.0f, regime=%s)",
                fired.value if fired else "?", prev.value, new.value,
                ev.drawdown * 100, equity, regime,
            )
        return new

    def check(self) -> bool:
        """Whether new trading is currently permitted.

        Returns:
            False if HALTED or a lock file is present; True otherwise.
        """
        if self.lock_path and self.lock_path.exists():
            return False
        return self.state is not RiskState.HALTED

    def reset_daily(self) -> None:
        """Clear the daily latch and accumulator (call at session open)."""
        self._daily_state = RiskState.NORMAL
        self._daily_ret = 0.0

    def reset_weekly(self) -> None:
        """Clear the weekly latch and accumulator (call at week open)."""
        self._weekly_state = RiskState.NORMAL
        self._weekly_ret = 0.0

    def get_history(self) -> list[BreakerEvent]:
        """Return the chronological breaker-event log."""
        return list(self._history)

    def _write_lock(self, dd: float, equity: float, regime: Optional[str]) -> None:
        """Write the trading-halted lock file (manual deletion required).

        Args:
            dd: Peak drawdown that triggered the halt.
            equity: Equity at the halt.
            regime: HMM regime active at the halt.
        """
        if not self.lock_path:
            return
        self.lock_path.write_text(
            f"TRADING HALTED — peak drawdown {dd:.2%} >= {self.config.max_dd_from_peak:.2%}\n"
            f"time={datetime.now(timezone.utc).isoformat()} equity={equity:.2f} regime={regime}\n"
            f"Delete this file manually to resume trading.\n"
        )
        logger.critical("Wrote halt lock file %s — manual deletion required", self.lock_path)


# ===========================================================================
# Risk manager
# ===========================================================================
class RiskManager:
    """Central gatekeeper for sizing, drawdown halts, and signal validation."""

    def __init__(self, config: RiskConfig) -> None:
        """Initialize the risk manager.

        Args:
            config: Risk limits and drawdown thresholds.
        """
        self.config = config
        self.state: RiskState = RiskState.NORMAL
        self._equity_peak: float = 0.0
        self._daily_trades: int = 0
        self.breaker = CircuitBreaker(config, lock_path=config.lock_file)

    # ----------------------------------------------------- sizing (shared) ---
    def position_size(
        self,
        equity: float,
        price: float,
        stop_distance: float,
        target_weight: float,
    ) -> SizingResult:
        """Size a position from per-trade risk and concentration caps.

        Two independent caps; the smaller wins:

        * **Risk cap** — risk no more than ``max_risk_per_trade`` of equity if
          the stop is hit: ``shares <= max_risk_per_trade * equity /
          stop_distance``.
        * **Concentration cap** — notional no more than
          ``min(target_weight, max_single_position) * equity``.

        Sizing is disabled entirely when ``HALTED``.

        Args:
            equity: Current account equity.
            price: Entry price per share.
            stop_distance: Price distance to stop (risk per share).
            target_weight: Desired portfolio weight for the symbol (the regime
                cap is folded in by the caller before this).

        Returns:
            `SizingResult` with approved quantity or rejection reason.
        """
        if self.state is RiskState.HALTED:
            return SizingResult(reason="risk state HALTED: no new exposure")
        if equity <= 0 or price <= 0 or stop_distance <= 0:
            return SizingResult(reason="invalid equity/price/stop inputs")

        weight = min(max(target_weight, 0.0), self.config.max_single_position)
        if self.state is RiskState.REDUCED:
            weight *= self.config.reduce_size_mult
        if weight <= 0:
            return SizingResult(reason="target weight collapsed to zero")

        risk_cap_shares = (self.config.max_risk_per_trade * equity) / stop_distance
        conc_cap_shares = (weight * equity) / price
        shares = int(min(risk_cap_shares, conc_cap_shares))
        if shares <= 0:
            return SizingResult(reason="caps round position to zero shares")

        binding = "risk-per-trade" if risk_cap_shares < conc_cap_shares else "concentration"
        return SizingResult(
            approved=True, shares=shares, notional=shares * price,
            reason=f"sized by {binding} cap (weight {weight:.2%})",
        )

    def check_exposure(
        self, current_gross: float, proposed_notional: float, equity: float
    ) -> bool:
        """Verify proposed trade keeps gross exposure within limits.

        Args:
            current_gross: Current gross exposure (notional).
            proposed_notional: Notional of the proposed trade.
            equity: Current equity.

        Returns:
            True if within `max_exposure` and `max_leverage`.
        """
        if equity <= 0:
            return False
        leverage = (current_gross + proposed_notional) / equity
        if leverage > self.config.max_leverage + 1e-9:
            return False
        if leverage > self.config.max_exposure * self.config.max_leverage + 1e-9:
            return False
        return True

    def check_concurrent(self, open_positions: int) -> bool:
        """Verify open-position count is below `max_concurrent`.

        Args:
            open_positions: Number of currently open positions.

        Returns:
            True if a new position may be opened.
        """
        return open_positions < self.config.max_concurrent

    def check_daily_trade_limit(self) -> bool:
        """Verify daily trade count is below `max_daily_trades`.

        Returns:
            True if another trade is permitted today.
        """
        return self._daily_trades < self.config.max_daily_trades

    def record_trade(self) -> None:
        """Increment the daily trade counter (call on each executed trade)."""
        self._daily_trades += 1

    def target_size_multiplier(self) -> float:
        """Allocation multiplier for the current posture (1.0/0.5/0.0)."""
        return STATE_SIZE_MULT[self.state]

    def update_drawdown_state(
        self,
        equity: float,
        daily_return: float,
        weekly_return: float,
    ) -> RiskState:
        """Recompute risk posture from drawdown breakers (per-bar, non-latching).

        Used by the **backtester** on daily bars: posture is recomputed from the
        current bar's daily/weekly returns plus the monotonic peak each call, so
        a recovered bar restores NORMAL (the peak breaker still provides a soft
        latch). For latched live behaviour use :class:`CircuitBreaker`.

        Args:
            equity: Current equity (updates the running peak).
            daily_return: Realized return this bar (signed).
            weekly_return: Realized return over the trailing week (signed).

        Returns:
            Updated `RiskState`.
        """
        if equity > self._equity_peak:
            self._equity_peak = equity
        peak_dd = (
            (self._equity_peak - equity) / self._equity_peak
            if self._equity_peak > 0 else 0.0
        )
        c = self.config
        daily_loss = -min(daily_return, 0.0)
        weekly_loss = -min(weekly_return, 0.0)

        state = RiskState.NORMAL
        reasons: list[str] = []

        def escalate(new: RiskState, why: str) -> None:
            nonlocal state
            if _ORDER[new] > _ORDER[state]:
                state = new
            if new is not RiskState.NORMAL:
                reasons.append(why)

        if daily_loss >= c.daily_dd_halt:
            escalate(RiskState.HALTED, f"daily DD {daily_loss:.2%}>=halt {c.daily_dd_halt:.2%}")
        elif daily_loss >= c.daily_dd_reduce:
            escalate(RiskState.REDUCED, f"daily DD {daily_loss:.2%}>=reduce {c.daily_dd_reduce:.2%}")

        if weekly_loss >= c.weekly_dd_halt:
            escalate(RiskState.HALTED, f"weekly DD {weekly_loss:.2%}>=halt {c.weekly_dd_halt:.2%}")
        elif weekly_loss >= c.weekly_dd_reduce:
            escalate(RiskState.REDUCED, f"weekly DD {weekly_loss:.2%}>=reduce {c.weekly_dd_reduce:.2%}")

        if peak_dd >= c.max_dd_from_peak:
            escalate(RiskState.HALTED, f"peak DD {peak_dd:.2%}>=max {c.max_dd_from_peak:.2%}")

        if state is not self.state:
            logger.info("Risk state %s -> %s (%s)", self.state.value, state.value,
                        "; ".join(reasons) or "recovered")
        self.state = state
        return state

    def reset_daily(self) -> None:
        """Reset per-day counters at session start."""
        self._daily_trades = 0
        self.breaker.reset_daily()

    def reset(self) -> None:
        """Fully reset posture, peak, and counters (e.g. between backtest runs)."""
        self.state = RiskState.NORMAL
        self._equity_peak = 0.0
        self._daily_trades = 0

    # ----------------------------------------------- signal validation (live) ---
    def validate_signal(
        self, signal: Signal, portfolio_state: PortfolioState
    ) -> RiskDecision:
        """Validate (and size/modify) a signal — the risk layer's absolute veto.

        Runs, in order: lock-file / halt veto, mandatory stop, tradeable +
        spread + duplicate order checks, daily-trade and concurrent limits,
        leverage rules, position sizing (1% risk, regime cap then 15%, $100
        floor, overnight 3x-gap budget), gross-exposure/leverage cap, buying
        power, correlation, and sector exposure. Any hard failure rejects;
        soft failures trim the signal and are recorded in ``modifications``.

        Args:
            signal: Proposed trade signal.
            portfolio_state: Current portfolio snapshot.

        Returns:
            A :class:`RiskDecision`.
        """
        c = self.config
        mods: list[str] = []

        def reject(reason: str) -> RiskDecision:
            logger.info("Signal REJECTED (%s): %s", signal.symbol, reason)
            return RiskDecision(False, None, reason, mods)

        # 0) hard halts -------------------------------------------------------
        if self.breaker.lock_path and self.breaker.lock_path.exists():
            return reject("trading halted: lock file present (manual reset required)")
        if portfolio_state.circuit_breaker_status is RiskState.HALTED or not self.breaker.check():
            return reject("trading halted by circuit breaker")

        # 1) mandatory stop ---------------------------------------------------
        if signal.direction is not Direction.LONG:
            return reject("long-only system: non-long signal")
        if signal.stop_loss is None or signal.stop_loss <= 0:
            return reject("missing stop loss: orders without a stop are refused")
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance <= 0:
            return reject("stop loss equals entry: zero stop distance")

        # 2) order validation -------------------------------------------------
        if signal.metadata.get("tradable", True) is False:
            return reject("symbol not tradeable")
        bid, ask = signal.metadata.get("bid"), signal.metadata.get("ask")
        if bid and ask and bid > 0:
            spread = (ask - bid) / ((ask + bid) / 2.0)
            if spread > c.max_spread_pct:
                return reject(f"bid-ask spread {spread:.2%} > {c.max_spread_pct:.2%}")
        now = signal.metadata.get("now_epoch", time.time())
        for sym, direction, ts in portfolio_state.recent_orders:
            if sym == signal.symbol and direction == signal.direction.value and now - ts < c.duplicate_window_sec:
                return reject(f"duplicate order within {c.duplicate_window_sec}s")

        # 3) count limits -----------------------------------------------------
        if not self.check_daily_trade_limit():
            return reject(f"daily trade limit ({c.max_daily_trades}) reached")
        held_symbols = {p.symbol for p in portfolio_state.positions}
        if signal.symbol not in held_symbols and not self.check_concurrent(len(portfolio_state.positions)):
            return reject(f"max concurrent positions ({c.max_concurrent}) reached")

        # 4) leverage rules ---------------------------------------------------
        leverage = min(signal.leverage, c.max_leverage)
        force_reasons: list[str] = []
        if signal.metadata.get("uncertainty") or signal.regime_probability < self.config_min_conf:
            force_reasons.append("regime uncertain")
        if self.state is not RiskState.NORMAL or portfolio_state.circuit_breaker_status is not RiskState.NORMAL:
            force_reasons.append("circuit breaker active")
        if len(portfolio_state.positions) >= 3:
            force_reasons.append("3+ positions open")
        if portfolio_state.flicker_rate > 4:
            force_reasons.append("high flicker rate")
        if force_reasons and leverage > 1.0:
            leverage = 1.0
            mods.append(f"leverage forced to 1.0x ({', '.join(force_reasons)})")

        # 5) position sizing --------------------------------------------------
        equity = portfolio_state.equity
        price = signal.entry_price
        # regime cap first, then portfolio single-position cap
        regime_weight = min(signal.position_size_pct, c.max_single_position)
        sized = self.position_size(equity, price, stop_distance, regime_weight)
        if not sized.approved:
            return reject(f"sizing failed: {sized.reason}")
        shares = sized.shares

        # overnight gap budget: 3x stop gap-through capped at 2% of equity
        gap_loss_per_share = c.overnight_gap_mult * stop_distance
        if gap_loss_per_share > 0:
            gap_cap_shares = int((c.overnight_gap_budget * equity) / gap_loss_per_share)
            if gap_cap_shares < shares:
                shares = gap_cap_shares
                mods.append(f"overnight {c.overnight_gap_mult:.0f}x-gap budget capped size to {shares} sh")

        notional = shares * price
        if notional < c.min_position_usd:
            return reject(f"position ${notional:.0f} below ${c.min_position_usd:.0f} minimum")

        # 6) exposure / buying power -----------------------------------------
        gross = sum(abs(p.market_value) for p in portfolio_state.positions)
        if not self.check_exposure(gross, notional * leverage, equity):
            # trim to fit the leverage/exposure ceiling
            room = c.max_leverage * equity - gross
            shares = max(0, int(room / (price * leverage))) if leverage > 0 else 0
            notional = shares * price
            if shares <= 0 or notional < c.min_position_usd:
                return reject("exceeds gross exposure / leverage ceiling")
            mods.append("size trimmed to gross-exposure ceiling")
        if portfolio_state.buying_power and notional * leverage > portfolio_state.buying_power:
            return reject("insufficient buying power")

        # 7) correlation ------------------------------------------------------
        corr = self._max_correlation(signal.symbol, portfolio_state)
        if corr is not None:
            if corr > c.corr_reject:
                return reject(f"correlation {corr:.2f} > {c.corr_reject:.2f} with held position")
            if corr > c.corr_reduce:
                shares = int(shares * 0.5)
                notional = shares * price
                mods.append(f"size halved: correlation {corr:.2f} > {c.corr_reduce:.2f}")
                if notional < c.min_position_usd:
                    return reject("correlation-trimmed position below minimum")

        # 8) sector exposure (skipped if no sector map) ----------------------
        sec = portfolio_state.sector_map.get(signal.symbol)
        if sec:
            sector_gross = sum(
                abs(p.market_value) for p in portfolio_state.positions
                if portfolio_state.sector_map.get(p.symbol) == sec
            )
            if (sector_gross + notional) / max(equity, 1e-9) > c.max_sector_exposure:
                return reject(f"sector '{sec}' exposure exceeds {c.max_sector_exposure:.0%}")

        modified = dataclasses.replace(
            signal, leverage=leverage,
            metadata={**signal.metadata, "approved_shares": shares,
                      "approved_notional": notional, "stop_distance": stop_distance},
        )
        logger.info("Signal APPROVED (%s): %d sh @ %.2f, lev %.2fx%s",
                    signal.symbol, shares, price, leverage,
                    f" [{'; '.join(mods)}]" if mods else "")
        return RiskDecision(True, modified, "", mods)

    @property
    def config_min_conf(self) -> float:
        """Min regime confidence to act (mirrors strategy.min_confidence)."""
        return getattr(self.config, "min_confidence", 0.55)

    def _max_correlation(
        self, symbol: str, portfolio_state: PortfolioState
    ) -> Optional[float]:
        """Max trailing correlation of ``symbol`` with any held position.

        Args:
            symbol: Candidate symbol.
            portfolio_state: Portfolio snapshot (needs ``price_history``).

        Returns:
            Max absolute correlation, or ``None`` if history is unavailable.
        """
        hist = portfolio_state.price_history
        if symbol not in hist:
            return None
        win = self.config.corr_window
        base = hist[symbol].tail(win)
        best: Optional[float] = None
        for p in portfolio_state.positions:
            other = hist.get(p.symbol)
            if other is None or p.symbol == symbol:
                continue
            joined = pd.concat([base, other.tail(win)], axis=1).dropna()
            if len(joined) < max(10, win // 2):
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if corr == corr:  # not NaN
                best = corr if best is None else max(best, corr)
        return best
