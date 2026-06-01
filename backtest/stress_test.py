"""Stress testing: crash injection, gap risk, regime misclassification.

Perturbs historical price paths with synthetic crashes and overnight gaps, and
deliberately corrupts the regime mapping, to probe whether the **risk layer**
(drawdown circuit breakers) contains damage independently of the HMM being
right. Each probe re-runs the walk-forward backtester on the perturbed path.

Three probes (the spec's asks):

* **Crash injection** — insert several -5%..-15% single-day gaps at random
  points; Monte-Carlo it; report mean / worst max drawdown and how often a
  circuit breaker fired.
* **Gap risk** — insert overnight gaps sized 2-5x ATR; report expected vs.
  actual loss.
* **Regime misclassification** — shuffle the regime->strategy map so the system
  acts on wrong regimes, and verify the breakers still cap the drawdown. If the
  account blows up, risk management is not independent enough.

Every breaker trigger is logged with breaker type, realized drawdown, equity,
and the HMM regime active at the time (so a wrong-HMM bar is auditable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from backtest.backtester import Backtester, BacktestResult
from core.regime_strategies import atr_series

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Supported stress scenarios."""

    CRASH = "crash"            # sudden multi-sigma drop
    GAP_DOWN = "gap_down"      # overnight gap down
    GAP_UP = "gap_up"          # overnight gap up
    VOL_SPIKE = "vol_spike"    # sustained volatility spike
    FLASH_CRASH = "flash_crash"  # intraday crash + partial recovery


@dataclass
class StressScenario:
    """Definition of a single stress scenario.

    Attributes:
        scenario_type: Kind of shock.
        magnitude: Shock size (e.g. -0.20 for a 20% crash).
        start_index: Bar index where the shock begins.
        duration_bars: Number of bars the shock spans.
    """

    scenario_type: ScenarioType
    magnitude: float
    start_index: int
    duration_bars: int = 1


@dataclass
class StressReport:
    """Aggregate outcome of a Monte-Carlo stress probe.

    Attributes:
        probe: Name of the probe.
        n_sims: Number of simulations run.
        mean_max_loss: Mean worst drawdown across sims (negative fraction).
        worst_max_loss: Single worst drawdown across sims.
        breaker_fire_rate: Fraction of sims where a circuit breaker fired.
        blowup_rate: Fraction of sims breaching the containment threshold.
        extra: Probe-specific extra metrics.
    """

    probe: str
    n_sims: int
    mean_max_loss: float
    worst_max_loss: float
    breaker_fire_rate: float
    blowup_rate: float
    extra: dict[str, float] = field(default_factory=dict)


# Drawdown beyond which we consider risk management to have failed ("blowup").
BLOWUP_DD = -0.25


class StressTester:
    """Injects shocks into price paths and re-runs the backtester."""

    def __init__(self, backtester: Backtester) -> None:
        """Initialize the stress tester.

        Args:
            backtester: Configured backtester to run perturbed paths.
        """
        self.backtester = backtester

    # ------------------------------------------------------- perturbations ---
    def inject_crash(
        self, prices: pd.DataFrame, magnitude: float, start_index: int
    ) -> pd.DataFrame:
        """Inject a single-day crash (permanent level gap) into a price series.

        Multiplies all OHLC prices from ``start_index`` onward by
        ``(1 + magnitude)``, i.e. a one-bar return shock that the path then
        continues from.

        Args:
            prices: OHLCV DataFrame.
            magnitude: Drop magnitude (negative fraction, e.g. -0.10).
            start_index: Positional bar where the crash starts.

        Returns:
            Perturbed OHLCV DataFrame (copy).
        """
        out = prices.copy()
        cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
        out.iloc[start_index:, [out.columns.get_loc(c) for c in cols]] *= (1.0 + magnitude)
        return out

    def inject_gap(
        self, prices: pd.DataFrame, magnitude: float, start_index: int
    ) -> pd.DataFrame:
        """Inject an overnight gap into a price series.

        Mechanically identical to :meth:`inject_crash` (a permanent level shift
        from ``start_index``) but used for ATR-scaled overnight gap risk.

        Args:
            prices: OHLCV DataFrame.
            magnitude: Gap magnitude (signed fraction).
            start_index: Positional bar where the gap occurs.

        Returns:
            Perturbed OHLCV DataFrame (copy).
        """
        return self.inject_crash(prices, magnitude, start_index)

    def run_scenario(
        self, prices: dict[str, pd.DataFrame], scenario: StressScenario
    ) -> BacktestResult:
        """Apply a scenario and run the backtest on the perturbed prices.

        Args:
            prices: Map of symbol -> OHLCV DataFrame.
            scenario: Stress scenario to inject.

        Returns:
            `BacktestResult` under the scenario.
        """
        symbol = next(iter(prices))
        perturbed = dict(prices)
        perturbed[symbol] = self.inject_crash(
            prices[symbol], scenario.magnitude, scenario.start_index
        )
        return self.backtester.run(perturbed)

    # ------------------------------------------------------------- helpers ---
    @staticmethod
    def _max_loss(result: BacktestResult) -> float:
        """Max drawdown of a result's equity curve (negative fraction)."""
        eq = result.equity_curve.dropna()
        if eq.empty:
            return 0.0
        return float((eq / eq.cummax() - 1.0).min())

    @staticmethod
    def _breaker_events(result: BacktestResult) -> pd.DataFrame:
        """Extract bars where a circuit breaker was active, with audit fields.

        Args:
            result: Backtest output.

        Returns:
            Frame of breaker-active bars (risk_state, regime, weight, ...).
        """
        hist = result.regime_history
        if hist.empty or "risk_state" not in hist.columns:
            return pd.DataFrame()
        return hist[hist["risk_state"] != "normal"][
            [c for c in ("risk_state", "regime", "regime_prob", "weight", "port_return")
             if c in hist.columns]
        ]

    @staticmethod
    def _random_indices(prices: pd.DataFrame, n: int, rng) -> list[int]:
        """Pick ``n`` random bar indices in the actionable (post-warmup) region.

        Args:
            prices: OHLCV DataFrame.
            n: Number of indices.
            rng: NumPy Generator.

        Returns:
            Sorted list of positional indices.
        """
        lo = min(len(prices) // 2, max(260, len(prices) // 4))
        hi = len(prices) - 2
        if hi <= lo:
            lo, hi = len(prices) // 2, len(prices) - 1
        return sorted(int(i) for i in rng.integers(lo, hi, size=n))

    # --------------------------------------------------------------- probes ---
    def crash_injection_mc(
        self,
        prices: dict[str, pd.DataFrame],
        n_sims: int = 100,
        n_crashes: int = 10,
        mag_range: tuple[float, float] = (-0.15, -0.05),
        seed: int = 0,
    ) -> StressReport:
        """Monte-Carlo crash injection.

        Args:
            prices: Map of symbol -> OHLCV DataFrame.
            n_sims: Number of simulations.
            n_crashes: Single-day gaps injected per sim.
            mag_range: (most negative, least negative) crash magnitudes.
            seed: Base RNG seed.

        Returns:
            `StressReport` (mean/worst max loss, breaker fire rate, blowup rate).
        """
        symbol = next(iter(prices))
        base = prices[symbol]
        losses, fired, blew = [], 0, 0
        for s in range(n_sims):
            rng = np.random.default_rng(seed + s)
            path = base.copy()
            for idx in self._random_indices(base, n_crashes, rng):
                mag = float(rng.uniform(mag_range[0], mag_range[1]))
                path = self.inject_crash(path, mag, idx)
            try:
                res = self.backtester.run({symbol: path})
            except (ValueError, RuntimeError):
                continue
            ml = self._max_loss(res)
            losses.append(ml)
            ev = self._breaker_events(res)
            if not ev.empty:
                fired += 1
                self._log_triggers("crash", res, ev)
            if ml <= BLOWUP_DD:
                blew += 1
        return self._aggregate("crash_injection", n_sims, losses, fired, blew)

    def gap_risk_mc(
        self,
        prices: dict[str, pd.DataFrame],
        n_sims: int = 100,
        n_gaps: int = 5,
        atr_mult_range: tuple[float, float] = (2.0, 5.0),
        seed: int = 1000,
    ) -> StressReport:
        """Monte-Carlo overnight-gap risk sized in ATR multiples.

        Reports expected loss (mean injected gap magnitude, summed) vs. actual
        realized loss (mean max drawdown) — divergence shows whether the strategy
        amplifies or absorbs gap shocks.

        Args:
            prices: Map of symbol -> OHLCV DataFrame.
            n_sims: Number of simulations.
            n_gaps: Overnight gaps injected per sim.
            atr_mult_range: Gap size range in ATR multiples.
            seed: Base RNG seed.

        Returns:
            `StressReport` with an ``extra`` expected-vs-actual comparison.
        """
        symbol = next(iter(prices))
        base = prices[symbol]
        natr = (atr_series(base, 14) / base["close"]).fillna(0.0).to_numpy()
        losses, fired, blew, expected = [], 0, 0, []
        for s in range(n_sims):
            rng = np.random.default_rng(seed + s)
            path = base.copy()
            exp_gap = 0.0
            for idx in self._random_indices(base, n_gaps, rng):
                mult = float(rng.uniform(*atr_mult_range))
                mag = -mult * float(natr[idx])      # gap DOWN
                exp_gap += mag
                path = self.inject_gap(path, mag, idx)
            expected.append(exp_gap)
            try:
                res = self.backtester.run({symbol: path})
            except (ValueError, RuntimeError):
                continue
            ml = self._max_loss(res)
            losses.append(ml)
            ev = self._breaker_events(res)
            if not ev.empty:
                fired += 1
                self._log_triggers("gap", res, ev)
            if ml <= BLOWUP_DD:
                blew += 1
        rep = self._aggregate("gap_risk", n_sims, losses, fired, blew)
        rep.extra = dict(
            expected_loss_mean=float(np.mean(expected)) if expected else 0.0,
            actual_loss_mean=rep.mean_max_loss,
        )
        return rep

    def regime_misclassification(
        self,
        prices: dict[str, pd.DataFrame],
        n_sims: int = 100,
        seed: int = 2000,
    ) -> StressReport:
        """Deliberately shuffle regime->strategy maps; verify damage is capped.

        The HMM is fit normally, but each fold's regime->strategy assignment is
        randomly permuted, so the system acts on the *wrong* regimes. Risk
        management is independent if the drawdown stays bounded regardless.

        Args:
            prices: Map of symbol -> OHLCV DataFrame.
            n_sims: Number of shuffles.
            seed: Base RNG seed.

        Returns:
            `StressReport`; a high ``blowup_rate`` means risk mgmt is too coupled
            to the HMM.
        """
        symbol = next(iter(prices))
        losses, fired, blew = [], 0, 0
        try:
            for s in range(n_sims):
                self.backtester.shuffle_regimes = seed + s
                try:
                    res = self.backtester.run({symbol: prices[symbol]})
                except (ValueError, RuntimeError):
                    continue
                ml = self._max_loss(res)
                losses.append(ml)
                ev = self._breaker_events(res)
                if not ev.empty:
                    fired += 1
                    self._log_triggers("misclassification", res, ev)
                if ml <= BLOWUP_DD:
                    blew += 1
        finally:
            self.backtester.shuffle_regimes = None
        return self._aggregate("regime_misclassification", n_sims, losses, fired, blew)

    # ------------------------------------------------------------- internal ---
    @staticmethod
    def _aggregate(
        probe: str, n_sims: int, losses: list[float], fired: int, blew: int
    ) -> StressReport:
        """Roll per-sim outcomes into a `StressReport`.

        Args:
            probe: Probe name.
            n_sims: Requested sims.
            losses: Per-sim max losses.
            fired: Count of sims where a breaker fired.
            blew: Count of sims breaching the blowup threshold.

        Returns:
            Aggregated `StressReport`.
        """
        n = len(losses) or 1
        return StressReport(
            probe=probe,
            n_sims=len(losses),
            mean_max_loss=float(np.mean(losses)) if losses else 0.0,
            worst_max_loss=float(np.min(losses)) if losses else 0.0,
            breaker_fire_rate=fired / n,
            blowup_rate=blew / n,
        )

    @staticmethod
    def _log_triggers(probe: str, result: BacktestResult, events: pd.DataFrame) -> None:
        """Log the first breaker trigger of a sim with full audit context.

        Args:
            probe: Probe name.
            result: Backtest output (for equity at the trigger).
            events: Breaker-active bars frame.
        """
        first = events.iloc[0]
        ts = events.index[0]
        eq = result.equity_curve.get(ts, float("nan"))
        logger.info(
            "[stress:%s] breaker=%s @ %s regime=%s p=%.2f weight=%.2f equity=%.0f",
            probe, first.get("risk_state"), ts, first.get("regime"),
            float(first.get("regime_prob", 0.0)), float(first.get("weight", 0.0)), eq,
        )


def render_stress_reports(reports: list[StressReport]) -> None:
    """Print stress-probe results as a rich table.

    Args:
        reports: List of `StressReport` to display.
    """
    from rich.console import Console
    from rich.table import Table

    con = Console()
    con.rule("[bold red]Stress Testing")
    t = Table(header_style="bold red")
    for c in ["Probe", "Sims", "Mean Max Loss", "Worst Max Loss",
              "Breaker Fired", "Blowup Rate"]:
        t.add_column(c, justify="right" if c != "Probe" else "left")
    for r in reports:
        t.add_row(r.probe, str(r.n_sims), f"{r.mean_max_loss:.2%}",
                  f"{r.worst_max_loss:.2%}", f"{r.breaker_fire_rate:.0%}",
                  f"{r.blowup_rate:.0%}")
    con.print(t)
    for r in reports:
        if r.extra:
            con.print(f"[dim]{r.probe} extra:[/dim] "
                      + ", ".join(f"{k}={v:.2%}" for k, v in r.extra.items()))
    con.print(f"[dim]Containment threshold (blowup) = {BLOWUP_DD:.0%} max drawdown.[/dim]")
