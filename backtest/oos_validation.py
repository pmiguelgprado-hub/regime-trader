"""Out-of-sample edge validation: one long walk-forward per asset, sliced by crisis.

The single-period anchor (SPY 2019-24, 52.8% / Sharpe 1.22) is ONE asset, ONE
(bull) period. This harness falsifies — not confirms — the edge by:

1. Running one true walk-forward per ETF over max history (no per-crisis re-runs;
   the backtester needs train 504 + ~450-bar warmup ≈ 4y, so short windows abort).
2. Slicing the resulting OOS equity/return series by crisis windows and recomputing
   sub-window-aware metrics (``total_return`` from compounded slice returns, NOT
   from ``initial_capital``; benchmarks recomputed per slice).
3. Comparing strategy vs buy&hold vs SMA200 **inside the crises**, where the
   de-risking thesis must pay if the edge is real (and not just bull-market beta
   from the halt floor staying invested).

ETFs only — single names inject survivorship bias. Regime inference is
floor-independent, so the floor sweep re-uses this same machinery per asset.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd

from backtest.backtester import BacktestConfig, Backtester, BacktestResult
from backtest.performance import PERIODS_PER_YEAR, PerformanceAnalyzer
from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import StrategyConfig
from core.risk_manager import RiskConfig, RiskManager
from data.feature_engineering import FeatureEngineer


# Crisis windows: where the de-risking thesis must pay (buy&hold suffers).
# "full" is filled in per run from the actual OOS span.
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "gfc_2008": ("2007-10-01", "2009-06-30"),
    "euro_2011": ("2011-07-01", "2011-12-31"),
    "china_2015": ("2015-08-01", "2016-02-29"),
    "q4_2018": ("2018-10-01", "2018-12-31"),
    "covid_2020": ("2020-02-15", "2020-04-30"),
    "bear_2022": ("2022-01-01", "2022-10-31"),
}


@dataclass
class SliceMetrics:
    """Strategy vs benchmarks over one time slice (base-independent)."""

    label: str
    n_bars: int
    strat_return: float
    strat_sharpe: float
    strat_mdd: float
    buy_hold_return: float
    sma200_return: float
    pct_halted: float

    def beats_bh(self) -> bool:
        return self.strat_return > self.buy_hold_return


def slice_metrics(
    result: BacktestResult,
    close: pd.Series,
    label: str,
    start: str | None = None,
    end: str | None = None,
    rf: float = 0.045,
) -> SliceMetrics | None:
    """Compute strategy + benchmark metrics over an OOS sub-window.

    Args:
        result: Full-history backtest output.
        close: Full close series of the traded asset (for SMA200, computed on the
            full series then sliced so the 200-bar lookback is causal-complete).
        label: Window name.
        start: ISO inclusive start (None = from first OOS bar).
        end: ISO inclusive end (None = to last OOS bar).
        rf: Annual risk-free rate.

    Returns:
        ``SliceMetrics`` or ``None`` if the window has < 2 OOS bars (e.g. the
        crisis predates this asset's first OOS bar).
    """
    an = PerformanceAnalyzer(risk_free_rate=rf)
    idx = result.returns.index
    lo = pd.Timestamp(start) if start else idx[0]
    hi = pd.Timestamp(end) if end else idx[-1]
    mask = (idx >= lo) & (idx <= hi)
    if mask.sum() < 2:
        return None

    strat_ret = result.returns[mask]
    asset_ret = result.asset_returns.reindex(idx).fillna(0.0)[mask]
    eq = result.equity_curve[mask]

    # SMA200 trend benchmark, recomputed on the slice (signal decided at t earns t->t+1)
    sma = close.rolling(200).mean()
    in_mkt = (close > sma).reindex(idx).fillna(False).astype(float).shift(1).fillna(0.0)[mask]
    sma_ret = in_mkt * asset_ret

    hist = result.regime_history
    if not hist.empty:
        hm = (hist.index >= lo) & (hist.index <= hi)
        pct_halted = float((hist["risk_state"][hm] == "halted").mean()) if hm.sum() else 0.0
    else:
        pct_halted = 0.0

    return SliceMetrics(
        label=label,
        n_bars=int(mask.sum()),
        strat_return=float((1.0 + strat_ret).prod() - 1.0),
        strat_sharpe=an.sharpe_ratio(strat_ret),
        strat_mdd=an.max_drawdown(eq / eq.iloc[0]),
        buy_hold_return=float((1.0 + asset_ret).prod() - 1.0),
        sma200_return=float((1.0 + sma_ret).prod() - 1.0),
        pct_halted=pct_halted,
    )


def build_backtester(config: dict, halt_floor_mult: float | None = None) -> Backtester:
    """Wire a backtester from parsed settings, optionally overriding the halt floor.

    Args:
        config: Parsed ``settings.yaml``.
        halt_floor_mult: If given, override ``RiskConfig.halt_floor_mult`` (for the
            exposure-vs-edge floor sweep). None keeps the configured default.

    Returns:
        Configured `Backtester`.
    """
    def bd(dc, d):
        fields = {f.name for f in dataclasses.fields(dc)}
        return dc(**{k: v for k, v in (d or {}).items() if k in fields})

    risk_cfg = bd(RiskConfig, config.get("risk", {}))
    if halt_floor_mult is not None:
        risk_cfg = dataclasses.replace(risk_cfg, halt_floor_mult=halt_floor_mult)
    return Backtester(
        bd(BacktestConfig, config.get("backtest", {})),
        HMMEngine(bd(HMMConfig, config.get("hmm", {}))),
        bd(StrategyConfig, config.get("strategy", {})),
        RiskManager(risk_cfg),
        FeatureEngineer(),
    )


def run_asset(
    config: dict,
    symbol: str,
    ohlcv: pd.DataFrame,
    halt_floor_mult: float | None = None,
) -> tuple[BacktestResult, list[SliceMetrics]]:
    """Run one full-history walk-forward and slice it into full + crisis windows.

    Args:
        config: Parsed settings.
        symbol: Ticker (for labelling).
        ohlcv: Full-history OHLCV.
        halt_floor_mult: Optional halt-floor override (floor sweep).

    Returns:
        ``(result, [SliceMetrics...])`` — first slice is "full", then each crisis
        window that overlaps this asset's OOS span.
    """
    bt = build_backtester(config, halt_floor_mult=halt_floor_mult)
    result = bt.run({symbol: ohlcv})
    close = ohlcv["close"].astype(float)
    rf = config.get("backtest", {}).get("risk_free_rate", 0.045)

    slices: list[SliceMetrics] = []
    full = slice_metrics(result, close, "full", None, None, rf)
    if full:
        slices.append(full)
    for name, (s, e) in CRISIS_WINDOWS.items():
        sm = slice_metrics(result, close, name, s, e, rf)
        if sm:
            slices.append(sm)
    return result, slices
