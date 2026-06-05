"""Performance analytics: Sharpe, drawdown, regime breakdown, benchmarks.

Computes risk-adjusted metrics from a backtest equity curve, decomposes
performance by regime and by HMM confidence bucket, compares against three
benchmarks (buy-and-hold, 200-SMA trend, random allocation), and reports
worst-case tail statistics. Renders rich tables to the terminal and exports the
canonical CSVs.

Conventions for the allocation model
------------------------------------
The backtester holds a target weight, not discrete trades, so:

* a **"trade"** is one *holding segment* between rebalances; its P&L is the
  compounded portfolio return while that weight was held;
* **regime / confidence breakdowns** are computed on *per-bar* portfolio
  returns grouped by the regime (or its filtered probability) active that bar.

Daily bars are assumed (``PERIODS_PER_YEAR = 252``) for annualization.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252
CONFIDENCE_BUCKETS = [(0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0001)]
EULER_MASCHERONI = 0.5772156649015329


# ----------------------------------------------- deflated Sharpe (Bailey/LdP) ---
def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Probability the true Sharpe exceeds ``sr_benchmark`` (Bailey & López de Prado).

    All Sharpe inputs are **per-observation (non-annualized)**. Corrects the
    Sharpe estimate for sample length and for the non-normality (skew, excess
    kurtosis) of returns — a high Sharpe on few, fat-tailed bars is less
    trustworthy than the raw number suggests.

    Args:
        sr: Observed per-bar Sharpe ratio.
        sr_benchmark: Benchmark per-bar Sharpe to beat (0 = "better than nothing").
        n_obs: Number of return observations.
        skew: Sample skewness of returns.
        kurt: Sample kurtosis of returns (normal = 3).

    Returns:
        Probability in ``[0, 1]`` (0.0 if fewer than 2 observations).
    """
    from scipy.stats import norm

    if n_obs < 2:
        return 0.0
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    stat = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(stat))


def expected_max_sharpe(n_trials: int, trials_sr_std: float) -> float:
    """Expected maximum per-bar Sharpe from ``n_trials`` independent trials.

    The deflation benchmark: under multiple testing, the best of many random
    strategies has a positive expected Sharpe even with no skill. Subtracting this
    is what turns a Probabilistic Sharpe into a *Deflated* Sharpe.

    Args:
        n_trials: Number of configurations tried (the multiple-testing count).
        trials_sr_std: Std of the per-bar Sharpe estimates across trials.

    Returns:
        Expected max per-bar Sharpe under the null (0.0 if <2 trials or no spread).
    """
    from scipy.stats import norm

    if n_trials < 2 or trials_sr_std <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return trials_sr_std * ((1.0 - g) * a + g * b)


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_trials: int = 1,
    trials_sr_std: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio: PSR against the multiple-testing benchmark.

    With ``n_trials == 1`` (frozen knobs, no sweep — the pre-registered case) the
    benchmark collapses to 0 and this is the Probabilistic Sharpe vs zero. With a
    sweep, pass ``n_trials`` and the spread of trial Sharpes to deflate harder.

    Args:
        sr: Observed per-bar Sharpe ratio.
        n_obs: Number of return observations.
        skew: Sample skewness of returns.
        kurt: Sample kurtosis of returns.
        n_trials: Configurations tried (1 = no multiple testing).
        trials_sr_std: Std of per-bar Sharpe across trials (only used if n_trials>1).

    Returns:
        DSR probability in ``[0, 1]``. ``> 0.5`` favours real skill; the gate
        requires ``> 0`` (i.e. not actively negative evidence).
    """
    sr0 = expected_max_sharpe(n_trials, trials_sr_std)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurt)


def pbo_cscv(returns_matrix, n_splits: int = 16) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey & López de Prado, 2015).

    Given per-bar returns for several strategy *configurations* (columns), the
    Combinatorially-Symmetric Cross-Validation procedure splits the ``T`` observations
    into ``n_splits`` contiguous blocks, forms every balanced in-sample / out-of-sample
    partition (half the blocks IS, half OOS), picks the **best in-sample-Sharpe** config in
    each partition, and records its out-of-sample rank. PBO is the fraction of partitions
    where that in-sample winner lands in the *bottom half* out-of-sample — how often
    selecting on backtest performance is expected to backfire.

    PBO near 0 means the in-sample winner generalizes; ``>= 0.5`` means the selection is
    overfit. The challenger gate requires ``< 0.5``. The pooled-block Sharpe is computed
    exactly from per-block sums/sum-of-squares (so the ~12.8k partitions of ``n_splits=16``
    stay cheap). Needs ``>= 2`` configs and an even ``n_splits >= 2``.

    Args:
        returns_matrix: ``(T, N)`` array-like of per-bar returns; columns = configs.
        n_splits: Even number of contiguous CSCV blocks (default 16).

    Returns:
        PBO probability in ``[0, 1]`` (``nan`` if inputs are insufficient).
    """
    from itertools import combinations

    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return float("nan")
    n_splits -= n_splits % 2
    if n_splits < 2:
        return float("nan")
    T, N = M.shape
    if T < n_splits:
        return float("nan")

    block = T // n_splits
    counts = np.array([block] * n_splits)
    s1 = np.vstack([M[i * block:(i + 1) * block].sum(axis=0) for i in range(n_splits)])
    s2 = np.vstack([(M[i * block:(i + 1) * block] ** 2).sum(axis=0) for i in range(n_splits)])

    def _pooled_sharpe(sel: tuple[int, ...]) -> np.ndarray:
        n = int(counts[list(sel)].sum())
        S1 = s1[list(sel)].sum(axis=0)
        S2 = s2[list(sel)].sum(axis=0)
        mean = S1 / n
        var = (S2 - S1 * S1 / n) / (n - 1)
        sd = np.sqrt(np.where(var > 0, var, np.nan))
        return mean / sd

    idx = list(range(n_splits))
    overfit = 0
    total = 0
    for is_sel in combinations(idx, n_splits // 2):
        oos_sel = tuple(i for i in idx if i not in set(is_sel))
        s_is = _pooled_sharpe(is_sel)
        if np.all(np.isnan(s_is)):
            continue
        s_oos = _pooled_sharpe(oos_sel)
        best = int(np.nanargmax(s_is))
        order = np.argsort(np.argsort(np.nan_to_num(s_oos, nan=-np.inf)))
        rank = float(order[best] + 1)          # 1 = worst OOS, N = best OOS
        omega = min(max(rank / (N + 1), 1e-6), 1 - 1e-6)
        overfit += 1 if math.log(omega / (1 - omega)) <= 0.0 else 0
        total += 1
    if total == 0:
        return float("nan")
    return overfit / total


@dataclass
class PerformanceReport:
    """Summary of backtest performance."""

    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_holding_period: float = 0.0
    regime_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    benchmark_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    worst_case: dict[str, float] = field(default_factory=dict)


class PerformanceAnalyzer:
    """Computes performance metrics from a backtest result."""

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        """Initialize the analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for risk-adjusted metrics.
        """
        self.risk_free_rate = risk_free_rate

    # --------------------------------------------------------------- core ---
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio.

        Args:
            returns: Period (per-bar) returns.

        Returns:
            Sharpe ratio (0.0 if std is zero or series empty).
        """
        r = returns.dropna()
        if r.empty or r.std(ddof=0) == 0:
            return 0.0
        excess = r - self.risk_free_rate / PERIODS_PER_YEAR
        return float(excess.mean() / r.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR))

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (downside deviation in the denominator).

        Args:
            returns: Period returns.

        Returns:
            Sortino ratio (0.0 if no downside or series empty).
        """
        r = returns.dropna()
        if r.empty:
            return 0.0
        excess = r - self.risk_free_rate / PERIODS_PER_YEAR
        downside = r[r < 0]
        dd = np.sqrt((downside**2).mean()) if not downside.empty else 0.0
        if dd == 0:
            return 0.0
        return float(excess.mean() / dd * np.sqrt(PERIODS_PER_YEAR))

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Worst peak-to-trough drawdown.

        Args:
            equity_curve: Equity over time.

        Returns:
            Max drawdown as a negative fraction (0.0 if empty).
        """
        eq = equity_curve.dropna()
        if eq.empty:
            return 0.0
        return float((eq / eq.cummax() - 1.0).min())

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Longest run (in bars) the equity stays below a prior peak.

        Args:
            equity_curve: Equity over time.

        Returns:
            Longest underwater stretch in bars.
        """
        eq = equity_curve.dropna()
        if eq.empty:
            return 0
        peak = eq.cummax()
        underwater = eq < peak
        longest = run = 0
        for uw in underwater:
            run = run + 1 if uw else 0
            longest = max(longest, run)
        return longest

    def cagr(self, equity_curve: pd.Series) -> float:
        """Compound annual growth rate.

        Args:
            equity_curve: Equity over time.

        Returns:
            CAGR (0.0 if insufficient data).
        """
        eq = equity_curve.dropna()
        if len(eq) < 2 or eq.iloc[0] <= 0:
            return 0.0
        years = len(eq) / PERIODS_PER_YEAR
        if years <= 0:
            return 0.0
        return float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)

    # --------------------------------------------------------- breakdowns ---
    def regime_breakdown(self, result: BacktestResult) -> dict[str, dict[str, float]]:
        """Per-regime performance breakdown.

        Args:
            result: Backtest output with per-bar regime history.

        Returns:
            Map of regime label -> metric dict (pct_time, return_contribution,
            avg_pnl, win_rate, sharpe).
        """
        hist = result.regime_history
        if hist.empty:
            return {}
        n = len(hist)
        out: dict[str, dict[str, float]] = {}
        for regime, grp in hist.groupby("regime"):
            pr = grp["port_return"]
            out[str(regime)] = dict(
                pct_time=len(grp) / n,
                return_contribution=float((1.0 + pr).prod() - 1.0),
                avg_pnl=float(pr.mean()),
                win_rate=float((pr > 0).mean()),
                sharpe=self.sharpe_ratio(pr),
            )
        return out

    def confidence_breakdown(self, result: BacktestResult) -> dict[str, dict[str, float]]:
        """Per-confidence-bucket breakdown (does HMM confidence add value?).

        Args:
            result: Backtest output with per-bar regime probabilities.

        Returns:
            Map of bucket label -> metric dict (bars, sharpe, win_rate, avg_pnl).
        """
        hist = result.regime_history
        if hist.empty:
            return {}
        out: dict[str, dict[str, float]] = {}
        for lo, hi in CONFIDENCE_BUCKETS:
            mask = (hist["regime_prob"] >= lo) & (hist["regime_prob"] < hi)
            grp = hist[mask]
            label = f"{int(lo*100)}-{int(min(hi,1.0)*100)}%" if hi <= 1.0 else f">{int(lo*100)}%"
            if lo == 0.0:
                label = f"<{int(hi*100)}%"
            elif hi > 1.0:
                label = f"{int(lo*100)}%+"
            pr = grp["port_return"]
            out[label] = dict(
                bars=int(len(grp)),
                sharpe=self.sharpe_ratio(pr) if len(grp) else 0.0,
                win_rate=float((pr > 0).mean()) if len(grp) else 0.0,
                avg_pnl=float(pr.mean()) if len(grp) else 0.0,
            )
        return out

    def trade_stats(self, result: BacktestResult) -> dict[str, float]:
        """Holding-segment ("trade") statistics for the allocation model.

        A trade is the stretch between two rebalances; its P&L is the
        compounded portfolio return while the weight was held.

        Args:
            result: Backtest output.

        Returns:
            Dict with total_trades, win_rate, avg_win, avg_loss, profit_factor,
            avg_holding_period.
        """
        ret = result.returns.dropna()
        trades = result.trades
        if ret.empty or trades.empty:
            return dict(total_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                       profit_factor=0.0, avg_holding_period=0.0)

        # Segment boundaries are rebalance timestamps; build segment returns.
        bounds = list(trades["timestamp"]) + [ret.index[-1]]
        seg_returns: list[float] = []
        seg_lengths: list[int] = []
        for i in range(len(bounds) - 1):
            seg = ret.loc[(ret.index >= bounds[i]) & (ret.index < bounds[i + 1])]
            if seg.empty:
                continue
            seg_returns.append(float((1.0 + seg).prod() - 1.0))
            seg_lengths.append(len(seg))

        if not seg_returns:
            return dict(total_trades=len(trades), win_rate=0.0, avg_win=0.0,
                       avg_loss=0.0, profit_factor=0.0, avg_holding_period=0.0)

        arr = np.array(seg_returns)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        return dict(
            total_trades=int(len(trades)),
            win_rate=float((arr > 0).mean()),
            avg_win=float(wins.mean()) if wins.size else 0.0,
            avg_loss=float(losses.mean()) if losses.size else 0.0,
            profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            avg_holding_period=float(np.mean(seg_lengths)),
        )

    def worst_case(self, result: BacktestResult) -> dict[str, float]:
        """Tail statistics: worst day/week/month, loss streak, time underwater.

        Args:
            result: Backtest output.

        Returns:
            Dict of worst-case metrics.
        """
        ret = result.returns.dropna()
        if ret.empty:
            return {}
        roll = lambda w: float((1.0 + ret).rolling(w).apply(np.prod, raw=True).min() - 1.0)
        # max consecutive losing bars
        streak = longest = 0
        for x in ret:
            streak = streak + 1 if x < 0 else 0
            longest = max(longest, streak)
        return dict(
            worst_day=float(ret.min()),
            worst_week=roll(5),
            worst_month=roll(21),
            max_consecutive_losses=int(longest),
            longest_underwater_bars=self.max_drawdown_duration(result.equity_curve),
        )

    # --------------------------------------------------------- benchmarks ---
    def _series_metrics(self, equity: pd.Series) -> dict[str, float]:
        """Compact metric set for a benchmark equity curve.

        Args:
            equity: Benchmark equity series.

        Returns:
            Dict of total_return, cagr, sharpe, max_drawdown.
        """
        eq = equity.dropna()
        if len(eq) < 2:
            return dict(total_return=0.0, cagr=0.0, sharpe=0.0, max_drawdown=0.0)
        rets = eq.pct_change().dropna()
        return dict(
            total_return=float(eq.iloc[-1] / eq.iloc[0] - 1.0),
            cagr=self.cagr(eq),
            sharpe=self.sharpe_ratio(rets),
            max_drawdown=self.max_drawdown(eq),
        )

    def benchmarks(
        self, result: BacktestResult, close: pd.Series, n_random: int = 100, seed: int = 0
    ) -> dict[str, dict[str, float]]:
        """Buy-and-hold, 200-SMA trend, and random-allocation benchmarks.

        Args:
            result: Backtest output (defines the OOS span + strategy metrics).
            close: Full close-price series (>= OOS span; needed for SMA200).
            n_random: Number of random-allocation seeds.
            seed: Base RNG seed.

        Returns:
            Map of benchmark name -> metric dict. ``random`` reports mean/std of
            total return and Sharpe across the seeds.
        """
        oos = result.equity_curve.index
        cap = result.initial_capital
        asset_ret = result.asset_returns.reindex(oos).fillna(0.0)

        out: dict[str, dict[str, float]] = {}
        out["strategy"] = dict(
            total_return=float(result.equity_curve.iloc[-1] / cap - 1.0),
            cagr=self.cagr(result.equity_curve),
            sharpe=self.sharpe_ratio(result.returns),
            max_drawdown=self.max_drawdown(result.equity_curve),
        )

        # a) buy-and-hold
        bh = cap * (1.0 + asset_ret).cumprod()
        out["buy_hold"] = self._series_metrics(bh)

        # b) 200-SMA trend: long when close>SMA200 (decided at t, earns t->t+1)
        sma = close.rolling(200).mean()
        in_mkt = (close > sma).reindex(oos).fillna(False).astype(float).shift(1).fillna(0.0)
        sma_ret = in_mkt * asset_ret
        out["sma200_trend"] = self._series_metrics(cap * (1.0 + sma_ret).cumprod())

        # c) random allocation at the same rebalance cadence + slippage
        levels = np.array([0.0, 0.60, 0.95, 1.1875])
        thr = 0.10
        slip = 0.0005
        tot, shp = [], []
        for s in range(n_random):
            rng = np.random.default_rng(seed + s)
            eq = cap
            held = 0.0
            rs = []
            for r in asset_ret.to_numpy():
                pr = held * float(r)
                eq *= (1.0 + pr)
                tgt = float(rng.choice(levels))
                slp = 0.0
                if abs(tgt - held) >= thr:
                    slp = abs(tgt - held) * slip
                    eq *= (1.0 - slp)
                    held = tgt
                rs.append((1.0 + pr) * (1.0 - slp) - 1.0)
            rser = pd.Series(rs, index=oos)
            tot.append(eq / cap - 1.0)
            shp.append(self.sharpe_ratio(rser))
        out["random"] = dict(
            total_return_mean=float(np.mean(tot)), total_return_std=float(np.std(tot)),
            sharpe_mean=float(np.mean(shp)), sharpe_std=float(np.std(shp)),
        )
        return out

    # ------------------------------------------------------------- analyze ---
    def analyze(
        self, result: BacktestResult, benchmark: pd.Series, with_benchmarks: bool = True
    ) -> PerformanceReport:
        """Produce a full performance report.

        Args:
            result: Backtest output.
            benchmark: Full close-price series of the traded asset (used for
                buy-and-hold and 200-SMA benchmarks).
            with_benchmarks: Whether to run the (slower) benchmark suite.

        Returns:
            `PerformanceReport`.
        """
        eq = result.equity_curve
        ts = self.trade_stats(result)
        mdd = self.max_drawdown(eq)
        cagr = self.cagr(eq)
        report = PerformanceReport(
            total_return=float(eq.iloc[-1] / result.initial_capital - 1.0) if not eq.empty else 0.0,
            cagr=cagr,
            sharpe=self.sharpe_ratio(result.returns),
            sortino=self.sortino_ratio(result.returns),
            max_drawdown=mdd,
            max_drawdown_duration=self.max_drawdown_duration(eq),
            calmar=(cagr / abs(mdd)) if mdd < 0 else 0.0,
            win_rate=ts["win_rate"],
            avg_win=ts["avg_win"],
            avg_loss=ts["avg_loss"],
            profit_factor=ts["profit_factor"],
            total_trades=int(ts["total_trades"]),
            avg_holding_period=ts["avg_holding_period"],
            regime_breakdown=self.regime_breakdown(result),
            confidence_breakdown=self.confidence_breakdown(result),
            worst_case=self.worst_case(result),
        )
        if with_benchmarks:
            report.benchmark_comparison = self.benchmarks(result, benchmark)
        return report


# ===========================================================================
# Reporting (rich tables) + CSV export
# ===========================================================================
def render_report(result: BacktestResult, report: PerformanceReport) -> None:
    """Print the full performance report as rich tables to the terminal.

    Args:
        result: Backtest output.
        report: Computed performance report.
    """
    from rich.console import Console
    from rich.table import Table

    con = Console()
    con.rule(f"[bold]Walk-Forward Backtest — {result.symbol}")

    core = Table(title="Core Metrics", title_justify="left", header_style="bold cyan")
    core.add_column("Metric"); core.add_column("Value", justify="right")
    rows = [
        ("Total Return", f"{report.total_return:.2%}"),
        ("CAGR", f"{report.cagr:.2%}"),
        ("Sharpe (ann.)", f"{report.sharpe:.2f}"),
        ("Sortino (ann.)", f"{report.sortino:.2f}"),
        ("Calmar", f"{report.calmar:.2f}"),
        ("Max Drawdown", f"{report.max_drawdown:.2%}"),
        ("Max DD Duration (bars)", f"{report.max_drawdown_duration}"),
        ("Win Rate (segments)", f"{report.win_rate:.2%}"),
        ("Avg Win / Avg Loss", f"{report.avg_win:.2%} / {report.avg_loss:.2%}"),
        ("Profit Factor", f"{report.profit_factor:.2f}"),
        ("Total Trades (rebalances)", f"{report.total_trades}"),
        ("Avg Holding (bars)", f"{report.avg_holding_period:.1f}"),
    ]
    for k, v in rows:
        core.add_row(k, v)
    con.print(core)

    if report.regime_breakdown:
        rt = Table(title="Per-Regime Breakdown", title_justify="left", header_style="bold magenta")
        for c in ["Regime", "% Time", "Return Contrib", "Avg Bar P&L", "Win Rate", "Sharpe"]:
            rt.add_column(c, justify="right" if c != "Regime" else "left")
        for regime, m in report.regime_breakdown.items():
            rt.add_row(regime, f"{m['pct_time']:.1%}", f"{m['return_contribution']:.2%}",
                       f"{m['avg_pnl']:.3%}", f"{m['win_rate']:.1%}", f"{m['sharpe']:.2f}")
        con.print(rt)

    if report.confidence_breakdown:
        ct = Table(title="Confidence Buckets", title_justify="left", header_style="bold green")
        for c in ["Confidence", "Bars", "Sharpe", "Win Rate", "Avg P&L"]:
            ct.add_column(c, justify="right" if c != "Confidence" else "left")
        for bucket, m in report.confidence_breakdown.items():
            ct.add_row(bucket, f"{m['bars']}", f"{m['sharpe']:.2f}",
                       f"{m['win_rate']:.1%}", f"{m['avg_pnl']:.3%}")
        con.print(ct)

    if report.worst_case:
        wt = Table(title="Worst Case", title_justify="left", header_style="bold red")
        wt.add_column("Metric"); wt.add_column("Value", justify="right")
        wc = report.worst_case
        wt.add_row("Worst Day", f"{wc['worst_day']:.2%}")
        wt.add_row("Worst Week", f"{wc['worst_week']:.2%}")
        wt.add_row("Worst Month", f"{wc['worst_month']:.2%}")
        wt.add_row("Max Consecutive Losses", f"{wc['max_consecutive_losses']}")
        wt.add_row("Longest Underwater (bars)", f"{wc['longest_underwater_bars']}")
        con.print(wt)

    if report.benchmark_comparison:
        bt = Table(title="Benchmark Comparison", title_justify="left", header_style="bold yellow")
        for c in ["Strategy/Benchmark", "Total Return", "CAGR", "Sharpe", "Max DD"]:
            bt.add_column(c, justify="right" if c != "Strategy/Benchmark" else "left")
        bc = report.benchmark_comparison
        for name in ["strategy", "buy_hold", "sma200_trend"]:
            if name in bc:
                m = bc[name]
                bt.add_row(name, f"{m['total_return']:.2%}", f"{m['cagr']:.2%}",
                           f"{m['sharpe']:.2f}", f"{m['max_drawdown']:.2%}")
        if "random" in bc:
            m = bc["random"]
            bt.add_row("random (100x) mean±std",
                       f"{m['total_return_mean']:.2%} ± {m['total_return_std']:.2%}",
                       "—", f"{m['sharpe_mean']:.2f} ± {m['sharpe_std']:.2f}", "—")
        con.print(bt)


def export_csvs(
    result: BacktestResult, report: PerformanceReport, outdir: str | Path
) -> dict[str, Path]:
    """Write equity_curve, trade_log, regime_history, benchmark_comparison CSVs.

    Args:
        result: Backtest output.
        report: Computed performance report (for benchmark comparison).
        outdir: Destination directory (created if absent).

    Returns:
        Map of artifact name -> written path.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p = out / "equity_curve.csv"
    result.equity_curve.to_frame("equity").assign(
        return_=result.returns, asset_return=result.asset_returns
    ).to_csv(p)
    paths["equity_curve"] = p

    p = out / "trade_log.csv"
    result.trades.to_csv(p, index=False)
    paths["trade_log"] = p

    p = out / "regime_history.csv"
    result.regime_history.to_csv(p)
    paths["regime_history"] = p

    if report.benchmark_comparison:
        p = out / "benchmark_comparison.csv"
        pd.DataFrame(report.benchmark_comparison).T.to_csv(p)
        paths["benchmark_comparison"] = p

    return paths
