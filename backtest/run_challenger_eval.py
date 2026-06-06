"""Directional eval harness: baseline 12-1 vs residual-momentum + vol-target challengers.

Runs the frozen baseline (raw 12-1 momentum + HMM gross overlay) against the challenger
variants (residual/idiosyncratic momentum with no / HMM / vol-target / both overlays) and
the universe-aware benchmarks (SPY cap-weight, equal-weight book), all through the SAME
net-of-cost engine (slippage on turnover + risk-free credit on idle cash). Reports Sharpe,
max drawdown, CAGR, Deflated Sharpe (multiple-testing-corrected across the variants), and a
CSCV Probability of Backtest Overfitting across the variant set.

NOT A GATE PASS. The historical universe is today's S&P 500 constituents, so this is
**survivorship-biased and only directional** — it tells you whether a swap helps *enough*
to be worth forward-paper testing, not whether it has a real edge. The real gate is
forward-paper ≥12 months vs EW-S&P500 + SPY (docs/analysis/2026-06-05-idio-momentum-
challenger-prereg.md). If a challenger does not beat the baseline even on biased history,
drop it now.

Usage:
    .venv/bin/python -m backtest.run_challenger_eval --limit 100 --start 2015-01-01
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from backtest.backtester import BacktestConfig, Backtester
from backtest.performance import PerformanceAnalyzer, deflated_sharpe_ratio, pbo_cscv
from core.cross_sectional_ranking import (
    make_book_weights,
    make_book_weights_challenger,
)
from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import StrategyConfig
from core.risk_manager import RiskConfig, RiskManager
from data.constituents import load_many, load_sector_map, load_sp500
from data.feature_engineering import FeatureEngineer
from data.market_data import load_ohlcv

PROXY = "SPY"
PERIODS_PER_YEAR = 252


def _build_backtester(fast: bool) -> Backtester:
    """Backtester wired with the real stack; a reduced HMM keeps the eval tractable."""
    hmm_cfg = (
        HMMConfig(n_candidates=[3], n_init=2)
        if fast
        else HMMConfig(n_candidates=[3, 4], n_init=5)
    )
    return Backtester(
        BacktestConfig(step_size=126, credit_cash_rf=True),
        HMMEngine(hmm_cfg),
        StrategyConfig(),
        RiskManager(RiskConfig()),
        FeatureEngineer(),
    )


def _per_bar_stats(ret: pd.Series, rf: float) -> tuple[float, float, float, int]:
    """Per-bar (non-annualized) Sharpe + skew + kurtosis + n_obs for the DSR."""
    from scipy.stats import kurtosis, skew

    r = ret.dropna()
    if len(r) < 3 or r.std(ddof=0) == 0:
        return 0.0, 0.0, 3.0, len(r)
    sr = float((r.mean() - rf / PERIODS_PER_YEAR) / r.std(ddof=0))
    return sr, float(skew(r)), float(kurtosis(r, fisher=False)), len(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100,
                    help="cap universe size (yfinance throttles; directional anyway)")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--fast", action="store_true", default=True,
                    help="reduced HMM for speed (default on)")
    ap.add_argument("--full-hmm", dest="fast", action="store_false")
    args = ap.parse_args()

    for name in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies",
                 "backtest.backtester", "data.market_data"):
        logging.getLogger(name).setLevel(logging.ERROR)

    print(f"Loading proxy {PROXY} + universe ({args.start}..{args.end or 'now'}) ...")
    spy = load_ohlcv(PROXY, start=args.start, end=args.end)
    universe = load_sp500(for_yfinance=True)[: args.limit]
    sector_map = load_sector_map()
    cons = load_many(universe, start=args.start, end=args.end)
    cons.pop(PROXY, None)
    print(f"  {len(cons)}/{len(universe)} names loaded; {len(spy)} proxy bars.")
    if len(cons) < 20:
        print("  Too few names loaded — aborting.")
        return

    market_close = spy["close"]
    frames_all = {PROXY: spy, **cons}     # SPY first -> drives the regime

    common = dict(
        frac=0.10, max_single=0.15, max_concurrent=50,
        sector_map=sector_map, max_sector_frac=0.30,
    )
    variants = {
        "baseline_raw_hmm": make_book_weights(
            cons, risk_on_gross=1.0, risk_off_gross=0.5, overlay="hmm", **common,
        ),
        "raw_none": make_book_weights(cons, overlay="none", **common),
        "raw_vol_target": make_book_weights(
            cons, overlay="vol_target", target_vol=0.12, **common,
        ),
        "raw_both": make_book_weights(cons, overlay="both", target_vol=0.12, **common),
        "resid_none": make_book_weights_challenger(
            cons, market_close, overlay="none", **common,
        ),
        "resid_hmm": make_book_weights_challenger(
            cons, market_close, overlay="hmm", **common,
        ),
        "resid_vol_target": make_book_weights_challenger(
            cons, market_close, overlay="vol_target", target_vol=0.12, **common,
        ),
        "resid_both": make_book_weights_challenger(
            cons, market_close, overlay="both", target_vol=0.12, **common,
        ),
    }

    bt = _build_backtester(args.fast)
    analyzer = PerformanceAnalyzer(risk_free_rate=bt.config.risk_free_rate)
    rf = bt.config.risk_free_rate

    results: dict[str, pd.Series] = {}
    oos_idx = None
    for label, wf in variants.items():
        print(f"Running {label} ...")
        eq = bt.run_portfolio(frames_all, weight_fn=wf)
        results[label] = eq
        oos_idx = eq.index

    # benchmarks on the same OOS index, through the SAME cost engine as the strategies
    # (matched slippage on turnover + rf credit on idle cash). A frictionless benchmark
    # would rig the comparison — the exact confound benchmarks.py warns about (it flipped a
    # prior validation 0/5 -> 1/5). EW is daily-reconstituted so it DOES pay turnover cost.
    from backtest.benchmarks import simulate_portfolio

    rf_daily = (bt.config.risk_free_rate / PERIODS_PER_YEAR) if bt.config.credit_cash_rf else 0.0
    cap = bt.config.initial_capital
    names = list(cons)
    ew_w = {s: 1.0 / len(names) for s in names}
    results["EW_universe"] = simulate_portfolio(
        cons, oos_idx, lambda t, hist: ew_w, bt.config.slippage_pct, rf_daily, cap,
    )
    results["SPY_hold"] = simulate_portfolio(
        {PROXY: spy}, oos_idx, lambda t, hist: {PROXY: 1.0},
        bt.config.slippage_pct, rf_daily, cap,
    )

    # ---- metrics ----
    strat_labels = list(variants)
    per_bar_sr = {}
    rows = []
    for label, eq in results.items():
        ret = eq.pct_change().fillna(0.0)
        sr, sk, ku, n = _per_bar_stats(ret, rf)
        per_bar_sr[label] = sr
        rows.append(dict(
            strategy=label,
            total_return=float(eq.iloc[-1] / eq.iloc[0] - 1.0),
            cagr=analyzer.cagr(eq),
            sharpe=analyzer.sharpe_ratio(ret),
            max_dd=analyzer.max_drawdown(eq),
            _sr=sr, _sk=sk, _ku=ku, _n=n,
        ))

    trials_sr_std = float(np.std([per_bar_sr[l] for l in strat_labels]))
    n_trials = len(strat_labels)
    for row in rows:
        row["dsr"] = deflated_sharpe_ratio(
            row.pop("_sr"), row.pop("_n"), row.pop("_sk"), row.pop("_ku"),
            n_trials=n_trials, trials_sr_std=trials_sr_std,
        ) if row["strategy"] in strat_labels else float("nan")
        for k in ("_sr", "_sk", "_ku", "_n"):
            row.pop(k, None)

    # PBO across the strategy variants only
    ret_mat = pd.DataFrame(
        {l: results[l].pct_change().fillna(0.0) for l in strat_labels}
    ).to_numpy()
    pbo = pbo_cscv(ret_mat, n_splits=16)

    # ---- render ----
    print("\n" + "=" * 78)
    print("DIRECTIONAL EVAL — survivorship-biased, NOT a gate pass")
    print("=" * 78)
    hdr = f"{'strategy':<20}{'tot_ret':>10}{'cagr':>9}{'sharpe':>9}{'max_dd':>9}{'dsr':>7}"
    print(hdr)
    print("-" * len(hdr))
    order = strat_labels + ["SPY_hold", "EW_universe"]
    for row in sorted(rows, key=lambda r: order.index(r["strategy"])):
        dsr = f"{row['dsr']:.2f}" if row["dsr"] == row["dsr"] else "  —"
        print(f"{row['strategy']:<20}{row['total_return']:>9.1%}{row['cagr']:>8.1%}"
              f"{row['sharpe']:>9.2f}{row['max_dd']:>8.1%}{dsr:>7}")
    print("-" * len(hdr))
    print(f"PBO (CSCV across {n_trials} variants): {pbo:.2f}  "
          f"(<0.5 = in-sample winner generalizes)")
    best = max((r for r in rows if r["strategy"] in strat_labels),
               key=lambda r: r["sharpe"])
    print(f"\nBest by Sharpe: {best['strategy']} "
          f"(Sharpe {best['sharpe']:.2f}, maxDD {best['max_dd']:.1%}, DSR {best['dsr']:.2f})")
    print("Reminder: directional only. Promote to forward-paper, do not deploy real money.")


if __name__ == "__main__":
    main()
