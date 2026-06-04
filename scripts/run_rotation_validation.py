"""Vía B validation: cross-asset regime rotation vs 60/40 + risk-parity.

Runs the PRE-REGISTERED test (docs/analysis/2026-06-04-rotation-prereg.md). Knobs
are frozen; this script only measures and applies the gate. A negative verdict is a
valid, expected outcome — nothing here is tuned to win.

Cost symmetry: rotation and both benchmarks run through the same per-bar engine
with credit_cash_rf=True (idle cash earns rf) and identical slippage.
"""

import dataclasses
import json
import sys

import numpy as np
import yaml
from scipy.stats import kurtosis, skew

sys.path.insert(0, ".")
from backtest.benchmarks import risk_parity_returns, static_mix_returns
from backtest.oos_validation import build_backtester
from backtest.performance import (
    PERIODS_PER_YEAR,
    PerformanceAnalyzer,
    deflated_sharpe_ratio,
)
from core.asset_rotation import RotationConfig
from data.market_data import load_ohlcv

START, END = "2004-01-01", "2024-12-31"
SPLIT = "2016-01-01"          # sub-period boundary (pre-registered)
ROT = RotationConfig()
TICKERS = sorted(set(ROT.symbols) | {"SPY", "TLT"})  # basket + 60/40 legs
OUT_JSON = "tmp/rotation_validation.json"
ANALYZER = PerformanceAnalyzer(risk_free_rate=0.045)


def metrics(equity, label):
    """Risk-adjusted metrics for an equity slice (base-independent where it matters)."""
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    if len(rets) < 30:
        return None
    sharpe = ANALYZER.sharpe_ratio(rets)                       # annualized
    mdd = ANALYZER.max_drawdown(eq / eq.iloc[0])               # re-based slice
    cagr = ANALYZER.cagr(eq / eq.iloc[0])
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    # per-bar (non-annualized) Sharpe for DSR
    sd = rets.std(ddof=1)
    sr_bar = float(rets.mean() / sd) if sd > 0 else 0.0
    dsr = deflated_sharpe_ratio(
        sr_bar, n_obs=len(rets),
        skew=float(skew(rets)), kurt=float(kurtosis(rets, fisher=False)),
        n_trials=1,  # frozen knobs, no sweep
    )
    return dict(label=label, n=len(rets), total_return=total, cagr=cagr,
                sharpe=sharpe, calmar=calmar, max_dd=mdd, dsr=dsr)


def run_books(frames, idx, rf_daily, slip):
    """Equity curves for rotation + benchmarks over a shared index."""
    bt = build_backtester(CFG)
    bt.config = dataclasses.replace(bt.config, credit_cash_rf=True)
    rot = bt.run_rotation(frames, ROT)
    idx = rot.index  # rotation defines the OOS span
    s6040 = static_mix_returns(frames, {"SPY": 0.6, "TLT": 0.4}, idx, slip, rf_daily)
    rp = risk_parity_returns({s: frames[s] for s in ROT.symbols}, idx, slip, rf_daily)
    return {"rotation": rot, "static_6040": s6040, "risk_parity": rp}


def gate(rot_m, b1_m, b2_m):
    """Rotation beats BOTH benchmarks on Sharpe AND Calmar (one slice)."""
    return (rot_m["sharpe"] > b1_m["sharpe"] and rot_m["calmar"] > b1_m["calmar"]
            and rot_m["sharpe"] > b2_m["sharpe"] and rot_m["calmar"] > b2_m["calmar"])


def main():
    global CFG
    CFG = yaml.safe_load(open("config/settings.yaml"))
    rf_daily = CFG.get("backtest", {}).get("risk_free_rate", 0.045) / PERIODS_PER_YEAR
    slip = CFG.get("backtest", {}).get("slippage_pct", 0.0005)

    print(f"Loading {TICKERS} {START}..{END} ...", flush=True)
    frames = {t: load_ohlcv(t, start=START, end=END, timeframe="1Day") for t in TICKERS}
    for t in TICKERS:
        print(f"  {t}: {len(frames[t])} bars "
              f"{frames[t].index.min().date()}..{frames[t].index.max().date()}", flush=True)

    print("Running rotation + benchmarks (production HMM, walk-forward)...", flush=True)
    books = run_books(frames, None, rf_daily, slip)
    idx = books["rotation"].index
    print(f"OOS span: {idx.min().date()}..{idx.max().date()} ({len(idx)} bars)", flush=True)

    periods = {
        "full": idx,
        "p1_2007_2015": idx[idx < SPLIT],
        "p2_2016_2024": idx[idx >= SPLIT],
    }
    report = {"periods": {}, "gate": {}}
    for pname, pidx in periods.items():
        m = {k: metrics(v.reindex(pidx).dropna(), f"{k}/{pname}") for k, v in books.items()}
        report["periods"][pname] = m
        passed = gate(m["rotation"], m["static_6040"], m["risk_parity"])
        dsr_ok = m["rotation"]["dsr"] > 0  # pre-registered floor
        report["gate"][pname] = {"beats_both_risk_adjusted": passed, "dsr_gt_0": dsr_ok}
        print(f"\n=== {pname} ===", flush=True)
        for k in ("rotation", "static_6040", "risk_parity"):
            x = m[k]
            print(f"  {k:14s} ret={x['total_return']:+.1%} CAGR={x['cagr']:+.2%} "
                  f"Sharpe={x['sharpe']:.2f} Calmar={x['calmar']:.2f} "
                  f"maxDD={x['max_dd']:.1%} DSR={x['dsr']:.3f}", flush=True)
        print(f"  GATE beats-both(Sharpe&Calmar)={passed}  DSR>0={dsr_ok}", flush=True)

    overall = all(report["gate"][p]["beats_both_risk_adjusted"]
                  and report["gate"][p]["dsr_gt_0"] for p in periods)
    report["verdict"] = "PASS" if overall else "FAIL"
    print(f"\n{'='*50}\nPRE-REGISTERED VERDICT: {report['verdict']}", flush=True)
    print("(PASS requires beating BOTH 60/40 and risk-parity on Sharpe AND Calmar "
          "in ALL THREE periods, DSR>0.)", flush=True)

    json.dump(report, open(OUT_JSON, "w"), indent=2, default=float)
    print(f"-> {OUT_JSON}", flush=True)
    return report


if __name__ == "__main__":
    main()
