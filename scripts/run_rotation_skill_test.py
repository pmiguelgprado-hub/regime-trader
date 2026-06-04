"""Skill-vs-exposure control for the cross-asset rotation (vía B).

The pre-registered gate PASSED (rotation beats 60/40 + risk-parity). Before
trusting that, this script answers the project's standing question: is the edge
the REGIME DETECTION, or just the static basket + vol-targeting riding the
2008-24 bond/gold bull?

Two controls, both reusing the SAME regime walk-forward (run once, cheap):

1. **Tier-permutation test** — relabel which volatility tier maps to which
   allocation. The real map is identity (calm->equities). If a random permutation
   (e.g. turbulent->equities) does as well, the vol/regime detection carries no
   allocation signal; the win is the basket, not the timing.
2. **Static basket** — equal-weight hold of the same 4 assets, no timing. If the
   rotation barely beats this, timing adds little over just holding the sleeve.
"""

import dataclasses
import itertools
import json
import sys

import yaml

sys.path.insert(0, ".")
from backtest.benchmarks import static_mix_returns
from backtest.oos_validation import build_backtester
from backtest.performance import PerformanceAnalyzer, PERIODS_PER_YEAR
from core.asset_rotation import RotationConfig
from core.regime_strategies import HIGH_VOL_MIN, LOW_VOL_MAX
from data.market_data import load_ohlcv

START, END = "2004-01-01", "2024-12-31"
ROT = RotationConfig()
TICKERS = sorted(set(ROT.symbols))
ANALYZER = PerformanceAnalyzer(risk_free_rate=0.045)
REP = {0: 0.0, 1: 0.5, 2: 1.0}   # tier index -> representative vol_rank


def tier_of(pos: float) -> int:
    if pos <= LOW_VOL_MAX:
        return 0
    if pos >= HIGH_VOL_MIN:
        return 2
    return 1


def perm_transform(perm: tuple[int, int, int]):
    """vol_rank transform that relabels tier i -> tier perm[i]."""
    return lambda pos: REP[perm[tier_of(pos)]]


def m(equity, label):
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    sharpe = ANALYZER.sharpe_ratio(rets)
    mdd = ANALYZER.max_drawdown(eq / eq.iloc[0])
    cagr = ANALYZER.cagr(eq / eq.iloc[0])
    calmar = cagr / abs(mdd) if mdd else 0.0
    return dict(label=label, sharpe=sharpe, calmar=calmar, max_dd=mdd,
                total_return=float(eq.iloc[-1] / eq.iloc[0] - 1.0))


def main():
    cfg = yaml.safe_load(open("config/settings.yaml"))
    rf_daily = cfg.get("backtest", {}).get("risk_free_rate", 0.045) / PERIODS_PER_YEAR
    slip = cfg.get("backtest", {}).get("slippage_pct", 0.0005)

    frames = {t: load_ohlcv(t, start=START, end=END, timeframe="1Day") for t in TICKERS}
    bt = build_backtester(cfg)
    bt.config = dataclasses.replace(bt.config, credit_cash_rf=True)

    print("Detecting regime once (proxy SPY walk-forward)...", flush=True)
    base = bt.run({"SPY": frames["SPY"]})

    out = {"permutations": {}, "static_basket": {}}
    identity = (0, 1, 2)
    results = []
    for perm in itertools.permutations(range(3)):
        eq = bt.run_rotation(frames, ROT, vr_transform=perm_transform(perm), base_result=base)
        tag = "IDENTITY(real)" if perm == identity else f"perm{perm}"
        res = m(eq, tag)
        res["perm"] = perm
        results.append(res)
        out["permutations"][str(perm)] = res
        print(f"  {tag:16s} Sharpe={res['sharpe']:.2f} Calmar={res['calmar']:.2f} "
              f"ret={res['total_return']:+.1%} maxDD={res['max_dd']:.1%}", flush=True)

    # static equal-weight basket (no timing)
    ew = {s: 1.0 / len(ROT.symbols) for s in ROT.symbols}
    idx = base.regime_history.index
    sb = static_mix_returns(frames, ew, idx, slip, rf_daily)
    out["static_basket"] = m(sb, "static_equal_weight")
    print(f"\n  static_equal_weight  Sharpe={out['static_basket']['sharpe']:.2f} "
          f"Calmar={out['static_basket']['calmar']:.2f} "
          f"ret={out['static_basket']['total_return']:+.1%} "
          f"maxDD={out['static_basket']['max_dd']:.1%}", flush=True)

    real = next(r for r in results if r["perm"] == identity)
    others = [r for r in results if r["perm"] != identity]
    best_other = max(others, key=lambda r: r["sharpe"])
    rank = 1 + sum(1 for r in others if r["sharpe"] > real["sharpe"])
    out["summary"] = dict(
        real_sharpe=real["sharpe"], best_other_sharpe=best_other["sharpe"],
        best_other_perm=best_other["perm"], real_rank_of_6=rank,
        beats_static_basket=real["sharpe"] > out["static_basket"]["sharpe"],
    )
    print(f"\n{'='*56}", flush=True)
    print(f"Real (identity) Sharpe {real['sharpe']:.2f} ranks #{rank} of 6 permutations.", flush=True)
    print(f"Best non-identity: perm{best_other['perm']} Sharpe {best_other['sharpe']:.2f}.", flush=True)
    print(f"Beats static equal-weight basket: {out['summary']['beats_static_basket']}.", flush=True)
    if rank == 1 and real["sharpe"] - best_other["sharpe"] > 0.1:
        print("=> Regime alignment carries signal (real clearly best).", flush=True)
    else:
        print("=> WEAK/NO regime skill: identity not decisively best => edge is "
              "basket/vol-target, not timing.", flush=True)
    json.dump(out, open("tmp/rotation_skill_test.json", "w"), indent=2, default=float)
    print("DONE_SKILL -> tmp/rotation_skill_test.json", flush=True)


if __name__ == "__main__":
    main()
