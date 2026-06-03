"""Skill test: does the regime->strategy map carry information?

At floor=1.0 the halt is a no-op, so this isolates the REGIME ALLOCATION overlay.
Compare real vol-rank mapping vs shuffled (random regime->strategy) over full
history. shuffled ~= real => no regime skill (overlay is dead weight).
"""
import json
import sys
import yaml

sys.path.insert(0, ".")
from backtest.oos_validation import build_backtester
from backtest.performance import PerformanceAnalyzer
from data.market_data import load_ohlcv

CFG = yaml.safe_load(open("config/settings.yaml"))
ohlcv = load_ohlcv("SPY", start="2004-01-01", end="2024-12-31", timeframe="1Day")
rf = CFG.get("backtest", {}).get("risk_free_rate", 0.045)
an = PerformanceAnalyzer(risk_free_rate=rf)
out = {}

def run(shuffle_seed):
    bt = build_backtester(CFG, halt_floor_mult=1.0)  # disable halt -> isolate regime overlay
    bt.shuffle_regimes = shuffle_seed
    res = bt.run({"SPY": ohlcv})
    return float(res.equity_curve.iloc[-1] / res.initial_capital - 1.0), an.sharpe_ratio(res.returns)

r_ret, r_shp = run(None)
out["real"] = dict(ret=r_ret, sharpe=r_shp)
print(f"REAL    : ret={r_ret:.1%} sharpe={r_shp:.2f}", flush=True)
for seed in [1, 2, 3]:
    sret, sshp = run(seed)
    out[f"shuffle_{seed}"] = dict(ret=sret, sharpe=sshp)
    print(f"SHUFFLE{seed}: ret={sret:.1%} sharpe={sshp:.2f}", flush=True)

json.dump(out, open("tmp/shuffle_results.json", "w"), indent=2)
print("DONE_SHUFFLE", flush=True)
