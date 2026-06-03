"""Tune halt re-entry on SPY ONLY (dev set). Grid over K x floor; pick best Sharpe.

Single process (R-4: cross-process numbers vary ~8%; intra-process is exact).
Writes the winner to tmp/reentry_tune.json. Does NOT touch the holdout ETFs.
"""
import json
import sys
import yaml

sys.path.insert(0, ".")
from backtest.oos_validation import build_backtester
from backtest.performance import PerformanceAnalyzer
from data.market_data import load_ohlcv

CFG = yaml.safe_load(open("config/settings.yaml"))
spy = load_ohlcv("SPY", start="2004-01-01", end="2024-12-31", timeframe="1Day")
rf = CFG.get("backtest", {}).get("risk_free_rate", 0.045)
an = PerformanceAnalyzer(risk_free_rate=rf)

results = []
last_res = None
for k in [0, 3, 5, 10]:
    for floor in [0.0, 0.25]:
        cfg = {**CFG, "risk": {**CFG.get("risk", {}),
                               "peak_reentry_calm_bars": k, "halt_floor_mult": floor}}
        bt = build_backtester(cfg)
        res = bt.run({"SPY": spy})
        last_res = res
        sharpe = an.sharpe_ratio(res.returns)
        mdd = an.max_drawdown(res.equity_curve)
        ret = float(res.equity_curve.iloc[-1] / res.initial_capital - 1.0)
        halted = float((res.regime_history["risk_state"] == "halted").mean())
        results.append(dict(k=k, floor=floor, sharpe=sharpe, mdd=mdd, ret=ret, halted=halted))
        print(f"K={k:>2} floor={floor}: sharpe={sharpe:.2f} mdd={mdd:.1%} ret={ret:.1%} "
              f"halted={halted:.1%}", flush=True)

bh_sharpe = an.sharpe_ratio(
    spy["close"].pct_change().reindex(last_res.returns.index).fillna(0.0)
)
best = max(results, key=lambda r: r["sharpe"])
out = dict(spy_buy_hold_sharpe=bh_sharpe, grid=results, best=best)
json.dump(out, open("tmp/reentry_tune.json", "w"), indent=2)
print(f"\nSPY buy&hold Sharpe (ref) = {bh_sharpe:.2f}")
print(f"BEST: K={best['k']} floor={best['floor']} sharpe={best['sharpe']:.2f}", flush=True)
print("DONE_TUNE", flush=True)
