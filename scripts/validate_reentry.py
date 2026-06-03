"""Validate the FROZEN re-entry combo on the 5 held-out ETFs (never tuned on).

Success per asset = Sharpe > buy&hold AND maxDD < buy&hold (strat_mdd > bh_mdd,
drawdowns negative). Single process. No re-tuning after seeing this output.
"""
import json
import sys
import yaml

sys.path.insert(0, ".")
from backtest.oos_validation import build_backtester, slice_metrics
from data.market_data import load_ohlcv

BEST_K = 3        # from scripts/tune_reentry.py BEST (SPY dev)
BEST_FLOOR = 0.25  # from scripts/tune_reentry.py BEST (SPY dev)

CFG = yaml.safe_load(open("config/settings.yaml"))
CFG = {**CFG, "risk": {**CFG.get("risk", {}),
                       "peak_reentry_calm_bars": BEST_K, "halt_floor_mult": BEST_FLOOR}}
HOLDOUT = ["QQQ", "IWM", "EFA", "EEM", "TLT"]
rf = CFG.get("backtest", {}).get("risk_free_rate", 0.045)

rows = []
for sym in HOLDOUT:
    ohlcv = load_ohlcv(sym, start="2004-01-01", end="2024-12-31", timeframe="1Day")
    bt = build_backtester(CFG)
    res = bt.run({sym: ohlcv})
    sm = slice_metrics(res, ohlcv["close"].astype(float), "full", rf=rf)
    passed = sm.beats_bh_risk_adjusted()
    rows.append(dict(symbol=sym, strat_sharpe=sm.strat_sharpe, bh_sharpe=sm.bh_sharpe,
                     strat_mdd=sm.strat_mdd, bh_mdd=sm.bh_mdd,
                     strat_ret=sm.strat_return, bh_ret=sm.buy_hold_return,
                     pct_halted=sm.pct_halted, passed=passed))
    print(f"{sym}: Sharpe {sm.strat_sharpe:.2f} vs bh {sm.bh_sharpe:.2f} | "
          f"maxDD {sm.strat_mdd:.1%} vs bh {sm.bh_mdd:.1%} | "
          f"ret {sm.strat_return:.0%} vs bh {sm.buy_hold_return:.0%} | PASS={passed}", flush=True)

n_pass = sum(r["passed"] for r in rows)
json.dump(dict(k=BEST_K, floor=BEST_FLOOR, rows=rows, n_pass=n_pass),
          open("tmp/reentry_validation.json", "w"), indent=2)
print(f"\nHOLDOUT: {n_pass}/5 assets beat bh on Sharpe AND maxDD", flush=True)
print("DONE_VALIDATE", flush=True)
