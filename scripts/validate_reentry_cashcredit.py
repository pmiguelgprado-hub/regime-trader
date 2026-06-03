"""Re-run the FROZEN re-entry combo with cash credited at rf (fair Sharpe).

Same K=3/floor=0.25 (NOT re-tuned). credit_cash_rf=True so idle cash earns rf and
the Sharpe comparison vs 100%-invested buy&hold is apples-to-apples. SPY shown for
reference (dev set); QQQ/IWM/EFA/EEM/TLT are the holdout. Single process.
"""
import json
import sys
import yaml

sys.path.insert(0, ".")
from backtest.oos_validation import build_backtester, slice_metrics
from data.market_data import load_ohlcv

BEST_K, BEST_FLOOR = 3, 0.25
CFG = yaml.safe_load(open("config/settings.yaml"))
CFG = {
    **CFG,
    "risk": {**CFG.get("risk", {}), "peak_reentry_calm_bars": BEST_K, "halt_floor_mult": BEST_FLOOR},
    "backtest": {**CFG.get("backtest", {}), "credit_cash_rf": True},
}
rf = CFG.get("backtest", {}).get("risk_free_rate", 0.045)
SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT"]  # SPY = dev ref; rest = holdout

rows = []
for sym in SYMBOLS:
    ohlcv = load_ohlcv(sym, start="2004-01-01", end="2024-12-31", timeframe="1Day")
    bt = build_backtester(CFG)
    res = bt.run({sym: ohlcv})
    sm = slice_metrics(res, ohlcv["close"].astype(float), "full", rf=rf)
    passed = sm.beats_bh_risk_adjusted()
    held = sm  # alias
    rows.append(dict(symbol=sym, strat_sharpe=sm.strat_sharpe, bh_sharpe=sm.bh_sharpe,
                     strat_mdd=sm.strat_mdd, bh_mdd=sm.bh_mdd,
                     strat_ret=sm.strat_return, bh_ret=sm.buy_hold_return,
                     pct_halted=sm.pct_halted, passed=passed))
    tag = "DEV " if sym == "SPY" else "HOLD"
    print(f"[{tag}] {sym}: Sharpe {sm.strat_sharpe:.2f} vs bh {sm.bh_sharpe:.2f} | "
          f"maxDD {sm.strat_mdd:.1%} vs bh {sm.bh_mdd:.1%} | "
          f"ret {sm.strat_return:.0%} vs bh {sm.buy_hold_return:.0%} | PASS={passed}", flush=True)

holdout = [r for r in rows if r["symbol"] != "SPY"]
n_pass = sum(r["passed"] for r in holdout)
json.dump(dict(k=BEST_K, floor=BEST_FLOOR, credit_cash_rf=True, rows=rows, n_pass_holdout=n_pass),
          open("tmp/reentry_validation_cashcredit.json", "w"), indent=2)
print(f"\nHOLDOUT (cash-credited): {n_pass}/5 beat bh on Sharpe AND maxDD", flush=True)
print("DONE_CC", flush=True)
