"""Run the OOS validation matrix: breadth (ETFs) + SPY halt-floor sweep.

Phase 1 (breadth): each ETF, full history, default floor → strategy vs bh/sma
over full span + each crisis window.
Phase 2 (floor sweep): SPY only, halt_floor_mult in {0, 0.25, 0.5, 1.0} →
isolate exposure vs edge. Writes JSON + prints markdown.
"""
import json
import sys
import traceback
import yaml

sys.path.insert(0, ".")
from backtest.oos_validation import run_asset
from data.market_data import load_ohlcv

CFG = yaml.safe_load(open("config/settings.yaml"))
START, END = "2004-01-01", "2024-12-31"
BREADTH = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT"]
FLOORS = [0.0, 0.25, 0.5, 1.0]
OUT = "tmp/oos_matrix_results.json"

results = {"breadth": {}, "floor_sweep": {}}


def sm_dict(sm):
    return dict(label=sm.label, n_bars=sm.n_bars, strat_return=sm.strat_return,
                strat_sharpe=sm.strat_sharpe, strat_mdd=sm.strat_mdd,
                buy_hold_return=sm.buy_hold_return, sma200_return=sm.sma200_return,
                pct_halted=sm.pct_halted, beats_bh=sm.beats_bh())


def flush():
    json.dump(results, open(OUT, "w"), indent=2)


# cache loaded data so the floor sweep reuses SPY
_data = {}
def get(sym):
    if sym not in _data:
        _data[sym] = load_ohlcv(sym, start=START, end=END, timeframe="1Day")
    return _data[sym]


print("=== PHASE 1: BREADTH (default floor 0.25) ===", flush=True)
for sym in BREADTH:
    try:
        ohlcv = get(sym)
        _, slices = run_asset(CFG, sym, ohlcv)  # default floor
        results["breadth"][sym] = [sm_dict(s) for s in slices]
        full = next((s for s in slices if s.label == "full"), None)
        print(f"{sym}: bars={len(ohlcv)} full_ret={full.strat_return:.2%} "
              f"sharpe={full.strat_sharpe:.2f} bh={full.buy_hold_return:.2%}", flush=True)
        flush()
    except Exception as e:
        print(f"{sym}: FAILED {e}", flush=True)
        traceback.print_exc()
        results["breadth"][sym] = {"error": str(e)}
        flush()

print("\n=== PHASE 2: SPY FLOOR SWEEP ===", flush=True)
spy = get("SPY")
for f in FLOORS:
    try:
        _, slices = run_asset(CFG, "SPY", spy, halt_floor_mult=f)
        results["floor_sweep"][str(f)] = [sm_dict(s) for s in slices]
        full = next((s for s in slices if s.label == "full"), None)
        print(f"floor={f}: full_ret={full.strat_return:.2%} sharpe={full.strat_sharpe:.2f} "
              f"halted={full.pct_halted:.2%}", flush=True)
        flush()
    except Exception as e:
        print(f"floor={f}: FAILED {e}", flush=True)
        traceback.print_exc()
        flush()

flush()
print(f"\nDONE -> {OUT}", flush=True)
