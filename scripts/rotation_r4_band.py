"""One-process sample of the rotation Sharpe (for the R-4 robustness band).

R-4 (documented): the HMM walk-forward is non-deterministic ACROSS processes on
long horizons. Same-process it is deterministic. So to size the band we re-run
THIS script in N separate processes (a shell loop) and collect the rotation
Sharpe each time, comparing to two R-4-invariant baselines (static equal-weight
and risk-parity use no HMM -> identical every process).

Prints one CSV row:  plain_sharpe, identity_perm_sharpe, static_eq, risk_parity
- plain vs identity_perm must match (confirms perm_transform identity is a no-op
  -> the cross-process swing is R-4, not a bug).
- static_eq / risk_parity must be constant across processes (the yardsticks).
"""

import dataclasses
import sys

import yaml

sys.path.insert(0, ".")
from backtest.benchmarks import risk_parity_returns, static_mix_returns
from backtest.oos_validation import build_backtester
from backtest.performance import PERIODS_PER_YEAR, PerformanceAnalyzer
from core.asset_rotation import RotationConfig
from core.regime_strategies import HIGH_VOL_MIN, LOW_VOL_MAX
from data.market_data import load_ohlcv

START, END = "2004-01-01", "2024-12-31"
ROT = RotationConfig()
TICKERS = sorted(set(ROT.symbols))
AN = PerformanceAnalyzer(risk_free_rate=0.045)
REP = {0: 0.0, 1: 0.5, 2: 1.0}


def identity_perm(pos):
    t = 0 if pos <= LOW_VOL_MAX else (2 if pos >= HIGH_VOL_MIN else 1)
    return REP[t]


def shp(eq):
    return AN.sharpe_ratio(eq.dropna().pct_change().dropna())


def main():
    cfg = yaml.safe_load(open("config/settings.yaml"))
    rf_daily = cfg.get("backtest", {}).get("risk_free_rate", 0.045) / PERIODS_PER_YEAR
    slip = cfg.get("backtest", {}).get("slippage_pct", 0.0005)
    frames = {t: load_ohlcv(t, start=START, end=END, timeframe="1Day") for t in TICKERS}

    bt = build_backtester(cfg)
    bt.config = dataclasses.replace(bt.config, credit_cash_rf=True)
    base = bt.run({"SPY": frames["SPY"]})
    idx = base.regime_history.index

    plain = shp(bt.run_rotation(frames, ROT, base_result=base))
    idp = shp(bt.run_rotation(frames, ROT, base_result=base, vr_transform=identity_perm))
    ew = {s: 1.0 / len(ROT.symbols) for s in ROT.symbols}
    seq = shp(static_mix_returns(frames, ew, idx, slip, rf_daily))
    rp = shp(risk_parity_returns(frames, idx, slip, rf_daily))

    print(f"ROW,{plain:.4f},{idp:.4f},{seq:.4f},{rp:.4f}", flush=True)


if __name__ == "__main__":
    main()
