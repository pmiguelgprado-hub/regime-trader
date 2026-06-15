"""Crypto time-series momentum sleeve (T2.3, Alpha Engine v2 phase 3).

A small, descorrelating sleeve: BTC/ETH on Alpaca crypto (free, 24/7, no shorts).
Time-series momentum (Moskowitz-Ooi-Pedersen) — hold a coin only while its OWN
trailing return is positive — sized by vol-targeting (reusing
``core.asset_rotation.vol_target_scale`` at a 365-day year), with the total gross
capped small (<=10% NAV). The prior is weaker than the equity sleeves; the value
is correlation, not standalone Sharpe. Own prereg + book + (eventually) account;
never touches the frozen equity books. Pure + unit-tested.
"""

from __future__ import annotations

import pandas as pd

from core.asset_rotation import vol_target_scale

CRYPTO_YEAR = 365          # crypto trades every day


def ts_momentum_signal(prices: pd.Series, lookback: int = 90) -> int:
    """1 if the trailing-``lookback`` return is positive, else 0 (long-only).

    Long-only because Alpaca crypto has no borrow; a negative trend means flat,
    not short. Returns 0 when there is less than ``lookback+1`` of history.
    """
    p = prices.astype(float).dropna()
    if len(p) < lookback + 1:
        return 0
    trailing = p.iloc[-1] / p.iloc[-1 - lookback] - 1.0
    return 1 if trailing > 0.0 else 0


def crypto_weights(frames: dict[str, pd.DataFrame], lookback: int = 90,
                   target_vol: float = 0.20, max_gross: float = 0.10,
                   vol_window: int = 60) -> dict[str, float]:
    """Vol-targeted long-only TS-momentum weights for the crypto sleeve.

    Each coin with positive momentum gets an equal share of the gross budget,
    scaled down by its vol-target factor; the total is clamped to ``max_gross``
    (the sleeve's NAV cap). Coins in a downtrend are excluded (weight 0).

    Args:
        frames: ``{symbol: OHLCV}`` per coin (daily bars).
        lookback: Momentum lookback (bars).
        target_vol: Annualized vol target for sizing.
        max_gross: Hard cap on total sleeve gross (fraction of NAV).
        vol_window: Trailing bars for the realized-vol estimate.

    Returns:
        ``{symbol: weight}`` (only positive-momentum coins; sum <= max_gross).
    """
    longs = {s: f for s, f in frames.items()
             if ts_momentum_signal(f["close"], lookback) == 1}
    if not longs:
        return {}
    per = max_gross / len(longs)
    weights: dict[str, float] = {}
    for sym, f in longs.items():
        rets = f["close"].astype(float).pct_change().dropna().iloc[-vol_window:]
        k = vol_target_scale(list(rets), target_vol=target_vol, cap=1.0, floor=0.0,
                             periods_per_year=CRYPTO_YEAR)
        w = per * k
        if w > 0:
            weights[sym] = w
    # numerical safety: never exceed the gross cap
    total = sum(weights.values())
    if total > max_gross and total > 0:
        weights = {s: w * max_gross / total for s, w in weights.items()}
    return weights
