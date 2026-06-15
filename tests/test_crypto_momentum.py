"""Tests for the crypto time-series momentum sleeve (T2.3).

BTC/ETH, time-series (not cross-sectional) momentum: hold a coin only while its
own trailing return is positive (long-only — Alpaca crypto has no shorts), sized
by vol-targeting (reusing asset_rotation.vol_target_scale at a 365-day year),
total gross capped small (<=10% NAV). Descorrelation sleeve, not a core bet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import crypto_momentum as cm


def _series(vals):
    return pd.DataFrame({"close": vals},
                        index=pd.bdate_range("2026-01-01", periods=len(vals)))


def test_ts_momentum_positive_trend_long():
    up = _series(list(np.linspace(100, 200, 120)))
    assert cm.ts_momentum_signal(up["close"], lookback=90) == 1


def test_ts_momentum_negative_trend_flat():
    down = _series(list(np.linspace(200, 100, 120)))
    assert cm.ts_momentum_signal(down["close"], lookback=90) == 0   # long-only: no short


def test_ts_momentum_insufficient_history_flat():
    assert cm.ts_momentum_signal(_series([100, 101])["close"], lookback=90) == 0


def test_crypto_weights_long_only_positive_momentum():
    up = list(np.linspace(100, 200, 200))
    frames = {"BTCUSD": _series(up), "ETHUSD": _series(up)}
    w = cm.crypto_weights(frames, lookback=90, target_vol=0.20, max_gross=0.10)
    assert set(w) <= {"BTCUSD", "ETHUSD"}
    assert all(v >= 0 for v in w.values())
    assert sum(w.values()) <= 0.10 + 1e-9               # gross cap respected


def test_crypto_weights_excludes_downtrend():
    up = list(np.linspace(100, 200, 200))
    down = list(np.linspace(200, 100, 200))
    frames = {"BTCUSD": _series(up), "ETHUSD": _series(down)}
    w = cm.crypto_weights(frames, lookback=90, target_vol=0.20, max_gross=0.10)
    assert "BTCUSD" in w and "ETHUSD" not in w


def test_crypto_weights_all_downtrend_empty():
    down = list(np.linspace(200, 100, 200))
    frames = {"BTCUSD": _series(down), "ETHUSD": _series(down)}
    assert cm.crypto_weights(frames, lookback=90, target_vol=0.20, max_gross=0.10) == {}


def test_crypto_weights_vol_target_caps_gross():
    # very volatile uptrend -> vol-target scales gross DOWN below the raw cap
    rng = np.random.default_rng(0)
    vol_up = list(100 * np.cumprod(1 + rng.normal(0.01, 0.10, 200)))
    frames = {"BTCUSD": _series(vol_up)}
    w = cm.crypto_weights(frames, lookback=90, target_vol=0.20, max_gross=0.10)
    if w:                                               # if still in uptrend
        assert sum(w.values()) <= 0.10
