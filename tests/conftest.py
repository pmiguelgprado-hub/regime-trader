"""Shared test fixtures: synthetic regime-switching OHLCV.

No network access — all market data for tests is generated locally with a
fixed seed so runs are deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(n: int = 1400, seed: int = 7) -> pd.DataFrame:
    """Generate regime-switching OHLCV (alternating low/high-vol blocks).

    Args:
        n: Number of bars.
        seed: RNG seed for reproducibility.

    Returns:
        OHLCV DataFrame indexed by business-day timestamps.
    """
    rng = np.random.default_rng(seed)
    block = 150
    vol = np.where((np.arange(n) // block) % 2 == 0, 0.008, 0.028)
    drift = np.where((np.arange(n) // block) % 2 == 0, 0.0005, -0.0004)
    ret = rng.normal(0.0, 1.0, n) * vol + drift
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + rng.uniform(0.0, 0.012, n))
    low = close * (1.0 - rng.uniform(0.0, 0.012, n))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    volume = rng.lognormal(mean=15.0, sigma=0.4, size=n)
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Default synthetic OHLCV frame (~1400 bars)."""
    return make_synthetic_ohlcv()
