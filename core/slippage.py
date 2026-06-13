"""Vol-aware slippage estimator (T5.1).

The backtester has a coded-but-dormant hook: per-trade slippage =
``slippage_vol_coeff * atr_pct`` (``config/settings.yaml`` ships it at 0.0). This
module estimates that coefficient from realized paper fills — regress realized
slippage (as a price fraction) on the bar's ATR% through the origin — so research
backtests charge more cost in turbulent regimes than calm ones.

**Scope: research backtests only.** The forward gates are judged on real fills,
and a backtest that is already gate evidence is never re-run (roadmap §T5.1). So
tuning this coefficient cannot alter any gate. Pure + unit-tested.
"""

from __future__ import annotations

from typing import Optional


def realized_slippage_bps(decision_price: float, fill_price: float, side: str) -> float:
    """Adverse slippage in basis points (positive = worse than the decision price).

    For a buy, paying above the decision price is adverse; for a sell, filling below
    it is adverse. Favorable fills come back negative.
    """
    if decision_price <= 0:
        return 0.0
    raw = (fill_price - decision_price) / decision_price
    signed = raw if side.lower() == "buy" else -raw
    return signed * 10_000.0


def estimate_vol_coeff(samples: list[dict], min_samples: int = 30) -> Optional[float]:
    """Least-squares slope (through the origin) of slippage-fraction on ATR%.

    Args:
        samples: ``[{slippage_bps, atr_pct}, ...]`` realized fills. ``atr_pct`` is a
            fraction (0.02 = 2% ATR); rows with non-positive ATR are dropped.
        min_samples: Minimum usable rows before a coefficient is returned.

    Returns:
        The coefficient for the backtester's ``slippage_vol_coeff`` hook, or ``None``
        when there are too few usable samples (don't tune on noise).
    """
    xy = sxx = 0.0
    n = 0
    for s in samples:
        atr = float(s.get("atr_pct", 0.0))
        if atr <= 0:
            continue
        slip_frac = float(s.get("slippage_bps", 0.0)) / 10_000.0
        xy += atr * slip_frac
        sxx += atr * atr
        n += 1
    if n < min_samples or sxx <= 0:
        return None
    return xy / sxx
