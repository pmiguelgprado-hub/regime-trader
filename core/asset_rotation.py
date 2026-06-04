"""Cross-asset regime rotation (vía B).

The HMM detects the market-wide *volatility regime* on a proxy (SPY); this module
turns that regime's volatility tier into a target allocation across **descorrelated
sleeves** — equities (risk-on), bonds + gold (defensive), and cash — instead of
sizing a single index up and down (which only modulates beta; see
docs/analysis/2026-06-04-markov-edge-redesign.md §1).

The map is **theory-driven and frozen** in
docs/analysis/2026-06-04-rotation-prereg.md: zero fitted parameters, so there is
nothing to overfit beyond the (pre-registered, un-swept) choice of knobs. A
volatility target replaces the binary halt — it de-risks continuously without the
latch pathology that froze the original backtester.

Tiers reuse the orchestrator terciles (``LOW_VOL_MAX=0.33``, ``HIGH_VOL_MIN=0.67``)
so the rotation reads the same vol-rank the single-asset strategy already computes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.regime_strategies import HIGH_VOL_MIN, LOW_VOL_MAX


@dataclass
class RotationConfig:
    """Frozen cross-asset rotation knobs (see the pre-registration doc).

    Attributes:
        equities: Equity sleeve tickers (risk-on).
        defensive: Defensive sleeve tickers (risk-off refuge: bonds + gold).
        risk_on_equity: Gross equity weight in the low-vol (risk-on) tier.
        mid_equity: Gross equity weight in the mid-vol tier.
        mid_defensive: Gross defensive weight in the mid-vol tier.
        risk_off_defensive: Gross defensive weight in the high-vol (risk-off) tier;
            the remainder up to 1.0 is the implicit cash sleeve.
        target_vol: Annualized portfolio volatility target.
        vol_window: Trailing bars used to estimate realized volatility.
        gross_cap: Maximum gross risky exposure after vol-targeting (1.0 = no leverage).
        gross_floor: Minimum gross risky exposure after vol-targeting.
        periods_per_year: Annualization factor (252 daily bars).
    """

    equities: list[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    defensive: list[str] = field(default_factory=lambda: ["TLT", "GLD"])
    risk_on_equity: float = 1.0
    mid_equity: float = 0.60
    mid_defensive: float = 0.40
    risk_off_defensive: float = 0.60
    target_vol: float = 0.10
    vol_window: int = 20
    gross_cap: float = 1.0
    gross_floor: float = 0.0
    periods_per_year: int = 252

    @property
    def symbols(self) -> list[str]:
        """All risky tickers (equities + defensive), de-duplicated, order-stable."""
        seen: dict[str, None] = {}
        for s in (*self.equities, *self.defensive):
            seen.setdefault(s, None)
        return list(seen)


def _split(weight: float, sleeve: list[str]) -> dict[str, float]:
    """Equal-split a sleeve gross ``weight`` across its tickers."""
    if weight <= 0.0 or not sleeve:
        return {}
    share = weight / len(sleeve)
    return {s: share for s in sleeve}


def rotation_weights(vol_rank_position: float, cfg: RotationConfig) -> dict[str, float]:
    """Per-symbol risky weights for a regime's volatility tier.

    The returned weights are the **base** allocation (before vol-targeting). Their
    sum is the gross risky exposure; ``1 - sum`` is the implicit cash sleeve. Cash
    is not returned as a key (it is the remainder, credited at the risk-free rate by
    the backtester).

    Args:
        vol_rank_position: Regime volatility rank in ``[0, 1]`` (0 = lowest vol).
        cfg: Frozen rotation configuration.

    Returns:
        ``{ticker: base_weight}`` for the held risky names (cash implicit).
    """
    if vol_rank_position <= LOW_VOL_MAX:           # risk-on
        return _split(cfg.risk_on_equity, cfg.equities)
    if vol_rank_position >= HIGH_VOL_MIN:          # risk-off
        return _split(cfg.risk_off_defensive, cfg.defensive)
    # mid: balanced equities + defensive
    out = _split(cfg.mid_equity, cfg.equities)
    out.update(_split(cfg.mid_defensive, cfg.defensive))
    return out


def regime_gross_scale(
    vol_rank_position: float,
    risk_on: float = 1.0,
    risk_off: float = 0.5,
) -> float:
    """HMM gross-exposure overlay for a book (vía C risk overlay).

    The expert role of an HMM in a trading system is a *risk filter*, not a signal
    generator (QuantStart, MDPI; matches this project's own falsification of HMM
    timing as alpha). Here it scales the **total gross exposure** of the cross-sectional
    book by the market volatility regime: full exposure in the low-vol (risk-on) tier,
    de-risked in the high-vol (risk-off) tier, linearly interpolated across the mid band.
    The alpha comes from the ranker; this only governs how much of it is on.

    Kept separable (a pure scalar map) so the pre-registered gate can measure whether the
    overlay adds value *incrementally* over the naked ranker — if it doesn't, drop it.

    Args:
        vol_rank_position: Regime volatility rank in ``[0, 1]`` (0 = lowest vol).
        risk_on: Gross multiplier in the low-vol tier (``<= LOW_VOL_MAX``).
        risk_off: Gross multiplier in the high-vol tier (``>= HIGH_VOL_MIN``).

    Returns:
        Gross-exposure scale factor.
    """
    if vol_rank_position <= LOW_VOL_MAX:
        return risk_on
    if vol_rank_position >= HIGH_VOL_MIN:
        return risk_off
    frac = (vol_rank_position - LOW_VOL_MAX) / (HIGH_VOL_MIN - LOW_VOL_MAX)
    return risk_on + frac * (risk_off - risk_on)


def vol_target_scale(
    trailing_returns,
    target_vol: float,
    cap: float,
    floor: float = 0.0,
    periods_per_year: int = 252,
    min_obs: int = 5,
) -> float:
    """Volatility-target scaling factor for the risky book (causal).

    Scales gross exposure so the book's annualized realized volatility approaches
    ``target_vol``: ``k = target_vol / realized_vol``, clamped to ``[floor, cap]``.
    This is the principled, non-latching replacement for the binary drawdown halt —
    it de-risks smoothly in turbulent windows and re-risks as volatility subsides.

    Degenerate inputs (too few points, zero volatility) return ``cap`` rather than
    dividing by zero — a near-zero-vol window is a calm market, so full exposure is
    the correct (and conservative-against-spurious-sizing) default.

    Args:
        trailing_returns: Recent per-bar portfolio returns (most recent last).
        target_vol: Annualized volatility target.
        cap: Maximum scale (1.0 = no leverage).
        floor: Minimum scale.
        periods_per_year: Annualization factor.
        min_obs: Minimum points required to estimate volatility.

    Returns:
        Scale factor in ``[floor, cap]``.
    """
    rets = list(trailing_returns)
    if len(rets) < min_obs:
        return cap
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    realized = math.sqrt(var) * math.sqrt(periods_per_year)
    if realized <= 0.0:
        return cap
    k = target_vol / realized
    return max(floor, min(cap, k))
