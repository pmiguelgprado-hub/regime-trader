"""Tail-hedge options overlay — selection/trigger/budget math (pure).

A *risk action* keyed to the one thing the HMM is validated to do: classify the
volatility regime. When the transition hazard (:func:`core.meta_overlay.high_tier_hazard`)
says the high-vol tier is imminent and the book is still exposed, buy a
defined-risk put debit spread on the proxy under a hard premium budget. This
spends a known premium to cap left-tail damage; it makes **no** direction or
return claim (short-vol structures stay out of v1 — see the triage memo).

Pre-registered v1 knobs (docs/analysis/2026-06-11-meta-overlay-triage.md §3,
chosen once, not swept): trigger hazard >= 0.35 two consecutive closes with book
gross > 0.6; long put ~4% OTM / short put ~10% OTM, 30-60 DTE nearest 45;
premium <= 25 bp/quarter and <= 100 bp/year of equity; max 1 structure; close
after hazard < 0.20 five consecutive closes (or expiry). Ships ``enabled=False``.

Broker I/O (chain fetch, quotes, multi-leg submission) lives in
:mod:`broker.options_executor`; this module stays pure and unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


@dataclass
class OptionsHedgeConfig:
    """Frozen v1 knobs for the tail-hedge overlay (mirrors ``options_hedge``)."""

    enabled: bool = False              # ships OFF; enabling = documented act
    trigger_hazard: float = 0.35       # open when hazard >= this ...
    trigger_days: int = 2              # ... this many consecutive closes
    release_hazard: float = 0.20       # close when hazard < this ...
    release_days: int = 5              # ... this many consecutive closes
    min_book_gross: float = 0.6        # only hedge a book that is still exposed
    long_otm: float = 0.04             # long put strike ~ spot * (1 - 4%)
    short_otm: float = 0.10            # short put strike ~ spot * (1 - 10%)
    dte_min: int = 30
    dte_max: int = 60
    dte_target: int = 45               # pick the expiry nearest this inside the window
    budget_quarter_bp: float = 25.0    # net premium cap per quarter (bp of equity)
    budget_year_bp: float = 100.0      # net premium cap per rolling year (bp of equity)
    max_structures: int = 1            # one live hedge at a time
    proxy: str = "SPY"                 # underlying hedged (the book's regime proxy)


@dataclass
class PutSpreadPlan:
    """A selected put debit spread (long the nearer strike, short the farther)."""

    long_symbol: str
    short_symbol: str
    long_strike: float
    short_strike: float
    expiry: date


@dataclass
class HedgeBudget:
    """Premium already spent (USD), tracked by the executor's state file."""

    spent_quarter: float = 0.0
    spent_year: float = 0.0

    def remaining(self, equity: float, cfg: OptionsHedgeConfig) -> float:
        """USD premium headroom: the binding one of the quarter/year caps.

        Args:
            equity: Current account equity (USD).
            cfg: Frozen hedge knobs.

        Returns:
            Max spendable premium now (>= 0).
        """
        cap_q = equity * cfg.budget_quarter_bp / 10_000.0 - self.spent_quarter
        cap_y = equity * cfg.budget_year_bp / 10_000.0 - self.spent_year
        return max(0.0, min(cap_q, cap_y))


def _tail_run(values: list[float], pred) -> int:
    """Length of the trailing run of values satisfying ``pred``."""
    n = 0
    for v in reversed(values):
        if pred(v):
            n += 1
        else:
            break
    return n


def should_open_hedge(
    hazard_history: list[float],
    book_gross: float,
    open_structures: int,
    cfg: OptionsHedgeConfig,
) -> bool:
    """Whether to open a new tail hedge at today's close.

    Requires the hazard to have held at/above ``trigger_hazard`` for the last
    ``trigger_days`` closes (a single spike is noise), a book that is still
    exposed (``book_gross > min_book_gross`` — if the gross overlay already
    de-risked, premium adds nothing), and no live structure.

    Args:
        hazard_history: Daily transition hazards, most recent last.
        book_gross: Current total gross exposure of the book.
        open_structures: Number of live hedge structures.
        cfg: Frozen hedge knobs.

    Returns:
        True when all open conditions hold.
    """
    if open_structures >= cfg.max_structures:
        return False
    if book_gross <= cfg.min_book_gross:
        return False
    if len(hazard_history) < cfg.trigger_days:
        return False
    return _tail_run(hazard_history, lambda h: h >= cfg.trigger_hazard) >= cfg.trigger_days


def should_close_hedge(hazard_history: list[float], cfg: OptionsHedgeConfig) -> bool:
    """Whether to close the live hedge: hazard calm for ``release_days`` closes.

    Args:
        hazard_history: Daily transition hazards, most recent last.
        cfg: Frozen hedge knobs.

    Returns:
        True when the trailing calm run is long enough.
    """
    if len(hazard_history) < cfg.release_days:
        return False
    return _tail_run(hazard_history, lambda h: h < cfg.release_hazard) >= cfg.release_days


def select_put_spread(
    chain: list[dict],
    spot: float,
    cfg: OptionsHedgeConfig,
) -> PutSpreadPlan | None:
    """Pick the put debit spread from a normalized chain.

    Expiry: the chain expiry inside ``[dte_min, dte_max]`` nearest ``dte_target``.
    Strikes: nearest listed strike to ``spot*(1-long_otm)`` (long leg) and to
    ``spot*(1-short_otm)`` (short leg). Both legs must land on distinct strikes
    with long > short (a collapsed spread protects nothing).

    Args:
        chain: Normalized put contracts: ``{symbol, strike, expiry, dte}`` dicts
            (see :func:`broker.options_executor.fetch_put_chain`).
        spot: Underlying spot price.
        cfg: Frozen hedge knobs.

    Returns:
        The selected :class:`PutSpreadPlan`, or None when no valid spread exists.
    """
    valid = [c for c in chain if cfg.dte_min <= int(c["dte"]) <= cfg.dte_max]
    if not valid or spot <= 0.0:
        return None
    expiry = min({c["expiry"] for c in valid},
                 key=lambda e: abs(next(c["dte"] for c in valid if c["expiry"] == e)
                                   - cfg.dte_target))
    legs = [c for c in valid if c["expiry"] == expiry]

    def nearest(target: float) -> dict:
        return min(legs, key=lambda c: abs(float(c["strike"]) - target))

    long_c = nearest(spot * (1.0 - cfg.long_otm))
    short_c = nearest(spot * (1.0 - cfg.short_otm))
    if float(long_c["strike"]) <= float(short_c["strike"]) or long_c["symbol"] == short_c["symbol"]:
        return None
    return PutSpreadPlan(
        long_symbol=str(long_c["symbol"]),
        short_symbol=str(short_c["symbol"]),
        long_strike=float(long_c["strike"]),
        short_strike=float(short_c["strike"]),
        expiry=expiry,
    )


def spread_contracts(budget_usd: float, net_debit: float) -> int:
    """Number of spreads affordable under the premium budget.

    ``floor(budget / (net_debit * 100))`` (option multiplier 100), never
    negative; degenerate quotes (debit <= 0) size to zero rather than divide.

    Args:
        budget_usd: Premium headroom in USD.
        net_debit: Net debit per spread per share (long mid - short mid).

    Returns:
        Whole spreads to buy (>= 0).
    """
    if budget_usd <= 0.0 or net_debit <= 0.0:
        return 0
    return int(math.floor(budget_usd / (net_debit * 100.0)))
