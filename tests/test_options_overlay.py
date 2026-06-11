"""Tests for the tail-hedge options overlay (risk action, NOT alpha).

Pre-registered v1 (docs/analysis/2026-06-11-meta-overlay-triage.md §3): when the
HMM transition hazard says risk-off is imminent and the book is exposed, buy a
defined-risk SPY put debit spread under a hard premium budget. Selection math is
pure and fixture-driven here; broker I/O lives in broker/options_executor.py.
Everything defaults OFF — the deployed book is untouched.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.options_overlay import (
    HedgeBudget,
    OptionsHedgeConfig,
    select_put_spread,
    should_close_hedge,
    should_open_hedge,
    spread_contracts,
)


def _cfg(**kw) -> OptionsHedgeConfig:
    return OptionsHedgeConfig(**kw)


# ------------------------------------------------------------ trigger logic ---
def test_opens_after_two_consecutive_hazard_closes() -> None:
    cfg = _cfg()
    assert should_open_hedge([0.1, 0.4, 0.45], book_gross=0.9, open_structures=0, cfg=cfg)


def test_single_hazard_spike_does_not_open() -> None:
    """One noisy close above the trigger must not buy protection."""
    cfg = _cfg()
    assert not should_open_hedge([0.4, 0.1, 0.45], book_gross=0.9, open_structures=0, cfg=cfg)


def test_derisked_book_does_not_hedge() -> None:
    """gross <= min_book_gross -> the overlay already de-risked; no premium spend."""
    cfg = _cfg()
    assert not should_open_hedge([0.5, 0.5], book_gross=0.4, open_structures=0, cfg=cfg)


def test_existing_structure_blocks_second_hedge() -> None:
    cfg = _cfg()
    assert not should_open_hedge([0.5, 0.5], book_gross=0.9, open_structures=1, cfg=cfg)


def test_closes_after_five_calm_closes() -> None:
    cfg = _cfg()
    assert should_close_hedge([0.5, 0.15, 0.1, 0.12, 0.18, 0.19], cfg=cfg)
    assert not should_close_hedge([0.15, 0.1, 0.12, 0.18, 0.25], cfg=cfg)  # one hot close resets


def test_short_history_is_conservative() -> None:
    """Too few observations -> neither open nor close fires."""
    cfg = _cfg()
    assert not should_open_hedge([0.9], book_gross=1.0, open_structures=0, cfg=cfg)
    assert not should_close_hedge([0.0], cfg=cfg)


# --------------------------------------------------------- strike selection ---
def _chain() -> list[dict]:
    """Synthetic SPY put chain around spot=500: strikes 430..505 step 5, two expiries."""
    out = []
    for dte, expiry in ((14, date(2026, 6, 25)), (45, date(2026, 7, 26))):
        for strike in range(430, 510, 5):
            out.append({
                "symbol": f"SPY{expiry:%y%m%d}P{strike * 1000:08d}",
                "strike": float(strike),
                "expiry": expiry,
                "dte": dte,
            })
    return out


def test_select_put_spread_targets_otm_strikes_and_dte_window() -> None:
    """long ~4% OTM, short ~10% OTM, expiry inside [30, 60] DTE."""
    plan = select_put_spread(_chain(), spot=500.0, cfg=_cfg())
    assert plan is not None
    assert plan.expiry == date(2026, 7, 26)            # 45 DTE; the 14-DTE expiry is out
    assert plan.long_strike == 480.0                   # 500*(1-0.04) = 480 exact
    assert plan.short_strike == 450.0                  # 500*(1-0.10) = 450 exact
    assert plan.long_symbol != plan.short_symbol


def test_select_put_spread_requires_strike_separation() -> None:
    """If rounding collapses both legs to one strike -> no plan (never same-strike)."""
    chain = [
        {"symbol": "A", "strike": 100.0, "expiry": date(2026, 7, 26), "dte": 45},
    ]
    assert select_put_spread(chain, spot=100.0, cfg=_cfg()) is None


def test_select_put_spread_no_valid_expiry() -> None:
    chain = [c for c in _chain() if c["dte"] == 14]    # only the too-near expiry
    assert select_put_spread(chain, spot=500.0, cfg=_cfg()) is None


# ------------------------------------------------------------ budget/sizing ---
def test_budget_caps_quarter_and_year() -> None:
    """25 bp/quarter, 100 bp/year on 100k -> 250/1000 USD caps; spend reduces both."""
    b = HedgeBudget(spent_quarter=100.0, spent_year=900.0)
    cfg = _cfg()
    # quarter headroom = 250-100 = 150; year headroom = 1000-900 = 100 -> binding 100
    assert b.remaining(equity=100_000.0, cfg=cfg) == pytest.approx(100.0)


def test_spread_contracts_floor_division_and_zero() -> None:
    """contracts = floor(budget / (debit*100)); never negative, zero when unaffordable."""
    assert spread_contracts(budget_usd=1000.0, net_debit=3.0) == 3   # 3*300=900 <= 1000
    assert spread_contracts(budget_usd=200.0, net_debit=3.0) == 0
    assert spread_contracts(budget_usd=500.0, net_debit=0.0) == 0    # degenerate quote
    assert spread_contracts(budget_usd=-10.0, net_debit=3.0) == 0


def test_disabled_by_default() -> None:
    """The overlay must ship OFF: enabling it is an explicit, documented act."""
    assert OptionsHedgeConfig().enabled is False
