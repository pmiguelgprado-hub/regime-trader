"""Tests for the options hedge executor (broker I/O wrapper, dry-run by default).

Stub clients, no network. The executor's job: persist hazard history + premium
budget across daily runs, evaluate the pre-registered open/close rules, build
the Alpaca multi-leg order — and submit ONLY when enabled AND not dry-run.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from core.options_overlay import OptionsHedgeConfig, PutSpreadPlan
from broker.options_executor import (
    OptionsHedgeExecutor,
    build_mleg_order,
    normalize_contracts,
)


# ----------------------------------------------------------- order building ---
def test_build_mleg_order_buys_long_sells_short_defined_risk() -> None:
    from alpaca.trading.enums import OrderClass, OrderSide

    plan = PutSpreadPlan("SPY260726P00480000", "SPY260726P00450000",
                         480.0, 450.0, date(2026, 7, 26))
    req = build_mleg_order(plan, qty=3, limit_debit=2.85)
    assert req.order_class == OrderClass.MLEG
    assert req.qty == 3
    assert req.limit_price == pytest.approx(2.85)
    sides = {leg.symbol: leg.side for leg in req.legs}
    assert sides[plan.long_symbol] == OrderSide.BUY
    assert sides[plan.short_symbol] == OrderSide.SELL
    assert all(leg.ratio_qty == 1 for leg in req.legs)


def test_normalize_contracts_computes_dte_and_floats() -> None:
    raw = [SimpleNamespace(symbol="SPY260726P00480000", strike_price="480",
                           expiration_date=date(2026, 7, 26))]
    out = normalize_contracts(raw, today=date(2026, 6, 11))
    assert out == [{"symbol": "SPY260726P00480000", "strike": 480.0,
                    "expiry": date(2026, 7, 26), "dte": 45}]


# ------------------------------------------------------------ executor flow ---
def _contract(strike: float, expiry: date) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=f"SPY{expiry:%y%m%d}P{int(strike * 1000):08d}",
        strike_price=str(strike),
        expiration_date=expiry,
    )


class _StubTrading:
    """Canned option chain + order recorder."""

    def __init__(self, contracts):
        self._contracts = contracts
        self.submitted: list = []

    def get_option_contracts(self, req):
        return SimpleNamespace(option_contracts=self._contracts, next_page_token=None)

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id="oid-1", status="accepted")


class _StubData:
    """Mid = (bid+ask)/2: long leg 3.50/3.70, short leg 0.70/0.90 -> debit 2.80."""

    def get_option_latest_quote(self, req):
        syms = req.symbol_or_symbols
        quotes = {}
        for s in syms:
            if "P00480000" in s:
                quotes[s] = SimpleNamespace(bid_price=3.50, ask_price=3.70)
            else:
                quotes[s] = SimpleNamespace(bid_price=0.70, ask_price=0.90)
        return quotes


def _executor(tmp_path, enabled=True):
    expiry = date(2026, 7, 26)                      # 45 DTE from the frozen today
    chain = [_contract(s, expiry) for s in (450.0, 460.0, 470.0, 480.0, 490.0)]
    trading, data = _StubTrading(chain), _StubData()
    cfg = OptionsHedgeConfig(enabled=enabled)
    ex = OptionsHedgeExecutor(
        trading, data, cfg, state_path=str(tmp_path / "hedge_state.json"),
        today_fn=lambda: date(2026, 6, 11),
    )
    return ex, trading


def test_dry_run_plans_but_never_submits(tmp_path) -> None:
    ex, trading = _executor(tmp_path)
    # two consecutive hazardous closes (state persists across calls)
    ex.run_check(hazard=0.40, equity=100_000.0, book_gross=0.9, spot=500.0, dry_run=True)
    out = ex.run_check(hazard=0.45, equity=100_000.0, book_gross=0.9, spot=500.0,
                       dry_run=True)
    assert out["action"] == "open_planned"
    assert out["plan"]["long_strike"] == 480.0
    assert out["plan"]["short_strike"] == 450.0
    assert out["plan"]["contracts"] == 0 or out["plan"]["contracts"] >= 1
    assert trading.submitted == []                 # dry-run NEVER touches the broker


def test_disabled_overlay_does_nothing(tmp_path) -> None:
    ex, trading = _executor(tmp_path, enabled=False)
    out = ex.run_check(hazard=0.99, equity=100_000.0, book_gross=1.0, spot=500.0,
                       dry_run=False)
    assert out["action"] == "disabled"
    assert trading.submitted == []


def test_budget_binds_contract_count(tmp_path) -> None:
    """25bp of 100k = $250 headroom; debit 2.80*100=280 -> 0 contracts affordable."""
    ex, _ = _executor(tmp_path)
    ex.run_check(hazard=0.40, equity=100_000.0, book_gross=0.9, spot=500.0, dry_run=True)
    out = ex.run_check(hazard=0.45, equity=100_000.0, book_gross=0.9, spot=500.0,
                       dry_run=True)
    assert out["plan"]["net_debit"] == pytest.approx(2.80)
    assert out["plan"]["contracts"] == 0           # honest: budget too small, no trade


def test_submit_path_requires_enabled_and_not_dry_run(tmp_path) -> None:
    """1M equity -> $2500 headroom -> 8 spreads at $280; live submit records order
    and the spend lands in the persisted budget."""
    ex, trading = _executor(tmp_path)
    ex.run_check(hazard=0.40, equity=1_000_000.0, book_gross=0.9, spot=500.0,
                 dry_run=True)
    out = ex.run_check(hazard=0.45, equity=1_000_000.0, book_gross=0.9, spot=500.0,
                       dry_run=False)
    assert out["action"] == "opened"
    assert len(trading.submitted) == 1
    state = json.loads((tmp_path / "hedge_state.json").read_text())
    assert state["budget"]["spent_quarter"] == pytest.approx(8 * 2.80 * 100.0)
    assert len(state["open_structures"]) == 1


def test_close_after_calm_run(tmp_path) -> None:
    ex, trading = _executor(tmp_path)
    ex.run_check(hazard=0.40, equity=1_000_000.0, book_gross=0.9, spot=500.0,
                 dry_run=True)
    ex.run_check(hazard=0.45, equity=1_000_000.0, book_gross=0.9, spot=500.0,
                 dry_run=False)                     # opens
    for h in (0.1, 0.1, 0.1, 0.1):
        out = ex.run_check(hazard=h, equity=1_000_000.0, book_gross=0.9, spot=500.0,
                           dry_run=False)
        assert out["action"] == "hold"
    out = ex.run_check(hazard=0.1, equity=1_000_000.0, book_gross=0.9, spot=500.0,
                       dry_run=False)               # 5th calm close
    assert out["action"] == "closed"
    assert len(trading.submitted) == 2             # open + close
    state = json.loads((tmp_path / "hedge_state.json").read_text())
    assert state["open_structures"] == []
