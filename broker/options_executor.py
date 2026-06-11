"""Options hedge executor: broker I/O for the tail-hedge overlay.

Thin wrapper around alpaca-py (verified against 0.43.4) that runs the
pre-registered hedge check once per close: persist hazard history + premium
budget in a state file, evaluate the pure rules in
:mod:`core.options_overlay`, and — only when the overlay is enabled AND the
caller passes ``dry_run=False`` — submit the multi-leg order. Every other path
plans and logs without touching the broker, mirroring ``main.run_rebalance``'s
dry-run-first discipline. Real money stays BLOCKED regardless: this trades the
paper account only until the pre-registered gate says otherwise.

State file (default ``options_hedge_state.json``)::

    {
      "hazard_history": [0.12, ...],          # most recent last, trimmed
      "open_structures": [{...}],             # live spreads (v1: max 1)
      "budget": {"quarter_key": "2026-Q2", "year_key": "2026",
                  "spent_quarter": 0.0, "spent_year": 0.0}
    }
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Callable

from core.options_overlay import (
    HedgeBudget,
    OptionsHedgeConfig,
    PutSpreadPlan,
    select_put_spread,
    should_close_hedge,
    should_open_hedge,
    spread_contracts,
)

logger = logging.getLogger(__name__)

_HISTORY_KEEP = 30  # daily hazards retained (>= release_days + trigger_days)


def normalize_contracts(contracts: list[Any], today: date) -> list[dict]:
    """SDK option contracts -> plain ``{symbol, strike, expiry, dte}`` dicts.

    Args:
        contracts: ``OptionContract``-like objects (``symbol``, ``strike_price``,
            ``expiration_date`` attributes).
        today: Reference date for DTE.

    Returns:
        Normalized chain rows for :func:`core.options_overlay.select_put_spread`.
    """
    out = []
    for c in contracts:
        expiry = c.expiration_date
        out.append({
            "symbol": str(c.symbol),
            "strike": float(c.strike_price),
            "expiry": expiry,
            "dte": (expiry - today).days,
        })
    return out


def build_mleg_order(plan: PutSpreadPlan, qty: int, limit_debit: float):
    """Multi-leg limit order for the put debit spread (defined risk).

    Buy the nearer (long) strike, sell the farther (short) strike, one contract
    each per spread, as a single MLEG limit order at the net debit — never two
    independent legs (a partial fill would leave a naked short put).

    Args:
        plan: Selected spread.
        qty: Number of spreads.
        limit_debit: Max net debit per spread per share.

    Returns:
        ``LimitOrderRequest`` ready for ``TradingClient.submit_order``.
    """
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

    legs = [
        OptionLegRequest(symbol=plan.long_symbol, side=OrderSide.BUY, ratio_qty=1),
        OptionLegRequest(symbol=plan.short_symbol, side=OrderSide.SELL, ratio_qty=1),
    ]
    return LimitOrderRequest(
        qty=qty,
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=legs,
        limit_price=round(limit_debit, 2),
    )


def _close_order(structure: dict):
    """MLEG order unwinding a live spread (sell the long leg, buy back the short)."""
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest

    legs = [
        OptionLegRequest(symbol=structure["long_symbol"], side=OrderSide.SELL, ratio_qty=1),
        OptionLegRequest(symbol=structure["short_symbol"], side=OrderSide.BUY, ratio_qty=1),
    ]
    return MarketOrderRequest(
        qty=int(structure["contracts"]),
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=legs,
    )


class OptionsHedgeExecutor:
    """Runs the daily hedge check against Alpaca's options API."""

    def __init__(
        self,
        trading_client: Any,
        data_client: Any,
        cfg: OptionsHedgeConfig,
        state_path: str = "options_hedge_state.json",
        today_fn: Callable[[], date] = date.today,
    ) -> None:
        """Wire the executor.

        Args:
            trading_client: ``TradingClient``-like (get_option_contracts,
                submit_order).
            data_client: ``OptionHistoricalDataClient``-like
                (get_option_latest_quote).
            cfg: Frozen hedge knobs.
            state_path: JSON state file path.
            today_fn: Injectable clock (tests freeze it).
        """
        self.trading = trading_client
        self.data = data_client
        self.cfg = cfg
        self.state_path = Path(state_path)
        self.today_fn = today_fn

    # ------------------------------------------------------------- state ---
    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                state = {}
        else:
            state = {}
        state.setdefault("hazard_history", [])
        state.setdefault("open_structures", [])
        state.setdefault("budget", {})
        today = self.today_fn()
        qkey = f"{today.year}-Q{(today.month - 1) // 3 + 1}"
        ykey = str(today.year)
        b = state["budget"]
        if b.get("quarter_key") != qkey:                  # quarter rollover
            b["quarter_key"], b["spent_quarter"] = qkey, 0.0
        if b.get("year_key") != ykey:                     # year rollover
            b["year_key"], b["spent_year"] = ykey, 0.0
        return state

    def _save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, default=str))

    # -------------------------------------------------------------- I/O ---
    def _fetch_chain(self, spot: float) -> list[dict]:
        """Fetch the put chain around the hedge strikes (single page, bounded)."""
        from alpaca.trading.enums import ContractType
        from alpaca.trading.requests import GetOptionContractsRequest
        from datetime import timedelta

        today = self.today_fn()
        req = GetOptionContractsRequest(
            underlying_symbols=[self.cfg.proxy],
            type=ContractType.PUT,
            expiration_date_gte=today + timedelta(days=self.cfg.dte_min),
            expiration_date_lte=today + timedelta(days=self.cfg.dte_max),
            strike_price_gte=str(round(spot * (1.0 - self.cfg.short_otm - 0.05), 2)),
            strike_price_lte=str(round(spot, 2)),
            limit=300,
        )
        res = self.trading.get_option_contracts(req)
        return normalize_contracts(list(res.option_contracts or []), today)

    def _net_debit(self, plan: PutSpreadPlan) -> float:
        """Net debit per share from latest mid quotes (long mid - short mid)."""
        from alpaca.data.requests import OptionLatestQuoteRequest

        req = OptionLatestQuoteRequest(
            symbol_or_symbols=[plan.long_symbol, plan.short_symbol]
        )
        quotes = self.data.get_option_latest_quote(req)

        def mid(sym: str) -> float:
            q = quotes[sym]
            return (float(q.bid_price) + float(q.ask_price)) / 2.0

        return mid(plan.long_symbol) - mid(plan.short_symbol)

    # ------------------------------------------------------------- check ---
    def run_check(
        self,
        hazard: float,
        equity: float,
        book_gross: float,
        spot: float,
        dry_run: bool = True,
    ) -> dict:
        """Evaluate the pre-registered hedge rules at today's close.

        Args:
            hazard: Today's transition hazard (core.meta_overlay).
            equity: Account equity (USD).
            book_gross: Current book gross exposure.
            spot: Proxy spot price.
            dry_run: True (default) plans + logs only; False submits (still
                requires ``cfg.enabled``).

        Returns:
            Action report: ``{action, hazard, ...}`` — actions are ``disabled``,
            ``hold``, ``open_planned``, ``opened``, ``close_planned``, ``closed``.
        """
        if not self.cfg.enabled:
            return {"action": "disabled", "hazard": hazard}

        state = self._load_state()
        hist = state["hazard_history"]
        hist.append(float(hazard))
        del hist[:-_HISTORY_KEEP]
        out: dict = {"action": "hold", "hazard": hazard}

        # 1) close path first: a live structure in a calm market gives back theta
        if state["open_structures"] and should_close_hedge(hist, self.cfg):
            structure = state["open_structures"][0]
            if dry_run:
                out = {"action": "close_planned", "hazard": hazard,
                       "structure": structure}
            else:
                self.trading.submit_order(_close_order(structure))
                state["open_structures"] = []
                out = {"action": "closed", "hazard": hazard, "structure": structure}
                logger.info("Hedge closed after calm run: %s", structure)
            self._save_state(state)
            return out

        # 2) open path
        if should_open_hedge(hist, book_gross, len(state["open_structures"]), self.cfg):
            chain = self._fetch_chain(spot)
            plan = select_put_spread(chain, spot, self.cfg)
            if plan is None:
                out = {"action": "hold", "hazard": hazard,
                       "reason": "no valid spread in chain"}
                self._save_state(state)
                return out
            debit = self._net_debit(plan)
            budget = HedgeBudget(
                spent_quarter=float(state["budget"].get("spent_quarter", 0.0)),
                spent_year=float(state["budget"].get("spent_year", 0.0)),
            )
            n = spread_contracts(budget.remaining(equity, self.cfg), debit)
            plan_dict = {
                "long_symbol": plan.long_symbol, "short_symbol": plan.short_symbol,
                "long_strike": plan.long_strike, "short_strike": plan.short_strike,
                "expiry": str(plan.expiry), "net_debit": round(debit, 4),
                "contracts": n,
            }
            if n < 1:
                out = {"action": "open_planned", "hazard": hazard, "plan": plan_dict,
                       "reason": "premium budget exhausted (no contracts affordable)"}
            elif dry_run:
                out = {"action": "open_planned", "hazard": hazard, "plan": plan_dict}
            else:
                # Limit at mid + a cent of give: defined max debit, never chase.
                self.trading.submit_order(build_mleg_order(plan, n, debit + 0.01))
                spent = n * debit * 100.0
                state["budget"]["spent_quarter"] = (
                    float(state["budget"].get("spent_quarter", 0.0)) + spent)
                state["budget"]["spent_year"] = (
                    float(state["budget"].get("spent_year", 0.0)) + spent)
                state["open_structures"].append({**plan_dict, "opened": str(self.today_fn()),
                                                 "premium_spent": round(spent, 2)})
                out = {"action": "opened", "hazard": hazard, "plan": plan_dict,
                       "premium_spent": round(spent, 2)}
                logger.info("Hedge opened: %s", plan_dict)

        self._save_state(state)
        return out
