"""Tests for the cross-sectional momentum ranker (vía C, v1 alpha signal).

The ranker is the *return predictor*: it scores S&P 500 names by cross-sectional
momentum (12-1) and selects the top decile. Pure, causal, no network, no fitted
parameters (the lookback/skip are fixed, not swept).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.asset_rotation import regime_gross_scale
from core.cross_sectional_ranking import (
    compute_book_targets,
    make_book_weights,
    momentum_score,
    plan_rebalance_orders,
    rank_universe,
    select_top,
    select_top_sector_capped,
    targets_to_orders,
)


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="close")


# ---------------------------------------------------------------- momentum ---
def test_momentum_score_uses_window_skipping_recent_bars() -> None:
    """12-1 style: return over [t-lookback, t-skip], ignoring the last `skip` bars."""
    # lookback=5, skip=1: start = close[-6], end = close[-2].
    # values: ...100 (idx -6) ... 110 (idx -2), 999 (idx -1, the skipped bar)
    close = _series([100.0, 101.0, 102.0, 105.0, 110.0, 999.0])
    score = momentum_score(close, lookback=5, skip=1)
    assert score == pytest.approx(110.0 / 100.0 - 1.0)   # 0.10; the 999 is skipped


def test_momentum_score_skip_ignores_recent_spike() -> None:
    """A spike inside the skip window must NOT inflate the score (reversal guard)."""
    flat = _series([100.0] * 5 + [500.0])          # last bar spikes
    assert momentum_score(flat, lookback=5, skip=1) == pytest.approx(0.0)


def test_momentum_score_insufficient_history_is_nan() -> None:
    """Too few bars to span the lookback -> nan (excluded from ranking)."""
    assert math.isnan(momentum_score(_series([100.0, 101.0]), lookback=5, skip=1))


def test_momentum_score_negative_for_downtrend() -> None:
    close = _series([110.0, 108.0, 106.0, 104.0, 100.0, 99.0])
    assert momentum_score(close, lookback=5, skip=1) < 0.0


# ------------------------------------------------------------------ ranking ---
def test_rank_universe_orders_by_descending_momentum() -> None:
    frames = {
        "WIN": pd.DataFrame({"close": _series([100, 100, 100, 100, 150, 9])}),   # +50%
        "MID": pd.DataFrame({"close": _series([100, 100, 100, 100, 120, 9])}),   # +20%
        "LOSE": pd.DataFrame({"close": _series([100, 100, 100, 100, 80, 9])}),   # -20%
    }
    ranked = rank_universe(frames, lookback=5, skip=1)
    assert ranked == ["WIN", "MID", "LOSE"]


def test_rank_universe_drops_names_with_insufficient_history() -> None:
    frames = {
        "GOOD": pd.DataFrame({"close": _series([100, 100, 100, 100, 130, 9])}),
        "SHORT": pd.DataFrame({"close": _series([100, 101])}),                    # too short -> nan
    }
    ranked = rank_universe(frames, lookback=5, skip=1)
    assert ranked == ["GOOD"]


# -------------------------------------------------------------- selection ---
def test_select_top_decile_rounds_up_to_at_least_one() -> None:
    ranked = [f"S{i}" for i in range(50)]
    top = select_top(ranked, frac=0.1)
    assert top == ranked[:5]                       # 10% of 50 = 5


def test_select_top_small_universe_keeps_one() -> None:
    assert select_top(["A", "B", "C"], frac=0.1) == ["A"]   # ceil(0.3) = 1


def test_select_top_empty() -> None:
    assert select_top([], frac=0.1) == []


# ---------------------------------------------------- sector cap (vía C) ---
def test_sector_cap_limits_dominant_sector_keeps_momentum_order() -> None:
    # 10 names, top decile-ish via frac=0.5 -> target 5; cap 0.40 -> max 2 per sector.
    ranked = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    sectors = {s: "TECH" for s in ranked}            # all tech...
    sectors.update({"C": "HEALTH", "E": "ENERGY", "G": "HEALTH", "I": "ENERGY"})
    out = select_top_sector_capped(ranked, sectors, frac=0.5, max_sector_frac=0.40)
    assert len(out) == 5
    # max 2 from any sector (cap = ceil(0.4*5)=2)
    from collections import Counter
    counts = Counter(sectors[s] for s in out)
    assert all(v <= 2 for v in counts.values())
    # strongest momentum that clears the cap come first: A,B (tech), C (health), E (energy), G (health)
    assert out == ["A", "B", "C", "E", "G"]           # D,F (tech) skipped: tech full at 2


def test_sector_cap_max_n_binds_cap_to_realized_book() -> None:
    # 100 names all TECH; frac=0.10 -> 10, but max_n=8 -> target 8, cap=ceil(0.4*8)=4.
    ranked = [f"S{i:03d}" for i in range(100)]
    sectors = {s: "TECH" for s in ranked}
    out = select_top_sector_capped(ranked, sectors, frac=0.10, max_sector_frac=0.40, max_n=8)
    assert len(out) == 4                                    # cap binds to min(10,8)=8 -> 4


def test_sector_cap_unknown_sector_bucketed() -> None:
    ranked = ["X", "Y", "Z"]
    out = select_top_sector_capped(ranked, {}, frac=1.0, max_sector_frac=0.34)
    # all UNKNOWN, cap = ceil(0.34*3)=2 -> only 2 admitted
    assert out == ["X", "Y"]


# ------------------------------------------------------ HMM gross overlay ---
def test_regime_gross_scale_tiers() -> None:
    """Full gross in risk-on, de-risked in risk-off, interpolated in between."""
    assert regime_gross_scale(0.1, risk_on=1.0, risk_off=0.5) == 1.0       # low-vol
    assert regime_gross_scale(0.9, risk_on=1.0, risk_off=0.5) == 0.5       # high-vol
    mid = regime_gross_scale(0.5, risk_on=1.0, risk_off=0.5)
    assert 0.5 < mid < 1.0                                                  # mid band


# -------------------------------------------------- book weight function ---
def _book_frames() -> dict[str, pd.DataFrame]:
    # lookback=3, skip=1 -> score uses close[-4]/close[-2]-1 (last bar skipped).
    return {
        "WIN": pd.DataFrame({"close": _series([100, 100, 110, 130, 9])}),   # strong up
        "OK": pd.DataFrame({"close": _series([100, 100, 105, 110, 9])}),    # mild up
        "BAD": pd.DataFrame({"close": _series([100, 100, 95, 80, 9])}),     # down
        "WORST": pd.DataFrame({"close": _series([100, 100, 90, 70, 9])}),   # worst
    }


def test_book_weights_pick_top_fraction_equal_weighted() -> None:
    frames = _book_frames()
    ts = frames["WIN"].index[-1]
    wf = make_book_weights(frames, lookback=3, skip=1, frac=0.5, max_single=0.6,
                           risk_on_gross=1.0)
    w = wf(ts, vol_rank=0.1)                     # risk-on -> gross 1.0
    assert set(w) == {"WIN", "OK"}               # top 50% of 4
    assert w["WIN"] == pytest.approx(0.5) and w["OK"] == pytest.approx(0.5)


def test_book_weights_overlay_de_risks_in_high_vol() -> None:
    frames = _book_frames()
    ts = frames["WIN"].index[-1]
    wf = make_book_weights(frames, lookback=3, skip=1, frac=0.5, max_single=0.6,
                           risk_on_gross=1.0, risk_off_gross=0.5)
    risk_off = wf(ts, vol_rank=0.9)              # high-vol -> gross 0.5
    assert risk_off["WIN"] == pytest.approx(0.25) and risk_off["OK"] == pytest.approx(0.25)


def test_book_weights_overlay_off_ignores_regime() -> None:
    frames = _book_frames()
    ts = frames["WIN"].index[-1]
    wf = make_book_weights(frames, lookback=3, skip=1, frac=0.5, max_single=0.6,
                           use_overlay=False)
    assert wf(ts, vol_rank=0.9) == wf(ts, vol_rank=0.1)   # regime ignored, gross 1.0


# --------------------------------------- backtest weight_fn integration ---
def test_run_portfolio_weight_fn_smoke() -> None:
    """The cross-sectional weight_fn path produces a sane, positive equity curve."""
    import logging

    from backtest.backtester import BacktestConfig, Backtester
    from core.hmm_engine import HMMConfig, HMMEngine
    from core.regime_strategies import StrategyConfig
    from core.risk_manager import RiskConfig, RiskManager
    from data.feature_engineering import FeatureEngineer
    from conftest import make_synthetic_ohlcv

    for _n in ("core.hmm_engine", "core.risk_manager", "core.regime_strategies"):
        logging.getLogger(_n).setLevel(logging.CRITICAL)

    ohlcv = make_synthetic_ohlcv()
    frames = {s: ohlcv for s in ("AAA", "BBB", "CCC", "DDD")}
    bt = Backtester(
        BacktestConfig(step_size=126, credit_cash_rf=True),
        HMMEngine(HMMConfig(n_candidates=[3], n_init=1)),
        StrategyConfig(), RiskManager(RiskConfig()), FeatureEngineer(),
    )
    wf = make_book_weights(frames, lookback=60, skip=5, frac=0.5, max_single=0.6)
    eq, w = bt.run_portfolio(frames, return_weights=True, weight_fn=wf)
    assert isinstance(eq, pd.Series) and len(eq) > 0
    assert eq.notna().all() and (eq > 0).all()
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()          # gross never exceeds 1.0


def test_compute_book_targets_one_shot_live() -> None:
    """The live one-shot rebalance picks the top decile and scales gross by regime."""
    frames = _book_frames()
    risk_on = compute_book_targets(frames, vol_rank=0.1, lookback=3, skip=1, frac=0.5,
                                   max_single=0.6)
    assert set(risk_on) == {"WIN", "OK"}
    assert sum(risk_on.values()) == pytest.approx(1.0)          # gross 1.0 in risk-on
    risk_off = compute_book_targets(frames, vol_rank=0.9, lookback=3, skip=1, frac=0.5,
                                    max_single=0.6, risk_off_gross=0.5)
    assert sum(risk_off.values()) == pytest.approx(0.5)         # de-risked in risk-off


def test_targets_to_orders_whole_shares_skips_unpriced() -> None:
    targets = {"AAA": 0.50, "BBB": 0.50, "NOPRICE": 0.50}
    plan = targets_to_orders(targets, equity=100_000.0,
                             prices={"AAA": 200.0, "BBB": 100.0, "NOPRICE": 0.0})
    by_sym = {o["symbol"]: o for o in plan}
    assert "NOPRICE" not in by_sym                       # no price -> skipped
    assert by_sym["AAA"]["shares"] == 250                # 50k / 200
    assert by_sym["BBB"]["shares"] == 500                # 50k / 100
    assert plan[0]["symbol"] in {"AAA", "BBB"}           # sorted by notional desc


# ------------------------------------------------- rebalance diff (vía C) ---
def test_plan_rebalance_orders_sells_before_buys_and_liquidates_dropped() -> None:
    targets = {"KEEP": 100, "GROW": 50, "NEW": 30}        # NEW not held; KEEP up; GROW down
    held = {"KEEP": 60, "GROW": 80, "DROP": 25}           # DROP not in target -> liquidate
    orders = plan_rebalance_orders(targets, held)
    sides = [o["side"] for o in orders]
    assert sides == sorted(sides, key=lambda s: 0 if s == "sell" else 1)  # all sells first
    by = {o["symbol"]: o for o in orders}
    assert by["DROP"] == {"symbol": "DROP", "side": "sell", "qty": 25}     # full liquidation
    assert by["GROW"] == {"symbol": "GROW", "side": "sell", "qty": 30}     # 80 -> 50
    assert by["KEEP"] == {"symbol": "KEEP", "side": "buy", "qty": 40}      # 60 -> 100
    assert by["NEW"] == {"symbol": "NEW", "side": "buy", "qty": 30}        # 0 -> 30


def test_plan_rebalance_orders_skips_zero_delta() -> None:
    orders = plan_rebalance_orders({"A": 10, "B": 5}, {"A": 10, "B": 3})
    assert orders == [{"symbol": "B", "side": "buy", "qty": 2}]            # A unchanged


def test_submit_market_orders_batch() -> None:
    """The executor batch submits each order as a plain market order."""
    from types import SimpleNamespace

    from broker.alpaca_client import AlpacaClient, AlpacaConfig
    from broker.order_executor import OrderExecutor, OrderStatus

    class _Trading:
        def __init__(self):
            self.submitted = []

        def submit_order(self, order_data):
            self.submitted.append(order_data)
            return SimpleNamespace(id=f"o{len(self.submitted)}", symbol=order_data.symbol,
                                   status="filled", filled_qty=order_data.qty,
                                   filled_avg_price=100.0, client_order_id=None, legs=None)

    client = AlpacaClient(AlpacaConfig("k", "s", paper=True))
    client._trading_client = _Trading()
    ex = OrderExecutor(client, fill_timeout_sec=0.0, poll_interval_sec=0.0)
    orders = [{"symbol": "DROP", "side": "sell", "qty": 25},
              {"symbol": "NEW", "side": "buy", "qty": 30},
              {"symbol": "ZERO", "side": "buy", "qty": 0}]          # skipped
    results = ex.submit_market_orders(orders)
    assert len(results) == 2                                        # ZERO skipped
    assert all(r.status is OrderStatus.FILLED for r in results)
    assert [o.symbol for o in client.trading.submitted] == ["DROP", "NEW"]  # order preserved
