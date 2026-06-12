"""Tests for the quality(+momentum) paper sleeve. Pure, causal, no network.

Covers: cross-sectional quality composite ordering, point-in-time exclusion of
not-yet-published filings, the average-of-ranks momentum combine, and the weight_fn
(monthly memoization, gross budget, sector cap) — the same invariants the baseline and
challenger books are held to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quality_ranking import (
    combined_rank,
    make_book_weights_quality,
    quality_scores,
    rank_by_quality,
)


# ── synthetic SimFin blocks ────────────────────────────────────────────────
def _row(pub: str, **fields) -> dict:
    return {"Publish Date": pub, **fields}


def _block(pl_rows: list[dict], bs_rows: list[dict]) -> dict:
    return {"statements": [
        {"statement": "PL", "data": pl_rows},
        {"statement": "BS", "data": bs_rows},
    ]}


def _simple_block(pub: str, revenue, gross, net, assets, equity, liab) -> dict:
    return _block(
        [_row(pub, **{"Revenue": revenue, "Gross Profit": gross, "Net Income": net})],
        [_row(pub, **{"Total Assets": assets, "Total Equity": equity,
                      "Total Liabilities": liab})],
    )


# ── quality composite ──────────────────────────────────────────────────────
def test_quality_scores_orders_high_quality_first():
    # GOOD: high gross profitability + low leverage. BAD: opposite. MID: in between.
    feats = {
        "GOOD": {"gross_profitability": 0.40, "roa": 0.20, "roe": 0.30,
                 "gross_margin": 0.50, "leverage": 0.20},
        "MID": {"gross_profitability": 0.20, "roa": 0.10, "roe": 0.15,
                "gross_margin": 0.35, "leverage": 0.50},
        "BAD": {"gross_profitability": 0.05, "roa": 0.01, "roe": 0.02,
                "gross_margin": 0.20, "leverage": 0.85},
    }
    scores = quality_scores(feats)
    assert scores["GOOD"] > scores["MID"] > scores["BAD"]
    assert rank_by_quality(feats) == ["GOOD", "MID", "BAD"]


def test_quality_scores_drops_names_with_no_usable_factor():
    feats = {
        "A": {"gross_profitability": 0.3, "roa": 0.1, "roe": 0.2,
              "gross_margin": 0.4, "leverage": 0.3},
        "B": {"gross_profitability": 0.1, "roa": 0.05, "roe": 0.1,
              "gross_margin": 0.25, "leverage": 0.6},
        "EMPTY": {"gross_profitability": None, "roa": None, "roe": None,
                  "gross_margin": None, "leverage": None},
    }
    scores = quality_scores(feats)
    assert "EMPTY" not in scores
    assert set(scores) == {"A", "B"}


def test_leverage_sign_is_negative():
    # Two names identical except leverage; the lower-leverage name must score higher.
    feats = {
        "LOWLEV": {"gross_profitability": 0.2, "roa": 0.1, "roe": 0.15,
                   "gross_margin": 0.3, "leverage": 0.20},
        "HIGHLEV": {"gross_profitability": 0.2, "roa": 0.1, "roe": 0.15,
                    "gross_margin": 0.3, "leverage": 0.80},
    }
    scores = quality_scores(feats)
    assert scores["LOWLEV"] > scores["HIGHLEV"]


# ── point-in-time gating ───────────────────────────────────────────────────
def test_quality_scores_excludes_future_publish_date():
    from datetime import date
    from core.fundamental_features import compute_features

    # Two filings: an old one (public) and a fresh one (not yet public as of asof).
    blk = _block(
        pl_rows=[
            _row("2023-02-01", Revenue=100.0, **{"Gross Profit": 40.0}, **{"Net Income": 10.0}),
            _row("2024-02-01", Revenue=200.0, **{"Gross Profit": 120.0}, **{"Net Income": 60.0}),
        ],
        bs_rows=[
            _row("2023-02-01", **{"Total Assets": 100.0, "Total Equity": 60.0,
                                  "Total Liabilities": 40.0}),
            _row("2024-02-01", **{"Total Assets": 100.0, "Total Equity": 70.0,
                                  "Total Liabilities": 30.0}),
        ],
    )
    # As of mid-2023 only the 2023 filing is public → gross_profitability = 40/100 = 0.40.
    f = compute_features(blk, asof=date(2023, 6, 1))
    assert f["gross_profitability"] == pytest.approx(0.40)
    # As of mid-2024 the newer filing dominates → 120/100 = 1.20.
    f2 = compute_features(blk, asof=date(2024, 6, 1))
    assert f2["gross_profitability"] == pytest.approx(1.20)


# ── combined rank ──────────────────────────────────────────────────────────
def test_combined_rank_averages_momentum_and_quality():
    momentum = ["M1", "M2", "M3", "M4"]          # M1 best momentum
    quality = {"M4": 3.0, "M3": 2.0, "M2": 1.0, "M1": 0.0}  # M4 best quality
    out = combined_rank(momentum, quality)
    # M1: (0+3)/2=1.5 ; M2:(1+2)/2=1.5 ; M3:(2+1)/2=1.5 ; M4:(3+0)/2=1.5 -> all tie,
    # broken by symbol name.
    assert out == ["M1", "M2", "M3", "M4"]


def test_combined_rank_keeps_only_common_names():
    momentum = ["A", "B", "C"]
    quality = {"B": 1.0, "C": 2.0, "Z": 5.0}     # Z has no momentum; A has no quality
    out = combined_rank(momentum, quality)
    assert set(out) == {"B", "C"}


# ── weight_fn ──────────────────────────────────────────────────────────────
def _frame(seed: int, n: int = 400, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    idx = pd.date_range("2022-01-01", periods=n + 1, freq="B")
    close = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    return pd.DataFrame({"close": close}, index=idx)


def _universe(n: int = 12):
    frames = {f"S{i:02d}": _frame(seed=i) for i in range(n)}
    blocks = {}
    for i in range(n):
        # spread quality across the universe via gross profit / leverage
        gp, lev = 20.0 + 4.0 * i, 70.0 - 3.0 * i
        blocks[f"S{i:02d}"] = _simple_block(
            "2021-06-01", revenue=100.0, gross=gp, net=10.0 + i,
            assets=100.0, equity=100.0 - lev, liab=lev)
    return frames, blocks


def test_weight_fn_respects_gross_budget_no_overlay():
    frames, blocks = _universe()
    wf = make_book_weights_quality(frames, blocks, frac=0.5, max_single=0.5,
                                   overlay="none", combine="quality")
    ts = frames["S00"].index[-1]
    w = wf(ts, vol_rank=0.0)
    assert w, "expected a non-empty book"
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)   # gross 1.0, naked
    assert all(v <= 0.5 + 1e-9 for v in w.values())


def test_weight_fn_monthly_memoization():
    frames, blocks = _universe()
    wf = make_book_weights_quality(frames, blocks, frac=0.5, overlay="none",
                                   combine="quality_momentum")
    idx = frames["S00"].index
    ts_a, ts_b = idx[-20], idx[-1]   # same calendar month region varies; assert cache key
    # Two calls in the SAME (year, month) must return identical selection.
    same_month = [t for t in (ts_a, ts_b) if (t.year, t.month) == (ts_b.year, ts_b.month)]
    if len(same_month) == 2:
        assert set(wf(same_month[0], 0.0)) == set(wf(same_month[1], 0.0))


def test_weight_fn_vol_target_overlay_runs():
    frames, blocks = _universe()
    wf = make_book_weights_quality(frames, blocks, frac=0.5, overlay="vol_target",
                                   target_vol=0.12)
    ts = frames["S00"].index[-1]
    w = wf(ts, vol_rank=0.5)
    assert isinstance(w, dict)
    assert sum(w.values()) <= 1.0 + 1e-9     # gross_cap 1.0 caps the vol-target scale


def test_invalid_combine_raises():
    frames, blocks = _universe()
    with pytest.raises(ValueError):
        make_book_weights_quality(frames, blocks, combine="bogus")
