"""Tests for the Streamlit dashboard's pure data layer (no Streamlit runtime).

The view (monitoring/streamlit_app.py) is verified by running it; these tests
pin the data loading/derivation it depends on so the panels never crash on
missing or empty state.
"""

from __future__ import annotations

import json

import pandas as pd

from monitoring.dashboard_data import (
    load_book_snapshot,
    load_equity_curve,
    load_regime_history,
    load_snapshot,
    regime_distribution,
    risk_panel,
)


def test_load_snapshot_missing_returns_empty(tmp_path) -> None:
    assert load_snapshot(str(tmp_path / "nope.json")) == {}


def test_load_snapshot_parses_state(tmp_path) -> None:
    p = tmp_path / "state_snapshot.json"
    p.write_text(json.dumps({"last_regime": "bull", "risk_state": "normal",
                             "equity_peak": 100000, "recent_signals": [{"symbol": "SPY"}]}))
    snap = load_snapshot(str(p))
    assert snap["last_regime"] == "bull"
    assert snap["recent_signals"][0]["symbol"] == "SPY"


def test_load_book_snapshot_missing_returns_empty(tmp_path) -> None:
    assert load_book_snapshot(str(tmp_path / "nope.json")) == {}


def test_load_book_snapshot_parses_book(tmp_path) -> None:
    p = tmp_path / "book_snapshot.json"
    p.write_text(json.dumps({
        "vol_rank": 0.0, "gross": 0.98, "dry_run": True, "mode": "PAPER",
        "targets": [{"symbol": "AVGO", "shares": 4, "price": 479.23, "weight": 0.02}],
        "executed": [],
    }))
    book = load_book_snapshot(str(p))
    assert book["gross"] == 0.98 and book["dry_run"] is True
    assert book["targets"][0]["symbol"] == "AVGO"


def test_risk_panel_defaults_when_empty() -> None:
    panel = risk_panel({})
    assert panel["risk_state"] == "—"
    assert panel["regime"] == "—"


def test_risk_panel_reads_snapshot() -> None:
    panel = risk_panel({"risk_state": "reduced", "last_regime": "bear",
                        "equity_peak": 99000, "daily_trades": 3})
    assert panel["risk_state"] == "reduced"
    assert panel["regime"] == "bear"
    assert panel["equity_peak"] == 99000
    assert panel["daily_trades"] == 3


def test_regime_distribution_counts_bars() -> None:
    df = pd.DataFrame({"regime": ["bull", "bull", "bear", "crash", "bear"]})
    dist = regime_distribution(df)
    assert dist["bull"] == 2
    assert dist["bear"] == 2
    assert dist["crash"] == 1


def test_regime_distribution_empty_is_empty() -> None:
    assert regime_distribution(None).empty
    assert regime_distribution(pd.DataFrame()).empty


def test_load_regime_history_missing_returns_none(tmp_path) -> None:
    assert load_regime_history("SPY", base=str(tmp_path)) is None


def test_load_regime_history_reads_csv(tmp_path) -> None:
    d = tmp_path / "SPY"
    d.mkdir()
    (d / "regime_history.csv").write_text(
        "timestamp,regime,regime_prob,weight\n2022-01-01,bull,0.9,0.95\n")
    df = load_regime_history("SPY", base=str(tmp_path))
    assert df is not None and list(df["regime"]) == ["bull"]


def test_load_equity_curve_missing_returns_none(tmp_path) -> None:
    assert load_equity_curve("SPY", base=str(tmp_path)) is None
