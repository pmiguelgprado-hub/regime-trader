"""Tests for the Alpaca portfolio-history pipeline (client wrapper + data layer).

The dashboard's portfolio-evolution chart reads the PAPER account's equity
series from Alpaca. Client wrapper returns plain types; the dashboard data
layer shapes them into a timestamp-indexed DataFrame and degrades to None on
any failure (the established dashboard_data contract).
"""

from __future__ import annotations

from types import SimpleNamespace

from broker.alpaca_client import AlpacaClient, AlpacaConfig
from monitoring.dashboard_data import portfolio_history_frame


class _StubTrading:
    def get_portfolio_history(self, req):
        return SimpleNamespace(
            timestamp=[1749600000, 1749686400, 1749772800],
            equity=[100000.0, 100500.0, 100250.0],
            profit_loss=[0.0, 500.0, -250.0],
            profit_loss_pct=[0.0, 0.005, -0.0025],
        )


def _client() -> AlpacaClient:
    c = AlpacaClient(AlpacaConfig("k", "s"))
    c._trading_client = _StubTrading()
    return c


def test_client_wrapper_returns_plain_lists() -> None:
    out = _client().get_portfolio_history(period="1M", timeframe="1D")
    assert out["equity"] == [100000.0, 100500.0, 100250.0]
    assert len(out["timestamp"]) == 3


def test_frame_is_timestamp_indexed_with_drawdown() -> None:
    raw = _client().get_portfolio_history()
    df = portfolio_history_frame(raw)
    assert list(df.columns) == ["equity", "profit_loss", "drawdown"]
    assert df.index.name == "timestamp"
    assert df["equity"].iloc[1] == 100500.0
    # drawdown: peak 100500 -> 100250 = -0.249% (running-peak definition)
    assert df["drawdown"].iloc[2] < 0.0
    assert df["drawdown"].iloc[1] == 0.0


def test_frame_handles_empty_or_none() -> None:
    assert portfolio_history_frame(None) is None
    assert portfolio_history_frame({"timestamp": [], "equity": []}) is None


def test_frame_drops_leading_zero_equity_rows() -> None:
    """Alpaca pads history with equity=0 before the account was funded; those
    rows poison returns/drawdown (division by zero) and must be dropped."""
    raw = {
        "timestamp": [1749600000, 1749686400, 1749772800, 1749859200],
        "equity": [0.0, 0.0, 100000.0, 100500.0],
        "profit_loss": [0.0, 0.0, 0.0, 500.0],
    }
    df = portfolio_history_frame(raw)
    assert len(df) == 2
    assert df["equity"].iloc[0] == 100000.0
    assert (df["drawdown"] <= 0.0).all() and df["drawdown"].notna().all()


def test_frame_all_zero_equity_is_none() -> None:
    assert portfolio_history_frame(
        {"timestamp": [1, 2], "equity": [0.0, 0.0]}) is None
