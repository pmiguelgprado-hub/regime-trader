"""Pure data layer for the Streamlit dashboard.

No Streamlit imports here on purpose: these loaders/derivations are unit-tested
and shared by ``monitoring/streamlit_app.py`` (the view). Everything degrades
gracefully — a missing snapshot or backtest artifact yields empty/None, never
an exception, so the dashboard renders placeholders instead of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DEFAULT_SNAPSHOT = "state_snapshot.json"
DEFAULT_BOOK_SNAPSHOT = "book_snapshot.json"
DEFAULT_BASE = "backtest_output"


def load_snapshot(path: str = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    """Load the live state snapshot written by ``TradingSystem.save_state``.

    Args:
        path: Snapshot JSON path.

    Returns:
        Parsed snapshot dict, or ``{}`` if absent/unreadable.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_book_snapshot(path: str = DEFAULT_BOOK_SNAPSHOT) -> dict[str, Any]:
    """Load the cross-sectional book snapshot written by ``main.run_rebalance``.

    Args:
        path: Book snapshot JSON path.

    Returns:
        Parsed dict (vol_rank, gross, targets, held, executed, ...), or ``{}`` if
        absent/unreadable (the dashboard then shows a placeholder).
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def risk_panel(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Extract the risk-control panel fields from a snapshot.

    Args:
        snapshot: Loaded snapshot dict (possibly empty).

    Returns:
        Dict with regime, risk_state, equity_peak, daily_trades, breaker_events
        — placeholder values when the snapshot is empty.
    """
    return {
        "regime": snapshot.get("last_regime") or "—",
        "risk_state": snapshot.get("risk_state") or "—",
        "equity_peak": snapshot.get("equity_peak", 0.0) or 0.0,
        "daily_trades": snapshot.get("daily_trades", 0) or 0,
        "breaker_events": snapshot.get("breaker_events", 0) or 0,
        "timestamp": snapshot.get("timestamp") or "—",
    }


def _read_csv(symbol: str, name: str, base: str) -> Optional[pd.DataFrame]:
    """Read a per-symbol backtest CSV (timestamp-indexed) or None if absent."""
    p = Path(base) / symbol / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, parse_dates=[0], index_col=0)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def load_regime_history(symbol: str, base: str = DEFAULT_BASE) -> Optional[pd.DataFrame]:
    """Load the per-bar regime history for the price/regime overlay panels.

    Args:
        symbol: Ticker.
        base: backtest_output base directory.

    Returns:
        DataFrame (regime, regime_prob, weight, returns, ...) or None if absent.
    """
    return _read_csv(symbol, "regime_history.csv", base)


def load_equity_curve(symbol: str, base: str = DEFAULT_BASE) -> Optional[pd.DataFrame]:
    """Load the equity curve for the portfolio-value panel.

    Args:
        symbol: Ticker.
        base: backtest_output base directory.

    Returns:
        DataFrame (equity, returns) or None if absent.
    """
    return _read_csv(symbol, "equity_curve.csv", base)


def live_account() -> Optional[dict[str, Any]]:
    """Pull live account state from Alpaca for the real-time panels.

    Returns:
        ``{mode, equity, cash, market_open}`` or None if creds are missing or
        the broker is unreachable (the dashboard then falls back to the
        snapshot).
    """
    try:
        from broker.alpaca_client import AlpacaClient, AlpacaConfig

        cfg = AlpacaConfig.from_env()
        client = AlpacaClient(cfg)
        client.connect()
        acct = client.get_account()
        return {
            "mode": "PAPER" if cfg.paper else "LIVE",
            "equity": float(acct["equity"]),
            "cash": float(acct["cash"]),
            "market_open": bool(client.is_market_open()),
        }
    except Exception:  # noqa: BLE001 - dashboard degrades to the snapshot
        return None


def live_positions() -> list[dict[str, Any]]:
    """Pull open positions from Alpaca (empty list if none/unreachable)."""
    try:
        from broker.alpaca_client import AlpacaClient, AlpacaConfig

        client = AlpacaClient(AlpacaConfig.from_env())
        client.connect()
        return list(client.get_positions() or [])
    except Exception:  # noqa: BLE001
        return []


def live_price(symbol: str, lookback_bars: int = 180,
               timeframe: str = "1Day") -> Optional[pd.DataFrame]:
    """Pull recent OHLCV from Alpaca for the price panel (None if unreachable)."""
    try:
        from broker.alpaca_client import AlpacaClient, AlpacaConfig
        from data.market_data import MarketData

        client = AlpacaClient(AlpacaConfig.from_env())
        client.connect()
        return MarketData(client).get_history(symbol, timeframe, lookback_bars)
    except Exception:  # noqa: BLE001
        return None


def regime_distribution(regime_history: Optional[pd.DataFrame]) -> pd.Series:
    """Count bars per regime for the learned-regimes panel.

    Args:
        regime_history: Frame with a ``regime`` column (or None/empty).

    Returns:
        Series of counts indexed by regime label (empty if no data).
    """
    if regime_history is None or regime_history.empty or "regime" not in regime_history:
        return pd.Series(dtype="int64")
    return regime_history["regime"].value_counts()
