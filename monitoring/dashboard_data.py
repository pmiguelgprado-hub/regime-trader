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


def portfolio_history_frame(raw: Optional[dict]) -> Optional[pd.DataFrame]:
    """Shape a portfolio-history payload into the evolution-chart frame.

    Args:
        raw: ``AlpacaClient.get_portfolio_history`` output (or None).

    Returns:
        DataFrame indexed by ``timestamp`` with ``equity``, ``profit_loss`` and
        ``drawdown`` (fraction from the running equity peak, <= 0), or None when
        the payload is missing/empty.
    """
    if not raw or not raw.get("timestamp") or not raw.get("equity"):
        return None
    try:
        idx = pd.to_datetime(pd.Series(raw["timestamp"], dtype="int64"), unit="s")
        eq = pd.Series([float(v) for v in raw["equity"]], index=idx, name="equity")
        pl = pd.Series(
            [float(v) for v in (raw.get("profit_loss") or [0.0] * len(eq))][: len(eq)],
            index=idx, name="profit_loss",
        )
        # Alpaca pads the window with equity=0 before the account was funded;
        # those rows poison returns/drawdown (divide-by-zero). Keep from the
        # first positive-equity bar onward.
        funded = eq > 0.0
        if not funded.any():
            return None
        start = funded.idxmax()
        eq, pl = eq.loc[start:], pl.loc[start:]
        dd = eq / eq.cummax() - 1.0
        df = pd.DataFrame({"equity": eq, "profit_loss": pl, "drawdown": dd})
        df.index.name = "timestamp"
        return df
    except (ValueError, TypeError):
        return None


def live_portfolio_history(period: str = "3M",
                           timeframe: str = "1D") -> Optional[pd.DataFrame]:
    """Pull the account's equity evolution from Alpaca (None if unreachable)."""
    try:
        from broker.alpaca_client import AlpacaClient, AlpacaConfig

        client = AlpacaClient(AlpacaConfig.from_env())
        client.connect()
        return portfolio_history_frame(
            client.get_portfolio_history(period=period, timeframe=timeframe)
        )
    except Exception:  # noqa: BLE001 - dashboard degrades to a placeholder
        return None


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


# --- gate countdown (T0.6) ----------------------------------------------------------

# Each forward gate: NAV column in the track record, its start date (prereg freeze),
# and the ledger family whose n_trials deflates its Sharpe.
GATES = [
    ("baseline", "book_nav", "2026-06-05", "momentum"),
    ("challenger", "challenger_nav", "2026-06-05", "momentum"),
    ("quality", "quality_nav", "2026-06-13", "quality"),
]
GATE_WINDOW_DAYS = 365


def _per_bar_dsr(nav: pd.Series, n_trials: int, min_obs: int) -> "tuple[Optional[float], int]":
    """Deflated Sharpe over a NAV level series ((dsr, n_obs); dsr None if too short)."""
    from backtest.performance import deflated_sharpe_ratio

    rets = nav.astype(float).pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    n = int(len(rets))
    if n < min_obs or rets.std(ddof=1) == 0:
        return None, n
    sr = float(rets.mean() / rets.std(ddof=1))         # per-bar Sharpe (DSR convention)
    dsr = deflated_sharpe_ratio(sr, n, skew=float(rets.skew()), kurt=float(rets.kurtosis() + 3.0),
                                n_trials=max(1, n_trials))
    return float(dsr), n


def gate_status(track_df: Optional[pd.DataFrame],
                n_trials_fn=None, today: Optional[str] = None,
                min_obs: int = 30) -> list[dict[str, Any]]:
    """Per-gate countdown + rolling DSR for the dashboard (T0.6).

    Args:
        track_df: Loaded track_record.csv (or None/empty -> []).
        n_trials_fn: ``family -> int`` (defaults to the research ledger).
        today: ISO date override (defaults to the last recorded row, then now).
        min_obs: Minimum return obs before a DSR is reported.

    Returns:
        One dict per gate with a usable NAV column: name, start, days_elapsed,
        days_remaining, window_days, n_obs, sharpe, dsr, n_trials.
    """
    if track_df is None or len(track_df) == 0 or "date" not in track_df:
        return []
    if n_trials_fn is None:
        from core import research_ledger as rl
        n_trials_fn = lambda fam: rl.n_trials(family=fam)  # noqa: E731
    ref = today or str(track_df.iloc[-1]["date"])[:10]
    ref_d = pd.Timestamp(ref)

    out: list[dict[str, Any]] = []
    for name, col, start, family in GATES:
        if col not in track_df.columns:
            continue
        nav = pd.to_numeric(track_df[col], errors="coerce").dropna()
        if nav.empty:
            continue                                   # sleeve not feeding yet
        n_trials = int(n_trials_fn(family) or 1)
        dsr, n_obs = _per_bar_dsr(nav, n_trials, min_obs)
        rets = nav.pct_change().dropna()
        sharpe = (float(rets.mean() / rets.std(ddof=1)) * (252 ** 0.5)
                  if len(rets) > 1 and rets.std(ddof=1) else None)
        elapsed = (ref_d - pd.Timestamp(start)).days
        out.append({
            "name": name, "start": start,
            "days_elapsed": elapsed,
            "days_remaining": GATE_WINDOW_DAYS - elapsed,
            "window_days": GATE_WINDOW_DAYS,
            "n_obs": n_obs, "sharpe": sharpe, "dsr": dsr, "n_trials": n_trials,
        })
    return out


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
