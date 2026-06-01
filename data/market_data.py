"""Market data: real-time and historical fetching.

Provides a unified interface over Alpaca historical bars and the live
websocket stream, with caching and gap handling.

NOTE: Skeleton only — no logic implemented yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# yfinance interval map (used by the offline backtest data loader).
_YF_INTERVAL = {
    "1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "1h", "1Day": "1d",
}


def load_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: str = "1Day",
    lookback_bars: Optional[int] = None,
) -> pd.DataFrame:
    """Load historical OHLCV from yfinance (offline backtest data source).

    The live loop uses Alpaca via :class:`MarketData`; backtests use this free,
    keyless source so ``main.py backtest`` runs without credentials.

    Args:
        symbol: Ticker.
        start: ISO start date (inclusive). Ignored if ``lookback_bars`` given.
        end: ISO end date (inclusive).
        timeframe: Bar size (``1Day`` default; mapped to a yfinance interval).
        lookback_bars: If set, fetch roughly this many recent bars instead of a
            date range.

    Returns:
        OHLCV DataFrame with lowercase columns (open, high, low, close, volume),
        indexed by timestamp.

    Raises:
        ImportError: If yfinance is not installed.
        ValueError: If no data is returned.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance required for backtest data: pip install yfinance") from exc

    interval = _YF_INTERVAL.get(timeframe, "1d")
    kw: dict = dict(interval=interval, auto_adjust=True, progress=False)
    if lookback_bars:
        # Over-fetch by ~1.5x calendar days to clear weekends/holidays.
        kw["period"] = f"{int(lookback_bars * 1.6) + 5}d"
    elif start:
        kw["start"], kw["end"] = start, end
    else:
        # No range specified -> default to a decade so a walk-forward fits.
        kw["period"] = "10y"

    raw = yf.download(symbol, **kw)
    if raw is None or raw.empty:
        raise ValueError(f"no data returned for {symbol} ({start}..{end})")

    # yfinance returns a column MultiIndex (field, ticker) for single symbols.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    cols = ["open", "high", "low", "close", "volume"]
    df = raw[[c for c in cols if c in raw.columns]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    if lookback_bars:
        df = df.tail(lookback_bars)
    return df


@dataclass
class Bar:
    """A single OHLCV bar.

    Attributes:
        timestamp: Bar timestamp.
        open: Open price.
        high: High price.
        low: Low price.
        close: Close price.
        volume: Traded volume.
    """

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketData:
    """Fetches and caches historical + live market data."""

    def __init__(self, client: AlpacaClient) -> None:
        """Initialize the data layer.

        Args:
            client: Connected Alpaca client.
        """
        self.client = client
        self._cache: dict[str, pd.DataFrame] = {}

    def get_historical_bars(
        self, symbol: str, timeframe: str, start: str, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars over a date range (live broker source).

        Gaps from weekends, holidays, and halts are left as-is (no synthetic
        fill) — the bar index simply skips non-trading sessions, which is what
        the causal feature pipeline expects.

        Args:
            symbol: Ticker.
            timeframe: Bar size.
            start: ISO start date.
            end: ISO end date (defaults to now).

        Returns:
            OHLCV DataFrame indexed by timestamp.
        """
        return self.client.get_historical_bars(symbol, timeframe, start, end)

    def get_history(
        self, symbol: str, timeframe: str, lookback_bars: int
    ) -> pd.DataFrame:
        """Fetch the most recent ``lookback_bars`` of OHLCV for a symbol.

        Args:
            symbol: Ticker.
            timeframe: Bar size.
            lookback_bars: Number of bars to retrieve.

        Returns:
            OHLCV DataFrame indexed by timestamp.
        """
        # Over-reach on the calendar to clear weekends/holidays, then tail.
        days = int(lookback_bars * 1.6) + 10
        start = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).isoformat()
        df = self.client.get_historical_bars(symbol, timeframe, start)
        return df.tail(lookback_bars)

    def get_history_multi(
        self, symbols: list[str], timeframe: str, lookback_bars: int
    ) -> dict[str, pd.DataFrame]:
        """Fetch recent OHLCV for multiple symbols.

        Args:
            symbols: Tickers.
            timeframe: Bar size.
            lookback_bars: Number of bars per symbol.

        Returns:
            Map of symbol -> OHLCV DataFrame (symbols with no data are skipped).
        """
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.get_history(sym, timeframe, lookback_bars)
            except Exception as exc:  # noqa: BLE001 - skip a bad symbol, keep the rest
                logger.warning("history fetch failed for %s: %s", sym, exc)
        return out

    def get_latest_bar(self, symbol: str) -> Bar:
        """Fetch the most recent completed bar.

        Args:
            symbol: Ticker.

        Returns:
            Latest `Bar`.
        """
        from alpaca.data.requests import StockLatestBarRequest

        req = StockLatestBarRequest(symbol_or_symbols=symbol)
        bars = self.client.data.get_stock_latest_bar(req)
        b = bars[symbol]
        return Bar(timestamp=pd.Timestamp(b.timestamp), open=float(b.open),
                   high=float(b.high), low=float(b.low), close=float(b.close),
                   volume=float(b.volume))

    def get_latest_quote(self, symbol: str) -> dict[str, float]:
        """Fetch the latest bid/ask (for spread checks).

        Args:
            symbol: Ticker.

        Returns:
            Quote dict (bid, ask, sizes, spread_pct).
        """
        return self.client.get_latest_quote(symbol)

    def get_snapshot(self, symbol: str) -> dict[str, Any]:
        """Fetch a full snapshot (latest trade/quote/bar) for a symbol.

        Args:
            symbol: Ticker.

        Returns:
            Snapshot dict with last price, bid/ask, and the latest daily bar.
        """
        from alpaca.data.requests import StockSnapshotRequest

        req = StockSnapshotRequest(symbol_or_symbols=symbol)
        snap = self.client.data.get_stock_snapshot(req)[symbol]
        q, b = snap.latest_quote, snap.daily_bar
        return {
            "last_price": float(snap.latest_trade.price) if snap.latest_trade else 0.0,
            "bid": float(q.bid_price) if q else 0.0,
            "ask": float(q.ask_price) if q else 0.0,
            "day_close": float(b.close) if b else 0.0,
        }

    def subscribe_bars(
        self, symbols: list[str], callback: Callable[[str, Bar], None]
    ) -> None:  # pragma: no cover - live socket
        """Subscribe to the live bar stream (thin WebSocket plumbing).

        Args:
            symbols: Tickers to subscribe.
            callback: Invoked as ``callback(symbol, Bar)`` per incoming bar.
        """
        from alpaca.data.live import StockDataStream

        stream = StockDataStream(self.client.config.api_key, self.client.config.secret_key)

        async def _on_bar(bar):
            callback(bar.symbol, Bar(
                timestamp=pd.Timestamp(bar.timestamp), open=float(bar.open),
                high=float(bar.high), low=float(bar.low), close=float(bar.close),
                volume=float(bar.volume)))

        stream.subscribe_bars(_on_bar, *symbols)
        stream.run()

    def subscribe_quotes(
        self, symbols: list[str], callback: Callable[[str, dict], None]
    ) -> None:  # pragma: no cover - live socket
        """Subscribe to the live quote stream for spread checks.

        Args:
            symbols: Tickers to subscribe.
            callback: Invoked as ``callback(symbol, quote_dict)`` per quote.
        """
        from alpaca.data.live import StockDataStream

        stream = StockDataStream(self.client.config.api_key, self.client.config.secret_key)

        async def _on_quote(q):
            bid, ask = float(q.bid_price), float(q.ask_price)
            mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
            callback(q.symbol, {"bid": bid, "ask": ask,
                                "spread_pct": (ask - bid) / mid if mid > 0 else float("inf")})

        stream.subscribe_quotes(_on_quote, *symbols)
        stream.run()

    # Backwards-compatible alias for the original skeleton method name.
    subscribe_stream = subscribe_bars
