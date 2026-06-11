"""Alpaca API wrapper.

Thin adapter around alpaca-py for account, market-data, and trading endpoints.
Centralizes auth, paper/live endpoint selection, and connection lifecycle.

**alpaca-py is imported lazily** (inside :meth:`AlpacaClient.connect` and the
data helpers) so the keyless backtest path — ``main.py backtest`` →
``data.market_data`` → this module — never hard-depends on the trading SDK.

**Safety:** paper trading is the default. Switching to live
(``paper=False``) requires an explicit typed confirmation
(``"YES I UNDERSTAND THE RISKS"``) via an injectable callback, so nothing in the
backtest/test path can silently instantiate a live trading client.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

LIVE_CONFIRM_PHRASE = "YES I UNDERSTAND THE RISKS"


@dataclass
class AlpacaConfig:
    """Connection config for Alpaca (sourced from .env / credentials.yaml)."""

    api_key: str
    secret_key: str
    paper: bool = True
    base_url: Optional[str] = None
    data_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        """Build config from environment variables (loads ``.env`` if present).

        Reads ``ALPACA_API_KEY``, ``ALPACA_SECRET_KEY``, ``ALPACA_PAPER``.
        Secrets are never hardcoded; ``.env`` is gitignored.

        Returns:
            Populated :class:`AlpacaConfig`.

        Raises:
            ValueError: If the API key or secret is missing.
        """
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover
            pass
        key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() != "false"
        if not key or not secret:
            raise ValueError(
                "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY (set them in .env)"
            )
        return cls(api_key=key, secret_key=secret, paper=paper)


class AlpacaClient:
    """Wrapper over Alpaca trading + data clients."""

    def __init__(
        self,
        config: AlpacaConfig,
        confirm_fn: Callable[[str], str] = input,
        max_retries: int = 5,
        backoff_base: float = 0.5,
    ) -> None:
        """Initialize the Alpaca client (does not connect yet).

        Args:
            config: API credentials and endpoint selection.
            confirm_fn: Prompt callback for the live-trading confirmation
                (injected in tests). Defaults to the builtin ``input``.
            max_retries: Max attempts for the health check / reconnect.
            backoff_base: Base seconds for exponential backoff.
        """
        self.config = config
        self.confirm_fn = confirm_fn
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._trading_client: Optional[Any] = None
        self._data_client: Optional[Any] = None

    # ----------------------------------------------------------- lifecycle ---
    def connect(self) -> None:
        """Instantiate the trading/data clients and verify auth.

        For live trading (``paper=False``) the user must type the exact
        confirmation phrase first. A health check (``get_account``) runs on
        startup, retried with exponential backoff.

        Raises:
            PermissionError: If live confirmation fails.
            ConnectionError: If the health check fails after all retries.
        """
        if not self.config.paper:
            self._require_live_confirmation()

        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self._trading_client = TradingClient(
            self.config.api_key, self.config.secret_key, paper=self.config.paper
        )
        self._data_client = StockHistoricalDataClient(
            self.config.api_key, self.config.secret_key
        )
        mode = "PAPER" if self.config.paper else "LIVE"
        logger.info("Alpaca client connecting in %s mode ...", mode)
        self._health_check()

    def _require_live_confirmation(self) -> None:
        """Gate live trading behind an exact typed confirmation phrase.

        Raises:
            PermissionError: If the typed phrase does not match exactly.
        """
        prompt = (
            "⚠️  LIVE TRADING MODE. "
            f"Type '{LIVE_CONFIRM_PHRASE}' to confirm: "
        )
        answer = self.confirm_fn(prompt)
        if answer.strip() != LIVE_CONFIRM_PHRASE:
            raise PermissionError("Live trading not confirmed; aborting connect")
        logger.warning("LIVE TRADING CONFIRMED by operator")

    def _health_check(self) -> None:
        """Verify connectivity by fetching the account, with backoff retries.

        Raises:
            ConnectionError: If all attempts fail.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                acct = self._trading_client.get_account()
                logger.info("Alpaca health check OK (status=%s, equity=%s)",
                            getattr(acct, "status", "?"), getattr(acct, "equity", "?"))
                return
            except Exception as exc:  # noqa: BLE001 - surface as ConnectionError
                last_exc = exc
                wait = self.backoff_base * (2 ** attempt)
                logger.warning("Health check attempt %d/%d failed: %s (retry in %.1fs)",
                               attempt + 1, self.max_retries, exc, wait)
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        raise ConnectionError(f"Alpaca health check failed after "
                              f"{self.max_retries} attempts: {last_exc}")

    def _retry(self, fn: Callable[[], Any], what: str) -> Any:
        """Run ``fn`` with exponential-backoff retries (auto-reconnect helper).

        Args:
            fn: Zero-arg callable performing the API call.
            what: Description for logging.

        Returns:
            The callable's result.

        Raises:
            ConnectionError: If all attempts fail.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = self.backoff_base * (2 ** attempt)
                logger.warning("%s failed (%d/%d): %s", what, attempt + 1,
                               self.max_retries, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        raise ConnectionError(f"{what} failed after {self.max_retries} attempts: {last_exc}")

    @property
    def trading(self) -> Any:
        """The underlying trading client (raises if not connected)."""
        if self._trading_client is None:
            raise RuntimeError("AlpacaClient not connected; call connect() first")
        return self._trading_client

    @property
    def data(self) -> Any:
        """The underlying market-data client (raises if not connected)."""
        if self._data_client is None:
            raise RuntimeError("AlpacaClient not connected; call connect() first")
        return self._data_client

    # ------------------------------------------------------------- account ---
    def get_account(self) -> dict[str, Any]:
        """Fetch account snapshot (equity, buying power, status).

        Returns:
            Account fields as a dict.
        """
        acct = self._retry(self.trading.get_account, "get_account")
        return {
            "equity": float(acct.equity),
            "last_equity": float(getattr(acct, "last_equity", 0.0) or 0.0),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "status": str(acct.status),
            "multiplier": float(getattr(acct, "multiplier", 1.0)),
            "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
        }

    def get_available_margin(self) -> float:
        """Available margin (buying power) for new positions.

        Returns:
            Buying power in account currency.
        """
        return self.get_account()["buying_power"]

    def get_positions(self) -> list[dict[str, Any]]:
        """Fetch all open positions.

        Returns:
            List of position dicts (symbol, qty, avg_entry_price, market_value,
            unrealized_pl, current_price).
        """
        positions = self._retry(self.trading.get_all_positions, "get_all_positions")
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "current_price": float(p.current_price),
                "side": str(p.side),
            }
            for p in positions
        ]

    def get_order_history(self, limit: int = 100, status: str = "all") -> list[dict[str, Any]]:
        """Fetch recent orders.

        Args:
            limit: Max orders to return.
            status: ``open``, ``closed``, or ``all``.

        Returns:
            List of order dicts.
        """
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }
        req = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.ALL), limit=limit)
        orders = self._retry(lambda: self.trading.get_orders(filter=req), "get_orders")
        return [
            {
                "id": str(o.id),
                "client_order_id": o.client_order_id,
                "symbol": o.symbol,
                "qty": float(o.qty) if o.qty else 0.0,
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
                "side": str(o.side),
                "type": str(o.order_type),
                "status": str(o.status),
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else 0.0,
            }
            for o in orders
        ]

    # ------------------------------------------------------------- market ---
    def get_clock(self) -> dict[str, Any]:
        """Fetch the market clock.

        Returns:
            Dict with ``is_open``, ``next_open``, ``next_close``.
        """
        clock = self._retry(self.trading.get_clock, "get_clock")
        return {
            "is_open": bool(clock.is_open),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    def is_market_open(self) -> bool:
        """Whether the market is currently open.

        Returns:
            True if open.
        """
        return self.get_clock()["is_open"]

    def get_historical_bars(
        self, symbol: str, timeframe: str, start: str, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from Alpaca.

        Args:
            symbol: Ticker.
            timeframe: Bar size (e.g. "1Day", "1Hour", "1Min").
            start: ISO start date.
            end: ISO end date (defaults to now).

        Returns:
            OHLCV DataFrame (lowercase columns) indexed by timestamp.
        """
        from alpaca.data.requests import StockBarsRequest

        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=_to_timeframe(timeframe),
            start=pd.Timestamp(start), end=pd.Timestamp(end) if end else None,
        )
        bars = self._retry(lambda: self.data.get_stock_bars(req), "get_stock_bars")
        df = bars.df
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex):  # (symbol, timestamp)
            df = df.xs(symbol, level=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]]

    def get_latest_quote(self, symbol: str) -> dict[str, float]:
        """Fetch the latest bid/ask for a symbol.

        Args:
            symbol: Ticker.

        Returns:
            Dict with ``bid``, ``ask``, ``bid_size``, ``ask_size``, ``spread_pct``.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self._retry(lambda: self.data.get_stock_latest_quote(req), "get_latest_quote")
        q = quotes[symbol]
        bid, ask = float(q.bid_price), float(q.ask_price)
        mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
        return {
            "bid": bid, "ask": ask,
            "bid_size": float(q.bid_size), "ask_size": float(q.ask_size),
            "spread_pct": (ask - bid) / mid if mid > 0 else float("inf"),
        }


def _to_timeframe(timeframe: str):
    """Map a settings timeframe string to an alpaca-py ``TimeFrame``.

    Args:
        timeframe: e.g. "1Min", "5Min", "15Min", "1Hour", "1Day".

    Returns:
        An alpaca ``TimeFrame`` instance.
    """
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    mapping = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    return mapping.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))
