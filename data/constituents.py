"""S&P 500 universe loader for the cross-sectional book (vía C, v1).

Loads **today's** S&P 500 constituents from a static CSV checked into the repo
(``data/sp500_constituents.csv``) — deliberately NOT a runtime scrape (a flaky web
call must never block a rebalance). Refresh the CSV periodically out-of-band.

Honesty note: this is the *current* membership. A historical backtest over it is
**survivorship-biased** (today's index excludes the names that were dropped/delisted)
and is therefore only a plumbing smoke test, never an edge claim — see
docs/analysis/2026-06-04-stock-picking-feasibility.md §4. The clean validation is the
forward paper track record on this list. A point-in-time, survivorship-free panel is a
paid-data upgrade (ML v2).

Ticker convention: the CSV uses dotted class shares (``BRK.B``); Alpaca expects the dot,
yfinance expects a dash (``BRK-B``). ``load_sp500(for_yfinance=True)`` converts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

_CSV_PATH = Path(__file__).resolve().parent / "sp500_constituents.csv"


def load_sp500(csv_path: Optional[str] = None, for_yfinance: bool = False) -> list[str]:
    """Load today's S&P 500 tickers from the static constituents CSV.

    Args:
        csv_path: Override path to the constituents CSV (defaults to the bundled file).
        for_yfinance: If True, convert dotted class shares (``BRK.B``) to the
            dash form yfinance uses (``BRK-B``). Leave False for Alpaca.

    Returns:
        De-duplicated ticker list in CSV order.
    """
    path = Path(csv_path) if csv_path else _CSV_PATH
    df = pd.read_csv(path)
    symbols = [str(s).strip() for s in df["Symbol"].tolist() if str(s).strip()]
    if for_yfinance:
        symbols = [s.replace(".", "-") for s in symbols]
    seen: dict[str, None] = {}
    for s in symbols:
        seen.setdefault(s, None)
    return list(seen)


def load_many(
    symbols: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: str = "1Day",
    lookback_bars: Optional[int] = None,
    loader: Optional[Callable[..., pd.DataFrame]] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for a list of symbols, skipping any that fail to load.

    A thin batch wrapper over the existing single-symbol loader (a missing/illiquid
    name should not abort a 500-name fetch). Names whose load raises or returns empty
    are silently dropped; the caller ranks only what loaded.

    Args:
        symbols: Tickers to fetch.
        start, end, timeframe, lookback_bars: Passed through to the per-symbol loader.
        loader: Per-symbol loader (defaults to ``data.market_data.load_ohlcv``);
            injectable for tests so the batch path needs no network.

    Returns:
        ``{symbol: OHLCV}`` for the names that loaded non-empty.
    """
    if loader is None:
        from data.market_data import load_ohlcv as loader  # lazy: keep import light

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = loader(sym, start=start, end=end, timeframe=timeframe,
                        lookback_bars=lookback_bars)
        except Exception:
            continue
        if df is not None and not df.empty:
            out[sym] = df
    return out
