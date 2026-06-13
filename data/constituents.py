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
_UNIVERSE_DIR = Path(__file__).resolve().parent / "universe"


def _resolve_csv(csv_path: Optional[str], as_of: Optional[str],
                 universe_dir: Optional[str]) -> Path:
    """Pick the constituents CSV to load (T5.3 point-in-time resolution).

    Precedence: an explicit ``csv_path`` always wins. Otherwise, if ``as_of`` is
    given, use the latest monthly snapshot under ``universe_dir`` whose month is
    ``<= as_of`` (``YYYY-MM-constituents.csv``); if none exists at/before that date,
    fall back to the bundled current CSV (documented survivorship degradation, not a
    crash). With no ``as_of``, use the bundled current CSV.
    """
    if csv_path:
        return Path(csv_path)
    if as_of:
        udir = Path(universe_dir) if universe_dir else _UNIVERSE_DIR
        target = str(as_of)[:7]                        # YYYY-MM
        if udir.exists():
            snaps = sorted(p for p in udir.glob("*-constituents.csv")
                           if p.name[:7] <= target)
            if snaps:
                return snaps[-1]
    return _CSV_PATH


def snapshot_universe(month: str, src_csv: Optional[str] = None,
                      universe_dir: Optional[str] = None) -> Path:
    """Write a point-in-time membership snapshot for ``month`` (``YYYY-MM``).

    Copies the current constituents CSV (Symbol + GICS Sector) into
    ``universe_dir/<month>-constituents.csv``. Run monthly: the past is
    survivorship-biased, but every snapshot from now on is survivorship-free
    forward. Idempotent (overwrites the same month).

    Returns:
        The snapshot path written.
    """
    src = Path(src_csv) if src_csv else _CSV_PATH
    udir = Path(universe_dir) if universe_dir else _UNIVERSE_DIR
    udir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    cols = [c for c in ("Symbol", "GICS Sector") if c in df.columns]
    out = udir / f"{month}-constituents.csv"
    df[cols].to_csv(out, index=False)
    return out


def ensure_snapshot(month: str, src_csv: Optional[str] = None,
                    universe_dir: Optional[str] = None) -> Path:
    """Freeze ``month``'s PIT universe iff not already frozen (write-if-absent).

    Call at the first rebalance of a month: the snapshot captures the membership as
    it was then, so a later refresh of the bundled CSV cannot retroactively change a
    month already under evaluation. Returns the (existing or newly written) path.
    """
    udir = Path(universe_dir) if universe_dir else _UNIVERSE_DIR
    out = udir / f"{month}-constituents.csv"
    if out.exists():
        return out
    return snapshot_universe(month, src_csv=src_csv, universe_dir=universe_dir)


def load_sp500(csv_path: Optional[str] = None, for_yfinance: bool = False,
               as_of: Optional[str] = None, universe_dir: Optional[str] = None) -> list[str]:
    """Load S&P 500 tickers from the constituents CSV (point-in-time aware).

    Args:
        csv_path: Override path to the constituents CSV (defaults to bundled/snapshot).
        for_yfinance: If True, convert dotted class shares (``BRK.B``) to the
            dash form yfinance uses (``BRK-B``). Leave False for Alpaca.
        as_of: ISO date — resolve to the latest monthly snapshot at/before it (T5.3).
        universe_dir: Override the snapshot directory (tests).

    Returns:
        De-duplicated ticker list in CSV order.
    """
    path = _resolve_csv(csv_path, as_of, universe_dir)
    df = pd.read_csv(path)
    symbols = [str(s).strip() for s in df["Symbol"].tolist() if str(s).strip()]
    if for_yfinance:
        symbols = [s.replace(".", "-") for s in symbols]
    seen: dict[str, None] = {}
    for s in symbols:
        seen.setdefault(s, None)
    return list(seen)


def load_sector_map(csv_path: Optional[str] = None, as_of: Optional[str] = None,
                    universe_dir: Optional[str] = None) -> dict[str, str]:
    """Map each S&P 500 ticker to its GICS sector (for the sector cap).

    Args:
        csv_path: Override path to the constituents CSV (defaults to bundled/snapshot).
        as_of: ISO date — resolve to the latest monthly snapshot at/before it (T5.3).
        universe_dir: Override the snapshot directory (tests).

    Returns:
        ``{ticker: sector}``; tickers with no sector fall back to ``"UNKNOWN"``.
    """
    path = _resolve_csv(csv_path, as_of, universe_dir)
    df = pd.read_csv(path)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip()
        if sym:
            sec = str(row.get("GICS Sector", "")).strip() or "UNKNOWN"
            out[sym] = sec
    return out


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
