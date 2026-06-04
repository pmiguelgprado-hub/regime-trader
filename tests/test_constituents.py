"""Tests for the S&P 500 universe loader (vía C, v1)."""

from __future__ import annotations

import pandas as pd

from data.constituents import load_many, load_sector_map, load_sp500


def test_load_sp500_returns_large_cap_names() -> None:
    """The bundled CSV yields ~500 names including the obvious mega-caps."""
    syms = load_sp500()
    assert 480 <= len(syms) <= 520
    for mega in ("AAPL", "MSFT", "NVDA"):
        assert mega in syms


def test_load_sp500_yfinance_converts_dotted_tickers() -> None:
    """Dotted class shares (BRK.B) become dash form (BRK-B) for yfinance."""
    alpaca = load_sp500()
    yf = load_sp500(for_yfinance=True)
    assert "BRK.B" in alpaca and "BRK.B" not in yf
    assert "BRK-B" in yf


def test_load_sector_map_maps_known_tickers() -> None:
    sm = load_sector_map()
    assert sm.get("NVDA") == "Information Technology"
    assert sm.get("JPM") == "Financials"
    assert len(sm) >= 480
    # every value is a non-empty sector string
    assert all(isinstance(v, str) and v for v in sm.values())


def test_load_many_skips_failures_and_empties() -> None:
    """A symbol that raises or returns empty is dropped, not fatal."""
    good = pd.DataFrame({"close": [1.0, 2.0]})

    def fake_loader(sym, **kwargs):
        if sym == "BOOM":
            raise RuntimeError("no data")
        if sym == "EMPTY":
            return pd.DataFrame()
        return good

    out = load_many(["AAA", "BOOM", "EMPTY", "BBB"], loader=fake_loader)
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"].equals(good)
