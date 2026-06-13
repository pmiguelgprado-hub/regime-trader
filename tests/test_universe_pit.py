"""Tests for the point-in-time universe snapshots (T5.3).

A backtest over *today's* constituents is survivorship-biased. We can't fix the
past, but we can start a perfect forward record now: monthly snapshots of the
membership under data/universe/, resolved by ``as_of`` (latest snapshot whose
month <= as_of). Without a snapshot for/before the date, fall back to the bundled
current CSV (documented degradation, not a crash).
"""

from __future__ import annotations

import pandas as pd

from data import constituents as c


def _write_universe(tmp_path, month: str, symbols: list[str]) -> None:
    d = tmp_path / "universe"
    d.mkdir(exist_ok=True)
    pd.DataFrame({"Symbol": symbols,
                  "GICS Sector": ["Information Technology"] * len(symbols)}
                 ).to_csv(d / f"{month}-constituents.csv", index=False)


def test_snapshot_writes_month_file(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "GICS Sector": ["IT", "IT"]}).to_csv(src, index=False)
    out = c.snapshot_universe("2026-06", src_csv=str(src), universe_dir=str(tmp_path / "universe"))
    assert out.exists()
    assert list(pd.read_csv(out)["Symbol"]) == ["AAPL", "MSFT"]


def test_as_of_resolves_latest_snapshot_at_or_before(tmp_path):
    _write_universe(tmp_path, "2026-03", ["AAPL", "OLD"])
    _write_universe(tmp_path, "2026-06", ["AAPL", "NEW"])
    # as-of May -> the March snapshot (latest <= May), NOT June
    syms = c.load_sp500(as_of="2026-05-15", universe_dir=str(tmp_path / "universe"))
    assert "OLD" in syms and "NEW" not in syms
    # as-of July -> the June snapshot
    syms = c.load_sp500(as_of="2026-07-01", universe_dir=str(tmp_path / "universe"))
    assert "NEW" in syms and "OLD" not in syms


def test_as_of_before_any_snapshot_falls_back_to_current(tmp_path):
    _write_universe(tmp_path, "2026-06", ["NEW"])
    # as-of before the earliest snapshot -> bundled current CSV (real ~500 names)
    syms = c.load_sp500(as_of="2020-01-01", universe_dir=str(tmp_path / "universe"))
    assert "AAPL" in syms and len(syms) >= 480


def test_as_of_none_uses_current_bundled(tmp_path):
    _write_universe(tmp_path, "2026-06", ["NEW"])
    syms = c.load_sp500(universe_dir=str(tmp_path / "universe"))   # no as_of
    assert "AAPL" in syms and "NEW" not in syms


def test_sector_map_honors_as_of(tmp_path):
    d = tmp_path / "universe"
    d.mkdir()
    pd.DataFrame({"Symbol": ["ZZZ"], "GICS Sector": ["Energy"]}
                ).to_csv(d / "2026-06-constituents.csv", index=False)
    sm = c.load_sector_map(as_of="2026-07-01", universe_dir=str(d))
    assert sm.get("ZZZ") == "Energy"
