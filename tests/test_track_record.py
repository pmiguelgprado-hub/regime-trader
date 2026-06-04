"""Tests for the daily track-record recorder (no network — pure + tmp CSV).

The recorder accumulates the three NAV series the frozen gate needs (book,
EW-S&P500, SPY), seeded equal at the book's day-1 equity so the curves are
directly comparable. Without this, ``book_snapshot.json`` is overwritten each run
and the 12-month forward test has no daily series to evaluate.
"""

from __future__ import annotations

import math

import pytest

from core import track_record as tr


def test_simple_return() -> None:
    assert tr.simple_return(100.0, 105.0) == pytest.approx(0.05)
    assert tr.simple_return(0.0, 105.0) == 0.0  # guard against div0


def test_append_seeds_three_levels_equal_on_first_day(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-04", book_equity=100_000.0, spy_ret=0.05, ew_ret=0.03)
    df = tr.load_track_record(str(p))
    assert len(df) == 1
    row = df.iloc[0]
    # day 1: no prior -> all three seeded at book equity (rets ignored)
    assert row["book_nav"] == 100_000.0
    assert row["spy_nav"] == 100_000.0
    assert row["ew_nav"] == 100_000.0


def test_append_chains_levels_from_prior_day(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-04", book_equity=100_000.0, spy_ret=0.0, ew_ret=0.0)
    tr.append_day(str(p), "2026-06-05", book_equity=101_000.0, spy_ret=0.02, ew_ret=0.01)
    df = tr.load_track_record(str(p))
    assert len(df) == 2
    last = df.iloc[-1]
    assert last["book_nav"] == 101_000.0            # book is the real equity level
    assert last["spy_nav"] == pytest.approx(102_000.0)   # 100k * 1.02
    assert last["ew_nav"] == pytest.approx(101_000.0)    # 100k * 1.01


def test_append_is_idempotent_on_repeated_date(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-04", book_equity=100_000.0, spy_ret=0.0, ew_ret=0.0)
    tr.append_day(str(p), "2026-06-04", book_equity=999_999.0, spy_ret=0.9, ew_ret=0.9)
    df = tr.load_track_record(str(p))
    assert len(df) == 1                              # no duplicate row for the same day
    assert df.iloc[0]["book_nav"] == 100_000.0       # first write wins, re-run is a no-op


def test_load_missing_file_is_empty(tmp_path) -> None:
    df = tr.load_track_record(str(tmp_path / "nope.csv"))
    assert len(df) == 0
