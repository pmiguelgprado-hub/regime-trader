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


# --- T0.1: additive challenger_nav column (gate feed) + T0.3 code_sha ---------------


def test_challenger_seeds_at_book_equity_on_adoption(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-12", book_equity=100_000.0, spy_ret=0.0, ew_ret=0.0,
                  challenger_ret=0.05)
    row = tr.load_track_record(str(p)).iloc[0]
    # first challenger observation: seeded at book equity, the day's ret is NOT applied
    assert row["challenger_nav"] == 100_000.0


def test_challenger_chains_from_prior_level(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-12", 100_000.0, spy_ret=0.0, ew_ret=0.0,
                  challenger_ret=0.0)
    tr.append_day(str(p), "2026-06-13", 101_000.0, spy_ret=0.0, ew_ret=0.0,
                  challenger_ret=0.02)
    last = tr.load_track_record(str(p)).iloc[-1]
    assert last["challenger_nav"] == pytest.approx(102_000.0)  # 100k * 1.02


def test_challenger_adoption_mid_series_preserves_old_rows(tmp_path) -> None:
    """Legacy 4-column CSV gains the challenger column additively (rows immutable)."""
    p = tmp_path / "track.csv"
    p.write_text("date,book_nav,spy_nav,ew_nav\n"
                 "2026-06-04,101089.66,101089.66,101089.66\n")
    tr.append_day(str(p), "2026-06-12", 97_000.0, spy_ret=0.01, ew_ret=0.01,
                  challenger_ret=0.03)
    df = tr.load_track_record(str(p))
    assert len(df) == 2
    # legacy row values untouched, challenger empty there
    assert df.iloc[0]["book_nav"] == 101089.66
    assert math.isnan(float(df.iloc[0]["challenger_nav"]))
    # adoption row: challenger seeded at that day's book equity (no prior level to chain)
    assert df.iloc[-1]["challenger_nav"] == 97_000.0
    # benchmark chaining still works across the legacy row
    assert df.iloc[-1]["spy_nav"] == pytest.approx(101089.66 * 1.01)


def test_challenger_none_records_gap_then_reseeds(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-12", 100_000.0, spy_ret=0.0, ew_ret=0.0)  # no challenger
    df = tr.load_track_record(str(p))
    assert math.isnan(float(df.iloc[0]["challenger_nav"]))
    tr.append_day(str(p), "2026-06-13", 99_000.0, spy_ret=0.0, ew_ret=0.0,
                  challenger_ret=0.04)
    df = tr.load_track_record(str(p))
    # cannot chain from NaN -> reseed at current book equity
    assert df.iloc[-1]["challenger_nav"] == 99_000.0


def test_code_sha_recorded_per_row(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-12", 100_000.0, spy_ret=0.0, ew_ret=0.0,
                  code_sha="abc1234")
    df = tr.load_track_record(str(p))
    assert df.iloc[0]["code_sha"] == "abc1234"


def test_portfolio_return_weights_times_rets() -> None:
    w = {"AAA": 0.5, "BBB": 0.3}                 # 0.2 implicit cash at 0 return
    r = {"AAA": 0.10, "BBB": -0.05}
    assert tr.portfolio_return(w, r) == pytest.approx(0.5 * 0.10 + 0.3 * -0.05)


def test_portfolio_return_missing_ret_counts_as_cash() -> None:
    w = {"AAA": 0.5, "BBB": 0.5}
    r = {"AAA": 0.10}                            # BBB ret unknown -> contributes 0
    assert tr.portfolio_return(w, r) == pytest.approx(0.05)


def test_portfolio_return_empty_weights_is_none() -> None:
    assert tr.portfolio_return({}, {"AAA": 0.1}) is None


def test_challenger_weights_from_snapshot(tmp_path) -> None:
    snap = tmp_path / "book_snapshot_challenger.json"
    snap.write_text('{"targets": [{"symbol": "GOOG", "weight": 0.1, "price": 365.76},'
                    ' {"symbol": "AMD", "weight": 0.2, "price": 466.38}]}')
    assert tr.challenger_weights(str(snap)) == {"GOOG": 0.1, "AMD": 0.2}


def test_challenger_weights_missing_file_is_empty(tmp_path) -> None:
    assert tr.challenger_weights(str(tmp_path / "nope.json")) == {}


# --- quality book NAV (T2.1 deploy): same additive pattern as challenger ------------


def test_quality_seeds_then_chains(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-13", 100_000.0, spy_ret=0.0, ew_ret=0.0,
                  quality_ret=0.03)
    tr.append_day(str(p), "2026-06-14", 101_000.0, spy_ret=0.0, ew_ret=0.0,
                  quality_ret=0.02)
    df = tr.load_track_record(str(p))
    assert df.iloc[0]["quality_nav"] == 100_000.0            # seed, ret not applied
    assert df.iloc[-1]["quality_nav"] == pytest.approx(102_000.0)  # 100k * 1.02


def test_quality_and_challenger_independent_columns(tmp_path) -> None:
    p = tmp_path / "track.csv"
    tr.append_day(str(p), "2026-06-13", 100_000.0, spy_ret=0.0, ew_ret=0.0,
                  challenger_ret=0.05, quality_ret=0.01)
    row = tr.load_track_record(str(p)).iloc[0]
    assert row["challenger_nav"] == 100_000.0 and row["quality_nav"] == 100_000.0


def test_snapshot_weights_alias_of_challenger_weights(tmp_path) -> None:
    snap = tmp_path / "book_snapshot_quality.json"
    snap.write_text('{"targets": [{"symbol": "AAPL", "weight": 0.2}]}')
    assert tr.snapshot_weights(str(snap)) == {"AAPL": 0.2}
    assert tr.snapshot_weights is tr.challenger_weights or \
        tr.snapshot_weights(str(snap)) == tr.challenger_weights(str(snap))
