"""Tests for the US macro event calendar (risk-timing helper, not alpha)."""

from __future__ import annotations

from datetime import date

from core.macro_calendar import (
    days_to_next_event,
    event_risk_scale,
    high_impact_events,
    in_event_window,
    next_event,
    nfp_dates,
)


def test_nfp_dates_are_twelve_first_fridays() -> None:
    days = nfp_dates(2026)
    assert len(days) == 12
    assert all(d.weekday() == 4 for d in days)          # all Fridays
    for d in days:                                       # each is the FIRST Friday
        assert d.day <= 7


def test_high_impact_events_merges_and_sorts() -> None:
    ev = high_impact_events(date(2026, 1, 1), date(2026, 12, 31))
    assert ("FOMC", ) and (date(2026, 1, 28), "FOMC") in ev
    names = {n for _, n in ev}
    assert {"FOMC", "NFP"} <= names
    dates = [d for d, _ in ev]
    assert dates == sorted(dates)                        # chronological


def test_next_event_and_days_to() -> None:
    # 2026-03-16 -> next high-impact event is the Mar 18 FOMC (2 days out).
    nxt = next_event(date(2026, 3, 16))
    assert nxt is not None and nxt[1] == "FOMC" and nxt[0] == date(2026, 3, 18)
    assert days_to_next_event(date(2026, 3, 16)) == 2


def test_in_event_window_flags_pre_event_days() -> None:
    flagged, label = in_event_window(date(2026, 3, 17), window_days=2)   # day before FOMC
    assert flagged and label == "FOMC"
    # A quiet stretch with no event within 2 days.
    assert in_event_window(date(2026, 3, 30), window_days=2)[0] is False


def test_event_risk_scale_derisks_only_in_window() -> None:
    assert event_risk_scale(date(2026, 3, 17), window_days=2, derisk=0.5) == 0.5
    assert event_risk_scale(date(2026, 3, 30), window_days=2, derisk=0.5) == 1.0
