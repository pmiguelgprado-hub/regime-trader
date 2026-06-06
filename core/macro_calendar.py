"""US macro event calendar — scheduled high-volatility events (risk timing, NOT alpha).

Public macro events (FOMC decisions, payrolls) are *known in advance* and reliably spike
realized volatility around the release. This module does NOT predict direction (that would
need information nobody legally has, and public news is priced in within seconds) — it only
flags *when* a scheduled high-vol event is imminent so the book can de-risk around it. That
is legitimate **risk management**: trim exposure before a known vol event, restore it after.

Sources of dates:
* **FOMC** decision days — the Fed publishes the schedule; 2026 dates are hard-coded
  (verified against the Fed's calendar; update yearly).
* **Nonfarm payrolls (NFP)** — released the **first Friday** of each month (deterministic
  rule, computed not hard-coded).

CPI and other releases follow less rigid rules; add them to ``EXTRA_EVENTS`` as published.
Everything is plain ``datetime.date`` so the module is pure, offline, and unit-testable.
"""

from __future__ import annotations

from datetime import date, timedelta

# FOMC rate-decision days (2nd day of each meeting). Verified vs the Fed 2026 schedule.
FOMC_DATES: dict[int, list[date]] = {
    2026: [
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
        date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
    ],
}

# Manually-curated extra high-impact releases (e.g. CPI), as (date, label). Extend as the
# official schedules publish; left empty so nothing here is a guessed date.
EXTRA_EVENTS: list[tuple[date, str]] = []


def nfp_dates(year: int) -> list[date]:
    """Nonfarm-payroll release days for ``year`` — the first Friday of each month."""
    out: list[date] = []
    for month in range(1, 13):
        d = date(year, month, 1)
        offset = (4 - d.weekday()) % 7          # Friday == weekday 4
        out.append(d + timedelta(days=offset))
    return out


def high_impact_events(start: date, end: date) -> list[tuple[date, str]]:
    """All known high-impact US events in ``[start, end]``, sorted by date.

    Merges FOMC decision days, monthly NFP releases, and any ``EXTRA_EVENTS``.
    """
    events: list[tuple[date, str]] = []
    for year in range(start.year, end.year + 1):
        for d in FOMC_DATES.get(year, []):
            events.append((d, "FOMC"))
        for d in nfp_dates(year):
            events.append((d, "NFP"))
    events.extend(EXTRA_EVENTS)
    events = [(d, n) for d, n in events if start <= d <= end]
    events.sort(key=lambda dn: dn[0])
    return events


def next_event(today: date, horizon_days: int = 60) -> tuple[date, str] | None:
    """The next high-impact event on/after ``today`` within ``horizon_days`` (or None)."""
    upcoming = high_impact_events(today, today + timedelta(days=horizon_days))
    return upcoming[0] if upcoming else None


def days_to_next_event(today: date, horizon_days: int = 60) -> int | None:
    """Calendar days until the next high-impact event (0 = today), or None if none soon."""
    nxt = next_event(today, horizon_days)
    return (nxt[0] - today).days if nxt else None


def in_event_window(today: date, window_days: int = 2) -> tuple[bool, str]:
    """Whether ``today`` is within ``window_days`` *before* (or on) a high-impact event.

    Returns ``(True, label)`` if an event lands in ``[today, today+window_days]`` — the
    pre-event window where vol tends to build — else ``(False, "")``.
    """
    nxt = next_event(today, horizon_days=window_days)
    if nxt and 0 <= (nxt[0] - today).days <= window_days:
        return True, nxt[1]
    return False, ""


def event_risk_scale(today: date, window_days: int = 2, derisk: float = 0.5) -> float:
    """Gross multiplier for the macro-event risk overlay (1.0 normally).

    In the ``window_days`` before a scheduled high-impact event, return ``derisk`` (e.g.
    0.5 = halve exposure into a known vol event); otherwise 1.0. Pure risk timing — it does
    not look at, or bet on, the *outcome* of the event.
    """
    flagged, _ = in_event_window(today, window_days)
    return derisk if flagged else 1.0
