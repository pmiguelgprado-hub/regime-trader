"""Tests for the gate-evidence heartbeat + Telegram alert channel (T0.5).

Unattended 12-month gates need a nervous system: if the daily track-record
recorder dies, every silent day is unrecoverable evidence loss. The heartbeat
(hosted in --risk-check, which launchd fires 24/7) alerts once per day when the
last recorded row is older than the configured business-day budget.
"""

from __future__ import annotations

import json

import pytest

from core import track_record as tr
from monitoring.alerts import AlertConfig, AlertManager, AlertSeverity


# --- staleness in business days ----------------------------------------------------


def _csv(tmp_path, last_date: str) -> str:
    p = tmp_path / "track.csv"
    p.write_text("date,book_nav,spy_nav,ew_nav\n"
                 f"{last_date},100000,100000,100000\n")
    return str(p)


def test_staleness_same_day_is_zero(tmp_path) -> None:
    assert tr.staleness_bdays(_csv(tmp_path, "2026-06-12"), "2026-06-12") == 0


def test_staleness_friday_to_monday_is_one_bday(tmp_path) -> None:
    # 2026-06-12 is a Friday; Monday is one business day later, weekend free
    assert tr.staleness_bdays(_csv(tmp_path, "2026-06-12"), "2026-06-15") == 1


def test_staleness_counts_business_days(tmp_path) -> None:
    # Fri 12th -> Thu 18th: Mon, Tue, Wed, Thu = 4 business days
    assert tr.staleness_bdays(_csv(tmp_path, "2026-06-12"), "2026-06-18") == 4


def test_staleness_missing_file_is_none(tmp_path) -> None:
    assert tr.staleness_bdays(str(tmp_path / "nope.csv"), "2026-06-12") is None


# --- telegram channel ---------------------------------------------------------------


def _manager(monkeypatch, **cfg_kwargs) -> tuple[AlertManager, list[str]]:
    sent: list[str] = []
    mgr = AlertManager(AlertConfig(**cfg_kwargs))
    monkeypatch.setattr(mgr, "_send_telegram", lambda text: sent.append(text))
    return mgr, sent


def test_telegram_dispatches_critical(monkeypatch) -> None:
    mgr, sent = _manager(monkeypatch, telegram_enabled=True,
                         telegram_token="t", telegram_chat_id="c")
    mgr.send("boom", "it broke", AlertSeverity.CRITICAL)
    assert len(sent) == 1 and "boom" in sent[0]


def test_telegram_skips_info_below_min_severity(monkeypatch) -> None:
    mgr, sent = _manager(monkeypatch, telegram_enabled=True,
                         telegram_token="t", telegram_chat_id="c")
    mgr.send("fyi", "nothing urgent", AlertSeverity.INFO)
    assert sent == []


def test_telegram_disabled_never_sends(monkeypatch) -> None:
    mgr, sent = _manager(monkeypatch, telegram_enabled=False,
                         telegram_token="t", telegram_chat_id="c")
    mgr.send("boom", "x", AlertSeverity.CRITICAL)
    assert sent == []


def test_telegram_reads_credentials_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "env-chat")
    mgr = AlertManager(AlertConfig(telegram_enabled=True))
    assert mgr.config.telegram_token == "env-token"
    assert mgr.config.telegram_chat_id == "env-chat"


def test_telegram_failure_never_raises(monkeypatch) -> None:
    mgr, _ = _manager(monkeypatch, telegram_enabled=True,
                      telegram_token="t", telegram_chat_id="c")
    monkeypatch.setattr(mgr, "_send_telegram",
                        lambda text: (_ for _ in ()).throw(OSError("net down")))
    assert mgr.send("boom", "x", AlertSeverity.CRITICAL) is True  # other channels fine


# --- heartbeat check (host glue, testable) ------------------------------------------


def test_heartbeat_alerts_once_per_day(tmp_path) -> None:
    import main as m

    csv = _csv(tmp_path, "2026-06-12")
    state = tmp_path / "hb.json"
    cfg: dict = {"monitoring": {}}
    # Thu 18th: 4 bdays stale > 2 -> alert + state written
    assert m._heartbeat_check(cfg, None, csv_path=csv, state_path=str(state),
                              today="2026-06-18") is True
    assert json.loads(state.read_text())["alerted"] == "2026-06-18"
    # same day again -> deduped
    assert m._heartbeat_check(cfg, None, csv_path=csv, state_path=str(state),
                              today="2026-06-18") is False
    # next day -> alerts again
    assert m._heartbeat_check(cfg, None, csv_path=csv, state_path=str(state),
                              today="2026-06-19") is True


def test_heartbeat_quiet_when_fresh(tmp_path) -> None:
    import main as m

    csv = _csv(tmp_path, "2026-06-12")
    state = tmp_path / "hb.json"
    assert m._heartbeat_check({"monitoring": {}}, None, csv_path=csv,
                              state_path=str(state), today="2026-06-15") is False
    assert not state.exists()


def test_heartbeat_quiet_when_no_file_yet(tmp_path) -> None:
    import main as m

    assert m._heartbeat_check({"monitoring": {}}, None,
                              csv_path=str(tmp_path / "nope.csv"),
                              state_path=str(tmp_path / "hb.json"),
                              today="2026-06-18") is False
