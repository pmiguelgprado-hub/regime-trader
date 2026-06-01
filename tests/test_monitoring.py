"""Tests for the monitoring package: logger, alerts, dashboard."""

from __future__ import annotations

import json

import pytest

from monitoring.alerts import AlertConfig, AlertManager, AlertSeverity
from monitoring.dashboard import Dashboard, DashboardConfig, DashboardState
from monitoring.logger import LoggerConfig, setup_logging


# ----------------------------------------------------------------- logger ---
def test_log_record_is_json_with_context(tmp_path) -> None:
    """Every record is JSON and carries the shared trading context."""
    tl = setup_logging(LoggerConfig(log_dir=str(tmp_path), console=False))
    tl.set_context(regime="bull", probability=0.72, equity=105230, positions=1, daily_pnl=340)
    tl.log(tl.trades, "rebalance", "SPY 60%->95%", symbol="SPY")

    line = (tmp_path / "trades.log").read_text().strip().splitlines()[-1]
    rec = json.loads(line)
    for k in ("timestamp", "regime", "probability", "equity", "positions", "daily_pnl", "event"):
        assert k in rec
    assert rec["event"] == "rebalance" and rec["regime"] == "bull" and rec["symbol"] == "SPY"


def test_rotating_handler_configured(tmp_path) -> None:
    """The file handler is a size-rotating handler (10 MB, 30 backups)."""
    from logging.handlers import RotatingFileHandler

    tl = setup_logging(LoggerConfig(log_dir=str(tmp_path), console=False,
                                    max_bytes=10 * 1024 * 1024, backup_count=30))
    h = next(h for h in tl.trades.handlers if isinstance(h, RotatingFileHandler))
    assert h.maxBytes == 10 * 1024 * 1024 and h.backupCount == 30


# ----------------------------------------------------------------- alerts ---
def test_alert_rate_limited_per_event(tmp_path) -> None:
    """One alert per event type per window; window expiry re-enables it."""
    clock = [1000.0]
    am = AlertManager(AlertConfig(rate_limit_minutes=15), clock=lambda: clock[0])
    assert am.send("regime_change", "x") is True
    assert am.send("regime_change", "x") is False      # within window
    assert am.send("circuit_breaker", "y") is True      # different event ok
    clock[0] += 15 * 60 + 1
    assert am.send("regime_change", "x") is True        # window elapsed


def test_alert_triggers_route_through_rate_limit() -> None:
    """Helper triggers respect the same per-event rate limit."""
    clock = [0.0]
    am = AlertManager(AlertConfig(), clock=lambda: clock[0])
    assert am.circuit_breaker("HALTED", 0.11) is True
    assert am.circuit_breaker("HALTED", 0.12) is False  # rate limited


# -------------------------------------------------------------- dashboard ---
def test_dashboard_renders_all_panels() -> None:
    """The renderable contains all six labelled panels."""
    db = Dashboard(DashboardConfig())
    state = DashboardState(
        regime_name="bull", regime_prob=0.72, stability_bars=14, flicker_rate=1,
        equity=105230, daily_pnl=340, daily_pnl_pct=0.0032, allocation=0.95, leverage=1.25,
        positions=[{"symbol": "SPY", "side": "LONG", "price": 520.3,
                    "pnl_pct": 0.012, "stop": 508, "holding": "3h"}],
        recent_signals=[{"time": "14:30", "symbol": "SPY", "change": "95%", "note": "Low vol"}],
        daily_dd=0.003, peak_dd=0.012, mode="PAPER",
    )
    out = db.render_to_str(state, width=90)
    for panel in ("REGIME", "PORTFOLIO", "POSITIONS", "RECENT SIGNALS", "RISK STATUS", "SYSTEM"):
        assert panel in out
    assert "PAPER" in out and "SPY" in out


def test_risk_bar_color_thresholds() -> None:
    """Risk bar marks scale green/amber/red with proximity to the limit."""
    from monitoring.dashboard import _risk_bar

    assert "✅" in str(_risk_bar(0.003, 0.03, "Daily DD").plain) or _risk_bar(0.003, 0.03, "x").style == "green"
    assert _risk_bar(0.02, 0.03, "x").style == "yellow"
    assert _risk_bar(0.04, 0.03, "x").style == "red"
