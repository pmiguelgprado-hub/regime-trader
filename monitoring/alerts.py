"""Alerts: rate-limited notifications for critical events.

Dispatches alerts to the console (always), a log file, and optionally email
(SMTP) and a webhook. Each event type is rate-limited to one alert per window
(default 15 min) so a flapping condition cannot spam every channel.

Triggers (helpers below): regime change, circuit breaker, large P&L, data feed
down, API connection lost, HMM retrained, flicker threshold exceeded.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("regime_trader.alerts")


class AlertSeverity(Enum):
    """Severity of an alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Configuration for alerting (channels + rate limiting)."""

    rate_limit_minutes: int = 15
    email_enabled: bool = False
    webhook_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    from_addr: Optional[str] = None
    to_addrs: list[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    # Telegram (T0.5): credentials come from env (TELEGRAM_BOT_TOKEN /
    # TELEGRAM_CHAT_ID — settings.yaml is committed, secrets stay in .env).
    telegram_enabled: bool = False
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_min_severity: str = "warning"     # info | warning | critical


class AlertManager:
    """Dispatches rate-limited alerts across channels."""

    def __init__(
        self,
        config: AlertConfig,
        trading_logger: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the alert manager.

        Args:
            config: Channel and rate-limit settings.
            trading_logger: Optional :class:`~monitoring.logger.TradingLogger`
                for the file channel (uses its ``alerts`` sink).
            clock: Time source (injectable for tests).
        """
        self.config = config
        self.trading_logger = trading_logger
        self._clock = clock
        self._last_sent: dict[str, float] = {}
        if config.telegram_enabled and not config.telegram_token:
            import os

            config.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
            config.telegram_chat_id = (config.telegram_chat_id
                                       or os.environ.get("TELEGRAM_CHAT_ID"))

    def send(
        self, event: str, message: str, severity: AlertSeverity = AlertSeverity.INFO
    ) -> bool:
        """Send an alert subject to per-event rate limiting.

        Args:
            event: Event key used for rate limiting.
            message: Alert body.
            severity: Alert severity.

        Returns:
            True if dispatched, False if suppressed by the rate limit.
        """
        if self._is_rate_limited(event):
            return False
        self._last_sent[event] = self._clock()

        line = f"[{severity.value.upper()}] {event}: {message}"
        # 1) console (always)
        print(line)
        # 2) log file
        if self.trading_logger is not None:
            self.trading_logger.log(self.trading_logger.alerts, event, message,
                                    level="WARNING", severity=severity.value)
        else:
            logger.warning(line)
        # 3) optional email / webhook (best-effort, never raise)
        if self.config.email_enabled:
            try:
                self._send_email(f"[regime-trader] {event}", message)
            except Exception as exc:  # noqa: BLE001
                logger.error("alert email failed: %s", exc)
        if self.config.webhook_enabled:
            try:
                self._send_webhook({"event": event, "message": message,
                                    "severity": severity.value})
            except Exception as exc:  # noqa: BLE001
                logger.error("alert webhook failed: %s", exc)
        # 4) Telegram (T0.5): WARNING+ by default — unattended gates need a push
        #    channel, but INFO-level chatter stays local.
        if self._telegram_wanted(severity):
            try:
                self._send_telegram(line)
            except Exception as exc:  # noqa: BLE001
                logger.error("alert telegram failed: %s", exc)
        return True

    _SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

    def _telegram_wanted(self, severity: AlertSeverity) -> bool:
        """Whether this severity clears the Telegram dispatch bar."""
        c = self.config
        if not (c.telegram_enabled and c.telegram_token and c.telegram_chat_id):
            return False
        floor = self._SEVERITY_ORDER.get(c.telegram_min_severity.lower(), 1)
        return self._SEVERITY_ORDER[severity.value] >= floor

    def _is_rate_limited(self, event: str) -> bool:
        """Whether an event key is within its rate-limit window.

        Args:
            event: Event key.

        Returns:
            True if suppressed.
        """
        last = self._last_sent.get(event)
        if last is None:
            return False
        return (self._clock() - last) < self.config.rate_limit_minutes * 60

    def _send_email(self, subject: str, body: str) -> None:  # pragma: no cover - network
        """Send an email alert via SMTP.

        Args:
            subject: Email subject.
            body: Email body.
        """
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.config.from_addr
        msg["To"] = ", ".join(self.config.to_addrs)
        msg.set_content(body)
        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as smtp:
            smtp.starttls()
            if self.config.username:
                smtp.login(self.config.username, self.config.password or "")
            smtp.send_message(msg)

    def _send_telegram(self, text: str) -> None:  # pragma: no cover - network
        """Send an alert via the Telegram Bot API (reuses the AIOS gateway bot).

        Args:
            text: Message body (plain text).
        """
        import urllib.parse
        import urllib.request

        data = urllib.parse.urlencode({
            "chat_id": self.config.telegram_chat_id,
            "text": f"📈 regime-trader\n{text}",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
            data=data,
        )
        urllib.request.urlopen(req, timeout=5)

    def _send_webhook(self, payload: dict) -> None:  # pragma: no cover - network
        """POST an alert payload to the configured webhook.

        Args:
            payload: JSON-serializable alert payload.
        """
        import urllib.request

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.config.webhook_url, data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)

    # ---------------------------------------------------- trigger helpers ---
    def regime_change(self, old: str, new: str, prob: float) -> bool:
        """Alert on a confirmed regime change."""
        return self.send("regime_change", f"{old} -> {new} (p={prob:.2f})",
                         AlertSeverity.INFO)

    def circuit_breaker(self, state: str, drawdown: float) -> bool:
        """Alert on a circuit-breaker state change."""
        return self.send("circuit_breaker", f"{state} (DD {drawdown:.2%})",
                         AlertSeverity.CRITICAL)

    def large_pnl(self, pnl_pct: float) -> bool:
        """Alert on an outsized daily P&L move."""
        return self.send("large_pnl", f"daily P&L {pnl_pct:+.2%}", AlertSeverity.WARNING)

    def data_feed_down(self, detail: str = "") -> bool:
        """Alert when the market data feed drops."""
        return self.send("data_feed_down", f"data feed down {detail}".strip(),
                         AlertSeverity.CRITICAL)

    def api_lost(self, detail: str = "") -> bool:
        """Alert when the broker API connection is lost."""
        return self.send("api_lost", f"broker API lost {detail}".strip(),
                         AlertSeverity.CRITICAL)

    def hmm_retrained(self, n_regimes: int, bic: float) -> bool:
        """Alert when the HMM is retrained."""
        return self.send("hmm_retrained", f"HMM retrained: {n_regimes} regimes, BIC {bic:.0f}",
                         AlertSeverity.INFO)

    def flicker_exceeded(self, rate: int, threshold: int) -> bool:
        """Alert when the regime flicker rate exceeds its threshold."""
        return self.send("flicker_exceeded", f"flicker {rate} > {threshold}",
                         AlertSeverity.WARNING)
