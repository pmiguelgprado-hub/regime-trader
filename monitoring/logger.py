"""Structured logging.

Configures JSON-structured logging across the app with consistent fields. Four
rotating sinks under ``logs/`` (10 MB per file, 30 backups): ``main.log``,
``trades.log``, ``alerts.log``, ``regime.log``. Every record carries a shared
trading context — ``regime``, ``probability``, ``equity``, ``positions``,
``daily_pnl`` — so any line can be correlated to the system state at the time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

MB = 1024 * 1024

# Context fields stamped onto every record (defaults until the loop sets them).
_CONTEXT_FIELDS = ("regime", "probability", "equity", "positions", "daily_pnl")


@dataclass
class LoggerConfig:
    """Configuration for structured logging."""

    level: str = "INFO"
    log_dir: str = "logs"
    json_format: bool = True
    max_bytes: int = 10 * MB
    backup_count: int = 30
    console: bool = True


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON with the trading context."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a JSON object.

        Args:
            record: The log record (may carry ``event`` and context extras).

        Returns:
            JSON string with timestamp, level, logger, event, message, context.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.name),
            "message": record.getMessage(),
        }
        for f in _CONTEXT_FIELDS:
            if hasattr(record, f):
                payload[f] = getattr(record, f)
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TradingLogger:
    """Bundle of the four rotating loggers plus shared trading context.

    The context (regime/probability/equity/positions/daily_pnl) is injected into
    every emitted record so logs are self-describing.
    """

    def __init__(self, config: LoggerConfig) -> None:
        """Initialize all sinks.

        Args:
            config: Logging configuration.
        """
        self.config = config
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self._context: dict[str, Any] = {
            "regime": None, "probability": None, "equity": None,
            "positions": None, "daily_pnl": None,
        }
        self.main = self._make_logger("regime_trader.main", "main.log")
        self.trades = self._make_logger("regime_trader.trades", "trades.log")
        self.alerts = self._make_logger("regime_trader.alerts", "alerts.log")
        self.regime = self._make_logger("regime_trader.regime", "regime.log")

    def _make_logger(self, name: str, filename: str) -> logging.Logger:
        """Build a named logger with a rotating JSON file handler.

        Args:
            name: Logger name.
            filename: Log filename under ``log_dir``.

        Returns:
            Configured logger (idempotent — no duplicate handlers).
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.config.level)
        logger.propagate = False
        # Reconfigure idempotently: drop any handlers from a prior setup so a
        # new log_dir / rotation config takes effect (named loggers are global).
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
        fmt = JSONFormatter() if self.config.json_format else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        fh = RotatingFileHandler(
            Path(self.config.log_dir) / filename,
            maxBytes=self.config.max_bytes, backupCount=self.config.backup_count,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        if self.config.console and name.endswith(".main"):
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger

    def set_context(self, **fields: Any) -> None:
        """Update the shared context stamped onto subsequent records.

        Args:
            **fields: Any of ``regime``, ``probability``, ``equity``,
                ``positions``, ``daily_pnl``.
        """
        self._context.update({k: v for k, v in fields.items() if k in self._context})

    def log(
        self, logger: logging.Logger, event: str, message: str = "",
        level: str = "INFO", **payload: Any,
    ) -> None:
        """Emit a structured record on ``logger`` with context + payload.

        Args:
            logger: Target logger (e.g. ``self.trades``).
            event: Event name.
            message: Human-readable message.
            level: Log level name.
            **payload: Extra structured fields.
        """
        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            message or event,
            extra={"event": event, "payload": payload, **self._context},
        )


def setup_logging(config: LoggerConfig) -> TradingLogger:
    """Configure and return the bundle of application loggers.

    Args:
        config: Logging level, directory, and rotation settings.

    Returns:
        A :class:`TradingLogger` with main/trades/alerts/regime sinks.
    """
    return TradingLogger(config)


def log_event(
    logger: logging.Logger, event: str, payload: dict[str, Any], level: str = "INFO"
) -> None:
    """Emit a single structured event record on a plain logger.

    Args:
        logger: Target logger.
        event: Event name (e.g. "order_filled", "regime_change").
        payload: Structured fields to attach.
        level: Log level.
    """
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        event, extra={"event": event, "payload": payload},
    )
