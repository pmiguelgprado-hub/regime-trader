"""Stream robustness primitives: reconnect-with-backoff + staleness watchdog (S-2).

A dropped WebSocket used to leave the bot silently blind. These pure helpers make
reconnection and stale-feed detection testable; the async stream loop that wires
them in (`TradingSystem.run_stream`) stays thin plumbing.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def reconnect_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff delay for a reconnect attempt.

    Args:
        attempt: Zero-based retry index.
        base: Delay for ``attempt == 0``.
        cap: Maximum delay.

    Returns:
        ``min(base * 2**attempt, cap)`` seconds.
    """
    return min(base * (2 ** attempt), cap)


def stream_is_stale(last_bar_age_sec: float, max_gap_sec: float, market_open: bool) -> bool:
    """Whether the bar feed has gone silent while the market is open.

    Args:
        last_bar_age_sec: Seconds since the last received bar.
        max_gap_sec: Tolerated silence before flagging stale.
        market_open: Whether the market is currently open.

    Returns:
        True only when the market is open AND the gap exceeds the tolerance.
    """
    return market_open and last_bar_age_sec > max_gap_sec


def run_with_reconnect(
    run_fn: Callable[[], T],
    max_retries: int = 5,
    base: float = 1.0,
    cap: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """Run ``run_fn``; on exception, retry with exponential backoff.

    Args:
        run_fn: Callable that connects and runs the stream (blocks until it ends
            or raises).
        max_retries: Max reconnect attempts after the first failure.
        base, cap: Backoff parameters (see :func:`reconnect_delay`).
        sleep: Sleep function (injected for tests).
        on_retry: Optional callback ``(attempt, exc)`` invoked before each retry.

    Returns:
        Whatever ``run_fn`` returns once it completes without raising.

    Raises:
        The last exception if all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return run_fn()
        except Exception as exc:  # noqa: BLE001 - reconnect on any stream failure
            if attempt >= max_retries:
                raise
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            sleep(reconnect_delay(attempt, base, cap))
    raise RuntimeError("unreachable")  # pragma: no cover
