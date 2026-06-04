"""SimFin fundamentals loader (ML v2 data foundation).

Pulls company fundamentals from the SimFin v3 REST API for the cross-sectional
book's ML predictor (the gated v2 that replaces the rules-based momentum signal).
The API key comes from ``SIMFIN_API_KEY`` in ``.env`` (never hard-coded). Auth is a
header: ``Authorization: <key>``.

Why fundamentals: the momentum (price) signal is v1; v2 adds value/quality/profitability
features (Gu-Kelly-Xiu show these add cross-sectional return predictability). SimFin
statements carry a **Publish Date** per period — use it to avoid look-ahead (a fundamental
is only usable from the date it became public, not its fiscal report date).

Honest limitation: SimFin's free history is the *current* companies' filings — training an
ML model on it still carries survivorship bias (same wall as the price data). So a v2 model
is for **forward** deployment, judged by the same pre-registered forward gate, not a
historical-backtest edge claim. See docs/analysis/2026-06-04-cross-sectional-prereg.md.

Network access is injected (``fetch`` arg) so the parsing/URL logic is unit-tested with no
network; ``main``/scripts pass the real fetcher.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

BASE = "https://backend.simfin.com/api/v3"

# (status_code, body_text) <- (url, api_key)
Fetcher = Callable[[str, str], "tuple[int, str]"]


def _api_key() -> str:
    """Read the SimFin API key from the environment (loaded from .env)."""
    key = os.environ.get("SIMFIN_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SIMFIN_API_KEY not set (add it to .env)")
    return key


def _default_fetch(url: str, key: str) -> tuple[int, str]:
    """Real HTTP GET against SimFin with header auth (stdlib only)."""
    req = urllib.request.Request(url, headers={"Authorization": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _get(path: str, params: dict[str, Any], key: Optional[str] = None,
         fetch: Optional[Fetcher] = None) -> Any:
    """GET a SimFin endpoint and parse JSON.

    Args:
        path: API path under :data:`BASE` (e.g. ``"/companies/general/verbose"``).
        params: Query parameters.
        key: API key (defaults to :func:`_api_key`).
        fetch: Injected fetcher ``(url, key) -> (status, body)`` (defaults to real HTTP).

    Returns:
        Parsed JSON.

    Raises:
        RuntimeError: on a non-200 status.
    """
    key = key or _api_key()
    fetch = fetch or _default_fetch
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{BASE}{path}?{query}"
    status, body = fetch(url, key)
    if status != 200:
        raise RuntimeError(f"SimFin {path} -> HTTP {status}: {body[:200]}")
    return json.loads(body)


def company_info(ticker: str, key: Optional[str] = None,
                 fetch: Optional[Fetcher] = None) -> dict[str, Any]:
    """Fetch general company info (sector, industry, description, ...).

    Returns:
        The first company record dict, or ``{}`` if none.
    """
    data = _get("/companies/general/verbose", {"ticker": ticker}, key, fetch)
    return data[0] if isinstance(data, list) and data else (data or {})


def statements(ticker: str, statements: str = "PL,BS,CF", period: str = "fy",
               fyear: Optional[int] = None, key: Optional[str] = None,
               fetch: Optional[Fetcher] = None) -> list[dict[str, Any]]:
    """Fetch financial statements (PL/BS/CF) for a ticker.

    Args:
        ticker: Symbol.
        statements: Comma list of ``PL`` (income), ``BS`` (balance), ``CF`` (cash flow).
        period: ``fy`` (full year) or quarters (``q1``..``q4``).
        fyear: Optional fiscal year filter (latest available if omitted).
        key, fetch: As in :func:`_get`.

    Returns:
        List of company-statement blocks (each with a ``statements`` list whose entries
        carry ``Report Date`` and **``Publish Date``** for point-in-time alignment).
    """
    params = {"ticker": ticker, "statements": statements, "period": period, "fyear": fyear}
    data = _get("/companies/statements/verbose", params, key, fetch)
    return data if isinstance(data, list) else [data]
