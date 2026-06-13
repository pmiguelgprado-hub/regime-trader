"""Macro risk-confirmation features (T1.3) — VIX term structure + FRED spreads.

Free data, used strictly as **risk confirmation, NOT return timing** (regime-as-
return-timer is falsified, R1). Two corroborators of elevated risk:

* **VIX term structure** — VIX / VIX3M. In calm markets the curve is in contango
  (VIX < VIX3M, ratio < 1); a flip into *backwardation* (front > back, ratio > 1)
  has preceded essentially every major drawdown. CBOE/Yahoo, free.
* **FRED credit/curve** — high-yield OAS (BAMLH0A0HYM2), the 10y-2y slope, NFCI.
  Free keyless CSV download.

**Hard guardrail (roadmap §T1.3 / §10):** these features NEVER enter the champion
HMM's feature panel — doing so would silently mutate the frozen gate. They are
shadow-only corroboration, logged separately, with the daily champion-hash assert
(``run_rebalance``) proving the live model is untouched. Pure scoring + parsing is
unit-tested; network fetch is injected.
"""

from __future__ import annotations

from typing import Callable, Optional

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def term_structure(vix: float, vix3m: float) -> dict:
    """VIX/VIX3M ratio + backwardation flag (ratio > 1 = front-month stress)."""
    ratio = float(vix) / float(vix3m) if vix3m else float("nan")
    return {"vix": float(vix), "vix3m": float(vix3m), "ratio": ratio,
            "backwardation": bool(vix3m and ratio > 1.0)}


def parse_fred_latest(csv_text: str) -> "tuple[Optional[float], Optional[str]]":
    """Latest valid (value, date) from a FRED CSV ('.' marks a missing observation)."""
    val = date = None
    for line in csv_text.strip().splitlines()[1:]:     # skip header
        parts = line.split(",")
        if len(parts) < 2:
            continue
        d, raw = parts[0].strip(), parts[1].strip()
        if raw and raw != ".":
            try:
                val, date = float(raw), d
            except ValueError:
                continue
    return val, date


def risk_confirmation(backwardation: bool, hy_oas: Optional[float],
                      hy_oas_hi: float = 5.0, hy_oas_lo: float = 3.0) -> float:
    """Composite risk-confirmation score in [0, 1] (higher = more risk corroborated).

    Blends the VIX backwardation flag (0/1) with a normalized HY-OAS stress level
    (``hy_oas_lo`` -> 0, ``hy_oas_hi`` -> 1). Equal weight; this is a de-risk
    *corroborator*, never a return signal. Missing HY OAS -> backwardation alone.
    """
    bw = 1.0 if backwardation else 0.0
    if hy_oas is None:
        return bw
    span = max(1e-9, hy_oas_hi - hy_oas_lo)
    oas = min(1.0, max(0.0, (float(hy_oas) - hy_oas_lo) / span))
    return 0.5 * bw + 0.5 * oas


def fetch_term_structure(loader: Optional[Callable] = None) -> dict:  # pragma: no cover - network default
    """Fetch VIX + VIX3M (yfinance ^VIX/^VIX3M by default) and build the term structure.

    Args:
        loader: ``symbol -> OHLCV DataFrame`` (injected for tests). Defaults to
            ``data.market_data.load_ohlcv`` with the yfinance source.
    """
    if loader is None:
        from data.market_data import load_ohlcv as loader
    vix = float(loader("^VIX", timeframe="1Day")["close"].iloc[-1])
    vix3m = float(loader("^VIX3M", timeframe="1Day")["close"].iloc[-1])
    return term_structure(vix, vix3m)


def fetch_fred_series(series: str, fetch: Optional[Callable] = None) -> "tuple[Optional[float], Optional[str]]":  # pragma: no cover - network default
    """Fetch the latest value of a FRED series via the keyless CSV endpoint."""
    if fetch is None:
        import urllib.request

        def fetch(url):  # noqa: ANN001
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read().decode("utf-8", "replace")
    return parse_fred_latest(fetch(FRED_CSV.format(series=series)))
