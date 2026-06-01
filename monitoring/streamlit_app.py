"""Streamlit dashboard for regime-trader — the video-faithful web UI.

Run it:

    streamlit run monitoring/streamlit_app.py     # http://localhost:8501

Panels (mirroring the reference build): detected regime + confidence, portfolio
value, learned regimes + allocation, risk controls (drawdown / leverage /
circuit breakers), the signal feed, a price + regime overlay, and volume /
confidence over time.

Live state comes from ``state_snapshot.json`` (written by the trading loop);
the overlay/distribution charts use the latest ``backtest_output/<symbol>/``
artifacts for context. All panels degrade to placeholders when data is absent
(e.g. a fresh system or a closed market), so the page never crashes.

The data layer lives in ``monitoring.dashboard_data`` (pure + unit-tested);
this module is only the view and is never imported by the test suite.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from monitoring.dashboard_data import (
    load_equity_curve,
    load_regime_history,
    load_snapshot,
    regime_distribution,
    risk_panel,
)

REGIME_COLORS = {
    "crash": "#b00020", "bear": "#e06c00", "neutral": "#9e9e9e",
    "bull": "#2e7d32", "euphoria": "#1565c0",
}


def _symbol() -> str:
    """Resolve the symbol to chart (sidebar selector, defaults to SPY)."""
    return st.sidebar.text_input("Symbol", value="SPY").strip().upper() or "SPY"


def _header(panel: dict) -> None:
    st.title("regime-trader — live dashboard")
    st.caption(f"snapshot: {panel['timestamp']}  ·  paper account")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected regime", str(panel["regime"]).upper())
    c2.metric("Risk state", str(panel["risk_state"]).upper())
    c3.metric("Equity peak", f"${panel['equity_peak']:,.0f}")
    c4.metric("Trades today", panel["daily_trades"])


def _risk_controls(panel: dict) -> None:
    st.subheader("Risk controls")
    state = str(panel["risk_state"]).lower()
    msg = {"normal": st.success, "reduced": st.warning, "halted": st.error}.get(state, st.info)
    msg(f"Circuit breaker: {state.upper()}  ·  breaker events: {panel['breaker_events']}")
    st.caption("Tiers: daily -2% reduce / -3% halt · weekly -5% / -7% · peak -10% halt+lock")


def _signal_feed(snapshot: dict) -> None:
    st.subheader("Signal feed")
    signals = snapshot.get("recent_signals") or []
    if not signals:
        st.info("No signals yet (fresh system or market closed).")
        return
    st.dataframe(pd.DataFrame(signals), use_container_width=True, hide_index=True)


def _price_regime_overlay(symbol: str, rh: pd.DataFrame | None) -> None:
    st.subheader("Price + regime overlay")
    if rh is None or rh.empty or "asset_return" not in rh:
        st.info(f"No regime history for {symbol} (run a backtest to populate).")
        return
    df = rh.copy()
    df["price"] = (1.0 + df["asset_return"].fillna(0.0)).cumprod() * 100.0
    df["regime"] = df["regime"].astype(str)
    try:
        import altair as alt

        chart = (
            alt.Chart(df.reset_index().rename(columns={df.index.name or "index": "date"}))
            .mark_circle(size=18)
            .encode(
                x=alt.X(f"{df.reset_index().columns[0]}:T", title="date"),
                y=alt.Y("price:Q", title="price (indexed=100)"),
                color=alt.Color("regime:N",
                                scale=alt.Scale(domain=list(REGIME_COLORS),
                                                range=list(REGIME_COLORS.values())),
                                title="regime"),
                tooltip=["regime:N", "price:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:  # noqa: BLE001 - charting is best-effort
        st.line_chart(df["price"])


def _volume_confidence(rh: pd.DataFrame | None) -> None:
    st.subheader("Allocation & confidence over time")
    if rh is None or rh.empty:
        st.info("No history to chart yet.")
        return
    cols = [c for c in ("weight", "regime_prob") if c in rh]
    if cols:
        st.line_chart(rh[cols])


def _learned_regimes(rh: pd.DataFrame | None) -> None:
    st.subheader("Learned regimes")
    dist = regime_distribution(rh)
    if dist.empty:
        st.info("Regimes appear once the model has classified some bars.")
        return
    st.bar_chart(dist)


def _portfolio(symbol: str) -> None:
    eq = load_equity_curve(symbol)
    if eq is not None and "equity" in eq:
        st.subheader("Equity curve (latest backtest)")
        st.line_chart(eq["equity"])


def main() -> None:
    """Render the full dashboard (called when run via ``streamlit run``)."""
    st.set_page_config(page_title="regime-trader", layout="wide")
    symbol = _symbol()
    snapshot = load_snapshot()
    panel = risk_panel(snapshot)
    rh = load_regime_history(symbol)

    _header(panel)
    st.divider()
    left, right = st.columns([2, 1])
    with left:
        _price_regime_overlay(symbol, rh)
        _volume_confidence(rh)
    with right:
        _risk_controls(panel)
        _learned_regimes(rh)
    st.divider()
    _portfolio(symbol)
    _signal_feed(snapshot)


main()
