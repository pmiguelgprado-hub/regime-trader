"""Streamlit dashboard for regime-trader — matches the reference build's UI.

Run it:

    streamlit run monitoring/streamlit_app.py     # http://localhost:8501

Layout mirrors the video's final dashboard: a sidebar (refresh interval,
primary symbol, view toggles), a metrics row (Mode / Equity / Cash / Market),
Regime Detection (regime, confidence, stability, vol rank + a confidence gauge
and runner-up states), Risk Status (drawdown / leverage + circuit-breaker
banner), the learned-regime table, the portfolio, and a price panel.

Live account / positions / price are pulled from Alpaca on every refresh (real
time); the regime + risk detail come from ``state_snapshot.json`` (written by
the daily ``--run-once`` cycle). Everything degrades to placeholders when a
source is missing, so the page never crashes.

The data layer is ``monitoring.dashboard_data`` (pure + unit-tested); this file
is the view and is never imported by the test suite.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from monitoring.dashboard_data import (
    live_account,
    live_positions,
    live_price,
    load_regime_history,
    load_snapshot,
    risk_panel,
)

st.set_page_config(page_title="Regime Trader", layout="wide",
                   initial_sidebar_state="expanded")


def _confidence_gauge(pct: float, regime: str) -> go.Figure:
    """Donut/gauge of the current regime confidence (green/orange/red arc)."""
    color = "#2ecc71" if pct >= 80 else "#e67e22" if pct >= 50 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": "#fafafa"}},
        title={"text": f"Regime: {regime.upper()}", "font": {"size": 16, "color": "#fafafa"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#888"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(231,76,60,0.15)"},
                {"range": [50, 80], "color": "rgba(230,126,34,0.15)"},
                {"range": [80, 100], "color": "rgba(46,204,113,0.15)"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"})
    return fig


def _metrics(acct: dict | None, snap: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    if acct:
        c1.metric("Mode", acct["mode"])
        c2.metric("Equity", f"${acct['equity']:,.2f}")
        c3.metric("Cash", f"${acct['cash']:,.2f}")
        c4.metric("Market", "OPEN" if acct["market_open"] else "CLOSED")
    else:
        c1.metric("Mode", "—")
        c2.metric("Equity", f"${snap.get('equity_peak', 0):,.2f}")
        c3.metric("Cash", "—")
        c4.metric("Market", "—")
        st.caption("⚠️ live account unavailable (no creds / market data) — showing last snapshot")


def _regime_detection(snap: dict) -> None:
    st.subheader("Regime Detection")
    reg = snap.get("regime") or {}
    name = reg.get("name", snap.get("last_regime") or "—")
    conf = float(reg.get("confidence", 0.0)) * 100.0
    a, b, c, d = st.columns(4)
    a.metric("Regime", str(name).upper())
    b.metric("Confidence", f"{conf:.1f}%")
    c.metric("Stability", f"{reg.get('stability_bars', 0)} bars")
    d.metric("Vol Rank", f"{reg.get('vol_rank', 0.0):.2f}")
    if reg.get("confirmed"):
        st.success("✓ CONFIRMED")
    elif reg:
        st.warning("… stabilizing")
    if reg:
        st.plotly_chart(_confidence_gauge(conf, str(name)), use_container_width=True)
        runners = reg.get("runner_ups") or {}
        others = [f"{k.upper()}: {v:.2e}" for k, v in list(runners.items())[1:5]]
        if others:
            st.caption("Runner-up states:  " + "   |   ".join(others))


def _risk_status(snap: dict) -> None:
    st.subheader("Risk Status")
    rk = snap.get("risk") or {}
    daily = rk.get("daily_dd", 0.0) * 100
    peak = rk.get("peak_dd", 0.0) * 100
    lev_lim = rk.get("leverage_limit", 1.25)
    a, b, c = st.columns(3)
    a.metric("Daily DD", f"{daily:.2f}%", help=f"limit {rk.get('daily_dd_limit', 0.03):.0%}")
    b.metric("Peak DD", f"{peak:.2f}%", help=f"limit {rk.get('peak_dd_limit', 0.10):.0%}")
    c.metric("Leverage cap", f"{lev_lim:.2f}x")
    state = str(rk.get("state", snap.get("risk_state", "—"))).lower()
    if rk.get("breakers_clear", state == "normal"):
        st.success("All circuit breakers clear")
    elif state == "reduced":
        st.warning("Circuit breaker: REDUCED — sizing halved")
    else:
        st.error(f"Circuit breaker: {state.upper()} — trading halted / liquidating")


def _regime_table(snap: dict) -> None:
    table = snap.get("regime_table") or []
    if table:
        st.dataframe(table, use_container_width=True, hide_index=True)


def _portfolio(positions: list) -> None:
    st.subheader("Portfolio")
    if not positions:
        st.info("No open positions.")
        return
    rows = [{
        "symbol": p.get("symbol"), "qty": p.get("qty"),
        "avg_entry": p.get("avg_entry_price"), "price": p.get("current_price"),
        "market_value": p.get("market_value"), "unrealized_pnl": p.get("unrealized_pl"),
    } for p in positions]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _signal_feed(snap: dict) -> None:
    st.subheader("Signal feed")
    sigs = snap.get("recent_signals") or []
    if sigs:
        st.dataframe(sigs, use_container_width=True, hide_index=True)
    else:
        st.info("No signals yet.")


def _price(symbol: str, show: bool) -> None:
    if not show:
        return
    st.subheader(f"{symbol} Price")
    df = live_price(symbol)
    if df is None or df.empty:
        df = load_regime_history(symbol)
    if df is not None and not df.empty and "close" in df:
        st.line_chart(df["close"])
    else:
        st.info(f"No price data for {symbol}.")


def render(symbol: str, toggles: dict) -> None:
    """Draw the whole dashboard from live + snapshot state (re-run on refresh)."""
    snap = load_snapshot()
    acct = live_account()
    positions = live_positions()

    st.title("⚡ Regime Trader")
    st.caption("HMM regime detection · Alpaca execution · walk-forward validated")
    _metrics(acct, snap)
    st.divider()
    left, right = st.columns([2, 1])
    with left:
        _regime_detection(snap)
    with right:
        _risk_status(snap)
    st.divider()
    _regime_table(snap)
    _portfolio(positions)
    if toggles.get("price"):
        _price(symbol, True)
    if toggles.get("regime_history"):
        rh = load_regime_history(symbol)
        if rh is not None and {"regime_prob", "weight"} & set(rh):
            st.subheader("Regime history")
            st.line_chart(rh[[c for c in ("regime_prob", "weight") if c in rh]])
    if toggles.get("logs"):
        _signal_feed(snap)


# ----------------------------------------------------------------- sidebar ---
st.sidebar.title("Regime Trader")
interval = st.sidebar.slider("Refresh interval (s)", 2, 60, 10)
symbol = st.sidebar.text_input("Primary symbol", value="SPY").strip().upper() or "SPY"
toggles = {
    "price": st.sidebar.checkbox("Show price chart", value=True),
    "regime_history": st.sidebar.checkbox("Show regime history", value=True),
    "logs": st.sidebar.checkbox("Show signal feed / logs", value=False),
}
st.sidebar.caption("Live account/positions/price refresh on the interval; "
                   "regime + risk come from the daily snapshot.")


@st.fragment(run_every=interval)
def _live():
    render(symbol, toggles)


_live()
