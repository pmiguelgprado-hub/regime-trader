"""Streamlit dashboard for regime-trader — matches the reference build's UI.

Run it:

    streamlit run monitoring/streamlit_app.py     # http://localhost:8501

Stock Streamlit **dark** theme (Source-Sans font, red accent — no custom
palette, matching the video), with: a sidebar (refresh interval, primary
symbol, view toggles), a metrics row (Mode / Equity / Cash / Market), Regime
Detection (regime, confidence, stability, vol rank + an orange confidence gauge
and runner-up states), Risk Status (drawdown / leverage + circuit-breaker
banner), the learned-regime table, the portfolio, a candlestick price panel,
and optional transition-matrix / model-info / logs panels.

Live account / positions / price are pulled from Alpaca on every refresh (real
time); the regime + risk detail come from ``state_snapshot.json`` (written by
the daily ``--run-once`` cycle). Everything degrades to placeholders when a
source is missing, so the page never crashes.

The data layer is ``monitoring.dashboard_data`` (pure + unit-tested); this file
is the view and is never imported by the test suite.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# `streamlit run monitoring/streamlit_app.py` puts the script's own dir on
# sys.path, not the project root, so the absolute `monitoring.*` package import
# below fails. Put the project root (this file's parent's parent) on the path so
# the documented launch command works from any cwd.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import plotly.graph_objects as go
import streamlit as st

from monitoring.dashboard_data import (
    live_account,
    live_positions,
    live_price,
    load_book_snapshot,
    load_regime_history,
    load_snapshot,
)

st.set_page_config(page_title="Regime Trader", layout="wide",
                   initial_sidebar_state="expanded")
# tighten Streamlit's default top gap so the metrics row sits near the top
st.markdown("<style>.block-container{padding-top:2.2rem;}</style>",
            unsafe_allow_html=True)

ORANGE = "#ff7f0e"
UP, DOWN = "#26a69a", "#ef5350"


def _confidence_gauge(pct: float, regime: str) -> go.Figure:
    """Orange confidence gauge — matches the video (thick amber arc, % center).

    Plotly animates the needle/number between values on each refresh.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": f"Regime: {regime.upper()}", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#777",
                     "ticksuffix": "%"},
            "bar": {"color": ORANGE, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 100], "color": "rgba(255,127,14,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        height=270, margin=dict(l=24, r=24, t=46, b=8),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"},
        transition={"duration": 600, "easing": "cubic-in-out"},
    )
    return fig


def _candles(symbol: str, df) -> None:
    """Candlestick price panel (green up / red down), dark template."""
    if df is None or df.empty or not {"open", "high", "low", "close"} <= set(df):
        st.info(f"No OHLC price data for {symbol}.")
        return
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=UP, decreasing_line_color=DOWN, name=symbol,
    ))
    fig.update_layout(template="plotly_dark", height=380,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_rangeslider_visible=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


def _transition_heatmap(tm: dict) -> None:
    labels, matrix = tm.get("labels"), tm.get("matrix")
    if not labels or not matrix:
        st.info("No transition matrix (model not trained).")
        return
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, colorscale="Oranges", zmin=0, zmax=1,
        text=matrix, texttemplate="%{text:.2f}", textfont={"size": 12},
        colorbar={"title": "P"},
    ))
    fig.update_layout(template="plotly_dark", height=360,
                      margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)",
                      yaxis={"title": "from"}, xaxis={"title": "to"})
    st.plotly_chart(fig, width="stretch")


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
        st.caption("⚠️ live account unavailable — showing last snapshot")


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
        st.plotly_chart(_confidence_gauge(conf, str(name)), width="stretch")
        runners = reg.get("runner_ups") or {}
        others = [f"{k.upper()}: {v:.2e}" for k, v in list(runners.items())[1:5]]
        if others:
            st.caption("Runner-up states:  " + "  |  ".join(others))


def _risk_status(snap: dict) -> None:
    st.subheader("Risk Status")
    rk = snap.get("risk") or {}
    a, b, c = st.columns(3)
    a.metric("Daily DD", f"{rk.get('daily_dd', 0.0) * 100:.2f}%",
             help=f"limit {rk.get('daily_dd_limit', 0.03):.0%}")
    b.metric("Peak DD", f"{rk.get('peak_dd', 0.0) * 100:.2f}%",
             help=f"limit {rk.get('peak_dd_limit', 0.10):.0%}")
    c.metric("Leverage cap", f"{rk.get('leverage_limit', 1.25):.2f}x")
    state = str(rk.get("state", snap.get("risk_state", "—"))).lower()
    if rk.get("breakers_clear", state == "normal"):
        st.success("All circuit breakers clear")
    elif state == "reduced":
        st.warning("Circuit breaker: REDUCED — sizing halved")
    else:
        st.error(f"Circuit breaker: {state.upper()} — trading halted / liquidating")


def _portfolio(positions: list) -> None:
    st.subheader("Portfolio — holdings & per-company return")
    if not positions:
        st.info("No open positions.")
        return

    def _ret_pct(p) -> float:
        entry = p.get("avg_entry_price") or 0.0
        price = p.get("current_price") or 0.0
        return (price / entry - 1.0) * 100.0 if entry > 0 else 0.0

    rows = [{
        "symbol": p.get("symbol"), "qty": p.get("qty"),
        "avg_entry": round(p.get("avg_entry_price") or 0.0, 2),
        "price": round(p.get("current_price") or 0.0, 2),
        "market_value": round(p.get("market_value") or 0.0, 2),
        "unrealized_pnl": round(p.get("unrealized_pl") or 0.0, 2),
        "return_%": round(_ret_pct(p), 2),
    } for p in positions]
    rows.sort(key=lambda r: r["return_%"], reverse=True)        # winners first
    total_pnl = sum(r["unrealized_pnl"] for r in rows)
    st.caption(f"{len(rows)} holdings · total unrealized P&L ${total_pnl:,.2f}. "
               "Queued book orders appear here once they fill (next market open).")
    st.dataframe(rows, width="stretch", hide_index=True)


def _model_info(snap: dict) -> None:
    mi = snap.get("model_info") or {}
    if not mi:
        st.info("No model info (train the HMM first).")
        return
    st.subheader("Model info")
    a, b, c, d = st.columns(4)
    a.metric("Regimes", mi.get("n_regimes", "—"))
    b.metric("BIC", mi.get("bic", "—"))
    c.metric("Log-likelihood", mi.get("log_likelihood", "—"))
    d.metric("Converged", "yes" if mi.get("converged") else "no")
    st.caption(f"iters {mi.get('n_iter', '—')} · features {mi.get('n_features', '—')} "
               f"· trained {mi.get('training_date', '—')}")


def _cross_sectional_book(book: dict) -> None:
    """Cross-sectional book panel (vía C): regime overlay + target holdings.

    Reads the snapshot written by ``main.run_rebalance``. The ranker (momentum) picks the
    names; the HMM ``vol_rank`` scales gross exposure. ``dry_run`` plans are shown the same
    way — the badge says whether orders were submitted.
    """
    st.subheader("Cross-Sectional Book (vía C)")
    if not book:
        st.info("No book snapshot yet. Run `python main.py --rebalance` to generate one.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Vol rank (regime)", f"{book.get('vol_rank', 0.0):.2f}",
             help="0 = low-vol/risk-on (full gross), 1 = high-vol/risk-off (de-risked)")
    b.metric("Gross exposure", f"{book.get('gross', 0.0) * 100:.0f}%",
             help="risk overlay scales total gross to the market's volatility")
    c.metric("Names", len(book.get("targets", [])))
    d.metric("Equity", f"${book.get('equity', 0.0):,.0f}")
    overlay = book.get("overlay", "—")
    cadence = ("monthly re-rank" if book.get("rebalanced") else "daily risk re-scale")
    st.caption(f"Book: **{book.get('book', 'baseline')}** · overlay **{overlay}** · "
               f"{cadence} · month {book.get('selection_month', '—')}")
    executed = book.get("executed") or []
    if book.get("dry_run", True):
        st.warning(f"DRY-RUN plan ({book.get('mode', '—')}) — no orders submitted")
    else:
        st.success(f"EXECUTED ({book.get('mode', '—')}) — {len(executed)} orders submitted")
    targets = book.get("targets") or []
    if targets:
        st.caption(f"Target book ({book.get('timestamp', '—')}):")
        st.dataframe([{
            "symbol": t.get("symbol"), "shares": t.get("shares"),
            "price": t.get("price"), "weight": t.get("weight"),
            "notional": t.get("notional"),
        } for t in targets], width="stretch", hide_index=True)


def _macro_events() -> None:
    """US macro calendar panel: upcoming scheduled high-vol events (risk timing, not alpha)."""
    from datetime import date, timedelta

    from core.macro_calendar import days_to_next_event, high_impact_events, in_event_window

    st.subheader("US Macro Calendar")
    today = date.today()
    flagged, label = in_event_window(today, window_days=2)
    d2n = days_to_next_event(today)
    if flagged:
        st.warning(f"⚠️ {label} within 2 days — pre-event vol window "
                   "(book de-risks here when the event overlay is enabled).")
    elif d2n is not None:
        st.caption(f"Next high-impact event in **{d2n}** day(s).")
    rows = [{"date": d.isoformat(), "event": n, "in_days": (d - today).days}
            for d, n in high_impact_events(today, today + timedelta(days=45))]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("FOMC + payrolls (scheduled). Public events are priced in within seconds — "
               "used for **risk timing** (de-risk into known vol), never for direction.")


def _news_panel() -> None:
    """Markets/macro headlines for operator awareness (no trading decision)."""
    st.subheader("Markets / Macro Headlines")

    @st.cache_data(ttl=300)                              # refetch at most every 5 min
    def _cached() -> list:
        from monitoring.news_feed import fetch_headlines
        try:
            return fetch_headlines(limit=12)
        except Exception:
            return []

    heads = _cached()
    if not heads:
        st.info("No headlines right now (feeds unreachable).")
        return
    for h in heads:
        src = h.get("source", "")
        title, link = h.get("title", ""), h.get("link", "")
        st.markdown(f"- [{title}]({link}) — *{src}*" if link else f"- {title} — *{src}*")
    st.caption("Awareness only. Public headlines carry no tradable edge for a retail bot "
               "(priced in instantly); this informs **you**, not the bot.")


def render(symbol: str, toggles: dict) -> None:
    """Draw the whole dashboard from live + snapshot state (re-run on refresh)."""
    snap = load_snapshot()
    acct = live_account()
    positions = live_positions()

    # Refresh heartbeat. Lives in the main fragment, not the sidebar: Streamlit
    # forbids a fragment (run_every) from writing to the sidebar.
    st.caption(f"Last refresh: {datetime.now():%H:%M:%S}")
    _metrics(acct, snap)
    st.divider()
    left, right = st.columns([2, 1])
    with left:
        _regime_detection(snap)
    with right:
        _risk_status(snap)
    st.divider()
    if snap.get("regime_table"):
        st.dataframe(snap["regime_table"], width="stretch", hide_index=True)
    _portfolio(positions)

    if toggles.get("cross_book"):
        st.divider()
        _cross_sectional_book(load_book_snapshot())

    if toggles.get("macro"):
        st.divider()
        mc, nf = st.columns(2)
        with mc:
            _macro_events()
        with nf:
            _news_panel()

    if toggles.get("price"):
        st.subheader(f"{symbol} Price")
        df = live_price(symbol)
        if df is None or df.empty:
            df = load_regime_history(symbol)
        _candles(symbol, df)
    if toggles.get("regime_history"):
        rh = load_regime_history(symbol)
        cols = [c for c in ("regime_prob", "weight") if rh is not None and c in rh]
        if cols:
            st.subheader("Regime history")
            st.line_chart(rh[cols])
    if toggles.get("transition"):
        st.subheader("Transition matrix")
        _transition_heatmap(snap.get("transition_matrix") or {})
    if toggles.get("model_info"):
        _model_info(snap)
    if toggles.get("logs"):
        st.subheader("Signal feed / logs")
        sigs = snap.get("recent_signals") or []
        st.dataframe(sigs, width="stretch", hide_index=True) if sigs \
            else st.info("No signals yet.")


def _symbol_options() -> list[str]:
    """Picker options: SPY (regime proxy) + every name in the current book."""
    book = load_book_snapshot()
    syms = {"SPY"}
    for t in book.get("targets", []):
        if t.get("symbol"):
            syms.add(str(t["symbol"]))
    return sorted(syms)


# ----------------------------------------------------------------- sidebar ---
st.sidebar.title("Regime Trader")
interval = st.sidebar.slider("Refresh interval (s)", 2, 60, 10)
_opts = _symbol_options()
symbol = st.sidebar.selectbox(
    "Stock — price chart", _opts,
    index=_opts.index("SPY") if "SPY" in _opts else 0,
    help="Pick any name in the book to see its candlestick chart. SPY = the "
         "regime proxy. The chart pulls live price from Alpaca.",
)
toggles = {
    "cross_book": st.sidebar.checkbox("Show cross-sectional book", value=True),
    "macro": st.sidebar.checkbox("Show macro calendar + news", value=True),
    "price": st.sidebar.checkbox("Show price chart", value=True),
    "regime_history": st.sidebar.checkbox("Show regime history", value=True),
    "transition": st.sidebar.checkbox("Show transition matrix", value=False),
    "logs": st.sidebar.checkbox("Show logs", value=False),
    "model_info": st.sidebar.checkbox("Show model info", value=False),
}
st.sidebar.caption("Live account/positions/price refresh on the interval; "
                   "regime + risk come from the daily snapshot.")


@st.fragment(run_every=interval)
def _live():
    render(symbol, toggles)


_live()
