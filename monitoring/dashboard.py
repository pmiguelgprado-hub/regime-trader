"""Terminal-based live dashboard.

Renders a live view of regime, portfolio, positions, recent signals, risk
status, and system health using ``rich``. The renderable is built from a plain
:class:`DashboardState` (pure, unit-testable); the live-refresh loop is thin
plumbing on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class DashboardConfig:
    """Configuration for the live dashboard."""

    refresh_seconds: int = 5


@dataclass
class DashboardState:
    """All values shown on one dashboard frame (decoupled from live objects)."""

    # regime
    regime_name: str = "—"
    regime_prob: float = 0.0
    stability_bars: int = 0
    flicker_rate: int = 0
    flicker_window: int = 20
    # portfolio
    equity: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    allocation: float = 0.0
    leverage: float = 1.0
    # positions: list of dicts(symbol, side, price, pnl_pct, stop, holding)
    positions: list[dict] = field(default_factory=list)
    # recent signals: list of dicts(time, symbol, change, note)
    recent_signals: list[dict] = field(default_factory=list)
    # risk
    daily_dd: float = 0.0
    daily_dd_limit: float = 0.03
    peak_dd: float = 0.0
    peak_dd_limit: float = 0.10
    # system
    data_ok: bool = True
    api_ok: bool = True
    api_ms: float = 0.0
    hmm_age: str = "—"
    mode: str = "PAPER"


def _risk_bar(value: float, limit: float, label: str) -> Text:
    """Render a color-coded ``used/limit`` risk indicator.

    Args:
        value: Current drawdown fraction (positive magnitude).
        limit: Limit fraction.
        label: Display label.

    Returns:
        A `rich` Text colored green/yellow/red by proximity to the limit.
    """
    ratio = value / limit if limit > 0 else 0.0
    if ratio < 0.5:
        color, mark = "green", "✅"
    elif ratio < 1.0:
        color, mark = "yellow", "⚠️"
    else:
        color, mark = "red", "🛑"
    return Text(f"{label}: {value:.1%}/{limit:.0%} {mark}", style=color)


class Dashboard:
    """Renders a live terminal dashboard via ``rich``."""

    def __init__(self, config: DashboardConfig) -> None:
        """Initialize the dashboard.

        Args:
            config: Refresh cadence.
        """
        self.config = config

    def build_renderable(self, state: DashboardState) -> Group:
        """Build the full 6-panel dashboard renderable from state.

        Args:
            state: The frame's values.

        Returns:
            A `rich` ``Group`` of panels (REGIME, PORTFOLIO, POSITIONS, RECENT
            SIGNALS, RISK STATUS, SYSTEM).
        """
        regime = Panel(
            f"[bold]{state.regime_name.upper()}[/bold] ({state.regime_prob:.0%})  |  "
            f"Stability: {state.stability_bars} bars  |  "
            f"Flicker: {state.flicker_rate}/{state.flicker_window}",
            title="REGIME", title_align="left",
        )

        pnl_color = "green" if state.daily_pnl >= 0 else "red"
        portfolio = Panel(
            f"Equity: ${state.equity:,.0f}  |  "
            f"Daily: [{pnl_color}]{state.daily_pnl:+,.0f} ({state.daily_pnl_pct:+.2%})[/{pnl_color}]\n"
            f"Allocation: {state.allocation:.0%}  |  Leverage: {state.leverage:.2f}x",
            title="PORTFOLIO", title_align="left",
        )

        pos_tbl = Table.grid(padding=(0, 2))
        for p in state.positions:
            pcolor = "green" if p.get("pnl_pct", 0) >= 0 else "red"
            pos_tbl.add_row(
                p.get("symbol", "?"), p.get("side", "LONG"),
                f"${p.get('price', 0):,.2f}",
                Text(f"{p.get('pnl_pct', 0):+.1%}", style=pcolor),
                f"Stop: ${p.get('stop', 0):,.2f}", p.get("holding", "—"),
            )
        if not state.positions:
            pos_tbl.add_row("(flat — no open positions)")
        positions = Panel(pos_tbl, title="POSITIONS", title_align="left")

        sig_tbl = Table.grid(padding=(0, 2))
        for s in state.recent_signals[-5:]:
            sig_tbl.add_row(s.get("time", ""), s.get("symbol", ""),
                            s.get("change", ""), s.get("note", ""))
        if not state.recent_signals:
            sig_tbl.add_row("(no signals yet)")
        signals = Panel(sig_tbl, title="RECENT SIGNALS", title_align="left")

        risk_tbl = Table.grid(padding=(0, 3))
        risk_tbl.add_row(_risk_bar(state.daily_dd, state.daily_dd_limit, "Daily DD"),
                         _risk_bar(state.peak_dd, state.peak_dd_limit, "From Peak"))
        risk = Panel(risk_tbl, title="RISK STATUS", title_align="left")

        data_mark = "✅" if state.data_ok else "🛑"
        api_mark = "✅" if state.api_ok else "🛑"
        mode_color = "yellow" if state.mode == "PAPER" else "red"
        system = Panel(
            f"Data: {data_mark}  |  API: {api_mark} {state.api_ms:.0f}ms  |  "
            f"HMM: {state.hmm_age}  |  [{mode_color}]{state.mode}[/{mode_color}]",
            title="SYSTEM", title_align="left",
        )

        return Group(regime, portfolio, positions, signals, risk, system)

    def render(self, state: DashboardState) -> None:
        """Print a single dashboard frame to the terminal.

        Args:
            state: The frame's values.
        """
        from rich.console import Console

        Console().print(self.build_renderable(state))

    def render_to_str(self, state: DashboardState, width: int = 80) -> str:
        """Render a frame to a string (for tests / capture).

        Args:
            state: The frame's values.
            width: Console width.

        Returns:
            The rendered frame as plain text.
        """
        import io

        from rich.console import Console

        con = Console(record=True, width=width, file=io.StringIO())
        con.print(self.build_renderable(state))
        return con.export_text()

    def run(
        self, state_provider: Callable[[], DashboardState]
    ) -> None:  # pragma: no cover - live loop
        """Start the live-refresh loop until interrupted.

        Args:
            state_provider: Zero-arg callable returning the latest state.
        """
        from rich.live import Live

        with Live(self.build_renderable(state_provider()),
                  refresh_per_second=1 / max(self.config.refresh_seconds, 1)) as live:
            try:
                while True:
                    import time

                    time.sleep(self.config.refresh_seconds)
                    live.update(self.build_renderable(state_provider()))
            except KeyboardInterrupt:
                return
