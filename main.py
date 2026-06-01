"""regime-trader — entry point and live orchestration.

Wires config, broker, data, core (HMM + strategy + risk), and monitoring into a
runnable trading system. Modes (mutually exclusive CLI flags):

* ``--backtest``    walk-forward backtester (``--compare`` adds benchmarks)
* ``--stress-test`` crash / gap / misclassification stress probes
* ``--train-only``  train the HMM and exit
* ``--dry-run``     full live pipeline on historical bars, **no orders** (the
                    only end-to-end integration path without broker credentials)
* ``--dashboard``   render the dashboard for the last saved state
* ``--live``        paper/live trading loop (default)

The live loop is structured as a :class:`TradingSystem` with a pure, testable
``process_symbol`` core; the WebSocket/broker plumbing around it is thin and
not unit-tested (no credentials / live market available).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal as signal_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

STATE_SNAPSHOT = "state_snapshot.json"
MODEL_DIR = "models"
HMM_MAX_AGE_DAYS = 7


# ===========================================================================
# Config / credentials
# ===========================================================================
def load_config(path: str = "config/settings.yaml") -> dict[str, Any]:
    """Load YAML settings into a nested dict.

    Args:
        path: Path to settings.yaml.

    Returns:
        Parsed configuration.
    """
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_credentials() -> dict[str, str]:
    """Load broker credentials from environment (``.env`` if present).

    Returns:
        Credential fields: ``api_key``, ``secret_key``, ``paper``.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return {
        "api_key": os.environ.get("ALPACA_API_KEY", ""),
        "secret_key": os.environ.get("ALPACA_SECRET_KEY", ""),
        "paper": os.environ.get("ALPACA_PAPER", "true"),
    }


def _build_dataclass(cls: type, section: dict[str, Any]):
    """Instantiate a dataclass from a settings section, ignoring unknown keys.

    Args:
        cls: Target dataclass type.
        section: Settings sub-dict.

    Returns:
        Constructed dataclass instance.
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (section or {}).items() if k in valid})


def _needs_retrain(model_path: str | Path, max_age_days: int = HMM_MAX_AGE_DAYS) -> bool:
    """Whether the HMM model is missing or older than ``max_age_days``.

    Args:
        model_path: Path to the pickled HMM.
        max_age_days: Max model age before a retrain is required.

    Returns:
        True if the model is absent or stale.
    """
    p = Path(model_path)
    if not p.exists():
        return True
    age_days = (datetime.now().timestamp() - p.stat().st_mtime) / 86400.0
    return age_days > max_age_days


# ===========================================================================
# Trading system (live orchestration)
# ===========================================================================
class TradingSystem:
    """Orchestrates the per-bar decision pipeline.

    ``process_symbol`` is the pure, testable core: features → filtered HMM →
    stability/flicker → strategy allocation → risk veto → (dry-run log | live
    order). Drawdown circuit-breaker resets are driven off bar-timestamp
    day/week rollovers. The broker/WebSocket plumbing lives in
    :meth:`run_stream` and is not unit-tested.
    """

    def __init__(
        self,
        config: dict[str, Any],
        hmm,
        orchestrator,
        risk_manager,
        feature_engineer,
        order_executor=None,
        position_tracker=None,
        tlogger=None,
        alerts=None,
        dashboard=None,
        dry_run: bool = True,
    ) -> None:
        """Wire the system components.

        Args:
            config: Parsed settings.
            hmm: Fitted :class:`~core.hmm_engine.HMMEngine`.
            orchestrator: :class:`~core.regime_strategies.StrategyOrchestrator`.
            risk_manager: :class:`~core.risk_manager.RiskManager`.
            feature_engineer: :class:`~data.feature_engineering.FeatureEngineer`.
            order_executor: Live order executor (None in dry-run).
            position_tracker: Live position tracker (None in dry-run).
            tlogger: Optional structured TradingLogger.
            alerts: Optional AlertManager.
            dashboard: Optional Dashboard.
            dry_run: If True, never submit orders.
        """
        self.config = config
        self.hmm = hmm
        self.orchestrator = orchestrator
        self.risk = risk_manager
        self.fe = feature_engineer
        self.executor = order_executor
        self.tracker = position_tracker
        self.tlog = tlogger
        self.alerts = alerts
        self.dashboard = dashboard
        self.dry_run = dry_run

        self.symbols: list[str] = config.get("broker", {}).get("symbols", [])
        self.initial_capital = float(config.get("backtest", {}).get("initial_capital", 100000))
        self.buffers: dict[str, pd.DataFrame] = {}
        self.recent_signals: list[dict] = []
        self.last_regime = None
        self._cur_day = None
        self._cur_week = None

    def ingest_bar(self, symbol: str, bar: pd.DataFrame) -> None:
        """Append a new bar (1-row OHLCV frame) to a symbol's rolling buffer.

        Args:
            symbol: Ticker.
            bar: Single-row OHLCV DataFrame indexed by timestamp.
        """
        buf = self.buffers.get(symbol)
        self.buffers[symbol] = bar if buf is None else pd.concat([buf, bar])

    def process_symbol(self, symbol: str) -> list[tuple]:
        """Run the decision pipeline for one symbol's current buffer.

        Args:
            symbol: Ticker to process.

        Returns:
            List of ``(Signal, RiskDecision)`` produced this bar.
        """
        buf = self.buffers.get(symbol)
        if buf is None or buf.empty:
            return []
        feats = self.fe.build_features(buf)
        if feats.empty:
            return []

        try:
            states = self.hmm.predict_regime_filtered(feats)
        except Exception as exc:  # noqa: BLE001 - HMM failure -> hold current regime
            logging.getLogger(__name__).error("HMM inference failed; holding regime: %s", exc)
            return []
        regime = states[-1]
        self.last_regime = regime
        flicker = self.hmm.is_flickering()
        if flicker and self.alerts:
            self.alerts.flicker_exceeded(self.hmm.get_regime_flicker_rate(),
                                         self.hmm.config.flicker_threshold)

        self._maybe_rollover(regime.timestamp)

        signals = self.orchestrator.generate_signals(
            [symbol], {symbol: buf}, regime, is_flickering=flicker
        )
        ps = self._portfolio_state(regime)
        out: list[tuple] = []
        for sig in signals:
            decision = self.risk.validate_signal(sig, ps)
            shares = (decision.modified_signal.metadata.get("approved_shares", 0)
                      if decision.approved and decision.modified_signal else 0)
            if decision.approved:
                if self.dry_run:
                    self._record_signal("would_submit", sig, shares, decision.modifications)
                else:  # pragma: no cover - live order path
                    self.executor.submit_signal(decision.modified_signal)
                    self.risk.record_trade()
                    self._record_signal("submitted", sig, shares, decision.modifications)
            else:
                self._record_signal("rejected", sig, 0, [decision.rejection_reason])
            out.append((sig, decision))

        self._update_dashboard_context(regime, ps)
        return out

    def update_trailing_stops(self) -> None:  # pragma: no cover - needs live positions
        """Tighten protective stops per the current regime's strategy.

        For each open position, recompute the strategy stop and call
        ``modify_stop`` (tighten-only). No-ops for positions whose bracket
        stop-leg id was not captured — see the go-live review.
        """
        if not self.tracker or not self.executor:
            return
        for sym, pos in (self.tracker._positions or {}).items():
            buf = self.buffers.get(sym)
            if buf is None or self.last_regime is None:
                continue
            sigs = self.orchestrator.generate_signals([sym], {sym: buf}, self.last_regime)
            if not sigs:
                continue
            new_stop = sigs[0].stop_loss
            if pos.stop_order_id and new_stop > pos.stop_level:
                self.executor.modify_stop(sym, new_stop, current_stop=pos.stop_level,
                                          stop_order_id=pos.stop_order_id)
                pos.stop_level = new_stop

    def _portfolio_state(self, regime):
        """Build the risk-layer PortfolioState for the current bar."""
        from core.risk_manager import PortfolioState

        if self.tracker and not self.dry_run:  # pragma: no cover - live broker
            snap = self.tracker.refresh()
            equity, positions = snap.equity, self.tracker.to_risk_positions()
        else:
            equity, positions = self.initial_capital, []
        return PortfolioState(
            equity=equity, positions=positions,
            flicker_rate=self.hmm.get_regime_flicker_rate(),
            circuit_breaker_status=self.risk.state,
        )

    def _maybe_rollover(self, ts) -> None:
        """Reset daily/weekly breaker latches on date/week rollover."""
        if ts is None:
            return
        ts = pd.Timestamp(ts)
        day = ts.date()
        week = ts.isocalendar()[:2]
        if self._cur_day is not None and day != self._cur_day:
            self.risk.reset_daily()
        if self._cur_week is not None and week != self._cur_week:
            self.risk.breaker.reset_weekly()
        self._cur_day, self._cur_week = day, week

    def _record_signal(self, action: str, sig, shares: int, notes: list) -> None:
        """Log a signal decision and append it to the recent-signals ring."""
        entry = {
            "time": str(sig.timestamp) if sig.timestamp else "",
            "symbol": sig.symbol, "action": action, "shares": shares,
            "change": f"{sig.position_size_pct:.0%}", "note": "; ".join(n for n in notes if n),
        }
        self.recent_signals.append(entry)
        self.recent_signals = self.recent_signals[-20:]
        if self.tlog:
            self.tlog.log(self.tlog.trades, f"signal_{action}", sig.reasoning,
                          symbol=sig.symbol, approved_shares=shares, regime=sig.regime_name)

    def _update_dashboard_context(self, regime, ps) -> None:
        """Push the latest regime/portfolio context into the logger."""
        if self.tlog:
            self.tlog.set_context(
                regime=regime.label.value if hasattr(regime.label, "value") else str(regime.label),
                probability=round(regime.probability, 4), equity=ps.equity,
                positions=len(ps.positions),
            )

    def run_stream(self, market_data) -> None:  # pragma: no cover - live WebSocket
        """Subscribe to live bars and drive the loop (thin plumbing).

        Args:
            market_data: A connected :class:`~data.market_data.MarketData`.
        """
        timeframe = self.config.get("broker", {}).get("timeframe", "1Day")

        def on_bar(symbol, bar):
            row = pd.DataFrame(
                [{"open": bar.open, "high": bar.high, "low": bar.low,
                  "close": bar.close, "volume": bar.volume}],
                index=[pd.Timestamp(bar.timestamp)],
            )
            self.ingest_bar(symbol, row)
            self.process_symbol(symbol)
            self.update_trailing_stops()

        market_data.subscribe_bars(self.symbols, on_bar)

    # ----------------------------------------------------- state snapshot ---
    def save_state(self, path: str = STATE_SNAPSHOT) -> None:
        """Persist recovery state to ``state_snapshot.json``.

        Args:
            path: Destination path.
        """
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_regime": (self.last_regime.label.value
                            if self.last_regime and hasattr(self.last_regime.label, "value")
                            else None),
            "equity_peak": self.risk._equity_peak,
            "daily_trades": self.risk._daily_trades,
            "risk_state": self.risk.state.value,
            "breaker_events": len(self.risk.breaker.get_history()),
            "recent_signals": self.recent_signals[-20:],
        }
        Path(path).write_text(json.dumps(state, indent=2, default=str))
        logging.getLogger(__name__).info("Saved state snapshot to %s", path)

    def load_state(self, path: str = STATE_SNAPSHOT) -> Optional[dict]:
        """Restore recovery state if a snapshot exists.

        Args:
            path: Snapshot path.

        Returns:
            The loaded state dict, or None if absent.
        """
        p = Path(path)
        if not p.exists():
            return None
        state = json.loads(p.read_text())
        self.risk._equity_peak = state.get("equity_peak", 0.0)
        self.recent_signals = state.get("recent_signals", [])
        logging.getLogger(__name__).info("Recovered state from %s (%s)", path,
                                         state.get("timestamp"))
        return state


# ===========================================================================
# Mode entry points
# ===========================================================================
def _core_configs(config: dict[str, Any]):
    """Build the core dataclass configs from settings (shared by modes)."""
    from core.hmm_engine import HMMConfig
    from core.regime_strategies import StrategyConfig
    from core.risk_manager import RiskConfig

    return (
        _build_dataclass(HMMConfig, config.get("hmm", {})),
        _build_dataclass(StrategyConfig, config.get("strategy", {})),
        _build_dataclass(RiskConfig, config.get("risk", {})),
    )


def run_train(config: dict[str, Any], symbols: list[str],
              start: Optional[str] = None, end: Optional[str] = None) -> str:
    """Train the HMM on historical data and save it; then exit.

    Args:
        config: Parsed settings.
        symbols: Tickers (first is used).
        start: ISO start date.
        end: ISO end date.

    Returns:
        Path to the saved model.
    """
    from core.hmm_engine import HMMEngine
    from data.feature_engineering import FeatureEngineer
    from data.market_data import load_ohlcv

    hmm_cfg, _, _ = _core_configs(config)
    symbol = symbols[0]
    timeframe = config.get("broker", {}).get("timeframe", "1Day")
    print(f"Training HMM on {symbol} ({timeframe}) ...")
    ohlcv = load_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    feats = FeatureEngineer().build_features(ohlcv)
    hmm = HMMEngine(hmm_cfg)
    hmm.fit(feats)
    Path(MODEL_DIR).mkdir(exist_ok=True)
    path = f"{MODEL_DIR}/hmm_{symbol}.pkl"
    hmm.save(path)
    print(f"  Trained: {hmm.n_regimes} regimes, BIC {hmm.metadata.bic:.0f} -> {path}")
    return path


def run_dry_run(config: dict[str, Any], symbols: list[str],
                start: Optional[str] = None, end: Optional[str] = None,
                stream_bars: int = 20) -> dict[str, int]:
    """Run the full live pipeline on historical daily bars — no orders.

    The only end-to-end integration path available without broker credentials:
    fits the HMM on history, then replays the final ``stream_bars`` as if
    streamed, exercising features → HMM → strategy → risk → (logged) decision.

    Args:
        config: Parsed settings.
        symbols: Tickers (first is traded).
        start: ISO start date.
        end: ISO end date.
        stream_bars: Number of trailing bars to replay.

    Returns:
        Summary counts: bars, signals, approved, rejected.
    """
    from core.hmm_engine import HMMConfig, HMMEngine
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import RiskManager
    from data.feature_engineering import FeatureEngineer
    from monitoring.logger import LoggerConfig, setup_logging

    hmm_cfg, strat_cfg, risk_cfg = _core_configs(config)
    # moderately reduced restarts keep the dry-run fast; pipeline is identical
    hmm_cfg = dataclasses.replace(hmm_cfg, n_init=min(hmm_cfg.n_init, 3))

    symbol = symbols[0]
    timeframe = config.get("broker", {}).get("timeframe", "1Day")
    from data.market_data import load_ohlcv

    print(f"[dry-run] loading {symbol} ({timeframe}) ...")
    ohlcv = load_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    fe = FeatureEngineer()
    feats_all = fe.build_features(ohlcv)
    if len(feats_all) <= stream_bars:
        raise ValueError(f"need > {stream_bars} usable bars, got {len(feats_all)}")

    # fit on everything before the streamed tail (no look-ahead into the replay)
    cutoff_ts = feats_all.index[-stream_bars]
    fit_feats = feats_all[feats_all.index < cutoff_ts]
    print(f"[dry-run] fitting HMM on {len(fit_feats)} bars ...")
    hmm = HMMEngine(hmm_cfg)
    hmm.fit(fit_feats)
    orch = StrategyOrchestrator(strat_cfg, hmm.regime_info)
    tlog = setup_logging(LoggerConfig(log_dir="logs", console=False))

    sys_ = TradingSystem(config, hmm, orch, RiskManager(risk_cfg), fe,
                         tlogger=tlog, dry_run=True)

    counts = {"bars": 0, "signals": 0, "approved": 0, "rejected": 0}
    start_i = len(ohlcv) - stream_bars
    for i in range(start_i, len(ohlcv)):
        sys_.buffers[symbol] = ohlcv.iloc[: i + 1]
        for sig, dec in sys_.process_symbol(symbol):
            counts["signals"] += 1
            if dec.approved and dec.modified_signal.metadata.get("approved_shares", 0) > 0:
                counts["approved"] += 1
            elif not dec.approved:
                counts["rejected"] += 1
        counts["bars"] += 1

    print(f"[dry-run] {counts['bars']} bars, {counts['signals']} signals, "
          f"{counts['approved']} approved, {counts['rejected']} rejected (NO orders sent)")
    return counts


def run_dashboard(config: dict[str, Any]) -> None:
    """Render the dashboard from the last saved state snapshot.

    Args:
        config: Parsed settings.
    """
    from monitoring.dashboard import Dashboard, DashboardConfig, DashboardState

    db = Dashboard(DashboardConfig(
        refresh_seconds=config.get("monitoring", {}).get("dashboard_refresh_seconds", 5)
    ))
    state = DashboardState(mode="PAPER")
    p = Path(STATE_SNAPSHOT)
    if p.exists():
        snap = json.loads(p.read_text())
        state.regime_name = snap.get("last_regime") or "—"
        state.recent_signals = snap.get("recent_signals", [])
    else:
        print("No state_snapshot.json found; showing an empty frame.")
    db.render(state)


def run_live(config: dict[str, Any], credentials: dict[str, str],
             dry_run: bool = False) -> None:  # pragma: no cover - live broker/market
    """Run the paper/live trading loop (startup → stream → shutdown).

    Args:
        config: Parsed settings.
        credentials: Broker credentials.
        dry_run: If True, run the pipeline without submitting orders.
    """
    from broker.alpaca_client import AlpacaClient, AlpacaConfig
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from core.hmm_engine import HMMEngine
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import RiskManager
    from data.feature_engineering import FeatureEngineer
    from data.market_data import MarketData
    from monitoring.alerts import AlertConfig, AlertManager
    from monitoring.logger import LoggerConfig, setup_logging

    tlog = setup_logging(LoggerConfig(log_dir="logs"))
    alerts = AlertManager(_build_dataclass(AlertConfig, config.get("monitoring", {})), tlog)
    hmm_cfg, strat_cfg, risk_cfg = _core_configs(config)
    symbols = config.get("broker", {}).get("symbols", [])
    symbol = symbols[0]
    timeframe = config.get("broker", {}).get("timeframe", "1Day")

    # --- STARTUP ---
    paper = str(credentials.get("paper", "true")).lower() != "false"
    client = AlpacaClient(AlpacaConfig(credentials["api_key"], credentials["secret_key"], paper=paper))
    client.connect()
    account = client.get_account()
    tlog.log(tlog.main, "startup", f"account equity {account['equity']}", mode="PAPER" if paper else "LIVE")

    if not client.is_market_open():
        tlog.log(tlog.main, "market_closed", "market closed; exiting", level="WARNING")
        return

    md = MarketData(client)
    model_path = f"{MODEL_DIR}/hmm_{symbol}.pkl"
    fe = FeatureEngineer()
    if _needs_retrain(model_path):
        tlog.log(tlog.main, "hmm_retrain", "model missing/stale; training")
        run_train(config, symbols)
    hmm = HMMEngine.load(model_path)
    alerts.hmm_retrained(hmm.n_regimes, hmm.metadata.bic if hmm.metadata else 0.0)

    risk = RiskManager(risk_cfg)
    risk._equity_peak = account["equity"]
    orch = StrategyOrchestrator(strat_cfg, hmm.regime_info)
    tracker = PositionTracker(client)
    tracker.sync_on_startup()
    executor = OrderExecutor(client)

    system = TradingSystem(config, hmm, orch, risk, fe, executor, tracker,
                           tlogger=tlog, alerts=alerts, dry_run=dry_run)
    system.load_state()
    tlog.log(tlog.main, "system_online", "System online")

    # --- SHUTDOWN handlers: keep positions (stops in place), save state ---
    def _shutdown(signum, frame):
        tlog.log(tlog.main, "shutdown", f"signal {signum}; saving state, keeping positions")
        system.save_state()
        raise SystemExit(0)

    signal_mod.signal(signal_mod.SIGINT, _shutdown)
    signal_mod.signal(signal_mod.SIGTERM, _shutdown)

    # --- MAIN LOOP (WebSocket-driven) ---
    try:
        system.run_stream(md)
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 - unhandled: log, save, alert
        logging.getLogger(__name__).exception("unhandled error")
        alerts.send("fatal_error", str(exc))
        system.save_state()
        raise


# ===========================================================================
# Backtest mode (Phase 4 — unchanged)
# ===========================================================================
def run_backtest(
    config: dict[str, Any],
    symbols: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    compare: bool = False,
    stress_test: bool = False,
) -> None:
    """Run a walk-forward backtest and print a performance report.

    Wires data (yfinance) -> features -> HMM -> strategy -> risk -> backtester,
    then renders metrics and writes CSV artifacts. Trades the first symbol (the
    backtester is a single-asset allocation sleeve).

    Args:
        config: Parsed settings.
        symbols: Tickers to backtest (first is traded).
        start: ISO start date (inclusive).
        end: ISO end date (inclusive).
        compare: Also run the benchmark suite (buy-hold / 200-SMA / random).
        stress_test: Run crash/gap/misclassification stress probes.
    """
    from backtest.backtester import BacktestConfig, Backtester
    from backtest.performance import PerformanceAnalyzer, export_csvs, render_report
    from core.hmm_engine import HMMConfig, HMMEngine
    from core.regime_strategies import StrategyConfig
    from core.risk_manager import RiskConfig, RiskManager
    from data.feature_engineering import FeatureEngineer
    from data.market_data import load_ohlcv

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    symbol = symbols[0]
    timeframe = config.get("broker", {}).get("timeframe", "1Day")
    print(f"Loading {symbol} ({start or 'max'}..{end or 'now'}, {timeframe}) ...")
    ohlcv = load_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    print(f"  {len(ohlcv)} bars loaded.")

    bt_cfg = _build_dataclass(BacktestConfig, config.get("backtest", {}))
    hmm_cfg = _build_dataclass(HMMConfig, config.get("hmm", {}))
    strat_cfg = _build_dataclass(StrategyConfig, config.get("strategy", {}))
    risk_cfg = _build_dataclass(RiskConfig, config.get("risk", {}))

    if stress_test:
        # Reduced HMM + coarser folds keep the Monte-Carlo runtime tractable.
        from backtest.stress_test import StressTester, render_stress_reports

        fast_hmm = HMMConfig(n_candidates=[3], n_init=2,
                             min_train_bars=hmm_cfg.min_train_bars)
        bt = Backtester(
            dataclasses.replace(bt_cfg, step_size=max(bt_cfg.test_window, 189)),
            HMMEngine(fast_hmm), strat_cfg, RiskManager(risk_cfg), FeatureEngineer(),
        )
        st = StressTester(bt)
        n = 30
        print(f"Running stress probes ({n} sims each; reduced HMM for speed) ...")
        reports = [
            st.crash_injection_mc({symbol: ohlcv}, n_sims=n),
            st.gap_risk_mc({symbol: ohlcv}, n_sims=n),
            st.regime_misclassification({symbol: ohlcv}, n_sims=n),
        ]
        render_stress_reports(reports)
        return

    bt = Backtester(bt_cfg, HMMEngine(hmm_cfg), strat_cfg, RiskManager(risk_cfg),
                    FeatureEngineer())
    print("Running walk-forward backtest ...")
    try:
        result = bt.run({symbol: ohlcv})
    except ValueError as exc:
        print(f"\nBacktest aborted: {exc}")
        print("  Hint: widen the date range — one fold needs "
              f"~{bt_cfg.train_window + bt_cfg.test_window} usable bars after the "
              "~450-bar feature warmup (≈4+ years of daily data).")
        return

    analyzer = PerformanceAnalyzer(risk_free_rate=bt_cfg.risk_free_rate)
    report = analyzer.analyze(result, ohlcv["close"], with_benchmarks=compare)
    render_report(result, report)

    outdir = f"backtest_output/{symbol}"
    paths = export_csvs(result, report, outdir)
    print(f"\nArtifacts written to {outdir}/:")
    for name, p in paths.items():
        print(f"  - {p}")


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments (one parser, mutually-exclusive mode flags).

    Modes::

        main.py --backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
        main.py --backtest --compare --symbols SPY --start ... --end ...
        main.py --stress-test
        main.py --train-only --symbols SPY
        main.py --dry-run --symbols SPY
        main.py --dashboard
        main.py --live           (or no flag -> default live)

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed args namespace.
    """
    parser = argparse.ArgumentParser(prog="regime-trader")
    parser.add_argument("--config", default="config/settings.yaml", help="path to settings.yaml")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--backtest", action="store_true", help="walk-forward backtest")
    mode.add_argument("--stress-test", action="store_true", dest="stress_test",
                      help="run crash/gap stress scenarios")
    mode.add_argument("--train-only", action="store_true", dest="train_only",
                      help="train the HMM and exit")
    mode.add_argument("--dry-run", action="store_true", dest="dry_run",
                      help="full pipeline on history, no orders")
    mode.add_argument("--dashboard", action="store_true", help="render dashboard for last state")
    mode.add_argument("--live", action="store_true", help="paper/live trading loop (default)")

    parser.add_argument("--compare", action="store_true", help="add benchmark comparison (backtest)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="tickers (default: broker.symbols from settings)")
    parser.add_argument("--start", default=None, help="ISO start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="ISO end date (YYYY-MM-DD)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Program entry point: parse args, load config, dispatch the mode.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    config = load_config(args.config)
    symbols = args.symbols or config.get("broker", {}).get("symbols", [])

    if args.backtest:
        run_backtest(config, symbols, args.start, args.end, args.compare, False)
    elif args.stress_test:
        run_backtest(config, symbols, args.start, args.end, False, True)
    elif args.train_only:
        run_train(config, symbols, args.start, args.end)
    elif args.dry_run:
        run_dry_run(config, symbols, args.start, args.end)
    elif args.dashboard:
        run_dashboard(config)
    else:  # default: live
        run_live(config, load_credentials())


if __name__ == "__main__":
    main()
