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
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# T0.4 (R-4 root cause): multithreaded BLAS makes floating-point reductions
# order-nondeterministic — two identical HMM fits in the same process diverged at
# ~5e-13, EM iteration amplified it, and near-tie restarts then picked different
# winners (the 0.37-0.49 Sharpe band across runs). Pin numeric libs to one thread
# BEFORE numpy/pandas load. setdefault, so an explicit env override still wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd
import yaml

STATE_SNAPSHOT = "state_snapshot.json"
BOOK_SNAPSHOT = "book_snapshot.json"
BOOK_SNAPSHOT_CHALLENGER = "book_snapshot_challenger.json"
BOOK_SNAPSHOT_QUALITY = "book_snapshot_quality.json"
TRACK_RECORD_CSV = "track_record.csv"
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


def load_pinned_champion(registry: "Any", symbol: str, legacy_path: str | Path,
                         train_fn: "Any") -> "tuple[Any, Optional[str]]":
    """Load the pinned champion HMM, bootstrapping the registry once (T0.4).

    Pin-champion operative amendment (2026-06-12): live runs load the registry
    champion instead of refitting by age — refit on end-date="now" data produced a
    different model every week, making the live book non-reproducible (R-4). On the
    first call with an empty registry, the legacy single pickle (``legacy_path``) is
    adopted: trained fresh if missing/stale (identical to what the old rule would
    have run that day), then versioned and promoted. Refit-by-drift arrives in T3.3;
    until then promotion is manual.

    Args:
        registry: :class:`core.model_registry.ModelRegistry`.
        symbol: Ticker namespace of the model.
        legacy_path: The pre-registry pickle (``models/hmm_<symbol>.pkl``).
        train_fn: Zero-arg callable that (re)writes ``legacy_path``.

    Returns:
        ``(engine, expected_hash)`` — the loaded champion and the transition hash
        recorded at promotion time (the reference for the daily drift assert).
    """
    from core.hmm_engine import HMMEngine

    hmm = registry.load_champion(symbol)
    if hmm is None:
        if _needs_retrain(legacy_path):
            train_fn()
        hmm = HMMEngine.load(legacy_path)
        version = registry.save_version(hmm, symbol)
        registry.promote(symbol, version)
        logging.getLogger(__name__).info(
            "model registry bootstrapped: %s promoted as champion for %s", version, symbol)
    return hmm, registry.champion_hash(symbol)


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
        registry=None,
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
        self.registry = registry

        self.symbols: list[str] = config.get("broker", {}).get("symbols", [])
        self.initial_capital = float(config.get("backtest", {}).get("initial_capital", 100000))
        self.buffers: dict[str, pd.DataFrame] = {}
        self.recent_signals: list[dict] = []
        self.last_regime = None
        self._cur_day = None
        self._cur_week = None
        self._pending_stops: dict[str, str] = {}  # symbol -> bracket stop-leg id awaiting its fill
        self._last_equity: Optional[float] = None  # for per-bar MtM return fed to the breaker
        self._last_bar_ts = None                    # S-2 watchdog heartbeat (last bar received)
        # Rolling-buffer cap (H3): min_train_bars + z-score(252)/SMA200 warmup margin.
        # Bounds memory and per-bar feature cost; also the live backfill depth.
        self._buffer_cap = int(config.get("hmm", {}).get("min_train_bars", 504)) + 260

    def ingest_bar(self, symbol: str, bar: pd.DataFrame) -> None:
        """Append a new bar (1-row OHLCV frame) to a symbol's rolling buffer.

        Args:
            symbol: Ticker.
            bar: Single-row OHLCV DataFrame indexed by timestamp.
        """
        buf = self.buffers.get(symbol)
        merged = bar if buf is None else pd.concat([buf, bar])
        self.buffers[symbol] = merged.tail(self._buffer_cap)  # H3: bound the rolling window

    def seed_buffers(self, market_data) -> int:
        """Pre-fill rolling buffers with history so features are ready on bar 1.

        Without this, the live loop starts with cold buffers and
        ``build_features`` returns empty for ~450 warmup bars (z-score 252 +
        SMA200), making the bot a silent no-op at startup (audit C1). Seeds each
        configured symbol with ``min_train_bars`` plus a warmup margin, fetched
        at the configured timeframe.

        Args:
            market_data: Source exposing ``get_history(symbol, timeframe,
                lookback_bars)`` (e.g. :class:`~data.market_data.MarketData`).

        Returns:
            The lookback (bar count) requested per symbol.
        """
        timeframe = self.config.get("broker", {}).get("timeframe", "1Day")
        lookback = self._buffer_cap  # same depth the rolling buffer is capped to (H3)
        for sym in self.symbols:
            try:
                hist = market_data.get_history(sym, timeframe, lookback)
            except Exception as exc:  # noqa: BLE001 - skip a bad symbol, keep the rest
                if self.tlog:
                    self.tlog.log(self.tlog.main, "backfill_warn",
                                  f"history fetch failed for {sym}: {exc}", level="WARNING")
                continue
            if hist is None or hist.empty:
                if self.tlog:
                    self.tlog.log(self.tlog.main, "backfill_warn",
                                  f"no history for {sym}", level="WARNING")
                continue
            self.buffers[sym] = hist
        if self.tlog:
            self.tlog.log(self.tlog.main, "backfill_done",
                          f"seeded {len(self.buffers)} buffers (~{lookback} bars)")
        return lookback

    # ----------------------------------------------------- model lifecycle ---
    def install_model(self, new_hmm) -> None:
        """Swap the live HMM and propagate its regime map to the orchestrator (A-1).

        A refit that is not pushed into the orchestrator leaves the vol-rank
        ``regime_to_strategy`` map pointing at the *previous* model's states.
        Always rewire via ``update_regime_infos`` after replacing the engine.

        Args:
            new_hmm: A freshly fitted :class:`~core.hmm_engine.HMMEngine`.
        """
        self.hmm = new_hmm
        self.orchestrator.update_regime_infos(new_hmm.regime_info)

    def retrain_from_buffer(self, symbol: str) -> bool:
        """Refit the HMM on a symbol's live buffer and install it (A-1).

        Args:
            symbol: Symbol whose rolling buffer supplies the training window.

        Returns:
            True if a new model was fitted and installed; False if there is too
            little history to train (current model left untouched).
        """
        from core.hmm_engine import HMMEngine

        buf = self.buffers.get(symbol)
        if buf is None or buf.empty:
            return False
        feats = self.fe.build_features(buf)
        cfg = self.hmm.config
        if len(feats) < cfg.min_train_bars:
            return False
        new = HMMEngine(cfg)
        try:
            new.fit(feats)
        except Exception as exc:  # noqa: BLE001 - keep the current model on failure
            logging.getLogger(__name__).error("retrain failed; keeping current model: %s", exc)
            if self.alerts:
                self.alerts.send("retrain_failed", str(exc))
            return False
        # Promotion gate 1: never auto-install a fit that did not converge.
        if not getattr(new.metadata, "converged", False):
            logging.getLogger(__name__).warning("retrain did not converge; keeping current model")
            if self.alerts:
                self.alerts.send("retrain_rejected", "challenger did not converge")
            return False
        # Promotion gate 2 (A-4 champion-challenger): keep the current champion
        # unless the challenger explains a recent holdout at least as well.
        champion = self.hmm
        if getattr(champion, "model", None) is not None:
            tol = float(self.config.get("hmm", {}).get("challenger_tol", 0.0))
            holdout = feats.tail(min(len(feats), 252))
            try:
                challenger_ll = new.mean_log_likelihood(holdout)
                champion_ll = champion.mean_log_likelihood(holdout)
            except Exception as exc:  # noqa: BLE001 - if scoring fails, stay safe
                logging.getLogger(__name__).error("challenger eval failed; keeping champion: %s", exc)
                return False
            # 1e-9 epsilon absorbs floating-point noise between two equivalent fits
            # so an essentially-identical refit still promotes.
            if challenger_ll < champion_ll - tol - 1e-9:
                logging.getLogger(__name__).warning(
                    "challenger underperforms champion on holdout (%.4f < %.4f); keeping champion",
                    challenger_ll, champion_ll)
                if self.alerts:
                    self.alerts.send("retrain_rejected", "challenger worse than champion on holdout")
                return False
        self.install_model(new)
        if self.registry is not None:
            version = self.registry.save_version(new, symbol)
            self.registry.promote(symbol, version)
        if self.tlog:
            self.tlog.log(self.tlog.main, "hmm_retrain_inloop",
                          f"refit {symbol} ({len(feats)} bars); propagated to orchestrator")
        return True

    def _model_age_days(self) -> float:
        """Age of the in-memory model in days (``inf`` if unknown)."""
        md = getattr(self.hmm, "metadata", None)
        if not md or not getattr(md, "training_date", None):
            return float("inf")
        try:
            trained = datetime.fromisoformat(md.training_date)
        except ValueError:
            return float("inf")
        if trained.tzinfo is None:
            trained = trained.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - trained).total_seconds() / 86400.0

    def maybe_retrain(self, symbol: str) -> bool:
        """Retrain in-loop when the in-memory model is stale by age (A-1).

        **Opt-in** via ``hmm.auto_retrain`` (default off). Unsupervised
        auto-promotion of a refit has no full quality gate yet (only the
        convergence floor in :meth:`retrain_from_buffer`; champion-challenger is
        A-4), so it stays disabled by default — and the deployed ``--run-once``
        path already refreshes by file age at startup, making this redundant
        there. It matters for a long-running stream process.

        Args:
            symbol: Symbol buffer to retrain from.

        Returns:
            True if a retrain happened this call.
        """
        hcfg = self.config.get("hmm", {})
        if not hcfg.get("auto_retrain", False):
            return False
        if self._model_age_days() <= float(hcfg.get("max_age_days", HMM_MAX_AGE_DAYS)):
            return False
        return self.retrain_from_buffer(symbol)

    @staticmethod
    def _rebalance_order(
        target_shares: int, held_shares: int,
        target_weight: float, current_weight: float, threshold: float,
    ) -> Optional[tuple[str, int]]:
        """Decide the delta order to move a holding toward its target (C4).

        Trades only the difference between target and held, gated by a weight
        drift threshold so repeated same-target bars don't over-accumulate. A
        target that collapses to zero always liquidates (``must_exit``), even
        below the threshold.

        Args:
            target_shares: Risk-approved target position size (shares).
            held_shares: Shares currently held for the symbol.
            target_weight: Target portfolio weight.
            current_weight: Current portfolio weight of the holding.
            threshold: Minimum |weight drift| to act on.

        Returns:
            ``(side, qty)`` with ``side`` in {"buy", "sell"} and ``qty`` > 0, or
            ``None`` when no trade is warranted.
        """
        must_exit = target_shares == 0 and held_shares > 0
        if abs(target_weight - current_weight) < threshold and not must_exit:
            return None
        delta = int(target_shares) - int(held_shares)
        if delta == 0:
            return None
        return ("buy" if delta > 0 else "sell", abs(delta))

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
        self._update_risk_posture(ps.equity, regime)   # C2: feed MtM drawdown to the breaker
        ps.circuit_breaker_status = self.risk.state     # so validate_signal sees the fresh posture
        out: list[tuple] = []
        for sig in signals:
            decision = self.risk.validate_signal(sig, ps)
            shares = (decision.modified_signal.metadata.get("approved_shares", 0)
                      if decision.approved and decision.modified_signal else 0)
            if decision.approved:
                if self.dry_run:
                    self._record_signal("would_submit", sig, shares, decision.modifications)
                else:  # pragma: no cover - live order path
                    self._submit_rebalanced(symbol, sig, decision, shares, ps)
            else:
                self._record_signal("rejected", sig, 0, [decision.rejection_reason])
            out.append((sig, decision))

        self._update_dashboard_context(regime, ps)
        return out

    def _submit_rebalanced(self, symbol, sig, decision, target_shares, ps) -> None:  # pragma: no cover - live order path
        """Submit only the delta toward the target position (C4).

        Reads held shares from the tracker, computes the target/current weights,
        and trades the difference: buy to increase, sell to reduce, hold on small
        drift. Liquidating sells route through the executor as FLAT-direction
        orders.
        """
        from core.regime_strategies import Direction

        held = self.tracker.get_position(symbol) if self.tracker else None
        held_shares = int(held.qty) if held else 0
        equity = ps.equity or self.initial_capital
        price = sig.entry_price or (held.current_price if held else 0.0)
        if equity <= 0 or price <= 0:
            return
        target_weight = target_shares * price / equity
        current_weight = held_shares * price / equity
        threshold = getattr(self.orchestrator.config, "rebalance_threshold", 0.10)
        order = self._rebalance_order(target_shares, held_shares,
                                      target_weight, current_weight, threshold)
        if order is None:
            self._record_signal("hold", sig, held_shares, decision.modifications)
            return
        side, qty = order
        if side == "buy":
            # C3: entries carry a protective stop (bracket); capture the stop leg
            # so update_trailing_stops can tighten it once the fill lands.
            delta_sig = dataclasses.replace(
                decision.modified_signal,
                metadata={**decision.modified_signal.metadata, "approved_shares": qty},
            )
            res = self.executor.submit_bracket_order(delta_sig)
            if getattr(res, "stop_order_id", None):
                self._pending_stops[symbol] = res.stop_order_id
                self._adopt_pending_stop(symbol, delta_sig.stop_loss)
        else:  # reduce / liquidate
            delta_sig = dataclasses.replace(
                decision.modified_signal, direction=Direction.FLAT,
                metadata={**decision.modified_signal.metadata, "approved_shares": qty},
            )
            self.executor.submit_signal(delta_sig)
        self.risk.record_trade()
        self._record_signal(f"submitted_{side}", sig, qty, decision.modifications)

    def _adopt_pending_stop(self, symbol, stop_level=0.0) -> None:  # pragma: no cover - live positions
        """Attach a captured bracket stop-leg id to its tracked position once it exists."""
        stop_id = self._pending_stops.get(symbol)
        if not stop_id or not self.tracker:
            return
        pos = self.tracker.get_position(symbol)
        if pos is not None:
            pos.stop_order_id = stop_id
            if stop_level:
                pos.stop_level = stop_level
            self._pending_stops.pop(symbol, None)

    def on_fill(self, fill) -> None:
        """Apply a live fill to position/P&L tracking only (C5).

        The circuit breaker is fed from **broker equity once per bar** (see
        :meth:`_update_risk_posture`), which already reflects realized *and*
        unrealized P&L. Feeding realized P&L here as well would double-count, so
        fills update the tracker (positions, average price, realized P&L) but do
        not touch the breaker.

        Args:
            fill: Normalized :class:`~broker.position_tracker.FillEvent`.
        """
        if self.tracker:
            self.tracker.on_fill(fill)

    def _update_risk_posture(self, equity, regime=None) -> bool:
        """Feed mark-to-market equity into the breaker once per bar (C2).

        Computes this bar's return from the running equity, latches the circuit
        breaker (daily/weekly accumulation + peak-drawdown), mirrors the posture
        into ``RiskManager.state`` so sizing/veto honor REDUCED/HALTED, and — on
        a HALT — liquidates everything (C6). Drawdown therefore halts trading
        even with no fills happening (the long-only hold case).

        Args:
            equity: Current account equity (realized + unrealized).
            regime: Active regime (audit tagging only).

        Returns:
            True if the breaker is HALTED after this update.
        """
        from core.risk_manager import RiskState

        if equity and equity > 0:
            prev = self._last_equity
            bar_ret = (equity / prev - 1.0) if prev else 0.0
            self._last_equity = equity
            label = (regime.label.value if regime and hasattr(regime.label, "value")
                     else None)
            self.risk.breaker.update(pnl=bar_ret, equity=equity, regime=label)
            self.risk.state = self.risk.breaker.state   # latching posture -> sizing/veto
        if self.tracker:
            self.tracker.advance_bar()
        if self.risk.breaker.state is RiskState.HALTED:
            if self.executor:
                self.executor.close_all_positions()     # C6: flatten on halt
            if self.alerts:
                self.alerts.send("circuit_breaker_halt",
                                 "HALTED: liquidating all positions")
            return True
        return False

    def update_trailing_stops(self) -> None:  # pragma: no cover - needs live positions
        """Tighten protective stops per the current regime's strategy.

        For each open position, recompute the strategy stop and call
        ``modify_stop`` (tighten-only). No-ops for positions whose bracket
        stop-leg id was not captured — see the go-live review.
        """
        if not self.tracker or not self.executor:
            return
        for sym in list(self._pending_stops):  # late-attach stop ids whose fills now landed
            self._adopt_pending_stop(sym)
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

    def check_stream_health(self, now_ts, market_open: bool) -> bool:
        """Whether the bar feed has gone stale (S-2 watchdog); alerts if so.

        Args:
            now_ts: Current timestamp.
            market_open: Whether the market is open.

        Returns:
            True if stale (no bar within ``broker.max_bar_gap_sec`` while open).
        """
        from broker.stream_supervisor import stream_is_stale

        if self._last_bar_ts is None:
            return False
        max_gap = float(self.config.get("broker", {}).get("max_bar_gap_sec", 300))
        age = (pd.Timestamp(now_ts) - pd.Timestamp(self._last_bar_ts)).total_seconds()
        stale = stream_is_stale(age, max_gap, market_open)
        if stale and self.alerts:
            self.alerts.send("stream_stale", f"no bar for {age:.0f}s while market open")
        return stale

    def run_stream(self, market_data) -> None:  # pragma: no cover - live WebSocket
        """Subscribe to live bars and drive the loop, reconnecting on drop (S-2).

        Args:
            market_data: A connected :class:`~data.market_data.MarketData`.
        """
        from broker.stream_supervisor import run_with_reconnect

        def on_bar(symbol, bar):
            row = pd.DataFrame(
                [{"open": bar.open, "high": bar.high, "low": bar.low,
                  "close": bar.close, "volume": bar.volume}],
                index=[pd.Timestamp(bar.timestamp)],
            )
            self._last_bar_ts = pd.Timestamp(bar.timestamp)   # watchdog heartbeat
            self.ingest_bar(symbol, row)
            self.process_symbol(symbol)
            self.update_trailing_stops()

        def _connect_and_run():
            market_data.subscribe_bars(self.symbols, on_bar)

        def _on_retry(attempt, exc):
            if self.alerts:
                self.alerts.send("stream_reconnect", f"attempt {attempt} after: {exc}")
            if self.tlog:
                self.tlog.log(self.tlog.main, "stream_reconnect",
                              f"reconnect attempt {attempt}: {exc}", level="WARNING")

        max_retries = int(self.config.get("broker", {}).get("max_reconnects", 10))
        run_with_reconnect(_connect_and_run, max_retries=max_retries, on_retry=_on_retry)

    def run_cycle(self, market_data, state_path: str = STATE_SNAPSHOT) -> list[tuple]:
        """Run ONE decision cycle on the latest closed bars (daily live path).

        The HMM/regimes/breakers are daily-calibrated, but Alpaca's bar
        websocket streams minute bars — the wrong cadence. So for a daily
        strategy this is the correct live entry point: refresh buffers from
        history (the freshly-closed daily bar), process each symbol once,
        tighten stops, and persist state. Schedule it once per trading day
        (see ``run_once``); do not drive it off the minute stream.

        Args:
            market_data: Source exposing ``get_history``.
            state_path: Where to write the recovery/dashboard snapshot.

        Returns:
            All ``(Signal, RiskDecision)`` produced this cycle.
        """
        self.seed_buffers(market_data)   # latest daily bars (capped, symbol-tolerant)
        if self.symbols:
            self.maybe_retrain(self.symbols[0])  # A-1: refresh stale in-memory model + propagate
        results: list[tuple] = []
        for sym in self.symbols:
            results.extend(self.process_symbol(sym))
        self.update_trailing_stops()
        self.save_state(state_path)
        return results

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
        try:
            state.update(self._dashboard_block())
        except Exception:  # noqa: BLE001 - dashboard extras are best-effort
            logging.getLogger(__name__).exception("dashboard block failed")
        Path(path).write_text(json.dumps(state, indent=2, default=str))
        logging.getLogger(__name__).info("Saved state snapshot to %s", path)

    def _dashboard_block(self) -> dict:
        """Rich live state for the Streamlit dashboard (regime detail, risk,
        per-regime table) — mirrors the reference video's panels."""
        infos = (getattr(self.hmm, "regime_info", {}) or {}) if self.hmm else {}
        block: dict = {}

        table = []
        for sid, ri in sorted(infos.items()):
            nm = ri.regime_name.value if hasattr(ri.regime_name, "value") else str(ri.regime_name)
            table.append({
                "id": sid, "name": nm,
                "exp_return": round(ri.expected_return, 4),
                "exp_vol": round(ri.expected_volatility, 4),
                "strategy": ri.recommended_strategy_type,
                "max_leverage": ri.max_leverage_allowed,
            })
        block["regime_table"] = table

        reg = self.last_regime
        if reg is not None:
            sp = getattr(reg, "state_probabilities", None)
            probs = list(sp) if sp is not None and len(sp) else []
            runner: dict[str, float] = {}
            for i in sorted(range(len(probs)), key=lambda j: probs[j], reverse=True):
                ri = infos.get(i)
                nm = ri.regime_name.value if (ri and hasattr(ri.regime_name, "value")) else f"state{i}"
                runner[nm] = float(probs[i])
            vols = sorted(ri.expected_volatility for ri in infos.values())
            cur = infos.get(reg.state_id)
            vol_rank = (vols.index(cur.expected_volatility) / (len(vols) - 1)
                        if cur and len(vols) > 1 else 0.0)
            block["regime"] = {
                "name": reg.label.value if hasattr(reg.label, "value") else str(reg.label),
                "confidence": float(reg.probability),
                "stability_bars": int(reg.consecutive_bars),
                "confirmed": bool(reg.is_confirmed),
                "vol_rank": round(vol_rank, 2),
                "runner_ups": runner,
            }
            # Fuzzy layer (core.meta_overlay): posterior-weighted rank + one-step
            # transition hazard + predictive entropy for the dashboard gauges.
            try:
                import numpy as np

                from core.meta_overlay import (high_tier_hazard,
                                               predictive_entropy_norm,
                                               prob_weighted_vol_rank)
                rank_map = {
                    sid: (vols.index(ri.expected_volatility) / (len(vols) - 1)
                          if len(vols) > 1 else 0.0)
                    for sid, ri in infos.items()
                }
                sp_arr = np.asarray(probs, dtype=float)
                A = self.hmm.get_transition_matrix()
                block["regime"]["vol_rank_prob"] = round(
                    prob_weighted_vol_rank(sp_arr, rank_map), 4)
                block["regime"]["transition_hazard"] = round(
                    high_tier_hazard(sp_arr, A, rank_map), 4)
                block["regime"]["predictive_entropy"] = round(
                    predictive_entropy_norm(sp_arr, A), 4)
            except Exception:  # noqa: BLE001 - dashboard extras are best-effort
                pass

        b = self.risk.breaker
        peak = getattr(b, "_peak_equity", 0.0) or 0.0
        eq = self._last_equity or self.initial_capital
        peak_dd = (peak - eq) / peak if peak > 0 else 0.0
        c = self.risk.config
        block["risk"] = {
            "state": self.risk.state.value,
            "daily_dd": round(-min(getattr(b, "_daily_ret", 0.0), 0.0), 4),
            "daily_dd_limit": c.daily_dd_halt,
            "weekly_dd": round(-min(getattr(b, "_weekly_ret", 0.0), 0.0), 4),
            "weekly_dd_limit": c.weekly_dd_halt,
            "peak_dd": round(max(peak_dd, 0.0), 4),
            "peak_dd_limit": c.max_dd_from_peak,
            "leverage_limit": c.max_leverage,
            "breakers_clear": self.risk.state.value == "normal",
        }

        # model info + transition matrix (for the dashboard's optional panels)
        meta = getattr(self.hmm, "metadata", None) if self.hmm else None
        if meta is not None:
            block["model_info"] = {
                "n_regimes": meta.n_regimes, "bic": round(meta.bic, 1),
                "log_likelihood": round(meta.log_likelihood, 1),
                "converged": meta.converged, "n_iter": meta.n_iter,
                "training_date": meta.training_date,
                "n_features": len(meta.feature_columns),
            }
        model = getattr(self.hmm, "model", None) if self.hmm else None
        if model is not None and getattr(model, "transmat_", None) is not None:
            tm = model.transmat_
            labels = []
            for i in range(len(tm)):
                lab = self.hmm.labels.get(i)
                labels.append(lab.value if hasattr(lab, "value") else str(lab))
            block["transition_matrix"] = {
                "labels": labels,
                "matrix": [[round(float(x), 3) for x in row] for row in tm],
            }
        return block

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

    # C2: enable the persistent peak-DD lock in live (backtests keep it null)
    risk_cfg.lock_file = risk_cfg.lock_file or "logs/trading_halted.lock"
    risk = RiskManager(risk_cfg)
    risk._equity_peak = account["equity"]
    orch = StrategyOrchestrator(strat_cfg, hmm.regime_info)
    tracker = PositionTracker(client)
    tracker.sync_on_startup()
    executor = OrderExecutor(client)

    system = TradingSystem(config, hmm, orch, risk, fe, executor, tracker,
                           tlogger=tlog, alerts=alerts, dry_run=dry_run)
    system.load_state()
    system.seed_buffers(md)  # C1: warm buffers so features are ready on bar 1

    # C5: route the broker fill stream into on_fill (feeds breaker + liquidation)
    def _regime_label():
        r = system.last_regime
        return (r.label.value if r and hasattr(r.label, "value") else None)

    fills_thread = threading.Thread(
        target=lambda: tracker.subscribe_fills(regime_provider=_regime_label,
                                               sink=system.on_fill),
        daemon=True, name="fills-stream",
    )
    fills_thread.start()
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


def run_once(config: dict[str, Any], credentials: dict[str, str],
             dry_run: bool = False) -> None:  # pragma: no cover - live broker/market
    """Run ONE daily decision cycle and exit (the daily live entry point).

    Correct cadence for the daily-calibrated strategy: schedule this once per
    trading day (cron/launchd) instead of holding a minute-bar websocket open.
    Connects, loads/trains the HMM, reconciles positions, runs one
    :meth:`TradingSystem.run_cycle` (decisions + bracket entries + rebalance +
    MtM breaker + stops), persists state, and returns. Orders submitted while
    the market is closed queue for the next open (paper), so this can run after
    close or before open.

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

    paper = str(credentials.get("paper", "true")).lower() != "false"
    client = AlpacaClient(AlpacaConfig(credentials["api_key"], credentials["secret_key"], paper=paper))
    client.connect()
    account = client.get_account()
    tlog.log(tlog.main, "cycle_start", f"account equity {account['equity']}",
             mode="PAPER" if paper else "LIVE", market_open=client.is_market_open())

    md = MarketData(client)
    model_path = f"{MODEL_DIR}/hmm_{symbol}.pkl"
    fe = FeatureEngineer()
    # T0.4 pin-champion: load the registry champion (no age-based refit — see
    # docs/analysis/2026-06-12-pin-champion-amendment.md). Hash logged every run;
    # mismatch vs the promotion-time hash = silent model swap -> CRITICAL alert.
    from core.model_registry import ModelRegistry
    registry = ModelRegistry(MODEL_DIR)
    hmm, expected_sha = load_pinned_champion(
        registry, symbol, model_path, lambda: run_train(config, symbols))
    actual_sha = hmm.transition_hash()
    tlog.log(tlog.main, "champion_hash",
             f"{symbol} champion={registry.champion_version(symbol)} sha={actual_sha}")
    if expected_sha and actual_sha != expected_sha:
        alerts.send("champion_drift",
                    f"{symbol} champion transition hash {actual_sha} != promoted "
                    f"{expected_sha} — model artifact changed outside promotion")

    # Dual-log (2-week equivalence check): where the old rule would have refit,
    # fit a throwaway shadow and log agreement instead of swapping models.
    if (bool(config.get("hmm", {}).get("dual_log_refit", True))
            and _needs_retrain(registry.champion_path(symbol) or model_path)):
        try:
            from core.shadow_refit import append_row, compare_engines
            from data.market_data import load_ohlcv
            shadow_feats = fe.build_features(
                load_ohlcv(symbol, timeframe=config.get("broker", {}).get("timeframe", "1Day")))
            shadow = HMMEngine(hmm_cfg)
            shadow.fit(shadow_feats)
            row = compare_engines(hmm, shadow, shadow_feats)
            append_row("logs/shadow_refit.csv", row)
            tlog.log(tlog.main, "shadow_refit",
                     f"agree={row['agree']} champion={row['champion_regime']} "
                     f"shadow={row['shadow_regime']} -> logs/shadow_refit.csv")
        except Exception as exc:  # noqa: BLE001 - measurement must never block the cycle
            logging.getLogger(__name__).warning("shadow refit failed (non-fatal): %s", exc)

    risk_cfg.lock_file = risk_cfg.lock_file or "logs/trading_halted.lock"
    risk = RiskManager(risk_cfg)
    risk._equity_peak = account["equity"]
    orch = StrategyOrchestrator(strat_cfg, hmm.regime_info)
    tracker = PositionTracker(client)
    tracker.sync_on_startup()
    executor = OrderExecutor(client)

    system = TradingSystem(config, hmm, orch, risk, fe, executor, tracker,
                           tlogger=tlog, alerts=alerts, dry_run=dry_run)
    system.load_state()
    try:
        results = system.run_cycle(md)
        tlog.log(tlog.main, "cycle_done", f"{len(results)} decisions")
    except Exception as exc:  # noqa: BLE001 - log, save, alert, re-raise
        logging.getLogger(__name__).exception("cycle error")
        alerts.send("fatal_error", str(exc))
        system.save_state()
        raise


def run_rebalance(config: dict[str, Any], credentials: dict[str, str],
                  dry_run: bool = True,
                  universe_limit: Optional[int] = None,
                  challenger: bool = False,
                  quality: bool = False) -> list[dict]:  # pragma: no cover - live broker/market
    """Compute (and, when un-gated, submit) the cross-sectional book rebalance (vía C).

    The monthly paper entry point for the cross-sectional return-predictor book: load
    today's S&P 500 constituents, fetch their history, rank by cross-sectional momentum
    (the v1 return predictor), take the top decile, scale total gross exposure by the
    current HMM volatility regime (the risk overlay), size to whole shares against account
    equity, and report the order plan. Schedule monthly (see
    ``deploy/com.regimetrader.rebalance.plist``); do NOT drive off the daily/minute cadence.

    Safety: ``dry_run=True`` (default) computes, logs, and snapshots the plan but submits
    nothing. ``dry_run=False`` diffs the plan against current holdings and submits market
    orders (sells before buys). The book trades the **paper** account; real money stays
    BLOCKED until the pre-registered gate passes
    (docs/analysis/2026-06-04-cross-sectional-prereg.md). NOTE: the diff liquidates any
    holding not in the book's target — if the daily ``--run-once`` SPY bot shares this
    account, retire/flatten it first (SPY is not an S&P 500 constituent, so the book would
    sell it and the two strategies would collide). Either way the snapshot is written for
    the dashboard.

    Args:
        config: Parsed settings (reads the ``cross_sectional`` block).
        credentials: Broker credentials.
        dry_run: If True (default), compute + log + snapshot only (no orders).
        universe_limit: Optional cap on the universe size (small-N safe testing).

    Returns:
        The order plan: ``[{symbol, weight, notional, price, shares}, ...]``.
    """
    from broker.alpaca_client import AlpacaClient, AlpacaConfig
    from broker.order_executor import OrderExecutor
    from core.cross_sectional_ranking import (book_targets_fixed_selection,
                                              compute_book_targets,
                                              compute_book_targets_challenger,
                                              drop_open_order_symbols,
                                              plan_rebalance_orders, targets_to_orders)
    from core.hmm_engine import HMMEngine
    from core.regime_strategies import StrategyOrchestrator
    from data.constituents import load_sector_map, load_sp500
    from data.feature_engineering import FeatureEngineer
    from data.market_data import MarketData
    from monitoring.logger import LoggerConfig, setup_logging

    tlog = setup_logging(LoggerConfig(log_dir="logs"))
    # Isolation invariant (roadmap §0): a new sleeve never trades the shared account —
    # the frozen baseline owns it, and both submitting would fight over positions. The
    # quality sleeve is dry-run + synthetic NAV (challenger pattern) until it has its own
    # paper account (T5.4). Force dry-run regardless of the flag; never silently execute.
    if quality and not dry_run:
        dry_run = True
        tlog.log(tlog.main, "quality_dryrun_forced",
                 "quality sleeve forced to dry-run (no separate paper account; T5.4)")
    cs = config.get("cross_sectional", {})
    _, strat_cfg, _ = _core_configs(config)
    proxy = cs.get("proxy", "SPY")
    lookback = int(cs.get("lookback", 252))
    skip = int(cs.get("skip", 21))
    frac = float(cs.get("top_fraction", 0.10))
    max_single = float(config.get("risk", {}).get("max_single_position", 0.15))
    max_concurrent = int(cs.get("max_concurrent", 50))

    paper = str(credentials.get("paper", "true")).lower() != "false"
    client = AlpacaClient(AlpacaConfig(credentials["api_key"], credentials["secret_key"], paper=paper))
    client.connect()
    equity = float(client.get_account()["equity"])
    md = MarketData(client)

    # Current market volatility regime (HMM on the proxy) -> gross-exposure overlay input.
    # T0.4 pin-champion: registry champion, hash-logged; never refit by age here.
    from core.model_registry import ModelRegistry
    registry = ModelRegistry(MODEL_DIR)
    hmm, expected_sha = load_pinned_champion(
        registry, proxy, f"{MODEL_DIR}/hmm_{proxy}.pkl", lambda: run_train(config, [proxy]))
    actual_sha = hmm.transition_hash()
    tlog.log(tlog.main, "champion_hash",
             f"{proxy} champion={registry.champion_version(proxy)} sha={actual_sha}")
    if expected_sha and actual_sha != expected_sha:
        logging.getLogger(__name__).error(
            "champion drift: %s hash %s != promoted %s — model artifact changed "
            "outside promotion", proxy, actual_sha, expected_sha)
    orch = StrategyOrchestrator(strat_cfg, hmm.regime_info)
    proxy_hist = md.get_history(proxy, config.get("broker", {}).get("timeframe", "1Day"),
                                lookback + skip + 300)
    feats = FeatureEngineer().build_features(proxy_hist)
    states = hmm.predict_regime_filtered(feats)
    # vol_rank is resolved below once the overlay mode is known (hmm_prob reads the
    # posterior-weighted rank; every legacy mode keeps the argmax rank).

    # Universe history -> cross-sectional rank -> top-decile targets -> share plan.
    # T5.3 point-in-time: the quality sleeve's prereg promises a PIT universe, so it
    # resolves the monthly snapshot at/before today (survivorship-free going forward).
    # The frozen baseline/challenger keep the current CSV their preregs specify — never
    # mutate a frozen book's universe mid-gate (roadmap §0).
    q_as_of = datetime.now(timezone.utc).date().isoformat() if quality else None
    if quality:                                         # freeze this month's PIT universe once
        from data.constituents import ensure_snapshot
        ensure_snapshot(f"{datetime.now(timezone.utc):%Y-%m}")
    universe = load_sp500(as_of=q_as_of)
    if universe_limit:                                  # safe-test override (small universe)
        universe = universe[:universe_limit]
    # The challenger's residual signal estimates beta over a long est_window (504) BEFORE
    # scoring the recent 12-1 sub-window, so it needs a deeper fetch than the baseline's
    # lookback+skip — otherwise residual_momentum_score silently fits on too few bars and
    # the live signal != the validated one.
    est_window = int(config.get("challenger", {}).get("est_window", 504))
    hist_depth = (max(lookback + skip, est_window) + 60) if challenger else (lookback + skip + 10)
    frames = md.get_history_multi(universe, config.get("broker", {}).get("timeframe", "1Day"),
                                  hist_depth)
    # Daily cadence: re-rank the (slow, monthly) momentum selection only on the FIRST run
    # of a new calendar month; on intra-month daily runs keep the month's names and only
    # re-scale total gross to today's market vol via the overlay (de-risk on vol spikes,
    # re-risk on calm). "Attentive every day" = daily risk management, monthly name turnover.
    snap_path = (BOOK_SNAPSHOT_QUALITY if quality else
                 BOOK_SNAPSHOT_CHALLENGER if challenger else BOOK_SNAPSHOT)
    month_key = f"{datetime.now(timezone.utc):%Y-%m}"
    prior_sel: list[str] = []
    prior_month = None
    if Path(snap_path).exists():
        try:
            prior = json.loads(Path(snap_path).read_text())
            prior_sel = [s for s in (prior.get("selected_symbols") or []) if isinstance(s, str)]
            prior_month = prior.get("selection_month")
        except Exception:
            prior_sel, prior_month = [], None
    src = (config.get("quality", {}) if quality else
           config.get("challenger", {}) if challenger else cs)
    ov = str(src.get("overlay", "vol_target" if (challenger or quality) else "hmm"))
    # Fuzzy layer: hmm_prob swaps the argmax rank for the posterior-weighted rank
    # (continuous de-risking, no cliff); every other mode keeps the argmax rank
    # untouched. Hazard/entropy are recorded for the dashboard either way.
    from core.meta_overlay import (high_tier_hazard, predictive_entropy_norm,
                                   vol_rank_for_overlay)
    last = states[-1]
    vol_rank = vol_rank_for_overlay(ov, last.state_probabilities, last.state_id,
                                    orch.vol_rank)
    hazard = high_tier_hazard(last.state_probabilities, hmm.get_transition_matrix(),
                              orch.vol_rank)
    entropy = predictive_entropy_norm(last.state_probabilities,
                                      hmm.get_transition_matrix())

    # T1.1 shadow regime log (baseline run only; shadow-only, never touches orders):
    # fit the Jump Model on the same panel and record HMM-vs-JM vol-rank agreement.
    if not challenger and not quality:
        try:
            from core.jump_model import JumpModel
            from core.shadow_regime import append_row, make_row
            jm = JumpModel(n_states=hmm.n_regimes, jump_penalty=30.0,
                           random_state=42).fit(feats)
            append_row("logs/shadow_regime.csv",
                       make_row(str(feats.index[-1])[:10], vol_rank, jm.vol_rank()))
        except Exception as exc:  # noqa: BLE001 - shadow must never break the rebalance
            logging.getLogger(__name__).warning("shadow regime log failed (non-fatal): %s", exc)
    tv = float(src.get("target_vol", 0.12))
    vw = int(src.get("vol_window", 126))
    gc = float(src.get("gross_cap", 1.0))
    gf = float(src.get("gross_floor", 0.0))
    reuse_selection = bool(prior_sel) and prior_month == month_key

    if reuse_selection:
        targets = book_targets_fixed_selection(
            frames, prior_sel, vol_rank, overlay=ov,
            risk_on_gross=float(cs.get("risk_on_gross", 1.0)),
            risk_off_gross=float(cs.get("risk_off_gross", 0.5)),
            target_vol=tv, vol_window=vw, gross_cap=gc, gross_floor=gf,
            weighting=str(cs.get("weighting", "equal")),
            max_single=max_single, max_concurrent=max_concurrent,
        )
        selected_symbols = prior_sel
    elif quality:
        # Quality(+momentum) sleeve (T2.1). Monthly rerank only — fundamentals are
        # fetched (EDGAR, cached) once a month here; intra-month days take the
        # reuse_selection branch above and need no fundamentals. Drop-in: the EDGAR
        # block shape equals the old SimFin stub, so make_book_weights_quality is
        # unchanged. Isolation: own snapshot, never --execute on the shared account.
        from core.quality_ranking import make_book_weights_quality
        from data.edgar_data import load_blocks
        q = config.get("quality", {})
        blocks = load_blocks(universe)
        tlog.log(tlog.main, "quality_fundamentals",
                 f"EDGAR blocks for {len(blocks)}/{len(universe)} names "
                 f"(coverage {len(blocks) / max(1, len(universe)):.0%})")
        weight_fn = make_book_weights_quality(
            frames, blocks, lookback=lookback, skip=skip, frac=frac,
            max_single=max_single, max_concurrent=max_concurrent,
            combine=str(q.get("combine", "quality_momentum")),
            overlay=ov,
            target_vol=tv, vol_window=vw, gross_cap=gc, gross_floor=gf,
            risk_on_gross=float(cs.get("risk_on_gross", 1.0)),
            risk_off_gross=float(cs.get("risk_off_gross", 0.5)),
            weighting=str(cs.get("weighting", "equal")),
            sector_map=load_sector_map(as_of=q_as_of),
            max_sector_frac=float(cs.get("max_sector_fraction", 0.30)),
        )
        targets = weight_fn(pd.Timestamp.now(tz=timezone.utc), vol_rank)
    elif challenger:
        ch = config.get("challenger", {})
        targets = compute_book_targets_challenger(
            frames, proxy_hist["close"], vol_rank, lookback=lookback, skip=skip,
            est_window=est_window, frac=frac,
            max_single=max_single, max_concurrent=max_concurrent,
            overlay=str(ch.get("overlay", "vol_target")),
            risk_on_gross=float(cs.get("risk_on_gross", 1.0)),
            risk_off_gross=float(cs.get("risk_off_gross", 0.5)),
            target_vol=float(ch.get("target_vol", 0.12)),
            vol_window=int(ch.get("vol_window", 126)),
            gross_cap=float(ch.get("gross_cap", 1.0)),
            gross_floor=float(ch.get("gross_floor", 0.0)),
            sector_map=load_sector_map(),
            max_sector_frac=float(cs.get("max_sector_fraction", 0.30)),
        )
    else:
        targets = compute_book_targets(
            frames, vol_rank, lookback=lookback, skip=skip, frac=frac,
            max_single=max_single, max_concurrent=max_concurrent,
            risk_on_gross=float(cs.get("risk_on_gross", 1.0)),
            risk_off_gross=float(cs.get("risk_off_gross", 0.5)),
            overlay=str(cs.get("overlay", "hmm")),
            target_vol=float(cs.get("target_vol", 0.12)),
            vol_window=int(cs.get("vol_window", 126)),
            gross_cap=float(cs.get("gross_cap", 1.0)),
            gross_floor=float(cs.get("gross_floor", 0.0)),
            weighting=str(cs.get("weighting", "equal")),
            sector_map=load_sector_map(),
            max_sector_frac=float(cs.get("max_sector_fraction", 0.30)),
        )
    if not reuse_selection:
        selected_symbols = list(targets)

    # Macro-event risk overlay (opt-in, risk timing not alpha): trim gross in the window
    # before a scheduled high-vol US event (FOMC/payrolls). Default off (event_derisk 0/1).
    ev_derisk = float(cs.get("event_derisk", 0.0) or 0.0)
    if 0.0 < ev_derisk < 1.0:
        from core.macro_calendar import event_risk_scale, in_event_window
        ev_win = int(cs.get("event_window_days", 2))
        ev_scale = event_risk_scale(datetime.now(timezone.utc).date(), ev_win, ev_derisk)
        if ev_scale < 1.0:
            flagged, ev_label = in_event_window(datetime.now(timezone.utc).date(), ev_win)
            targets = {s: w * ev_scale for s, w in targets.items()}
            tlog.log(tlog.main, "event_derisk",
                     f"{ev_label} within {ev_win}d -> gross x{ev_scale:.2f}")

    prices = {s: float(frames[s]["close"].iloc[-1]) for s in targets if s in frames}
    plan = targets_to_orders(targets, equity, prices)

    tlog.log(tlog.main, "rebalance_plan",
             f"vol_rank={vol_rank:.2f} names={len(plan)} gross={sum(o['weight'] for o in plan):.2f} "
             f"{'reuse' if reuse_selection else 'rerank'} month={month_key} overlay={ov}",
             mode="PAPER" if paper else "LIVE", dry_run=dry_run)
    for o in plan:
        tlog.log(tlog.main, "rebalance_target",
                 f"{o['symbol']}: {o['shares']}sh @ {o['price']:.2f} (w={o['weight']:.3f})")

    # Current holdings (for the dashboard + the rebalance diff).
    held = {p["symbol"]: int(float(p["qty"])) for p in client.get_positions()}

    executed: list[dict] = []
    if not dry_run:
        # Diff target vs held -> sells before buys; plain market orders (cash book,
        # risk = diversification + gross overlay, no per-name stops).
        target_shares = {o["symbol"]: int(o["shares"]) for o in plan}
        orders = plan_rebalance_orders(target_shares, held)
        # Idempotency guard: never re-submit a name that already has a pending order
        # (a re-run in the fill gap would double-submit; held doesn't yet reflect it).
        # limit=500: the book holds ~100 names; the default 100 could truncate.
        open_syms = {o["symbol"]
                     for o in client.get_order_history(limit=500, status="open")}
        if open_syms:
            kept = drop_open_order_symbols(orders, open_syms)
            tlog.log(tlog.main, "rebalance_open_order_guard",
                     f"skipped {len(orders) - len(kept)} names with pending orders: "
                     f"{sorted(open_syms)}")
            orders = kept
        results = OrderExecutor(client).submit_market_orders(orders)
        executed = [{"symbol": r.symbol, "status": r.status.value,
                     "filled_qty": r.filled_qty} for r in results]
        tlog.log(tlog.main, "rebalance_executed", f"{len(executed)} orders submitted",
                 mode="PAPER" if paper else "LIVE")

    # Tail-hedge overlay check (options_hedge block; ships disabled). Risk action
    # keyed to the transition hazard — pre-registered knobs, premium budget capped,
    # double-gated: orders need BOTH --execute (not dry_run) AND allow_orders: true.
    hedge_out: dict = {"action": "disabled"}
    oh = config.get("options_hedge", {}) or {}
    if bool(oh.get("enabled", False)) and not challenger and not quality:
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient

            from broker.options_executor import OptionsHedgeExecutor
            from core.options_overlay import OptionsHedgeConfig

            fields = OptionsHedgeConfig.__dataclass_fields__
            hcfg = OptionsHedgeConfig(**{k: v for k, v in oh.items()
                                         if k in fields and k != "proxy"}, proxy=proxy)
            odata = OptionHistoricalDataClient(credentials["api_key"],
                                               credentials["secret_key"])
            hedge_exec = OptionsHedgeExecutor(client.trading, odata, hcfg)
            hedge_out = hedge_exec.run_check(
                hazard=hazard, equity=equity,
                book_gross=sum(o["weight"] for o in plan),
                spot=float(proxy_hist["close"].iloc[-1]),
                dry_run=dry_run or not bool(oh.get("allow_orders", False)),
            )
            tlog.log(tlog.main, "options_hedge",
                     f"action={hedge_out.get('action')} hazard={hazard:.2f}")
        except Exception:  # noqa: BLE001 - the hedge must never break the rebalance
            logging.getLogger(__name__).exception("options hedge check failed")
            hedge_out = {"action": "error"}

    # Persist the book snapshot for the dashboard (dry-run writes it too).
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "book": "quality" if quality else "challenger" if challenger else "baseline",
        "mode": "PAPER" if paper else "LIVE",
        "dry_run": dry_run,
        "selection_month": month_key,
        "selected_symbols": selected_symbols,
        "rebalanced": not reuse_selection,
        "overlay": ov,
        "vol_rank": vol_rank,
        "transition_hazard": round(hazard, 4),
        "predictive_entropy": round(entropy, 4),
        "options_hedge": hedge_out,
        "gross": round(sum(o["weight"] for o in plan), 4),
        "equity": equity,
        "universe_size": len(frames),
        "targets": plan,
        "held": [{"symbol": s, "shares": q} for s, q in sorted(held.items())],
        "executed": executed,
    }
    Path(snap_path).write_text(json.dumps(snapshot, indent=2, default=str))
    tlog.log(tlog.main, "book_snapshot", f"wrote {snap_path}")
    return plan


def run_record_track(config: dict[str, Any], credentials: dict[str, str],
                     path: str = TRACK_RECORD_CSV) -> None:  # pragma: no cover - live broker/market
    """Append today's book / EW-S&P500 / SPY NAV to the track record (daily gate plumbing).

    Schedule this daily after the close. It reads the book's account equity and the latest
    two daily closes for SPY (cap-weight benchmark) and RSP (Invesco S&P 500 Equal Weight —
    the investable, net-of-fee equal-weight benchmark), computes their one-day returns, and
    appends a single row via :mod:`core.track_record`. It submits no orders and touches no
    signal or construction knob — pure measurement, so the frozen forward gate has a clean
    daily NAV series for the book and **both** benchmarks to evaluate Sharpe / maxDD / DSR at
    month 12. Idempotent per day (safe to re-run).

    Args:
        config: Parsed settings.
        credentials: Broker credentials.
        path: Track-record CSV (defaults to :data:`TRACK_RECORD_CSV`).
    """
    from broker.alpaca_client import AlpacaClient, AlpacaConfig
    from core import track_record as tr
    from data.market_data import MarketData
    from monitoring.logger import LoggerConfig, setup_logging

    tlog = setup_logging(LoggerConfig(log_dir="logs"))
    proxy = config.get("cross_sectional", {}).get("proxy", "SPY")
    ew_proxy = config.get("cross_sectional", {}).get("ew_proxy", "RSP")
    timeframe = config.get("broker", {}).get("timeframe", "1Day")

    paper = str(credentials.get("paper", "true")).lower() != "false"
    client = AlpacaClient(AlpacaConfig(credentials["api_key"], credentials["secret_key"], paper=paper))
    client.connect()
    equity = float(client.get_account()["equity"])
    md = MarketData(client)

    def _last_ret(symbol: str) -> "tuple[float, str]":
        hist = md.get_history(symbol, timeframe, 6)
        ret = tr.simple_return(float(hist["close"].iloc[-2]), float(hist["close"].iloc[-1]))
        return ret, str(hist.index[-1])[:10]          # return + closed-bar date (not wall clock)

    spy_ret, date = _last_ret(proxy)
    ew_ret, _ = _last_ret(ew_proxy)

    # Gap 5 data-quality sentinel: a stale feed / split error / NaN day on the benchmark
    # series silently corrupts the gate evidence. Check before the row lands; alert, never
    # auto-correct (patching would itself contaminate the gate).
    try:
        from core import data_quality as dq
        from monitoring.alerts import AlertConfig, AlertManager, AlertSeverity
        sentinel_issues = []
        for sym in (proxy, ew_proxy):
            sentinel_issues += dq.check_price_series(sym, md.get_history(sym, timeframe, 30), date)
        if sentinel_issues:
            alerts = AlertManager(_build_dataclass(AlertConfig, config.get("monitoring", {})),
                                  trading_logger=tlog)
            alerts.send("data_quality", f"track-record data issues {dq.summary(sentinel_issues)}: "
                        + "; ".join(f"{i.symbol}/{i.kind}: {i.detail}" for i in sentinel_issues),
                        AlertSeverity.WARNING)
    except Exception as exc:  # noqa: BLE001 - the sentinel must never block the recorder
        logging.getLogger(__name__).warning("data-quality sentinel failed (non-fatal): %s", exc)

    # Dry-run sleeve gate feed (T0.1 challenger, T2.1 quality): synthesize each sleeve's
    # daily return by marking its snapshot target weights to market — neither has a broker
    # account of its own (challenger pattern). A name with no bar contributes 0 (cash).
    def _sleeve_ret(snapshot_path: str, label: str) -> "Optional[float]":
        weights = tr.snapshot_weights(snapshot_path)
        if not weights:
            return None
        rets: dict[str, float] = {}
        for sym in weights:
            try:
                rets[sym] = _last_ret(sym)[0]
            except Exception as exc:                      # missing bar -> cash (0) that day
                tlog.log(tlog.main, "track_record", f"{label} ret fetch failed {sym}: {exc}")
        return tr.portfolio_return(weights, rets)

    ch_ret = _sleeve_ret(BOOK_SNAPSHOT_CHALLENGER, "challenger")
    q_ret = _sleeve_ret(BOOK_SNAPSHOT_QUALITY, "quality")

    # T0.3 evidence audit trail: which checked-out code produced this row.
    try:
        code_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, timeout=10,
                                  cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        code_sha = None

    tr.append_day(path, date, equity, spy_ret, ew_ret,
                  challenger_ret=ch_ret, quality_ret=q_ret, code_sha=code_sha)
    tlog.log(tlog.main, "track_record",
             f"{date} book={equity:.0f} spy_ret={spy_ret:+.4f} ew_ret={ew_ret:+.4f} "
             f"challenger_ret={'n/a' if ch_ret is None else f'{ch_ret:+.4f}'} "
             f"quality_ret={'n/a' if q_ret is None else f'{q_ret:+.4f}'} "
             f"sha={code_sha} -> {path}",
             mode="PAPER" if paper else "LIVE")

    # Gap 6 tamper-evidence: chain today's hash of every gate-evidence file.
    from core import evidence as ev
    row = ev.append_chain(EVIDENCE_CHAIN, date, {
        "track_record": path,
        "book_snapshot": BOOK_SNAPSHOT,
        "book_snapshot_challenger": BOOK_SNAPSHOT_CHALLENGER,
        "book_snapshot_quality": BOOK_SNAPSHOT_QUALITY,
        "champion_sha_SPY": f"{MODEL_DIR}/SPY/champion_sha.txt",
    })
    if row is not None:
        tlog.log(tlog.main, "evidence_chain", f"{date} chain={row['chain'][:16]}")


RISK_STATE_FILE = "risk_monitor_state.json"
HEARTBEAT_STATE_FILE = "logs/heartbeat_state.json"
EVIDENCE_CHAIN = "evidence/chain.jsonl"  # committed (git-anchored); logs/ is gitignored


def _heartbeat_check(config: dict[str, Any], tlog: Any,
                     csv_path: str = TRACK_RECORD_CSV,
                     state_path: str = HEARTBEAT_STATE_FILE,
                     today: Optional[str] = None) -> bool:
    """Alert (once per day) when the gate-evidence feed has gone stale (T0.5).

    Hosted in ``--risk-check`` because launchd fires it 24/7 — the recorder
    cannot watchdog itself. A last track-record row older than
    ``monitoring.heartbeat_max_bdays`` business days means the 12-month gates
    are silently starving; every silent day is unrecoverable evidence loss.

    Args:
        config: Parsed settings.
        tlog: Trading logger (or None).
        csv_path: Track-record CSV to check.
        state_path: Once-per-day dedup state file.
        today: ISO date override (tests).

    Returns:
        True if a CRITICAL alert was dispatched.
    """
    from zoneinfo import ZoneInfo

    from core import track_record as tr
    from monitoring.alerts import AlertConfig, AlertManager, AlertSeverity

    today = today or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    stale = tr.staleness_bdays(csv_path, today)
    max_bdays = int(config.get("monitoring", {}).get("heartbeat_max_bdays", 2))
    if stale is None or stale <= max_bdays:
        return False
    sp = Path(state_path)
    if sp.exists():
        try:
            if json.loads(sp.read_text()).get("alerted") == today:
                return False                     # already screamed today
        except (json.JSONDecodeError, OSError):
            pass
    alerts = AlertManager(_build_dataclass(AlertConfig, config.get("monitoring", {})),
                          trading_logger=tlog)
    alerts.send("track_record_stale",
                f"last track-record row is {stale} business days old "
                f"(max {max_bdays}) — the gate evidence feed may be down",
                AlertSeverity.CRITICAL)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"alerted": today}))
    return True


def run_risk_check(config: dict[str, Any], credentials: dict[str, str],
                   execute: bool = False,
                   state_path: str = RISK_STATE_FILE) -> None:  # pragma: no cover - live broker/market
    """One intraday risk-check cycle for the cross-sectional book (schedule every 15 min).

    RISK-ONLY: watches the account's intraday drawdown vs the prior close and walks the
    escalation ladder in :mod:`core.risk_monitor` (alert -> derisk -> flatten). It never
    buys, never trades intraday alpha (the signal stack is daily-calibrated and LOCKED,
    M3). Escalation is monotonic within a session via a small state file; the next
    session resets it, so the latch auto-releases (the old halt-latch lesson).

    Orders are submitted only when BOTH the ``--execute`` flag and the
    ``risk_monitor.allow_orders`` config key are on — default is observe+alert only,
    because an intraday kill-switch was not part of the frozen pre-registration and
    silently enabling it would contaminate the 12-month forward gate.

    Args:
        config: Parsed settings.
        credentials: Broker credentials.
        execute: CLI gate for order submission.
        state_path: Session-latch state file.
    """
    from broker.alpaca_client import AlpacaClient, AlpacaConfig
    from broker.order_executor import OrderExecutor
    from core.risk_monitor import (RiskThresholds, assess_intraday_risk, escalate,
                                   plan_derisk_orders, plan_flatten_orders)
    from monitoring.alerts import AlertConfig, AlertManager, AlertSeverity
    from monitoring.logger import LoggerConfig, setup_logging

    tlog = setup_logging(LoggerConfig(log_dir="logs"))
    rm_cfg = config.get("risk_monitor", {})
    thresholds = RiskThresholds(
        alert_dd=float(rm_cfg.get("alert_dd", 0.02)),
        derisk_dd=float(rm_cfg.get("derisk_dd", 0.04)),
        flatten_dd=float(rm_cfg.get("flatten_dd", 0.08)),
        derisk_scale=float(rm_cfg.get("derisk_scale", 0.5)),
    )
    allow_orders = execute and bool(rm_cfg.get("allow_orders", False))

    paper = str(credentials.get("paper", "true")).lower() != "false"
    client = AlpacaClient(AlpacaConfig(credentials["api_key"], credentials["secret_key"],
                                       paper=paper))
    client.connect()
    # T0.5 heartbeat: check the evidence feed BEFORE the market-closed exit —
    # staleness is exactly the thing that shows up off-hours.
    _heartbeat_check(config, tlog)
    if not client.is_market_open():
        return                                  # launchd fires 24/7; quiet no-op when closed

    acct = client.get_account()
    equity = float(acct["equity"])
    last_equity = float(acct.get("last_equity", 0.0))
    intraday_ret = (equity / last_equity - 1.0) if last_equity > 0 else 0.0

    # Session latch: only escalate within the same trading day (NY session date).
    from zoneinfo import ZoneInfo
    session = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    state = {"session": session, "action": "ok"}
    sp = Path(state_path)
    if sp.exists():
        try:
            prev = json.loads(sp.read_text())
            if prev.get("session") == session:
                state["action"] = prev.get("action", "ok")
        except (json.JSONDecodeError, OSError):
            pass                                # unreadable state -> start the day at "ok"

    assessed = assess_intraday_risk(equity, last_equity, thresholds)
    action = escalate(state["action"], assessed)
    already_acted = action == state["action"] and action in ("derisk", "flatten")
    tlog.log(tlog.main, "risk_check",
             f"intraday {intraday_ret:+.2%} equity={equity:.0f} -> {action}"
             f"{' (latched)' if already_acted else ''}",
             mode="PAPER" if paper else "LIVE")

    if action in ("alert", "derisk", "flatten") and not already_acted:
        alerts = AlertManager(AlertConfig(), trading_logger=tlog)
        sev = AlertSeverity.WARNING if action == "alert" else AlertSeverity.CRITICAL
        alerts.send(f"intraday_{action}",
                    f"intraday return {intraday_ret:+.2%} breached {action} threshold "
                    f"(equity {equity:.0f} vs prior close {last_equity:.0f})", sev)

    if action in ("derisk", "flatten") and not already_acted:
        held = {p["symbol"]: int(float(p["qty"])) for p in client.get_positions()}
        orders = (plan_flatten_orders(held) if action == "flatten"
                  else plan_derisk_orders(held, thresholds.derisk_scale))
        if allow_orders:
            results = OrderExecutor(client).submit_market_orders(orders)
            tlog.log(tlog.main, "risk_check_executed",
                     f"{action}: {len(results)} sell orders submitted",
                     mode="PAPER" if paper else "LIVE")
        else:
            tlog.log(tlog.main, "risk_check_observe",
                     f"{action}: would submit {len(orders)} sell orders "
                     f"(allow_orders off — observe mode)")

    state["action"] = action
    sp.write_text(json.dumps(state))


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
    mode.add_argument("--run-once", action="store_true", dest="run_once",
                      help="one daily decision cycle then exit (schedule this for daily)")
    mode.add_argument("--rebalance", action="store_true",
                      help="cross-sectional book rebalance (dry-run by default; "
                           "schedule monthly). Add --execute to submit paper orders.")
    mode.add_argument("--record-track", action="store_true", dest="record_track",
                      help="append today's book/EW-S&P500/SPY NAV to the track record "
                           "(schedule daily; gate-evaluation plumbing, no orders)")
    mode.add_argument("--risk-check", action="store_true", dest="risk_check",
                      help="one intraday risk-check cycle (schedule every 15 min; "
                           "alert/derisk/flatten ladder; orders only with --execute "
                           "AND risk_monitor.allow_orders)")
    mode.add_argument("--verify-evidence", action="store_true", dest="verify_evidence",
                      help="walk the evidence hash-chain and report the first "
                           "broken link, if any (gap-6 tamper-evidence audit)")
    mode.add_argument("--shadow-report", action="store_true", dest="shadow_report",
                      help="generate the monthly HMM-vs-JumpModel shadow regime "
                           "report from logs/shadow_regime.csv (T1.4)")
    mode.add_argument("--live", action="store_true", help="paper/live trading loop (default)")

    parser.add_argument("--execute", action="store_true",
                        help="with --rebalance: actually submit the paper orders (default dry-run)")
    parser.add_argument("--limit", type=int, default=None,
                        help="with --rebalance: cap universe size (small-N safe testing)")
    parser.add_argument("--challenger", action="store_true",
                        help="with --rebalance: run the residual-momentum + vol-target "
                             "challenger book (parallel to the frozen baseline; GATED, "
                             "writes a separate snapshot)")
    parser.add_argument("--quality", action="store_true",
                        help="with --rebalance: run the EDGAR quality(+momentum) sleeve "
                             "(T2.1; parallel, own snapshot, ALWAYS dry-run + synthetic "
                             "NAV until a 2nd paper account exists — never executes)")

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
    elif args.run_once:
        run_once(config, load_credentials())
    elif args.rebalance:
        run_rebalance(config, load_credentials(), dry_run=not args.execute,
                      universe_limit=args.limit, challenger=args.challenger,
                      quality=args.quality)
    elif args.record_track:
        run_record_track(config, load_credentials())
    elif args.risk_check:
        run_risk_check(config, load_credentials(), execute=args.execute)
    elif args.verify_evidence:
        from core.evidence import verify_chain
        ok, bad = verify_chain(EVIDENCE_CHAIN)
        print(f"evidence chain: {'OK' if ok else f'BROKEN at row {bad}'} ({EVIDENCE_CHAIN})")
        if not ok:
            raise SystemExit(1)
    elif args.shadow_report:
        from core.shadow_regime import monthly_report, report_markdown
        month = (args.end or datetime.now(timezone.utc).date().isoformat())[:7]
        md = report_markdown(monthly_report("logs/shadow_regime.csv", month))
        out = Path(f"docs/analysis/{month}-shadow-regime-report.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"shadow report -> {out}")
    else:  # default: live
        run_live(config, load_credentials())


if __name__ == "__main__":
    main()
