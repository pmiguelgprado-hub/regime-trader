"""Walk-forward allocation backtester.

This is an **allocation-based** walk-forward backtester. It does *not* track
individual trade entries/exits — each bar it sets a single target portfolio
weight (gross equity exposure, leverage included) from the detected volatility
regime, and only rebalances when the target drifts from the held weight by more
than ``rebalance_threshold``. That is how real systematic allocation strategies
operate, and it sidesteps the brittle entry/exit bookkeeping that makes most toy
backtests untrustworthy.

Causality / look-ahead
----------------------
* Features are computed **once** on the full causal series (every transform in
  :mod:`data.feature_engineering` is trailing, so a global compute equals a
  per-prefix compute — provable with ``FeatureEngineer.assert_no_lookahead``).
* The walk-forward loop refits the HMM on each in-sample window and runs the
  **forward-algorithm filtered** inference on the out-of-sample window (never
  Viterbi), seeded with the in-sample history as warmup.
* A weight decided using the close at bar ``t`` earns the ``t -> t+1`` return.
  Slippage is charged on turnover at the moment of rebalance.

Window sizing (deviation from the original "IS=252" spec)
---------------------------------------------------------
``HMMEngine.fit`` requires ``min_train_bars`` (504) *usable* rows, so a 252-bar
in-sample window cannot train the model — every fold would raise. The default
``train_window`` is therefore **504** (~2y). Override via config if you raise
``min_train_bars`` accordingly.

Scope: single-asset sleeve. ``run`` accepts a ``{symbol: OHLCV}`` map but trades
the **first** symbol (the canonical CLI call is ``--symbols SPY``). The regime is
market-wide and the gross-exposure/single-name diversification caps in
:class:`RiskManager` are intentionally not applied to a one-asset sleeve — the
drawdown **circuit breakers** (the safety layer the stress tests probe) are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine, Regime
from core.portfolio import portfolio_target_weights
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from core.risk_manager import RiskManager
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for the walk-forward backtester (mirrors `backtest`)."""

    slippage_pct: float = 0.0005   # base slippage per unit turnover (flat floor)
    slippage_vol_coeff: float = 0.0  # extra slippage per unit ATR% (R-1; 0 = flat, legacy)
    initial_capital: float = 100000.0
    train_window: int = 504        # usable rows for HMM fit (>= hmm.min_train_bars)
    test_window: int = 126
    step_size: int = 126
    risk_free_rate: float = 0.045
    credit_cash_rf: bool = False   # credit idle cash at rf (and charge rf on levered
                                   #   excess). Default False = legacy (idle cash earns 0).
                                   #   Fair Sharpe vs 100%-invested buy&hold needs this on.
    max_leverage: float = 1.25     # hard ceiling on target weight


@dataclass
class BacktestResult:
    """Output of a backtest run.

    Attributes:
        equity_curve: Equity over time (net of slippage), indexed by timestamp.
        returns: Net per-bar portfolio returns.
        trades: Rebalance ledger (one row per rebalance event).
        regime_labels: Regime label per OOS bar.
        fold_boundaries: ``(test_start, test_end)`` positional index of each fold.
        regime_history: Per-bar frame: regime, probability, weight, risk state.
        asset_returns: Per-bar return of the traded asset (benchmark base).
        symbol: Traded symbol.
        initial_capital: Starting equity.
    """

    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_labels: pd.Series = field(default_factory=pd.Series)
    fold_boundaries: list[tuple[int, int]] = field(default_factory=list)
    regime_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    asset_returns: pd.Series = field(default_factory=pd.Series)
    symbol: str = ""
    initial_capital: float = 0.0


class Backtester:
    """Walk-forward backtester wiring HMM + strategy + risk."""

    def __init__(
        self,
        config: BacktestConfig,
        hmm_engine: HMMEngine,
        strategy_config: StrategyConfig,
        risk_manager: RiskManager,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """Initialize the backtester.

        Args:
            config: Backtest parameters.
            hmm_engine: Regime-detection engine (refit per fold).
            strategy_config: Allocation-strategy parameters (the orchestrator
                is rebuilt per fold from each fit's regime characterization).
            risk_manager: Drawdown circuit-breaker / sizing gatekeeper.
            feature_engineer: Causal feature builder.
        """
        self.config = config
        self.hmm_engine = hmm_engine
        self.strategy_config = strategy_config
        self.risk_manager = risk_manager
        self.feature_engineer = feature_engineer
        # Stress-test hook: when set to a seed, the per-fold regime->strategy
        # map is randomly permuted (deliberate misclassification) so the stress
        # tester can verify the risk layer contains damage independently of the
        # HMM. ``None`` (default) leaves the correct vol-rank mapping intact.
        self.shuffle_regimes: int | None = None

    # ------------------------------------------------------------------ run ---
    def run(self, prices: dict[str, pd.DataFrame]) -> BacktestResult:
        """Execute the full walk-forward backtest.

        Args:
            prices: Map of symbol -> OHLCV DataFrame. The first symbol is traded.

        Returns:
            `BacktestResult` with the equity curve and ledgers.
        """
        if not prices:
            raise ValueError("no price data supplied")
        symbol = next(iter(prices))
        bars = prices[symbol].sort_index()
        if len(prices) > 1:
            logger.info("Multiple symbols supplied; trading first only: %s", symbol)

        feats = self.feature_engineer.build_features(bars, dropna=True)
        if feats.empty:
            raise ValueError("feature matrix is empty after warmup drop")
        close = bars["close"].reindex(feats.index).astype(float)
        asset_ret = close.pct_change().fillna(0.0)

        folds = self._generate_folds(len(feats))
        if not folds:
            raise ValueError(
                f"not enough usable bars ({len(feats)}) for one fold "
                f"(need train {self.config.train_window} + test 1)"
            )

        cap = self.config.initial_capital
        equity = cap
        held_weight = 0.0
        rf_daily = (self.config.risk_free_rate / 252.0) if self.config.credit_cash_rf else 0.0
        self.risk_manager.reset()  # fresh posture/peak for this run

        eq_idx: list[pd.Timestamp] = []
        eq_val: list[float] = []
        ret_val: list[float] = []
        rows: list[dict] = []          # per-bar regime history
        trades: list[dict] = []        # rebalance events
        fold_bounds: list[tuple[int, int]] = []

        port_ret_hist: list[float] = []  # for trailing weekly return

        for (tr_s, tr_e, te_s, te_e) in folds:
            train_feats = feats.iloc[tr_s:tr_e]
            try:
                self.hmm_engine.fit(train_feats)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Fold train %d:%d skipped (HMM fit failed: %s)", tr_s, tr_e, exc)
                continue

            orch = StrategyOrchestrator(self.strategy_config, self.hmm_engine.regime_info)
            if self.shuffle_regimes is not None:
                self._permute_strategy_map(orch, seed=self.shuffle_regimes + len(fold_bounds))

            # Causal filtered inference: warm up on train history, act on test slice.
            infer_feats = feats.iloc[tr_s:te_e]
            states = self.hmm_engine.predict_regime_filtered(infer_feats)
            test_states = states[-(te_e - te_s):]
            flicker_flags = self._flicker_flags(states)[-(te_e - te_s):]

            fold_bounds.append((te_s, te_e))

            for k, pos in enumerate(range(te_s, te_e)):
                ts = feats.index[pos]
                state = test_states[k]

                # 1) realize prior bar's weight against this bar's asset return
                #    (idle cash earns rf / leverage is charged rf when credit_cash_rf)
                r = float(asset_ret.iloc[pos])
                port_ret = self._cash_credited_return(held_weight * r, held_weight, rf_daily)
                equity *= (1.0 + port_ret)

                # 2) update circuit breakers from realized P&L
                weekly = self._trailing_return(port_ret_hist + [port_ret], 5)
                self.risk_manager.update_drawdown_state(
                    equity=equity, daily_return=port_ret, weekly_return=weekly,
                    calm=self._calm_flag(orch, state.state_id),
                )

                # 3) target weight for this bar (earned t -> t+1)
                sigs = orch.generate_signals(
                    [symbol], {symbol: bars.loc[:ts]}, state, is_flickering=flicker_flags[k]
                )
                raw = sigs[0].position_size_pct * sigs[0].leverage if sigs else 0.0
                raw = min(raw, self.config.max_leverage)
                target = raw * self.risk_manager.target_size_multiplier()

                # 4) rebalance on meaningful drift, or always when de-risking to 0
                delta = target - held_weight
                must_exit = target == 0.0 and held_weight > 0.0
                slip_cost = 0.0
                if abs(delta) >= self.strategy_config.rebalance_threshold or must_exit:
                    px = float(close.iloc[pos])
                    atr = sigs[0].metadata.get("atr") if sigs else None
                    atr_pct = (atr / px) if (atr and px > 0) else 0.0
                    rate = self._slippage_rate(
                        self.config.slippage_pct, self.config.slippage_vol_coeff, atr_pct
                    )
                    slip_cost = abs(delta) * rate
                    equity *= (1.0 - slip_cost)
                    self.risk_manager.record_trade()
                    trades.append(
                        dict(
                            timestamp=ts, symbol=symbol,
                            from_weight=round(held_weight, 4), to_weight=round(target, 4),
                            delta=round(delta, 4), price=float(close.iloc[pos]),
                            slippage_cost=round(slip_cost, 6),
                            regime=state.label.value if isinstance(state.label, Regime) else str(state.label),
                            regime_prob=round(state.probability, 4),
                            risk_state=self.risk_manager.state.value,
                        )
                    )
                    held_weight = target

                net_ret = (1.0 + port_ret) * (1.0 - slip_cost) - 1.0
                port_ret_hist.append(port_ret)

                eq_idx.append(ts)
                eq_val.append(equity)
                ret_val.append(net_ret)
                rows.append(
                    dict(
                        timestamp=ts,
                        regime=state.label.value if isinstance(state.label, Regime) else str(state.label),
                        regime_id=state.state_id,
                        regime_prob=state.probability,
                        confirmed=state.is_confirmed,
                        flickering=bool(flicker_flags[k]),
                        weight=held_weight,
                        vol_rank=orch.vol_rank.get(state.state_id, 1.0),
                        risk_state=self.risk_manager.state.value,
                        asset_return=r,
                        port_return=net_ret,
                    )
                )

        equity_curve = pd.Series(eq_val, index=pd.DatetimeIndex(eq_idx), name="equity")
        returns = pd.Series(ret_val, index=equity_curve.index, name="return")
        regime_hist = pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
        trades_df = pd.DataFrame(trades)
        regime_labels = regime_hist["regime"] if not regime_hist.empty else pd.Series(dtype=object)
        asset_returns = asset_ret.reindex(equity_curve.index)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades_df,
            regime_labels=regime_labels,
            fold_boundaries=fold_bounds,
            regime_history=regime_hist,
            asset_returns=asset_returns,
            symbol=symbol,
            initial_capital=cap,
        )

    def run_portfolio(self, frames: dict[str, pd.DataFrame], return_weights: bool = False):
        """Multi-asset backtest (E-1): regime sets the gross budget, split across names.

        The market-wide volatility regime (detected on the **primary** symbol via
        the existing single-asset walk-forward) sets HOW MUCH gross exposure the
        book runs each bar; :func:`~core.portfolio.portfolio_target_weights` splits
        that budget equally across the universe, capped per name
        (``max_single_position``) and by count (``max_concurrent``). Portfolio
        return is the weighted sum of per-symbol returns; slippage is charged on
        per-name turnover. v1 = equal-weight, shared regime (see the optimization
        roadmap doc for the design + forks).

        Args:
            frames: ``{symbol: OHLCV}``; the first symbol drives the regime.
            return_weights: If True, also return the per-bar weight DataFrame.

        Returns:
            Portfolio equity ``Series`` (or ``(equity, weights_df)`` if requested).
        """
        symbols = list(frames)
        primary = symbols[0]
        base = self.run({primary: frames[primary]})   # regime + gross budget per OOS bar
        budget = base.regime_history["weight"]
        idx = budget.index
        rc = self.risk_manager.config
        rets = {
            s: frames[s]["close"].pct_change().reindex(idx).fillna(0.0).to_numpy()
            for s in symbols
        }

        equity = self.config.initial_capital
        eq_val: list[float] = []
        weight_rows: list[dict] = []
        prev_w = {s: 0.0 for s in symbols}

        for t in range(len(idx)):
            # 1) realize prior weights against this bar's returns
            port_ret = sum(prev_w[s] * rets[s][t] for s in symbols)
            equity *= (1.0 + port_ret)
            # 2) new target weights from this bar's regime budget
            w = portfolio_target_weights(
                float(budget.iloc[t]), symbols, rc.max_single_position, rc.max_concurrent
            )
            # 3) slippage on per-name turnover
            turnover = sum(abs(w.get(s, 0.0) - prev_w[s]) for s in symbols)
            equity *= (1.0 - turnover * self.config.slippage_pct)
            prev_w = {s: w.get(s, 0.0) for s in symbols}
            eq_val.append(equity)
            weight_rows.append(dict(prev_w))

        eq = pd.Series(eq_val, index=idx, name="equity")
        if return_weights:
            return eq, pd.DataFrame(weight_rows, index=idx)
        return eq

    def run_rotation(
        self,
        frames: dict[str, pd.DataFrame],
        rot_cfg: "RotationConfig",
        proxy: str | None = None,
        return_weights: bool = False,
    ):
        """Cross-asset regime rotation backtest (vía B).

        The market regime is detected on the **proxy** via the existing single-asset
        walk-forward (causal, refit per fold). Each OOS bar, the proxy regime's
        *clean volatility tier* (``vol_rank``, NOT the halt-adjusted weight) maps to a
        cross-asset allocation via :func:`~core.asset_rotation.rotation_weights`
        (equities in risk-on, bonds+gold+cash in risk-off). A volatility target scales
        the risky book; the unallocated remainder is an explicit cash sleeve credited
        at the risk-free rate. Slippage is charged on per-name turnover.

        Taking only the tier (not the proxy's halt-adjusted weight) keeps the rotation
        book's risk to a single, non-latching layer (the vol target) and avoids
        re-importing the halt-latch pathology this project escaped.

        Args:
            frames: ``{symbol: OHLCV}`` for every ``rot_cfg.symbols`` and the proxy.
            rot_cfg: Frozen rotation configuration (the pre-registered knobs).
            proxy: Regime proxy symbol (defaults to the first frame key).
            return_weights: If True, also return the per-bar weight DataFrame.

        Returns:
            Portfolio equity ``Series`` (or ``(equity, weights_df)`` if requested).
        """
        from core.asset_rotation import rotation_weights, vol_target_scale

        proxy = proxy or next(iter(frames))
        base = self.run({proxy: frames[proxy]})       # causal regime + vol_rank per bar
        vr = base.regime_history["vol_rank"]
        idx = vr.index
        symbols = rot_cfg.symbols
        missing = [s for s in symbols if s not in frames]
        if missing:
            raise ValueError(f"missing price frames for rotation symbols: {missing}")
        rets = {
            s: frames[s]["close"].pct_change().reindex(idx).fillna(0.0).to_numpy()
            for s in symbols
        }
        rf_daily = (self.config.risk_free_rate / 252.0) if self.config.credit_cash_rf else 0.0

        equity = self.config.initial_capital
        eq_val: list[float] = []
        weight_rows: list[dict] = []
        port_hist: list[float] = []
        prev_w = {s: 0.0 for s in symbols}

        for t in range(len(idx)):
            # 1) realize prior weights against this bar's returns; idle cash earns rf
            risky_ret = sum(prev_w[s] * rets[s][t] for s in symbols)
            cash_w = 1.0 - sum(prev_w.values())
            port_ret = risky_ret + cash_w * rf_daily
            equity *= (1.0 + port_ret)
            port_hist.append(port_ret)

            # 2) target weights for next bar: tier -> rotation map, scaled to vol target
            base_w = rotation_weights(float(vr.iloc[t]), rot_cfg)
            k = vol_target_scale(
                port_hist[-rot_cfg.vol_window:],
                rot_cfg.target_vol, rot_cfg.gross_cap, rot_cfg.gross_floor,
                rot_cfg.periods_per_year,
            )
            w = {s: base_w.get(s, 0.0) * k for s in symbols}

            # 3) slippage on per-name turnover
            turnover = sum(abs(w[s] - prev_w[s]) for s in symbols)
            equity *= (1.0 - turnover * self.config.slippage_pct)
            prev_w = w
            eq_val.append(equity)
            weight_rows.append(dict(prev_w))

        eq = pd.Series(eq_val, index=idx, name="equity")
        if return_weights:
            return eq, pd.DataFrame(weight_rows, index=idx)
        return eq

    # -------------------------------------------------------------- helpers ---
    def _generate_folds(self, n_bars: int) -> list[tuple[int, int, int, int]]:
        """Compute rolling (train_start, train_end, test_start, test_end) folds.

        Args:
            n_bars: Total number of usable (post-warmup) bars available.

        Returns:
            List of fold index tuples; the final test window is truncated to
            the available data.
        """
        tw, te, step = self.config.train_window, self.config.test_window, self.config.step_size
        folds: list[tuple[int, int, int, int]] = []
        start = 0
        while start + tw < n_bars:
            tr_s, tr_e = start, start + tw
            te_s, te_e = tr_e, min(tr_e + te, n_bars)
            if te_e <= te_s:
                break
            folds.append((tr_s, tr_e, te_s, te_e))
            start += step
        return folds

    @staticmethod
    def _slippage_rate(base_pct: float, vol_coeff: float, atr_pct: float) -> float:
        """Slippage rate per unit turnover, scaled by volatility (R-1).

        Flat ``base_pct`` plus a volatility premium ``vol_coeff * atr_pct``. With
        ``vol_coeff == 0`` this reduces to the legacy flat rate (no behaviour
        change). Higher ATR%% (volatile bars) cost more to rebalance, as in real
        markets where spreads and impact widen with volatility.

        Args:
            base_pct: Flat base slippage rate (e.g. 0.0005 = 5 bps).
            vol_coeff: Premium per unit of ATR fraction.
            atr_pct: ATR as a fraction of price at the rebalance bar (>= 0).

        Returns:
            Slippage rate to charge on ``abs(delta)`` turnover.
        """
        return base_pct + vol_coeff * max(0.0, atr_pct)

    def _apply_slippage(self, price: float, side: str) -> float:
        """Adjust a fill price for slippage.

        Args:
            price: Reference price.
            side: "buy" or "sell".

        Returns:
            Slippage-adjusted fill price (buys fill higher, sells lower).
        """
        s = self.config.slippage_pct
        return price * (1.0 + s) if side == "buy" else price * (1.0 - s)

    @staticmethod
    def _trailing_return(returns: list[float], window: int) -> float:
        """Compound the trailing ``window`` per-bar returns.

        Args:
            returns: Per-bar return history (most recent last).
            window: Number of trailing bars to compound.

        Returns:
            Compounded return over the trailing window (0.0 if empty).
        """
        tail = returns[-window:]
        if not tail:
            return 0.0
        comp = 1.0
        for x in tail:
            comp *= (1.0 + x)
        return comp - 1.0

    @staticmethod
    def _permute_strategy_map(orch: StrategyOrchestrator, seed: int) -> None:
        """Randomly permute a fold's regime->strategy map (misclassification).

        Reassigns each regime id to a randomly chosen strategy among the
        orchestrator's three vol-tier strategies, breaking the vol-rank logic on
        purpose. Used only by the stress tester.

        Args:
            orch: Orchestrator whose mapping is permuted in place.
            seed: RNG seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        choices = [orch._low, orch._mid, orch._high]
        for rid in list(orch.regime_to_strategy):
            orch.regime_to_strategy[rid] = choices[int(rng.integers(0, 3))]

    @staticmethod
    def _cash_credited_return(port_ret: float, weight: float, rf_daily: float) -> float:
        """Add the cash/financing term to a bar's portfolio return.

        Idle capital ``(1 - weight)`` earns the daily risk-free rate; levered
        exposure (``weight > 1``) is charged rf on the borrowed excess. With
        ``rf_daily == 0`` this is the identity (legacy: idle cash earns nothing).
        Needed for a fair Sharpe vs a 100%-invested buy&hold, since the Sharpe
        hurdle subtracts rf from both but a de-risked book holds cash.

        Args:
            port_ret: Invested P&L this bar (``weight * asset_return``).
            weight: Held portfolio weight this bar.
            rf_daily: Daily risk-free rate (0 to disable).

        Returns:
            Portfolio return including the cash/financing term.
        """
        return port_ret + (1.0 - weight) * rf_daily

    @staticmethod
    def _calm_flag(orch: "StrategyOrchestrator", state_id: int) -> bool:
        """True when the current regime's vol tier is below the high-vol cutoff.

        Drives the risk manager's vol-normalization re-entry: the peak-DD halt
        releases only after K consecutive *calm* bars. Unknown regimes default to
        not-calm (conservative — won't trigger re-entry).

        Args:
            orch: The fold's strategy orchestrator (holds ``vol_rank`` per regime).
            state_id: Current regime state id.

        Returns:
            ``True`` if ``vol_rank[state_id] < HIGH_VOL_MIN``.
        """
        from core.regime_strategies import HIGH_VOL_MIN
        return orch.vol_rank.get(state_id, 1.0) < HIGH_VOL_MIN

    @staticmethod
    def _flicker_flags(states: list) -> list[bool]:
        """Per-bar flicker flag: True when the regime is switching too rapidly.

        Mirrors :meth:`HMMEngine.is_flickering` but evaluated causally at every
        bar (trailing window of raw state-id changes).

        Args:
            states: Ordered list of ``RegimeState`` from filtered inference.

        Returns:
            List of booleans, one per state.
        """
        window, threshold = 20, 4
        flags: list[bool] = []
        for i in range(len(states)):
            lo = max(0, i - window + 1)
            seg = states[lo : i + 1]
            changes = sum(
                1 for a, b in zip(seg[:-1], seg[1:]) if a.state_id != b.state_id
            )
            flags.append(changes > threshold)
        return flags
