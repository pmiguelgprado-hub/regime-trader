# regime-trader — Go-Live Review

_Last updated: 2026-06-01. Phases 1–8 implemented. 102 tests passing, 0 skipped._

**Read this before funding anything.** The test suite proves the *units and the
backtest/dry-run pipeline*. It does **not** prove the system works live: the bot
has never connected to Alpaca, never received a live bar, never placed an order.
Below is the honest list of what must be revised or added before it trades real
money — ordered by severity.

---

## ✅ What is actually proven

- Backtest stack (walk-forward, performance, stress) — runs end-to-end on real
  daily data via `--backtest` / `--stress-test`.
- Risk layer (`validate_signal` veto, latching `CircuitBreaker`, sizing,
  leverage rules, correlation) — unit-tested to contract.
- Broker adapters (client/executor/tracker) — unit-tested against a **mocked**
  Alpaca SDK (paper default, live-confirm gate, backoff, order parsing,
  partial-fill netting, reconcile).
- Monitoring (JSON logs, rate-limited alerts, dashboard render) — unit-tested.
- `--dry-run` — full pipeline (features → HMM → strategy → risk → decision) on
  real daily SPY, **no orders**. This is the only integration coverage.

---

## 🔴 CRITICAL — must resolve before any capital (even paper is premature)

1. **The strategy has no proven edge — gates REAL MONEY (not paper).** On SPY
   2019–2024 the strategy returned **6.9% vs buy-and-hold 69.9%**, and lost to a
   random-allocation benchmark (27%). The mechanics are correct; the *strategy*
   underperforms — it de-risks through 2020/2022 volatility and misses the
   rebounds. **Do not fund this with real capital until the edge is fixed or the
   project is re-scoped to research.** Note this gates *funding*, not *paper
   trading* — paper is free and is the cheapest way to shake out the untested
   live bugs below (#3, #4, #6). Paper-trade first; fund last.

2. **Timeframe mismatch (daily vs 5-min).** Everything — the HMM,
   `min_train_bars=504` (~2 years of *daily* bars), the volatility regimes, the
   strategy parameters, and every backtest result — is calibrated on **daily**
   bars. The Phase-7 spec defaults the live loop to **5-minute** bars, where
   regimes and vol-ranks mean something entirely different and have never been
   validated. `config/settings.yaml` currently sets `timeframe: 1Day`.
   **Decision required:** run live on **daily** bars (matches the backtest), or
   re-run the entire walk-forward + stress validation at 5-min before trusting
   intraday signals. Do not run intraday on daily-calibrated parameters.

3. **Live execution OVER-ACCUMULATES — it has no rebalance/delta gate.** The
   backtester is *allocation-based*: each bar it sets a target weight and trades
   the **delta** only when `abs(target − current) ≥ rebalance_threshold` (buys
   to add, **sells to reduce**). The live loop
   (`TradingSystem.process_symbol` → `order_executor.submit_signal`) is missing
   that gate entirely. `orchestrator.generate_signals` emits a full
   target-allocation Signal **every bar regardless of what is already held**,
   and `submit_signal` buys the full `approved_shares` each time with no check
   against the current position. Consequences:
   - On repeated same-direction bars the bot **re-buys the full target every
     bar and over-accumulates** — piling up toward the gross-exposure cap (≈7–8×
     the intended ~13% position) until `validate_signal`'s exposure check finally
     starts rejecting. This is a serious live bug, not just a missing feature.
   - It also **cannot reduce** exposure when the target drops (no sell side).

   Why the tests didn't catch it: `--dry-run` runs with `dry_run=True` and a
   fresh empty `positions=[]` each bar, so nothing ever accumulates — the
   integration test validates the *decision* pipeline but never the
   *execution/position-accounting* loop.

   **Fix:** before submitting, compute `delta = target_weight −
   current_weight` from the position tracker and only trade when
   `abs(delta) ≥ rebalance_threshold`, issuing a buy for `delta > 0` and a
   **sell** for `delta < 0` (port the backtester's logic). The fix must handle
   **both directions** — fixing only the sell side still leaves the over-buy.

---

## 🟠 HIGH — required before live (paper can start once #1–3 are addressed)

4. **Trailing stops can't fire (`modify_stop` has no order id).**
   `update_trailing_stops` needs `Position.stop_order_id`; the field now exists
   but **nothing populates it**, because the bracket order's child stop-leg id
   is never captured back from Alpaca. "Update trailing stops per regime" is
   currently a no-op live. **Wire it:** after `submit_bracket_order`, query the
   child orders (`get_order_history` / order legs) and store the stop leg's id
   on the tracked position.

5. **Data-feed entitlement (verify before funding).** Confirm your Alpaca plan
   returns **real-time** bars for the chosen timeframe. Free-tier market data
   has historically been delayed/limited; intraday signals on delayed data act
   on stale prices. Not something the code can check for you.

6. **The entire live path is untested against real Alpaca.** `connect`, the
   WebSocket stream, order round-trips, fill notifications, reconnect-on-drop —
   all `pragma: no cover`. **Run a multi-day paper session and watch the logs**
   before trusting any of it.

7. **HMM model lifecycle.** Live loads `models/hmm_<symbol>.pkl`; if missing it
   trains on startup. Run `python main.py --train-only --symbols SPY` first and
   confirm the pickle round-trips (`HMMEngine.save`/`load`). The 7-day staleness
   retrain is implemented but never exercised live.

---

## 🟡 MEDIUM — inconsistencies / gaps to clean up

8. **Single-asset vs multi-name risk caps.** The backtester trades only the
   **first** symbol. The live loop iterates all `broker.symbols`, but the risk
   layer's `max_single_position` (15%), `max_concurrent`, sector, and
   correlation caps assume a multi-name portfolio that was never backtested.
   Either backtest the multi-asset allocation or restrict live to one symbol.

9. **`core/signal_generator.py` is a dead skeleton.** The Phase-7 pipeline uses
   `StrategyOrchestrator` → `validate_signal` directly; `SignalGenerator`/
   `TradeSignal` are unused and unimplemented. Delete it or implement it — don't
   leave it looking required.

10. **`--dashboard` is not a live attach.** It renders the last
    `state_snapshot.json`, not a running instance (no IPC between processes).
    For a true live dashboard, run it inside the live loop or add a shared state
    file the dashboard polls.

11. **State snapshot is minimal.** It restores `equity_peak` and recent signals,
    not open positions — recovery relies on `tracker.sync_on_startup()`
    reconciling against the broker (acceptable, but know that's the source of
    truth on restart).

12. **`PortfolioState.price_history` is empty live**, so the correlation check
    silently skips until you feed it return series. Intentional/graceful, but
    means correlation limits are inactive until wired.

---

## Quick reference — CLI

```bash
python main.py --backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
python main.py --backtest --compare --symbols SPY --start 2019-01-01 --end 2024-12-31
python main.py --stress-test
python main.py --train-only --symbols SPY
python main.py --dry-run --symbols SPY          # full pipeline, no orders
python main.py --dashboard
python main.py --live                            # paper by default; needs .env
```

## Path to a *paper* run (recommended next step — catches the live bugs cheaply)

1. Decide timeframe (#2) and make config + calibration consistent.
2. Add the delta/rebalance gate, both directions (#3) — without it the paper bot
   over-accumulates immediately.
3. Wire `stop_order_id` so trailing stops fire (#4).
4. `cp .env.example .env`, add **paper** keys, `python main.py --train-only
   --symbols SPY`, then `python main.py --live` and **watch the logs for several
   sessions**. Paper is where #3/#4/#6 actually get shaken out — do this before
   any funding decision.

## Path to *real money* (do NOT skip)

5. Only after a clean multi-session paper run **and** a fixed strategy edge (#1)
   — the backtest currently says the strategy loses to buy-and-hold and to
   random. No edge, no funding.
