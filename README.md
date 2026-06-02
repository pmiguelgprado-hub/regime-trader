# regime-trader

HMM-driven, volatility-regime-based allocation trading bot on Alpaca.
Detects market regimes with a Gaussian Hidden Markov Model, maps each regime
to a target allocation/leverage, enforces hard risk limits and drawdown
circuit breakers, and executes via Alpaca (paper by default).

> **Status: PAPER-READY** (132 tests passing). The live loop was audited and
> repaired (see [`docs/audit/2026-06-01-senior-audit.md`](docs/audit/2026-06-01-senior-audit.md),
> §10): buffer backfill (C1), delta/rebalance gate with a sell branch (C4),
> bracket stops on entries (C3), and the drawdown safety slice — mark-to-market
> circuit breaker → halt → liquidation (C2/C5/C6) — are all wired and tested.
> Verified against a live Alpaca **paper** account (no position left behind):
> connection + account, daily-data entitlement at the **full 764-bar seed depth**
> (so C1 isn't a live no-op), and the **real bracket entry path** — submit +
> stop-leg-id capture + cancel. Still unverified until the **first market-open
> session**: an actual fill arriving on the stream and the per-bar MtM breaker
> feed from live equity (both wired + unit-tested, not yet exercised live).
> Dashboard is Streamlit (`streamlit run monitoring/streamlit_app.py`).
>
> **⚠️ Paper-ready ≠ profitable, and NOT cleared for real money.** In backtest
> the strategy underperforms buy-and-hold (no proven edge) — that gates *real
> capital*, not paper. Run paper for ≥1 month and watch every rebalance first.
> Known limits: single-symbol by default (multi-asset risk caps unvalidated,
> M4/M5); no stream auto-reconnect yet (H5). See `docs/go-live-review.md`.
>
> **Safety:** paper trading is the default; live mode requires typing
> `YES I UNDERSTAND THE RISKS`. Secrets come from `.env` (gitignored).
>
> **Window note:** the in-sample window is **504 bars (~2y)**, not the 252 in
> the original spec — `HMMEngine.fit` needs `min_train_bars` (504) usable rows,
> so a 252-bar IS window cannot train the model.

## Backtest CLI

```bash
# walk-forward backtest on SPY (writes CSVs to backtest_output/SPY/)
python main.py --backtest --symbols SPY --start 2019-01-01 --end 2024-12-31

# add benchmarks: buy-and-hold, 200-SMA trend, random (100 seeds)
python main.py --backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare

# crash / gap / regime-misclassification stress probes
python main.py --stress-test
```

## Architecture

```
regime-trader/
├── config/         # settings.yaml (all params) + credentials example
├── core/           # HMM engine, regime strategies, risk manager
├── broker/         # Alpaca client, order executor, position tracker
├── data/           # market data fetching, feature engineering (causal)
├── monitoring/     # structured logging, terminal dashboard, alerts
├── backtest/       # walk-forward backtester, performance, stress tests
├── tests/          # hmm, look-ahead, strategies, risk, orders
└── main.py         # entry point (--live | --backtest | --dry-run | ...)
```

### Pipeline

1. **data** fetches OHLCV → **feature_engineering** builds a strictly causal
   feature matrix.
2. **hmm_engine** detects the current regime (with confidence, stability, and
   flicker filtering).
3. **regime_strategies** maps regime + trend → target allocation/leverage
   and emits the concrete signals.
4. **risk_manager** sizes positions and enforces exposure/leverage/drawdown
   limits.
5. **broker** executes orders and tracks positions/P&L.
6. **monitoring** logs, renders a live dashboard, and fires alerts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                       # fill in Alpaca keys
cp config/credentials.yaml.example config/credentials.yaml   # optional
```

`.env` and `credentials.yaml` are gitignored — never commit secrets.

## Configuration

All parameters live in `config/settings.yaml`, grouped by section:
`broker`, `hmm`, `strategy`, `risk`, `backtest`, `monitoring`.
Broker secrets come from `.env` (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`ALPACA_PAPER`).

## Usage

```bash
python main.py --backtest            # walk-forward backtest
python main.py --stress-test         # crash/gap stress scenarios
python main.py --train-only          # train the HMM and exit
python main.py --dry-run             # full pipeline on history, no orders
python main.py --dashboard           # legacy terminal dashboard (rich)
python main.py --live                # paper/live trading loop (paper by default)
```

Modes are mutually exclusive; `--live` is the default if none is given.
Shared flags: `--symbols`, `--start`, `--end`, `--compare` (backtest only),
`--config`.

### Running it daily (paper)

The strategy is **daily**-calibrated, and Alpaca's bar websocket only streams
*minute* bars — the wrong cadence. So the live entry point is **one decision
cycle per trading day** on the freshly-closed daily bar, not a held-open stream:

```bash
python main.py --run-once     # connect, decide on the latest daily bar, place
                              # bracket orders, update risk, persist state, exit
```

Decisions are made on the daily **close**; bracket entry orders **queue and
fill at the next open**. Schedule `--run-once` once per trading day with the
bundled launchd agent (macOS):

```bash
cp deploy/com.regimetrader.runonce.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.regimetrader.runonce.plist   # weekdays 22:15 local
launchctl unload ~/Library/LaunchAgents/com.regimetrader.runonce.plist # to stop
```

> **Run it at most once per day.** Re-running `--run-once` before the prior
> entry order has *filled* would see a flat tracker and submit a duplicate
> (the C4 delta gate only suppresses re-buys once the position exists). The
> daily schedule (one run/day, fills happen at the next open in between) avoids
> this; don't manually re-run between a queued order and its fill. launchd
> cannot skip US market holidays — on a holiday the unchanged bar yields no
> new order (hold).

`python main.py --live` holds the minute-bar websocket open instead; it is
**not** the right path for a daily strategy (kept for an eventual intraday
re-validation, H5/M3).

### Dashboard (Streamlit web)

The primary dashboard is a Streamlit web app (regime + confidence, portfolio
value, learned regimes, risk controls, signal feed, price/regime overlay):

```bash
streamlit run monitoring/streamlit_app.py     # http://localhost:8501
```

It reads live state from `state_snapshot.json` (written by the trading loop)
and overlay charts from `backtest_output/<symbol>/`. `python main.py --dashboard`
keeps the older terminal renderer as a headless fallback.

## Testing

```bash
pytest
```

Includes a dedicated `test_lookup_ahead.py` to guard against look-ahead bias —
a non-negotiable invariant for any backtest result to be trustworthy.

## Safety

- **Paper trading is the default** (`broker.paper_trading: true`).
- Hard caps on per-trade risk, exposure, leverage, concentration, and trade
  counts.
- Tiered drawdown circuit breakers (daily / weekly / peak) that reduce sizing
  then halt trading.
- Trade live only after backtests, stress tests, and a sustained paper run.

## Disclaimer

For research/education. Trading involves substantial risk of loss. Use at your
own risk. No warranty.
