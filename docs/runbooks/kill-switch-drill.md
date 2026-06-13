# Runbook — Kill-switch drill (gap 2)

Quarterly drill (paper = free dress rehearsal). Goal: verify that the three
failure modes a 12-month unattended gate must survive are detected, alerted, and
recoverable — **before** they happen for real. Paper account; real money is
BLOCKED regardless.

Cadence: once per quarter, and after any change to `core/risk_monitor.py`,
`monitoring/alerts.py`, or the launchd plists.

---

## Pre-checks (5 min)

```bash
launchctl list | grep regimetrader          # runonce, rebalance, recordtrack, challenger, quality, riskcheck loaded
python main.py --verify-evidence            # evidence chain intact
python scripts/flatten_preview.py           # current book + the sell orders that would flatten it
```

Confirm the heartbeat + Telegram channel are live: the last `track_record.csv`
row is ≤2 business days old (else the heartbeat would already be alerting), and a
test alert reaches Telegram (`monitoring/alerts.py` WARNING+ → bot).

---

## Drill 1 — Alpaca API outage

**Simulate:** temporarily point at a bad endpoint or revoke the key in `.env`
(keep a backup), then run one cycle:

```bash
python main.py --risk-check        # observe mode; should log the failure, not crash silently
```

**Expect:** the cycle raises/logs a connection error and an `api_lost` /
`fatal_error` alert fires (console + Telegram). No partial state written.
**Verify:** alert received; `risk_monitor_state.json` unchanged; restoring the key
returns the next cycle to normal.

## Drill 2 — Bad-data day

**Simulate:** the data-quality sentinel (`core/data_quality.py`) is wired into
`--record-track`. Feed it a stale/NaN/jump series (unit-tested in
`tests/test_data_quality.py`); for a live drill, run `--record-track` on a day
the feed is known-thin.

**Expect:** a `data_quality` WARNING alert listing the issues; the row is still
recorded (flag, don't drop — the operator decides whether to quarantine it).
**Verify:** alert names the symbol + kind (stale/jump/nonpositive); the evidence
chain row is written so the anomaly is itself tamper-evident.

## Drill 3 — Runaway / unexpected position

**Simulate:** manually buy an off-book name in the Alpaca paper console (e.g. a
large SPY lot), then run the risk monitor:

```bash
python main.py --risk-check        # observe mode
python scripts/flatten_preview.py  # confirm the flatten plan now includes the rogue lot
```

**Expect:** intraday drawdown ladder (`alert 2% → derisk 4% → flatten 8%`)
evaluates correctly; in observe mode it logs what it *would* submit. The flatten
preview lists the rogue position.
**Verify:** the ladder is monotonic within the session and the latch resets next
session (the old halt-latch lesson); manual flatten clears the book.

---

## Manual flatten (real emergency)

This project intentionally has **no automated all-in flatten**. To flatten by hand:

1. `python scripts/flatten_preview.py` — review the exact sell orders.
2. Submit them via the **Alpaca console** (paper or live), or
3. Enable the gated intraday flatten tier: set `risk_monitor.allow_orders: true`
   in `config/settings.yaml` and run `python main.py --risk-check --execute`.
   **This amends the frozen pre-registration** — document the amendment first
   (it was deliberately observe-only during the gate).

---

## Post-drill

- Restore `.env` / config to the pre-drill state; re-run `--verify-evidence`.
- Note the drill date + any gaps found in this file's changelog below.
- If a detection or alert failed to fire, that is a P1 fix before the next gate day.

### Changelog

- 2026-06-13 — runbook created; flatten preview tool added; drills not yet exercised live.
