# research/ — hypothesis & trials ledger (T4.1/T4.2)

The multiple-testing control for the whole program. Every research avenue costs
statistical credibility whether it ships or not; the Deflated Sharpe Ratio
discounts a track record by how many things were tried. Without an auditable
count, mass factor mining (LLM loops included) silently collapses the DSR of
everything. **This ledger precedes all new research** (roadmap invariant).

## Files

- `registry.jsonl` — append-only event log (`core/research_ledger.py`). Two
  event kinds: `registered` (charges `n_configs` to a `family` at freeze time)
  and `verdict` (later outcome referencing the trial id). Rows are never
  rewritten; corrections are new events.

## Workflow

```bash
# new hypothesis: generate the frozen-prereg skeleton AND charge the ledger atomically
python scripts/new_prereg.py --slug quality-edgar --family quality \
    --hypothesis "EDGAR-PIT quality+momentum sleeve beats EW S&P500 net" \
    --n-configs 2 --universe "S&P 500" --cadence monthly

# inspect the budget
python -c "from core import research_ledger as rl; print('momentum:', rl.n_trials(family='momentum'))"

# record an outcome (append-only)
python -c "from core import research_ledger as rl; rl.record_verdict(rl.LEDGER_PATH, '<trial-id>', 'falsified', note='lost to baseline')"
```

`n_trials(family=...)` is what `backtest/performance.py::deflated_sharpe_ratio`
should be fed. Within-family counting is complete; cross-family effective-trials
judgment is documented in each prereg.

## Current charges (backfilled 2026-06-13, basis=estimate)

| Family | n_trials | Books |
|---|---|---|
| momentum | 3 | baseline deployed book (1) + idiosyncratic-momentum challenger (2) |
| hedge | 1 | tail-hedge options overlay |

Backfill basis is `estimate` (reconstructed from the frozen preregs, not charged
at their original freeze). All future charges are `prereg`-basis at freeze time.

## Research-factory rails (T4.4 — rails built, autonomous loop GATED)

`core/research_runner.py` is the hard scaffolding the LLM-alpha-mining literature
ignores. Invoked manually for now (Pablo's decision: manual for weeks, nightly
plist later):

- **Blocklist** (`default_blocklist`) — already-falsified ideas (R1 HMM return
  timer, rotation via B, regime-conditional shorts, direct hmm_prob deploy,
  paid-data order-flow/VPIN, HF pairs) are refused before any evaluation.
- **Weekly budget** — `weekly_budget_ok` caps hypotheses/week (trial discipline >
  throughput); runs logged to `research/runlog.jsonl`.
- **Evaluation harness** — `evaluate_candidate` charges the ledger then runs
  CPCV + deflated-Sharpe, writing a verdict to `research/vault/<id>.md` for a
  HUMAN to adjudicate. A pass is necessary, not sufficient; promotion = new prereg
  + forward book (T2 pattern).

**NOT built (the gated step):** autonomous code generation, a sandbox that writes
trader state, and the nightly scheduler. The rails exist so that when the loop is
eventually turned on, it cannot mine the DSR into the ground.
