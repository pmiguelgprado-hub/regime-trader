---
type: analysis
status: done
tags: [regime-trader, cross-sectional, via-c, operations, validity, track-record, senior-review]
created: 2026-06-04
related: ["[[2026-06-04-cross-sectional-prereg]]", "[[2026-06-04-ml-v2-deferral-decision]]", "[[2026-06-03-oos-validation]]"]
---

# Senior review — current bot situation + improvements

Triggered by: *analyze the current bot, propose improvements; if nothing genuinely
beats the current 50-name book, leave it as is.* This is the review and what was done.

## Current state (verified live, not from memory)

- **Book v1:** 50 names, top-decile 12-1 momentum, equal-weight, 30% sector cap, HMM
  gross-exposure overlay. **Zero fitted parameters.** Executed once on paper 2026-06-04
  (~$101.3k equity by EOD, ~50 names held).
- **Daily SPY bot:** retired (not in `launchctl list`) — no collision.
- **Frozen pre-registration** governs the book: ≥12-month forward paper, beats EW-S&P500
  **and** SPY risk-adjusted net-of-cost, DSR > 0, overlay must add value. Real money BLOCKED.

## Alpha — leave as is (categorical, not "found nothing")

The frozen pre-registration forbids **any** change to signal or construction until the
baseline reports — every alpha idea resets the 12-month clock *by construction*. So this is
not "I looked and found nothing"; it is "the discipline forbids all of them now."

Best documented candidate for the **next** pre-registration cycle (after the baseline
speaks): **Barroso–Santa-Clara constant-volatility momentum** (scale momentum exposure by
its own recent realized volatility — the most-replicated momentum enhancement, removes the
2009-style crash). Deferred because it mutates frozen knobs, needs its own validation, and is
redundant with the HMM overlay already under test. The ML predictor is separately deferred
([[2026-06-04-ml-v2-deferral-decision]]).

## What actually improves the bot now: operational validity (zero overfit, zero gate mutation)

The highest-value work was protecting the integrity of the running experiment, not adding
return. Two gaps would have quietly invalidated the 12-month wait.

| # | Finding | Disposition |
|---|---------|-------------|
| **F1** | **No track-record recorder.** `book_snapshot.json` is overwritten each run; nothing accumulated a daily book / EW-S&P500 / SPY NAV series — the gate would have had nothing to evaluate at month 12. | **BUILT** `core/track_record.py` + `main --record-track` + daily plist (23:00). Verified live (book=101303, spy_ret +0.33%, ew_ret +0.74%). |
| **F2** | **Scheduled agent ran dry-run** (`--rebalance`), so the forward test would have measured buy-and-hold of the June decile, not the pre-registered monthly-rebalanced book. Direction of effect unknown (the OOS record even leans "active doesn't beat static"), but it is the wrong strategy under test. | **Pablo chose auto-execute.** Flipped plist to `--rebalance --execute` (PAPER). |
| **F3** | **No open-order guard** — a re-run in the fill gap would double-submit (the diff is vs *held*, which excludes pending orders). | **BUILT** `drop_open_order_symbols`, wired into the execute path. Prereq for F2. |

`get_orders` already existed; benchmarks (`backtest/benchmarks.py`) are backtest-only — neither
recorded a live series, confirming F1.

## Net

Found something better — but it is **measurement/operational plumbing, not alpha**. Alpha is
correctly left untouched. The book now (a) auto-rebalances monthly on paper, faithful to the
pre-registration, (b) refuses to double-submit, and (c) records a clean daily three-NAV
series so the gate is runnable at month 12. 286 tests green. Commit `5ec34d3`.

**Pending Pablo (carried):** rotate the SimFin key (leaked prior session); first real
auto-execute fires 2026-07-01 — eyeball the fills.
