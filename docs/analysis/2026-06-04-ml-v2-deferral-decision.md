---
type: decision
status: frozen
tags: [regime-trader, cross-sectional, via-c, ml-v2, decision, defer, overfit]
created: 2026-06-04
related: ["[[2026-06-04-cross-sectional-prereg]]", "[[2026-06-04-stock-picking-feasibility]]", "[[2026-06-03-oos-validation]]"]
---

# Decision — Defer ML v2. Keep the v1 momentum book. Build nothing now.

> **Verdict:** Do **not** build the Phase 3 ML predictor (gradient-boosted trees on the
> momentum + quality panel). Keep the deployed v1 cross-sectional momentum book (50 names,
> zero fitted parameters) running untouched on paper under its frozen pre-registration. Do
> not substitute a "lighter" non-ML quality screen either. **Build nothing now.**

## Context

Phase 1–2 built the fundamentals data foundation: a bulk SimFin loader and a
cross-sectional feature panel (rows = S&P 500 names; cols = v1 price momentum + 5
quality factors, point-in-time). Phase 3 was slated to train a `HistGradientBoostingRegressor`
on that panel to predict forward 1-month cross-sectionally-demeaned returns and replace the
v1 momentum rule as the ranker. The design was scoped, both modelling forks were chosen, and
the build was ready to start.

Before building, the operator asked the decisive question: *does the ML model actually
improve the bot — and if the current 50-name book is better, do not implement it.*

## Decision: do not build it

Three dispositive reasons, any one of which is sufficient; together, decisive.

1. **It cannot be validated by the project's own bar.** The free fundamentals history is the
   *current* constituents' filings — survivorship-biased. The frozen v1 pre-registration
   already states the ML predictor "es un v2 con su propio pre-registro y **datos de pago**."
   Any backtest on this data is a plumbing smoke, never evidence of edge. Building a model we
   already know we cannot validate is busywork, not progress.

2. **There is no measured baseline to improve on.** v1's edge is **unknown** — its ≥12-month
   forward-paper clock is still running (deployed 2026-06-04). You cannot measure an
   "improvement" against a baseline you have not yet measured. Worse, swapping in (or forking
   for) an ML ranker now **resets or forks that clock**, leaving two unvalidated strategies
   compared against each other instead of one clean hypothesis under test.

3. **Too few independent observations to fit *and* validate without overfitting.** Under
   current-constituent (survivorship-biased) data with a handful of features and ~500 names,
   the panel is far from the regime where this kind of model earns its keep. The documented
   prior bites here: a *published* HMM + neural-net + Black-Litterman stock system still
   returned IR −0.1 vs the S&P; Gu-Kelly-Xiu's results rest on ~60 years × thousands of names
   × ~94 features of **paid point-in-time** data. Overfit is, and has always been, THE risk
   on this project (three prior avenues — timing, re-entry, rotation — were each falsified
   exactly this way).

**What this is NOT a claim about.** It is *not* a claim that the quality factors are
signal-less. Gross profitability and the other quality metrics are documented-robust
*cross-sectional* factors (Novy-Marx); they are static in time (slow-moving fundamentals) but
vary across names at every rebalance. The objection is **timing + data**, not factor
worthlessness: we lack the validatable data and the measured baseline to add them
responsibly right now.

## Also rejected: the "lighter" non-ML quality screen

A tempting consolation prize — skip the ML model, just add a simple non-fitted quality
screen/composite to the momentum ranker — is **also rejected**. It carries the *same*
survivorship-biased-fundamentals contamination and the *same* unmeasured-baseline problem,
and it still mutates the frozen v1 knobs (multiple-testing, invalidates the gate). It is the
same mistake in a smaller package. The disciplined answer is *build nothing*, not *build
something lighter*.

## Disposition of Phase 1–2 — parked infrastructure, not waste

The bulk adapter (`data/simfin_bulk.py`) and the feature panel (`core/feature_panel.py`) are
correct, tested (276 green), and committed. They are **parked, ready-to-use infrastructure**.
They become worth activating **if and when both** of these hold:

- the v1 momentum book **passes** its frozen forward-paper gate (≥12 months, beats EW-S&P500
  and SPY risk-adjusted net-of-cost, DSR > 0); **and**
- **paid point-in-time** fundamentals data is acquired (removing the survivorship bias that
  makes any ML validation non-evidential).

At that point — and only then — an ML/quality layer earns its **own** pre-registration. Until
then, the simplest deployed hypothesis runs untouched.

## What happens now

- **No code is written for Phase 3.** The brainstorming/implementation flow is closed.
- The v1 book continues on paper, unchanged, accumulating its forward track record.
- The roadmap no longer implies Phase 3 is being built; the next agent must **not** pick it up
  by default. Re-opening requires the two preconditions above to be met first.

**Status:** ML v2 DEFERRED (not cancelled — gated behind v1-gate-pass + paid-PIT-data).
Real money remains BLOCKED. The disciplined move is to wait for the baseline to speak.
