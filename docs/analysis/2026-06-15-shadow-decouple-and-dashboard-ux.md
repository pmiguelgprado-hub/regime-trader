# 2026-06-15 — Shadow logging decoupling + dashboard UX

Two pieces of work this session. Both verified empirically before declaring done.
Books/champion frozen throughout; real money stays BLOCKED.

## 1. Decouple shadow logging from the executor (`fe82e3d`, pushed)

### Problem
The closing note of the prior roadmap session flagged that the nightly `--execute`
baseline now fit the Jump Model + BOCPD and fetched yfinance/FRED inline. Investigation
showed it was worse than the note implied: those measurement blocks ran **upstream of
order submission**, not after.

- `run_rebalance` (weekday 22:30 `--execute`): JM fit + BOCPD (T1.1/T1.2) and the macro
  VIX/FRED fetch (T1.3) ran before `targets` were computed.
- `run_once` (daily): the champion-vs-refit HMM fit (T0.4) ran before `run_cycle`.

`try/except` catches exceptions (after the client's internal timeout) but not a hang with
no timeout. So a slow/hung feed could delay or block paper-order submission.

### Decision
Extract all three blocks into a standalone **read-only** command `run_shadow_log`
(`--shadow-log`): it pulls price history + free macro feeds only, never reads the account
for sizing, never submits. It reproduces `run_rebalance`'s baseline `vol_rank` (overlay
`hmm` = champion argmax) and history depth so `logs/shadow_regime.csv` stays continuous
across the cutover. New plist `com.regimetrader.shadowlog.plist` (Mon-Fri 22:15) keeps the
T1.x shadow study on its daily cadence.

Rejected alternative: reorder blocks to run after order submission. Still couples the
executor process to feed availability and risks a hang after orders submit; a separate
read-only job is the clean fix.

### Verification
- **Island purity** by grep + AST (not eyeball on a ~1000-line function): every variable
  bound inside the three blocks is unreferenced downstream → deletion can't change order
  logic.
- `tests/test_shadow_log_decoupled.py` (AST guard): executor paths free of the feed/fit
  symbols; `run_shadow_log` free of order-submission symbols.
- Live smoke into an isolated tmp dir (real Alpaca/yfinance/FRED): writes the expected
  rows, **zero orders**, refit branch correctly skipped (not retrain-due).
- Full suite **567 passed**; model/book/config artifacts byte-identical (champion
  `8c807d916bf62af5` frozen).
- Routing check: `parse_args(['--shadow-log'])` sets only `shadow_log=True` and dispatches
  before the `else: run_live` fallback (advisor caught the risk that a broken route would
  start the trading loop on the now-loaded job).

### Gated action taken
Installed + loaded the shadowlog plist (read-only, reversible `launchctl unload`) so the
shadow study does not go dark after removing the inline logging. Surfaced loudly.

### Open
- `logs/rebalance.err.log` + `logs/shadowlog.err.log` — glance after the first scheduled
  runs to close the empirical loop on the executor paths under the new code.
- Known minor deviation: the macro block now stamps `now()` instead of the last-bar date
  (`feats` is scoped in the regime try). Immaterial — `shadow_macro.csv` has no reader yet.

## 2. Dashboard UX (`baf590b`, local)

Pablo's localhost cockpit (`monitoring/streamlit_app.py`) had four UX problems. Diagnosed
against the running app on `:8501` via headless Playwright screenshots (before/after).

| Problem | Root cause (seen in screenshot) | Fix |
|---|---|---|
| SPY chart unreadable, tiny scale | Plotly auto-ranged the y-axis off one bad bar (low ~100), crushing the $640-760 band into the top sliver | Robust y-range from 1st/99th percentiles of low/high; taller panel; 1M/3M/6M/all range buttons |
| Probability gauges shifted left | Gauges lived inside the `left` column of `st.columns([2,1])` → confined to the left two-thirds | Pulled out to full width (new `_regime_gauges`), centered/balanced |
| News wall (low value) | 12-link markdown list | Kept (Pablo: don't delete) but de-emphasized to a 6-item collapsed digest |
| Metrics opaque/technical | Vol Rank, transition hazard, DSR, drawdown unexplained | Always-visible plain-Spanish explanations (not hover tooltips), Spanish labels, green/amber/red cues; numbers stay large |

View only; not imported by the test suite (verified). `py_compile` clean; rendering
confirmed by before/after screenshots.

## Durable learnings (filed to the AIOS vault)
- `wiki/concepts/desacoplar-medicion-de-ejecucion.md`
- `wiki/concepts/diagnostico-visual-screenshot-primero.md`
