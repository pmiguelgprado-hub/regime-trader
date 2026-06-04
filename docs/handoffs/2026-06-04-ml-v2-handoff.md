---
type: handoff
status: superseded
tags: [regime-trader, handoff, ml-v2, cross-sectional, via-c]
created: 2026-06-04
next_focus: "NONE — Phase 2 done, Phase 3 (ML) DEFERRED. v1 momentum book runs untouched."
---

> **SUPERSEDED 2026-06-04.** Phase 2 (bulk adapter + feature panel) is **DONE** (commits
> `9661525`, `8cf40c5`; 276 tests; fundamentals coverage 477/503). Phase 3 (the ML
> predictor) is **DEFERRED — do not build it.** Reasoning (can't-validate on survivorship
> free data / no measured baseline while v1's 12mo clock runs / too few obs to fit+validate):
> `docs/analysis/2026-06-04-ml-v2-deferral-decision.md`. **Next agent: do NOT pick up Phase 3.**
> The Phase 1–2 panel is parked infra — activate only if v1 passes its gate AND paid
> point-in-time data is acquired. The sections below are the original (now historical) plan.

# Handoff — regime-trader ML v2 (Phase 2)

## 1. Goal of next session
Build **ML v2 Phase 2**: a cross-sectional **feature panel** (rows = S&P 500 names, cols =
momentum + the quality factors) for the whole universe, loaded via **bulk download / cache**
(not per-ticker REST — see blocker). This panel feeds Phase 3 (the model). Stay phase-by-phase
with a checkpoint after each (Pablo's explicit choice).

## 2. State of play
**v1 book = DONE + LIVE on paper.** Cross-sectional momentum ranker (alpha) + HMM `vol_rank`
gross overlay (risk), monthly rebalance, sector cap, dashboard. Architecture rationale, the
3 falsified avenues, and "why this won't be known until ≥12mo forward" are in the analysis
docs (§5) — do not re-litigate.
- **Executed on paper:** daily SPY bot retired (`launchctl unload …runonce`); 50 book orders
  submitted; **market opened mid-session and they began filling** (cash 85.6k→82.6k). Monthly
  agent `com.regimetrader.rebalance` loaded (fires 1st of month).
- **Sector cap** live: top sector held to 30% (was 44% IT). Config `cross_sectional.max_sector_fraction`.
- **Dashboard** live at `localhost:8501`: stock-picker dropdown + per-company return panel +
  "Cross-Sectional Book" panel. Server runs in background; restart = `streamlit run monitoring/streamlit_app.py`.
- **ML v2 Phase 1 = DONE:** `core/fundamental_features.py` (gross_profitability/ROE/ROA/margin/
  leverage, **point-in-time** via Publish Date) + `data/simfin_data.py` (SimFin v3 loader).
  Proven on real AAPL/NVDA. 263 tests green.

## 3. Open decisions (next agent must resolve)
- **Data scaling (BLOCKER):** SimFin free tier 429s after ~10 per-ticker calls → can't fetch
  ~500 names that way. Decide: (a) `simfin` **python package** bulk dataset download *[lean — built for this]*; (b) local cache (fetch monthly, store); (c) yfinance fundamentals. Quota resets (~daily).
- **Value factor:** current features need no market cap. Adding value (earnings/book yield)
  needs shares outstanding (field not yet located in SimFin BS) — decide whether to add.
- **Model (Phase 3):** start with gradient-boosted trees (GKX), NOT a neural net. Keep degrees
  of freedom low — overfit is THE project risk.
- **Honest framing (locked):** free history = current-constituent filings → survivorship-biased
  training → v2 is FORWARD-deployed, judged by the **same pre-registered gate**, never a
  backtest edge claim. Validate with walk-forward + DSR/PBO (Phase 4).

## 4. Action items (immediate)
- **Rotate the SimFin API key** — it leaked into a test's stdout this session (now in transcript). Then update `.env` `SIMFIN_API_KEY`.
- **Monday:** verify the 49 book buys filled (some may reject on buying-power; re-run `python main.py --rebalance --execute` to top up). The Monday fills are the *pre-cap* book; the sector cap applies from the next rebalance.

## 5. Skills to use
- `claude-mem:mem-search` / read the memory file first — fastest way to absorb full context.
- `superpowers:test-driven-development` — every module here was TDD'd; keep it (the gate discipline).
- `td-train-test-split` + `statistical-analyst` — Phase 3/4 model split + DSR/PBO significance.
- `superpowers:verification-before-completion` — run the real path, not py_compile (this session's repeated lesson: caught NameError + vol_rank glue + collision only by running it).
- `handoff:cs-handoff` — to hand off again at the next checkpoint.

## 6. Artifacts (paths/URLs only)
- Memory: `~/.claude/projects/-Users-pablomiguelgonzalezprado-AIOS/memory/project-regime-trader.md` (full chronology; read first).
- Code: `core/cross_sectional_ranking.py`, `core/fundamental_features.py`, `data/simfin_data.py`,
  `data/constituents.py`, `main.py::run_rebalance`, `monitoring/streamlit_app.py`, `backtest/backtester.py::run_portfolio`.
- Frozen gate: `docs/analysis/2026-06-04-cross-sectional-prereg.md`.
- Analysis: `docs/analysis/2026-06-04-stock-picking-feasibility.md`, `2026-06-04-markov-edge-redesign.md`, `2026-06-04-rotation-results.md`, `2026-06-03-oos-validation.md`.
- Commits: `c160b91` (v1) · `43902e9` (execution+dashboard) · `0502d7e` (sector cap) · `011ca06` (picker) · `6b2092d` (SimFin loader) · `ee0d6b8` (Phase 1 features). Branch `feat/high-priority-improvements`.
- Run: `.venv/bin/python -m pytest -q` (263); `python main.py --rebalance` (dry-run); deploy plists in `deploy/`.
- AIOS vault note: `~/AIOS/conversations/regime-trader/2026-06-04-cross-sectional-and-ml-v2.md`.
