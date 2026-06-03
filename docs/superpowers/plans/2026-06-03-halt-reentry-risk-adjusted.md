# Halt Re-entry (Vol-Normalization) + Risk-Adjusted Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the peak-DD halt latch by releasing the halt after K consecutive calm (low-vol-regime) bars instead of waiting for equity to recover a monotonic peak, then validate the risk-adjusted result (Sharpe > bh AND maxDD < bh) on a 5-ETF holdout tuned only on SPY.

**Architecture:** Backtester-only change. `RiskManager.update_drawdown_state` gains a `calm` flag and a calm-streak counter; once the peak-DD halt is engaged, K consecutive calm bars release it and reset the equity-peak reference. The backtester derives `calm` from `orch.vol_rank[regime] < HIGH_VOL_MIN`. Re-entry is opt-in via `RiskConfig.peak_reentry_calm_bars` (default 0 = legacy). The live `CircuitBreaker` is untouched.

**Tech Stack:** Python 3.14, pytest, numpy/pandas, hmmlearn. Run tests: `PYTHONPATH=. .venv/bin/python -m pytest -q`.

---

## File Structure

- `core/risk_manager.py` — add `RiskConfig.peak_reentry_calm_bars`, `RiskManager._calm_streak`, re-entry logic in `update_drawdown_state`. (Modify)
- `backtest/backtester.py` — `_calm_flag` staticmethod + pass `calm` into the risk update. (Modify)
- `backtest/oos_validation.py` — extend `SliceMetrics`/`slice_metrics` with `bh_sharpe`, `bh_mdd` (needed for the success criterion). (Modify)
- `config/settings.yaml` — document `peak_reentry_calm_bars: 0` in the `risk` block. (Modify)
- `tests/test_risk.py`, `tests/test_oos_validation.py`, `tests/test_backtest.py` — new tests. (Modify)
- `scripts/tune_reentry.py` — SPY grid (K × floor), pick best Sharpe. (Create)
- `scripts/validate_reentry.py` — frozen combo on 5 ETFs. (Create)

---

### Task 1: RiskConfig field + RiskManager calm-streak state

**Files:**
- Modify: `core/risk_manager.py` (RiskConfig ~94, `__init__` ~363-367, `reset` ~547-551)
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_risk.py`:

```python
def test_reentry_config_default_disabled_and_streak_resets():
    from core.risk_manager import RiskConfig, RiskManager
    cfg = RiskConfig()
    assert cfg.peak_reentry_calm_bars == 0          # default = legacy/disabled
    rm = RiskManager(cfg)
    assert rm._calm_streak == 0
    rm._calm_streak = 4
    rm.reset()
    assert rm._calm_streak == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_risk.py::test_reentry_config_default_disabled_and_streak_resets -v`
Expected: FAIL — `AttributeError: 'RiskConfig' object has no attribute 'peak_reentry_calm_bars'`

- [ ] **Step 3: Add the config field**

In `core/risk_manager.py`, in `RiskConfig` immediately after the `halt_floor_mult` field (line ~94-95):

```python
    peak_reentry_calm_bars: int = 0    # bars of calm (regime vol_rank < HIGH_VOL_MIN)
                                       #   before the peak-DD halt RELEASES, instead of
                                       #   waiting to recover the (monotonic) equity peak
                                       #   that reduced exposure can't reach. 0 = disabled
                                       #   (legacy: peak-halt releases only on recovery).
                                       #   Backtester-only; live keeps the hard halt.
```

- [ ] **Step 4: Add `_calm_streak` to `__init__`**

In `RiskManager.__init__` (after `self._daily_trades: int = 0`, line ~366):

```python
        self._calm_streak: int = 0
```

- [ ] **Step 5: Reset `_calm_streak` in `reset`**

In `RiskManager.reset` (after `self._daily_trades = 0`, line ~551):

```python
        self._calm_streak = 0
```

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_risk.py::test_reentry_config_default_disabled_and_streak_resets -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add core/risk_manager.py tests/test_risk.py
git commit -m "feat(risk): peak_reentry_calm_bars config + calm-streak state (disabled default)"
```

---

### Task 2: Re-entry logic in `update_drawdown_state` (the core fix)

**Files:**
- Modify: `core/risk_manager.py::update_drawdown_state` (signature ~482-487, peak block ~503-534)
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_risk.py`:

```python
def _drive_to_peak_halt(rm):
    """Push equity down ~12% from a 100k peak so the peak-DD breaker engages."""
    from core.risk_manager import RiskState
    rm.update_drawdown_state(equity=100_000.0, daily_return=0.0, weekly_return=0.0)
    st = rm.update_drawdown_state(equity=88_000.0, daily_return=-0.12, weekly_return=-0.12)
    assert st is RiskState.HALTED
    return rm


def test_legacy_peak_halt_latches_when_disabled():
    """K=0 (default): with reduced exposure equity stays below peak -> stays HALTED."""
    from core.risk_manager import RiskConfig, RiskManager, RiskState
    rm = _drive_to_peak_halt(RiskManager(RiskConfig()))  # K=0
    # many flat-ish calm bars below peak: legacy never releases the peak halt
    for _ in range(20):
        st = rm.update_drawdown_state(equity=89_000.0, daily_return=0.0,
                                      weekly_return=0.0, calm=True)
    assert st is RiskState.HALTED


def test_calm_streak_releases_peak_halt_and_resets_peak():
    """K=3: 3 consecutive calm bars release the peak halt without recovering the peak."""
    from core.risk_manager import RiskConfig, RiskManager, RiskState
    rm = _drive_to_peak_halt(RiskManager(RiskConfig(peak_reentry_calm_bars=3)))
    s1 = rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=True)
    s2 = rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=True)
    assert s1 is RiskState.HALTED and s2 is RiskState.HALTED   # streak 1,2 < 3
    s3 = rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=True)
    assert s3 is RiskState.NORMAL                              # streak hit 3 -> released
    assert rm._equity_peak == 89_000.0                        # peak reset to current
    assert rm._calm_streak == 0


def test_high_vol_does_not_release_and_resets_streak():
    """Non-calm bars never accumulate; high-vol keeps the peak halt engaged."""
    from core.risk_manager import RiskConfig, RiskManager, RiskState
    rm = _drive_to_peak_halt(RiskManager(RiskConfig(peak_reentry_calm_bars=3)))
    rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=True)
    rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=True)
    # high-vol bar resets the streak
    st = rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=False)
    assert st is RiskState.HALTED and rm._calm_streak == 0
    # two calm bars again: still < 3, still halted
    rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=True)
    st = rm.update_drawdown_state(equity=89_000.0, daily_return=0.0, weekly_return=0.0, calm=True)
    assert st is RiskState.HALTED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_risk.py -k "release or latches or high_vol" -v`
Expected: FAIL — `update_drawdown_state() got an unexpected keyword argument 'calm'`

- [ ] **Step 3: Add the `calm` parameter to the signature**

In `core/risk_manager.py`, change the `update_drawdown_state` signature (line ~482-487) to:

```python
    def update_drawdown_state(
        self,
        equity: float,
        daily_return: float,
        weekly_return: float,
        calm: bool = True,
    ) -> RiskState:
```

- [ ] **Step 4: Replace the peak-DD escalation block with re-entry logic**

In `update_drawdown_state`, replace this block (currently lines ~533-534):

```python
        if peak_dd >= c.max_dd_from_peak:
            escalate(RiskState.HALTED, f"peak DD {peak_dd:.2%}>=max {c.max_dd_from_peak:.2%}")
```

with:

```python
        peak_breached = peak_dd >= c.max_dd_from_peak
        # Vol-normalization re-entry (opt-in, K>0): once the peak-DD halt is engaged,
        # release it after `peak_reentry_calm_bars` consecutive calm bars instead of
        # waiting for equity to recover the monotonic peak — which reduced exposure
        # prevents, causing the permanent-flat trap. On release, reset the peak to the
        # current equity (fresh reference) so a stale high peak can't re-halt instantly;
        # a fresh 10% drawdown from here re-engages (protection preserved).
        if peak_breached and c.peak_reentry_calm_bars > 0:
            self._calm_streak = self._calm_streak + 1 if calm else 0
            if self._calm_streak >= c.peak_reentry_calm_bars:
                self._equity_peak = equity
                self._calm_streak = 0
                peak_breached = False
        else:
            self._calm_streak = 0

        if peak_breached:
            escalate(RiskState.HALTED, f"peak DD {peak_dd:.2%}>=max {c.max_dd_from_peak:.2%}")
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_risk.py -k "release or latches or high_vol" -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full risk suite for non-regression**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_risk.py tests/test_risk_validate.py -q`
Expected: PASS (all existing + new). If any legacy test that loops `update_drawdown_state` breaks, it means default K!=0 leaked — confirm `RiskConfig().peak_reentry_calm_bars == 0`.

- [ ] **Step 7: Commit**

```bash
git add core/risk_manager.py tests/test_risk.py
git commit -m "feat(risk): release peak-DD halt after K calm bars (breaks the latch trap)"
```

---

### Task 3: Wire `calm` flag from the regime vol-tier in the backtester

**Files:**
- Modify: `backtest/backtester.py` (add `_calm_flag` staticmethod; the risk-update call ~200-203)
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_backtest.py`:

```python
def test_calm_flag_uses_vol_rank_cutoff():
    from backtest.backtester import Backtester
    from core.regime_strategies import HIGH_VOL_MIN

    class _Orch:
        vol_rank = {0: 0.10, 1: 0.50, 2: 0.90}

    orch = _Orch()
    assert Backtester._calm_flag(orch, 0) is True            # low vol -> calm
    assert Backtester._calm_flag(orch, 1) is True            # mid vol (<0.67) -> calm
    assert Backtester._calm_flag(orch, 2) is False           # high vol (>=0.67) -> not calm
    assert Backtester._calm_flag(orch, 99) is False          # unknown -> conservative (not calm)
    assert HIGH_VOL_MIN == 0.67
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_backtest.py::test_calm_flag_uses_vol_rank_cutoff -v`
Expected: FAIL — `AttributeError: type object 'Backtester' has no attribute '_calm_flag'`

- [ ] **Step 3: Add the `_calm_flag` staticmethod**

In `backtest/backtester.py`, add inside `class Backtester` (next to the other staticmethods, e.g. after `_flicker_flags`):

```python
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
```

- [ ] **Step 4: Pass `calm` into the risk update**

In `Backtester.run`, change the risk-update call (currently lines ~200-203):

```python
                weekly = self._trailing_return(port_ret_hist + [port_ret], 5)
                self.risk_manager.update_drawdown_state(
                    equity=equity, daily_return=port_ret, weekly_return=weekly
                )
```

to:

```python
                weekly = self._trailing_return(port_ret_hist + [port_ret], 5)
                self.risk_manager.update_drawdown_state(
                    equity=equity, daily_return=port_ret, weekly_return=weekly,
                    calm=self._calm_flag(orch, state.state_id),
                )
```

- [ ] **Step 5: Run the test + full backtest suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_backtest.py -q`
Expected: PASS (new test + existing; default K=0 means behaviour is unchanged, so existing backtest assertions hold).

- [ ] **Step 6: Commit**

```bash
git add backtest/backtester.py tests/test_backtest.py
git commit -m "feat(backtest): feed regime calm flag to the risk manager for halt re-entry"
```

---

### Task 4: Extend `SliceMetrics` with `bh_sharpe` + `bh_mdd` (success criterion needs them)

**Files:**
- Modify: `backtest/oos_validation.py` (`SliceMetrics` dataclass; `slice_metrics`)
- Test: `tests/test_oos_validation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_oos_validation.py`:

```python
def test_slice_reports_buy_hold_sharpe_and_mdd():
    from backtest.performance import PerformanceAnalyzer
    res, close = _make_result()
    sm = slice_metrics(res, close, "mid", "2018-06-01", "2018-12-31", rf=0.045)
    mask = (res.asset_returns.index >= pd.Timestamp("2018-06-01")) & (
        res.asset_returns.index <= pd.Timestamp("2018-12-31")
    )
    an = PerformanceAnalyzer(risk_free_rate=0.045)
    bh_ret = res.asset_returns[mask]
    bh_eq = (1.0 + bh_ret).cumprod()
    assert abs(sm.bh_sharpe - an.sharpe_ratio(bh_ret)) < 1e-12
    assert abs(sm.bh_mdd - an.max_drawdown(bh_eq)) < 1e-12
    # success predicate helper: beats bh on risk-adjusted terms
    assert sm.beats_bh_risk_adjusted() == (sm.strat_sharpe > sm.bh_sharpe and sm.strat_mdd > sm.bh_mdd)
```

(Note: `strat_mdd > bh_mdd` because drawdowns are negative fractions — "smaller drawdown" means a larger, i.e. less-negative, value.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_oos_validation.py::test_slice_reports_buy_hold_sharpe_and_mdd -v`
Expected: FAIL — `AttributeError: 'SliceMetrics' object has no attribute 'bh_sharpe'`

- [ ] **Step 3: Add fields to `SliceMetrics`**

In `backtest/oos_validation.py`, add to the `SliceMetrics` dataclass (after `sma200_return`):

```python
    bh_sharpe: float = 0.0
    bh_mdd: float = 0.0
```

And add a predicate method to the dataclass (after `beats_bh`):

```python
    def beats_bh_risk_adjusted(self) -> bool:
        """Success criterion: higher Sharpe AND smaller drawdown than buy&hold.

        Drawdowns are negative fractions, so "smaller drawdown" is the larger
        (less negative) value: ``strat_mdd > bh_mdd``.
        """
        return self.strat_sharpe > self.bh_sharpe and self.strat_mdd > self.bh_mdd
```

- [ ] **Step 4: Compute them in `slice_metrics`**

In `backtest/oos_validation.py::slice_metrics`, after `asset_ret` is built and before the `return SliceMetrics(...)`, add:

```python
    bh_equity = (1.0 + asset_ret).cumprod()
    bh_sharpe = an.sharpe_ratio(asset_ret)
    bh_mdd = an.max_drawdown(bh_equity)
```

Then add to the `SliceMetrics(...)` constructor call:

```python
        bh_sharpe=bh_sharpe,
        bh_mdd=bh_mdd,
```

- [ ] **Step 5: Run the oos_validation tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_oos_validation.py -q`
Expected: PASS (existing 5 + new).

- [ ] **Step 6: Commit**

```bash
git add backtest/oos_validation.py tests/test_oos_validation.py
git commit -m "feat(oos): slice metrics report buy&hold Sharpe + maxDD for risk-adjusted gate"
```

---

### Task 5: settings.yaml entry + tuning harness (SPY grid → best Sharpe)

**Files:**
- Modify: `config/settings.yaml` (`risk` block)
- Create: `scripts/tune_reentry.py`

- [ ] **Step 1: Document the config knob**

In `config/settings.yaml`, in the `risk:` block, after the `lock_file:` line add:

```yaml
  peak_reentry_calm_bars: 0         # bars of calm regime (vol_rank<0.67) before the
                                    #   peak-DD halt RELEASES (0 = legacy; backtester-only).
                                    #   Tuned on SPY, validated on held-out ETFs.
```

- [ ] **Step 2: Write the tuning script**

Create `scripts/tune_reentry.py`:

```python
"""Tune halt re-entry on SPY ONLY (dev set). Grid over K x floor; pick best Sharpe.

Single process (R-4: cross-process numbers vary ~8%; intra-process is exact).
Writes the winner to tmp/reentry_tune.json. Does NOT touch the holdout ETFs.
"""
import json
import sys
import yaml

sys.path.insert(0, ".")
from backtest.oos_validation import build_backtester
from backtest.performance import PerformanceAnalyzer
from data.market_data import load_ohlcv

CFG = yaml.safe_load(open("config/settings.yaml"))
spy = load_ohlcv("SPY", start="2004-01-01", end="2024-12-31", timeframe="1Day")
rf = CFG.get("backtest", {}).get("risk_free_rate", 0.045)
an = PerformanceAnalyzer(risk_free_rate=rf)

results = []
for k in [0, 3, 5, 10]:
    for floor in [0.0, 0.25]:
        cfg = {**CFG, "risk": {**CFG.get("risk", {}),
                               "peak_reentry_calm_bars": k, "halt_floor_mult": floor}}
        bt = build_backtester(cfg)
        res = bt.run({"SPY": spy})
        sharpe = an.sharpe_ratio(res.returns)
        mdd = an.max_drawdown(res.equity_curve)
        ret = float(res.equity_curve.iloc[-1] / res.initial_capital - 1.0)
        halted = float((res.regime_history["risk_state"] == "halted").mean())
        results.append(dict(k=k, floor=floor, sharpe=sharpe, mdd=mdd, ret=ret, halted=halted))
        print(f"K={k:>2} floor={floor}: sharpe={sharpe:.2f} mdd={mdd:.1%} ret={ret:.1%} halted={halted:.1%}",
              flush=True)

bh_sharpe = an.sharpe_ratio(spy["close"].pct_change().reindex(
    res.returns.index).fillna(0.0))
best = max(results, key=lambda r: r["sharpe"])
out = dict(spy_buy_hold_sharpe=bh_sharpe, grid=results, best=best)
json.dump(out, open("tmp/reentry_tune.json", "w"), indent=2)
print(f"\nSPY buy&hold Sharpe (ref) = {bh_sharpe:.2f}")
print(f"BEST: K={best['k']} floor={best['floor']} sharpe={best['sharpe']:.2f}", flush=True)
print("DONE_TUNE", flush=True)
```

- [ ] **Step 3: Run the tuning grid (single process)**

Run: `cd ~/AIOS/projects/regime-trader && PYTHONPATH=. .venv/bin/python -u scripts/tune_reentry.py 2>&1 | grep -E "K=|buy&hold|BEST|DONE_TUNE"`
Expected: 8 grid rows + a BEST line. Record the winning `(K, floor)` and whether its Sharpe beats SPY buy&hold. (This is dev-set tuning — not the verdict.)

- [ ] **Step 4: Commit**

```bash
git add config/settings.yaml scripts/tune_reentry.py
git commit -m "feat(scripts): re-entry tuning grid on SPY dev set + settings knob"
```

---

### Task 6: Holdout validation (frozen combo on 5 ETFs) + results doc

**Files:**
- Create: `scripts/validate_reentry.py`
- Create: `docs/analysis/2026-06-04-reentry-validation.md` (date = run date)

- [ ] **Step 1: Write the validation script**

Create `scripts/validate_reentry.py` (reads the tuned combo, runs the 5 held-out ETFs, applies the risk-adjusted gate). Replace `BEST_K`/`BEST_FLOOR` with the winners from Task 5 before running:

```python
"""Validate the FROZEN re-entry combo on the 5 held-out ETFs (never tuned on).

Success per asset = Sharpe > buy&hold AND maxDD < buy&hold (strat_mdd > bh_mdd,
drawdowns negative). Single process. No re-tuning after seeing this output.
"""
import json
import sys
import yaml

sys.path.insert(0, ".")
from backtest.oos_validation import build_backtester, slice_metrics
from data.market_data import load_ohlcv

BEST_K = 5        # <-- set from scripts/tune_reentry.py BEST
BEST_FLOOR = 0.0  # <-- set from scripts/tune_reentry.py BEST

CFG = yaml.safe_load(open("config/settings.yaml"))
CFG = {**CFG, "risk": {**CFG.get("risk", {}),
                       "peak_reentry_calm_bars": BEST_K, "halt_floor_mult": BEST_FLOOR}}
HOLDOUT = ["QQQ", "IWM", "EFA", "EEM", "TLT"]

rows = []
for sym in HOLDOUT:
    ohlcv = load_ohlcv(sym, start="2004-01-01", end="2024-12-31", timeframe="1Day")
    bt = build_backtester(CFG)
    res = bt.run({sym: ohlcv})
    sm = slice_metrics(res, ohlcv["close"].astype(float), "full",
                       rf=CFG.get("backtest", {}).get("risk_free_rate", 0.045))
    passed = sm.beats_bh_risk_adjusted()
    rows.append(dict(symbol=sym, strat_sharpe=sm.strat_sharpe, bh_sharpe=sm.bh_sharpe,
                     strat_mdd=sm.strat_mdd, bh_mdd=sm.bh_mdd,
                     strat_ret=sm.strat_return, bh_ret=sm.buy_hold_return, passed=passed))
    print(f"{sym}: Sharpe {sm.strat_sharpe:.2f} vs bh {sm.bh_sharpe:.2f} | "
          f"maxDD {sm.strat_mdd:.1%} vs bh {sm.bh_mdd:.1%} | PASS={passed}", flush=True)

n_pass = sum(r["passed"] for r in rows)
json.dump(dict(k=BEST_K, floor=BEST_FLOOR, rows=rows, n_pass=n_pass),
          open("tmp/reentry_validation.json", "w"), indent=2)
print(f"\nHOLDOUT: {n_pass}/5 assets beat bh on Sharpe AND maxDD", flush=True)
print("DONE_VALIDATE", flush=True)
```

- [ ] **Step 2: Run validation (single process)**

Run: `cd ~/AIOS/projects/regime-trader && PYTHONPATH=. .venv/bin/python -u scripts/validate_reentry.py 2>&1 | grep -E "Sharpe|HOLDOUT|DONE_VALIDATE"`
Expected: 5 per-asset lines + an `n_pass/5` summary.

- [ ] **Step 3: Write the results doc**

Create `docs/analysis/2026-06-04-reentry-validation.md` with: the tuned combo (from `tmp/reentry_tune.json`), the SPY dev result, the 5-ETF holdout table (Sharpe/maxDD/return strat-vs-bh, PASS per asset), and an honest verdict against the success criterion (≥4/5 = success; otherwise report the failure and conclude the overlay lacks the skill — do NOT re-tune). Snapshot `tmp/reentry_tune.json` + `tmp/reentry_validation.json` into `docs/analysis/data/`.

- [ ] **Step 4: Run the full suite + commit**

```bash
cd ~/AIOS/projects/regime-trader
PYTHONPATH=. .venv/bin/python -m pytest -q   # expect all green
git add scripts/validate_reentry.py docs/analysis/2026-06-04-reentry-validation.md docs/analysis/data/
git commit -m "docs(oos): halt re-entry holdout validation on 5 ETFs (risk-adjusted gate)"
```

- [ ] **Step 5: advisor before declaring done** (per project lessons) — reconcile the verdict (edge vs exposure, holdout integrity, R-4 caveat) before claiming success.

---

## Notes for the executor

- **Determinism (R-4):** keep each grid/holdout run in ONE process; absolute numbers carry ~±8% cross-process. The gate compares strat vs bh **within the same run**, so the comparison is exact.
- **Do not re-tune after the holdout.** If the gate fails, that is the result — it means the regime overlay lacks risk-adjusted skill, and the honest next step is to reconsider the thesis, not to fish for a passing combo.
- **Live path untouched.** This entire change is the backtester sizing path (`update_drawdown_state` + `target_size_multiplier`). The live `CircuitBreaker` hard-halt is unchanged; aligning them is a separate, deferred decision.
