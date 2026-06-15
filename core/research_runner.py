"""Research-factory RAILS (T4.4 — rails only; the autonomous loop is GATED).

The ambition is to industrialize the hypothesis→falsification cycle. The central
risk the LLM-alpha-mining papers (Chain-of-Alpha, AlphaAgent) ignore: mass factor
mining explodes n_trials and silently collapses the program's DSR. So the RAILS
come first and the autonomous nightly loop comes later (Pablo's explicit decision:
manual for weeks, plist after).

This module is the rails — invoked manually for now:

* **Blocklist** — already-falsified ideas (R1 HMM return timer, cross-asset
  rotation via B, regime-conditional shorts, direct hmm_prob deploy, paid-data
  order-flow/VPIN, HF pairs) are refused before any evaluation.
* **Weekly budget** — at most N hypotheses/week (trial discipline > throughput).
* **Evaluation harness** — CPCV + DSR + PBO on a candidate's returns, charging the
  ledger, producing a verdict for HUMAN adjudication.

What is deliberately NOT here: autonomous code generation, a sandbox that writes
trader state, and the nightly scheduler. Those are the gated step. Pure + tested.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

VAULT_DIR = "research/vault"
RUNLOG = "research/runlog.jsonl"


def default_blocklist() -> list[str]:
    """Ideas falsified out-of-sample — never re-explore (checked before each run)."""
    return [
        "r1",                      # HMM as a return timer (falsified OOS)
        "hmm return timer",
        "rotation via b",          # cross-asset rotation avenue B (falsified)
        "cross-asset rotation",
        "regime-conditional short", "regime conditional short", "shorts by regime",
        "hmm_prob deploy",         # direct deploy of the fuzzy posterior (lost to crash_only)
        "order flow", "vpin",      # paid-data microstructure (no free source)
        "hf pairs", "pairs hf",    # retail latency can't compete
    ]


def is_blocked(hypothesis: str, blocklist: list[str]) -> bool:
    """Whether the hypothesis text matches any blocklisted idea (case-insensitive)."""
    h = hypothesis.lower()
    return any(term.lower() in h for term in blocklist)


def _week_key(d: str) -> str:
    iso = datetime.fromisoformat(str(d)[:10]).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def record_run(runlog: str, run_id: str, day: str) -> None:
    """Append a run record (for the weekly budget accounting)."""
    p = Path(runlog)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"id": run_id, "date": str(day)[:10],
                            "week": _week_key(day)}) + "\n")


def weekly_budget_ok(runlog: str, max_per_week: int, today: Optional[str] = None) -> bool:
    """Whether this week's run count is below ``max_per_week``."""
    today = today or datetime.now(timezone.utc).date().isoformat()
    wk = _week_key(today)
    p = Path(runlog)
    if not p.exists() or p.stat().st_size == 0:
        return True
    n = sum(1 for line in p.read_text().splitlines()
            if line.strip() and json.loads(line).get("week") == wk)
    return n < max_per_week


def run_guard(hypothesis: str, blocklist: list[str], runlog: str,
              max_per_week: int, today: Optional[str] = None) -> "tuple[bool, str]":
    """Combined gate: blocklist THEN weekly budget. Returns ``(allowed, reason)``."""
    if is_blocked(hypothesis, blocklist):
        return False, "refused: matches the falsified-ideas blocklist"
    if not weekly_budget_ok(runlog, max_per_week, today):
        return False, "refused: weekly hypothesis budget exhausted"
    return True, "ok"


def evaluate_candidate(returns, family: str, n_configs: int,
                       ledger_path: str, dsr_min: float = 0.5,
                       pbo_max: float = 0.5) -> dict:
    """Evaluate a candidate's returns with CPCV + DSR (+ ledger charge) -> verdict.

    Charges ``n_configs`` to the ledger family FIRST (multiple-testing honesty),
    then judges: DSR (deflated by the family's running n_trials) above ``dsr_min``
    and the CPCV distribution not dominated by negative paths.

    Returns:
        ``{family, n_obs, sharpe_ann, dsr, n_trials, cpcv, verdict}``.
    """
    from backtest.cpcv import cpcv_summary
    from backtest.performance import deflated_sharpe_ratio
    from core import research_ledger as rl

    rl.register(ledger_path, family=family, hypothesis="candidate eval",
                n_configs=n_configs, basis="prereg")
    n_trials = rl.n_trials(ledger_path, family=family)

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    sd = r.std(ddof=1) if n > 1 else 0.0
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    dsr = deflated_sharpe_ratio(sr, n, skew=0.0, kurt=3.0, n_trials=max(1, n_trials)) if n > 1 else 0.0
    cpcv = cpcv_summary(r)
    verdict = "pass" if (dsr >= dsr_min and cpcv.get("prob_negative", 1.0) <= pbo_max) else "fail"
    return {
        "family": family, "n_obs": n,
        "sharpe_ann": sr * (252 ** 0.5),
        "dsr": dsr, "n_trials": n_trials, "cpcv": cpcv, "verdict": verdict,
    }


def write_verdict(run_id: str, result: dict, vault_dir: str = VAULT_DIR) -> Path:
    """Write a candidate verdict to ``research/vault/<id>.md`` for human review."""
    out = Path(vault_dir) / f"{run_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    cpcv = result.get("cpcv", {})
    out.write_text(
        f"# Candidate {run_id}\n\n"
        f"- family: {result.get('family')}\n"
        f"- verdict: **{result.get('verdict', '?').upper()}**\n"
        f"- DSR: {result.get('dsr', 0):.3f} (n_trials {result.get('n_trials', '?')})\n"
        f"- Sharpe (ann): {result.get('sharpe_ann', 0):.2f}\n"
        f"- CPCV mean Sharpe: {cpcv.get('mean_sharpe', 0):.3f}, "
        f"prob_negative {cpcv.get('prob_negative', 0):.2f}\n\n"
        f"_Rails verdict — a HUMAN adjudicates promotion; a pass is necessary, not "
        f"sufficient. Promotion = new prereg + forward book (T2 pattern)._\n"
    )
    return out
