"""Tests for the research-factory RAILS (T4.4 — rails only, loop is GATED).

The hard guardrails the LLM-alpha-mining literature ignores: a blocklist of
already-falsified ideas checked before any run, a weekly hypothesis budget
(discipline of trials > throughput), and an evaluation harness that charges the
ledger and runs CPCV+DSR+PBO. The autonomous nightly loop + code generation are
deliberately NOT built (Pablo's gated decision: manual first). Pure + tested.
"""

from __future__ import annotations

import numpy as np

from core import research_runner as rr


def test_blocklist_blocks_falsified_idea():
    bl = rr.default_blocklist()
    assert rr.is_blocked("hmm return timer (R1)", bl)
    assert rr.is_blocked("cross-asset rotation via B", bl)
    assert rr.is_blocked("regime-conditional shorts", bl)
    assert not rr.is_blocked("edgar quality value sleeve", bl)


def test_blocklist_matches_on_keywords():
    bl = ["order flow"]
    assert rr.is_blocked("institutional ORDER FLOW signal", bl)
    assert not rr.is_blocked("momentum signal", bl)


def test_weekly_budget_gate(tmp_path):
    log = str(tmp_path / "runlog.jsonl")
    # under budget -> allowed (Mon 2026-06-15 .. Sun = ISO week 25)
    assert rr.weekly_budget_ok(log, max_per_week=3, today="2026-06-15") is True
    for i in range(3):
        rr.record_run(log, f"id{i}", "2026-06-15")
    # at budget, same week -> blocked
    assert rr.weekly_budget_ok(log, max_per_week=3, today="2026-06-16") is False
    # next week -> allowed again
    assert rr.weekly_budget_ok(log, max_per_week=3, today="2026-06-23") is True


def test_evaluate_candidate_charges_ledger_and_verdicts(tmp_path):
    ledger = str(tmp_path / "registry.jsonl")
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 300)
    res = rr.evaluate_candidate(rets, family="momentum", n_configs=2,
                                ledger_path=ledger)
    assert "dsr" in res and "cpcv" in res and "verdict" in res
    assert res["verdict"] in ("pass", "fail")
    from core import research_ledger as rl
    assert rl.n_trials(ledger, family="momentum") == 2     # charged


def test_evaluate_candidate_fail_on_negative_series(tmp_path):
    ledger = str(tmp_path / "registry.jsonl")
    rng = np.random.default_rng(1)
    rets = rng.normal(-0.002, 0.01, 300)                   # losing strategy
    res = rr.evaluate_candidate(rets, family="momentum", n_configs=1,
                                ledger_path=ledger)
    assert res["verdict"] == "fail"


def test_write_verdict_creates_vault_doc(tmp_path):
    out = rr.write_verdict("20260614-abc", {"family": "momentum", "verdict": "fail",
                                            "dsr": 0.3, "cpcv": {"mean_sharpe": 0.1}},
                           vault_dir=str(tmp_path / "vault"))
    assert out.exists()
    assert "verdict" in out.read_text().lower()


def test_run_guard_blocks_then_allows(tmp_path):
    log = str(tmp_path / "runlog.jsonl")
    bl = ["r1 timer"]
    ok, reason = rr.run_guard("a fresh momentum idea", bl, log, max_per_week=2,
                              today="2026-06-14")
    assert ok is True
    blocked, reason = rr.run_guard("the R1 timer again", bl, log, max_per_week=2,
                                   today="2026-06-14")
    assert blocked is False and "blocklist" in reason.lower()
