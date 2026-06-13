"""Tests for the global hypothesis/trials ledger (T4.1).

Mass factor mining explodes n_trials and silently collapses the DSR of the
whole program — the ledger is THE control. Append-only JSONL: registrations
charge configs to a family, verdicts arrive as later events referencing the
trial id (no row is ever rewritten). ``n_trials`` is what
``performance.deflated_sharpe_ratio`` should be fed, auditable.
"""

from __future__ import annotations

import json

import pytest

from core import research_ledger as rl


def test_register_charges_trials(tmp_path) -> None:
    p = str(tmp_path / "registry.jsonl")
    row = rl.register(p, family="momentum", hypothesis="12-1 cross-sectional",
                      n_configs=4, prereg="docs/analysis/x-prereg.md")
    assert row["id"] and row["event"] == "registered"
    assert rl.n_trials(p) == 4
    assert rl.n_trials(p, family="momentum") == 4
    assert rl.n_trials(p, family="sentiment") == 0


def test_trials_accumulate_across_registrations(tmp_path) -> None:
    p = str(tmp_path / "registry.jsonl")
    rl.register(p, family="momentum", hypothesis="a", n_configs=3)
    rl.register(p, family="quality", hypothesis="b", n_configs=2)
    rl.register(p, family="momentum", hypothesis="c", n_configs=5)
    assert rl.n_trials(p) == 10
    assert rl.n_trials(p, family="momentum") == 8


def test_verdict_appends_never_rewrites(tmp_path) -> None:
    p = str(tmp_path / "registry.jsonl")
    row = rl.register(p, family="regime", hypothesis="jump model", n_configs=6)
    rl.record_verdict(p, row["id"], "falsified", note="lost to HMM on flicker")
    lines = open(p).read().strip().splitlines()
    assert len(lines) == 2                       # append-only: 2 events
    first = json.loads(lines[0])
    assert first["event"] == "registered"        # original row untouched
    last = json.loads(lines[1])
    assert last["event"] == "verdict" and last["id"] == row["id"]
    assert last["verdict"] == "falsified"
    # verdicts charge nothing
    assert rl.n_trials(p) == 6


def test_verdict_unknown_id_raises(tmp_path) -> None:
    p = str(tmp_path / "registry.jsonl")
    rl.register(p, family="x", hypothesis="y", n_configs=1)
    with pytest.raises(KeyError):
        rl.record_verdict(p, "nope-000", "falsified")


def test_status_merges_latest_verdict(tmp_path) -> None:
    p = str(tmp_path / "registry.jsonl")
    a = rl.register(p, family="x", hypothesis="alpha", n_configs=1)
    b = rl.register(p, family="x", hypothesis="beta", n_configs=1)
    rl.record_verdict(p, a["id"], "falsified")
    status = {r["id"]: r for r in rl.status(p)}
    assert status[a["id"]]["verdict"] == "falsified"
    assert status[b["id"]]["verdict"] == "open"


def test_n_trials_missing_file_is_zero(tmp_path) -> None:
    assert rl.n_trials(str(tmp_path / "none.jsonl")) == 0
