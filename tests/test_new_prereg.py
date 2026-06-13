"""Tests for the prereg generator (T4.2): doc emitted + ledger charged atomically."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "new_prereg", Path(__file__).resolve().parent.parent / "scripts" / "new_prereg.py")
new_prereg = importlib.util.module_from_spec(_SPEC)
sys.modules["new_prereg"] = new_prereg
_SPEC.loader.exec_module(new_prereg)

from core import research_ledger as rl


def test_create_emits_doc_and_charges_ledger(tmp_path) -> None:
    ledger = str(tmp_path / "registry.jsonl")
    out = new_prereg.create("quality-edgar", "quality",
                            "EDGAR-PIT quality sleeve beats EW S&P500 net",
                            n_configs=2, universe="S&P 500", cadence="monthly",
                            ledger_path=ledger, out_dir=str(tmp_path / "docs"))
    assert out.exists()
    body = out.read_text()
    assert "quality-edgar" in body and "n_trials = 2" in body
    assert rl.n_trials(ledger, family="quality") == 2
    # the generated doc references its own ledger trial id
    trial_id = rl.load(ledger)[0]["id"]
    assert trial_id in body


def test_create_refuses_to_overwrite(tmp_path) -> None:
    ledger = str(tmp_path / "registry.jsonl")
    kw = dict(n_configs=1, universe="x", cadence="y",
              ledger_path=ledger, out_dir=str(tmp_path / "docs"))
    new_prereg.create("dup", "momentum", "h", **kw)
    with pytest.raises(SystemExit):
        new_prereg.create("dup", "momentum", "h", **kw)
