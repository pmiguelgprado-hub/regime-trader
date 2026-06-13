"""Global hypothesis/trials ledger — the multiple-testing control (T4.1).

Every research avenue costs statistical credibility whether it ships or not:
the Deflated Sharpe Ratio discounts a track record by how many things were
tried. Mass factor mining (LLM loops included — Chain-of-Alpha et al. ignore
this) explodes ``n_trials`` and silently collapses the DSR of the *entire*
program. This ledger makes the count auditable, and it **precedes all new
research** by roadmap invariant: alfa nueva = prereg nuevo + cargo aquí.

Append-only JSONL at ``research/registry.jsonl``. Two event kinds:

* ``registered`` — charges ``n_configs`` trials to a ``family`` at freeze time.
* ``verdict`` — later outcome (falsified / passed / abandoned…) referencing the
  trial id. Rows are never rewritten; corrections are new events.

``n_trials(family=...)`` feeds ``performance.deflated_sharpe_ratio`` per family
(complete within-family counting; cross-family effective-trials judgment stays
documented in each prereg, per roadmap).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

LEDGER_PATH = "research/registry.jsonl"


def _append(path: str | Path, row: dict) -> dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def load(path: str | Path = LEDGER_PATH) -> list[dict]:
    """All ledger events, oldest first ([] if the ledger does not exist yet)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(line) for line in p.read_text().strip().splitlines()]


def register(path: str | Path = LEDGER_PATH, *, family: str, hypothesis: str,
             n_configs: int, prereg: Optional[str] = None,
             basis: str = "prereg") -> dict:
    """Charge a new hypothesis (and its config count) to the ledger.

    Args:
        path: Ledger JSONL.
        family: Trial family (momentum / sentiment / quality / regime / …).
        hypothesis: One-line statement of what is being tried.
        n_configs: Number of configurations this freeze charges.
        prereg: Path to the frozen pre-registration doc, when one exists.
        basis: ``prereg`` (exact, from a frozen doc) or ``estimate``
            (historical backfill — honest but approximate).

    Returns:
        The appended ``registered`` event (with generated ``id``).
    """
    row = {
        "event": "registered",
        "id": f"{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:6]}",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "family": family,
        "hypothesis": hypothesis,
        "n_configs": int(n_configs),
        "prereg": prereg,
        "basis": basis,
    }
    return _append(path, row)


def record_verdict(path: str | Path, trial_id: str, verdict: str,
                   note: Optional[str] = None) -> dict:
    """Append the outcome of a registered trial (never edits the original row).

    Args:
        path: Ledger JSONL.
        trial_id: ``id`` of the ``registered`` event.
        verdict: Outcome label (falsified / passed / abandoned / …).
        note: Optional evidence pointer or one-liner.

    Raises:
        KeyError: If ``trial_id`` was never registered.
    """
    known = {r["id"] for r in load(path) if r.get("event") == "registered"}
    if trial_id not in known:
        raise KeyError(f"trial id not in ledger: {trial_id}")
    row = {
        "event": "verdict",
        "id": trial_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "verdict": verdict,
        "note": note,
    }
    return _append(path, row)


def n_trials(path: str | Path = LEDGER_PATH, family: Optional[str] = None) -> int:
    """Total configs charged (optionally within one family) — the DSR input."""
    return sum(int(r.get("n_configs", 0)) for r in load(path)
               if r.get("event") == "registered"
               and (family is None or r.get("family") == family))


def status(path: str | Path = LEDGER_PATH) -> list[dict]:
    """One merged record per trial: registration fields + latest verdict.

    Trials without a verdict yet carry ``verdict: "open"``.
    """
    merged: dict[str, dict] = {}
    for r in load(path):
        if r.get("event") == "registered":
            merged[r["id"]] = {**r, "verdict": "open"}
        elif r.get("event") == "verdict" and r.get("id") in merged:
            merged[r["id"]]["verdict"] = r.get("verdict")
            if r.get("note"):
                merged[r["id"]]["verdict_note"] = r["note"]
    return list(merged.values())
