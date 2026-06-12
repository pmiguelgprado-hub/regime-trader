"""Tests for the evidence hash-chain (gap 6 — tamper-evidence for gate files).

The 12-month gates rest on files a future self (or a bug) could quietly edit:
track_record.csv above all. Each day the recorder appends one chain row —
SHA-256 of the evidence files plus the previous row's chain hash — so any
retroactive edit breaks every subsequent link and is detectable in O(n).
This is tamper-EVIDENCE, not tamper-proofing: the auto-red-teamer's receipt.
"""

from __future__ import annotations

import json

from core import evidence as ev


def _write(tmp_path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_chain_grows_and_verifies(tmp_path) -> None:
    chain = str(tmp_path / "chain.jsonl")
    csv = _write(tmp_path, "track.csv", "day1\n")
    ev.append_chain(chain, "2026-06-12", {"track": csv})
    csv2 = _write(tmp_path, "track.csv", "day1\nday2\n")  # legitimate append
    ev.append_chain(chain, "2026-06-13", {"track": csv2})
    ok, bad = ev.verify_chain(chain)
    assert ok is True and bad is None
    rows = [json.loads(l) for l in open(chain)]
    assert len(rows) == 2
    assert rows[1]["prev"] == rows[0]["chain"]     # linked


def test_chain_append_idempotent_per_date(tmp_path) -> None:
    chain = str(tmp_path / "chain.jsonl")
    csv = _write(tmp_path, "track.csv", "day1\n")
    ev.append_chain(chain, "2026-06-12", {"track": csv})
    ev.append_chain(chain, "2026-06-12", {"track": csv})
    assert len(open(chain).readlines()) == 1


def test_tampered_chain_row_detected(tmp_path) -> None:
    chain = str(tmp_path / "chain.jsonl")
    csv = _write(tmp_path, "track.csv", "day1\n")
    ev.append_chain(chain, "2026-06-12", {"track": csv})
    ev.append_chain(chain, "2026-06-13", {"track": csv})
    rows = open(chain).readlines()
    doctored = json.loads(rows[0])
    doctored["files"]["track"] = "0" * 64           # rewrite history
    rows[0] = json.dumps(doctored) + "\n"
    open(chain, "w").writelines(rows)
    ok, bad = ev.verify_chain(chain)
    assert ok is False and bad == 0                 # first broken link reported


def test_broken_link_detected(tmp_path) -> None:
    chain = str(tmp_path / "chain.jsonl")
    csv = _write(tmp_path, "track.csv", "day1\n")
    ev.append_chain(chain, "2026-06-12", {"track": csv})
    ev.append_chain(chain, "2026-06-13", {"track": csv})
    rows = open(chain).readlines()
    doctored = json.loads(rows[1])
    doctored["prev"] = "f" * 64                     # cut the link
    rows[1] = json.dumps(doctored) + "\n"
    open(chain, "w").writelines(rows)
    ok, bad = ev.verify_chain(chain)
    assert ok is False and bad == 1


def test_missing_evidence_file_hashes_as_absent(tmp_path) -> None:
    chain = str(tmp_path / "chain.jsonl")
    ev.append_chain(chain, "2026-06-12", {"gone": str(tmp_path / "nope.csv")})
    row = json.loads(open(chain).readline())
    assert row["files"]["gone"] == "ABSENT"
    assert ev.verify_chain(chain) == (True, None)
