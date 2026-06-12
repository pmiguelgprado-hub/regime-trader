"""Evidence hash-chain: tamper-evidence for the gate's daily files (gap 6).

The forward gates are adjudicated from files that live on a laptop for 12
months — ``track_record.csv`` above all. A hash-chain is the cheapest honest
receipt: every day the recorder appends one JSONL row holding the SHA-256 of
each evidence file *plus* the previous row's chain hash. Editing any historical
file (or chain row) breaks every later link, so ``verify_chain`` pinpoints the
first broken day in O(n). Tamper-EVIDENCE, not tamper-proofing — the
auto-red-teamer's defence against his own future self, per the roadmap.

Chain row: ``{"date", "files": {name: sha256|ABSENT}, "prev", "chain"}`` where
``chain = sha256(prev + canonical-json(files) + date)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

_GENESIS = "0" * 64


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex of a file's bytes; ``"ABSENT"`` when the file is missing."""
    p = Path(path)
    if not p.exists():
        return "ABSENT"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _row_hash(prev: str, files: dict[str, str], date: str) -> str:
    canon = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev + canon + date).encode()).hexdigest()


def append_chain(chain_path: str | Path, date: str,
                 named_files: dict[str, str]) -> Optional[dict]:
    """Append one chain row hashing the named evidence files (idempotent on date).

    Args:
        chain_path: JSONL chain file.
        date: ISO date of the evidence row (dedup key).
        named_files: Logical name -> filesystem path of each evidence file.

    Returns:
        The appended row, or None when the date was already chained.
    """
    p = Path(chain_path)
    prev = _GENESIS
    if p.exists() and p.stat().st_size > 0:
        lines = p.read_text().strip().splitlines()
        last = json.loads(lines[-1])
        if str(last.get("date")) == str(date):
            return None                          # same-day re-run: no duplicate
        prev = last["chain"]
    files = {name: file_sha256(path) for name, path in named_files.items()}
    row = {"date": date, "files": files, "prev": prev,
           "chain": _row_hash(prev, files, date)}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def verify_chain(chain_path: str | Path) -> tuple[bool, Optional[int]]:
    """Walk the chain; report the first broken link.

    Returns:
        ``(True, None)`` if every row's hash and back-link check out (or the
        chain is empty/absent); ``(False, i)`` with the 0-based index of the
        first inconsistent row otherwise.
    """
    p = Path(chain_path)
    if not p.exists() or p.stat().st_size == 0:
        return True, None
    prev = _GENESIS
    for i, line in enumerate(p.read_text().strip().splitlines()):
        row = json.loads(line)
        if row.get("prev") != prev:
            return False, i
        if _row_hash(prev, row.get("files", {}), str(row.get("date"))) != row.get("chain"):
            return False, i
        prev = row["chain"]
    return True, None
