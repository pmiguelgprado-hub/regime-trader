# evidence/ — git-anchored tamper-evidence (gap 6)

`chain.jsonl` is an append-only hash-chain (`core/evidence.py`) written daily by
`main.py --record-track`. Each row holds the SHA-256 of every gate-evidence file
(`track_record.csv`, both book snapshots, the champion sha) plus the previous
row's chain hash, so any retroactive edit to a historical evidence file breaks
every later link.

`track_record.csv` and `logs/` are gitignored (live, regenerated). This directory
is **committed** so the chain is anchored in git history: rewriting past evidence
would require rewriting both the local chain and the remote git history, and the
mismatch is detectable.

Audit: `python main.py --verify-evidence` (exit 1 on the first broken row).
