"""Pin-champion dual-log: compare the pinned champion against a shadow refit (T0.4).

Operative amendment 2026-06-12 (docs/analysis/2026-06-12-pin-champion-amendment.md):
the live HMM is **pinned** to the registry champion — the old age-based weekly refit
is suppressed, because refitting on data whose end-date is "now" produced a different
model every week (and multithreaded BLAS made even same-data fits diverge, R-4).

The gate measures the book's NAV, not the model's internals, so pinning is an
*operational* amendment, not a strategy change. To demonstrate (or refute) that
pinned ≈ refit, while the old rule would have refit we instead fit a throwaway
*shadow* engine on fresh data and append one comparison row per day here. Two weeks
of rows adjudicate: high agreement → keep the pin and drop the dual-log; sustained
disagreement → drift trigger territory (T3.3), human decides.

The shadow engine is never saved, never drives orders, and never touches the
registry. Pure measurement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.hmm_engine import HMMEngine

COLUMNS = ["date", "champion_hash", "shadow_hash", "champion_regime", "shadow_regime",
           "champion_conf", "shadow_conf", "agree"]


def compare_engines(champion: HMMEngine, shadow: HMMEngine,
                    features: pd.DataFrame) -> dict:
    """One comparison row: what each engine says about the latest bar.

    Args:
        champion: The pinned engine (drives orders).
        shadow: Freshly refit engine (measurement only).
        features: Feature frame whose last row is "today".

    Returns:
        Row dict (see :data:`COLUMNS`); ``agree`` is True when both engines put
        today in the same *labelled* regime (label space, not raw state index —
        state indices are permutation-arbitrary across fits).
    """
    c_last = champion.predict_regime_filtered(features)[-1]
    s_last = shadow.predict_regime_filtered(features)[-1]
    c_label = champion.labels[c_last.state_id].value
    s_label = shadow.labels[s_last.state_id].value
    return {
        "date": str(features.index[-1])[:10],
        "champion_hash": champion.transition_hash(),
        "shadow_hash": shadow.transition_hash(),
        "champion_regime": c_label,
        "shadow_regime": s_label,
        "champion_conf": round(float(c_last.probability), 4),
        "shadow_conf": round(float(s_last.probability), 4),
        "agree": c_label == s_label,
    }


def append_row(path: str, row: dict) -> None:
    """Append one row to the dual-log CSV (idempotent on date, append-only)."""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)
        if not df.empty and str(df.iloc[-1]["date"]) == str(row["date"]):
            return
    else:
        df = pd.DataFrame(columns=COLUMNS)
    out = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
