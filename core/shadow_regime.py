"""Shadow regime log: HMM champion vs Jump Model challenger (T1.1/T1.4).

Each day the live rebalance records what the deployed HMM and the shadow Jump
Model (:mod:`core.jump_model`) say about the current volatility regime, into
``logs/shadow_regime.csv``. The Jump Model never touches orders — this is pure
measurement to decide, over months, whether its lower flicker (and any downside
edge) justifies a NEW pre-registered book. ``monthly_report`` aggregates the log
into the agreement rate + per-engine flicker counts for the T1.4 report.

Agreement is defined on the **risk decision**, not raw labels: both engines map
the day to a vol_rank in [0, 1]; they "agree" when both are on the same side of
0.5 (risk-on vs risk-off), which is what actually drives gross exposure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

COLUMNS = ["date", "hmm_vol_rank", "jm_vol_rank", "agree", "next_ret"]


def make_row(date: str, hmm_vol_rank: float, jm_vol_rank: float,
             next_ret: Optional[float] = None) -> dict:
    """One shadow row; ``agree`` = both engines on the same side of risk-on/off (0.5)."""
    agree = (hmm_vol_rank < 0.5) == (jm_vol_rank < 0.5)
    return {"date": date, "hmm_vol_rank": round(float(hmm_vol_rank), 4),
            "jm_vol_rank": round(float(jm_vol_rank), 4), "agree": bool(agree),
            "next_ret": next_ret}


def append_row(path: str, row: dict) -> None:
    """Append a shadow row (idempotent on date, append-only)."""
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


def _switches(series: pd.Series) -> int:
    side = (series.astype(float) >= 0.5).astype(int)
    return int((side.to_numpy()[1:] != side.to_numpy()[:-1]).sum())


def monthly_report(path: str, month: str) -> dict:
    """Aggregate one month of the shadow log (agreement + per-engine flicker)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"month": month, "n_days": 0}
    df = pd.read_csv(p)
    df = df[df["date"].astype(str).str.startswith(month)]
    if df.empty:
        return {"month": month, "n_days": 0}
    return {
        "month": month,
        "n_days": int(len(df)),
        "agreement_rate": float(df["agree"].astype(bool).mean()),
        "hmm_switches": _switches(df["hmm_vol_rank"]),
        "jm_switches": _switches(df["jm_vol_rank"]),
        "hmm_mean_vol_rank": float(df["hmm_vol_rank"].astype(float).mean()),
        "jm_mean_vol_rank": float(df["jm_vol_rank"].astype(float).mean()),
    }


def report_markdown(rep: dict) -> str:
    """Render a monthly shadow report dict as markdown (T1.4)."""
    if rep.get("n_days", 0) == 0:
        return f"# Shadow regime report — {rep.get('month')}\n\nNo shadow data this month.\n"
    return (
        f"# Shadow regime report — {rep['month']}\n\n"
        f"- Days logged: **{rep['n_days']}**\n"
        f"- Risk-side agreement (HMM vs Jump Model): **{rep['agreement_rate']:.0%}**\n"
        f"- Risk-side flips — HMM: **{rep['hmm_switches']}**, "
        f"Jump Model: **{rep['jm_switches']}** "
        f"({'JM steadier' if rep['jm_switches'] <= rep['hmm_switches'] else 'HMM steadier'})\n"
        f"- Mean vol_rank — HMM {rep['hmm_mean_vol_rank']:.2f}, "
        f"JM {rep['jm_mean_vol_rank']:.2f}\n\n"
        f"Shadow only — the Jump Model drives no orders. Promotion would require a new "
        f"pre-registered book (roadmap §0).\n"
    )
