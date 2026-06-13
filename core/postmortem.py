"""Monthly postmortem report (T4.5) + live regime attribution (gap 4).

Cheap, zero gate-risk: aggregates the month's evidence — per-book NAV stats,
gate countdown, shadow-regime agreement, ledger trial counts, alert counts, and
a breakdown of book returns conditioned on the live regime — into a markdown
narrative at ``docs/analysis/YYYY-MM-postmortem.md``. Read-only over the track
record and logs; touches no signal, knob, or order path.

Pure functions (inputs injected) so the report is unit-tested; the live glue
(loading the CSVs, counting alert lines) is thin and lives in ``main``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def book_stats(track_df: pd.DataFrame, col: str) -> dict:
    """Return / Sharpe / maxDD over a NAV column ({n_obs:0} if absent/empty)."""
    if col not in track_df.columns:
        return {"n_obs": 0}
    nav = pd.to_numeric(track_df[col], errors="coerce").dropna()
    if len(nav) < 2:
        return {"n_obs": int(len(nav))}
    rets = nav.pct_change().dropna()
    peak = nav.cummax()
    max_dd = float((nav / peak - 1.0).min())
    sd = rets.std(ddof=1)
    sharpe = float(rets.mean() / sd * (252 ** 0.5)) if sd > 0 else 0.0
    return {
        "n_obs": int(len(nav)),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "max_drawdown": max_dd,
        "sharpe_ann": sharpe,
    }


def regime_attribution(nav: pd.Series, regimes: list[str]) -> dict[str, dict]:
    """Mean daily book return grouped by the live regime label (gap 4).

    Args:
        nav: Book NAV level series.
        regimes: Regime label per row of ``nav`` (same length; the trailing return
            of each day is attributed to that day's regime).

    Returns:
        ``{regime: {mean_ret, n}}`` (empty if lengths mismatch or too short).
    """
    nav = pd.to_numeric(pd.Series(nav), errors="coerce")
    rets = nav.pct_change().dropna()
    labels = list(regimes)[1:len(rets) + 1]            # align to returns (drop first)
    if len(labels) != len(rets) or rets.empty:
        return {}
    by: dict[str, list[float]] = {}
    for lab, r in zip(labels, rets):
        by.setdefault(str(lab), []).append(float(r))
    return {lab: {"mean_ret": float(sum(v) / len(v)), "n": len(v)} for lab, v in by.items()}


def monthly_postmortem_markdown(month: str, track_df: pd.DataFrame,
                                ledger_counts: dict, alert_counts: dict,
                                shadow: dict,
                                regimes: Optional[list[str]] = None) -> str:
    """Render the monthly postmortem as markdown (T4.5)."""
    books = [("baseline", "book_nav"), ("challenger", "challenger_nav"),
             ("quality", "quality_nav")]
    lines = [f"# Postmortem — {month}", ""]

    lines.append("## Book performance (forward paper)")
    any_book = False
    for name, col in books:
        s = book_stats(track_df, col)
        if s.get("n_obs", 0) >= 2:
            any_book = True
            lines.append(f"- **{name}**: ret {s['total_return']:+.2%}, "
                         f"Sharpe(ann) {s['sharpe_ann']:.2f}, maxDD {s['max_drawdown']:.2%} "
                         f"({s['n_obs']} obs)")
    if not any_book:
        lines.append("- (insufficient track-record data this month)")

    lines += ["", "## Gates (n_trials charged to the ledger)"]
    if ledger_counts:
        for fam, n in sorted(ledger_counts.items()):
            lines.append(f"- {fam}: {n} trials")
    else:
        lines.append("- (ledger empty)")

    lines += ["", "## Shadow regime (HMM vs Jump Model)"]
    if shadow.get("n_days", 0):
        lines.append(f"- agreement {shadow.get('agreement_rate', 0):.0%} over "
                     f"{shadow['n_days']} days; flips HMM {shadow.get('hmm_switches', 0)} / "
                     f"JM {shadow.get('jm_switches', 0)}")
    else:
        lines.append("- (no shadow data this month)")

    if regimes is not None and "book_nav" in track_df.columns:
        attr = regime_attribution(track_df["book_nav"], regimes)
        if attr:
            lines += ["", "## Live regime attribution (baseline book)"]
            for lab, v in sorted(attr.items()):
                lines.append(f"- {lab}: mean daily ret {v['mean_ret']:+.3%} (n={v['n']})")

    lines += ["", "## Alerts this month"]
    if alert_counts:
        for k, n in sorted(alert_counts.items()):
            lines.append(f"- {k}: {n}")
    else:
        lines.append("- none")

    lines += ["", "_Read-only postmortem — no signal, knob, or order touched._"]
    return "\n".join(lines) + "\n"
