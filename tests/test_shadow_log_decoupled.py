"""Regression: shadow-signal logging must stay OUT of the executor paths.

2026-06-15: the Jump Model fit, BOCPD, and the VIX/FRED macro fetch used to run
INLINE in ``run_rebalance`` (daily 22:30 --execute) and the champion-vs-refit HMM
fit ran inline in ``run_once`` — all BEFORE order submission. A slow or hung
yfinance/FRED call (no exception for the try/except to catch) could therefore sit
upstream of paper-order submission. They were moved to the standalone, read-only
``run_shadow_log`` (--shadow-log). Dry-run tests never exercise the live broker
paths (``# pragma: no cover``), so this AST guard is what keeps the feed fetches
from creeping back upstream of trading.
"""
import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent / "main.py"

# Symbols that pull external feeds or fit shadow models — must never live in a
# function that submits orders.
SHADOW_SYMBOLS = {
    "JumpModel",
    "changepoint_score",
    "fetch_fred_series",
    "fetch_term_structure",
    "compare_engines",
}
# Order-submission surface — must never appear in the read-only shadow logger.
ORDER_SYMBOLS = {
    "OrderExecutor",
    "plan_rebalance_orders",
    "targets_to_orders",
    "submit_order",
}


def _funcs() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(MAIN.read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _symbols(fn: ast.FunctionDef) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            out.update(a.name for a in node.names)
    return out


def test_executor_paths_have_no_shadow_feed_fetches() -> None:
    funcs = _funcs()
    for name in ("run_once", "run_rebalance"):
        assert name in funcs, f"{name} not found in main.py"
        leaked = SHADOW_SYMBOLS & _symbols(funcs[name])
        assert not leaked, (
            f"{name} references shadow feed/fit symbols {sorted(leaked)} — these run "
            "upstream of order submission and must live in run_shadow_log instead"
        )


def test_shadow_log_exists_and_owns_the_shadow_signals() -> None:
    funcs = _funcs()
    assert "run_shadow_log" in funcs, "run_shadow_log was removed"
    have = _symbols(funcs["run_shadow_log"])
    missing = SHADOW_SYMBOLS - have
    assert not missing, f"run_shadow_log no longer logs {sorted(missing)}"


def test_shadow_log_is_read_only() -> None:
    funcs = _funcs()
    used = ORDER_SYMBOLS & _symbols(funcs["run_shadow_log"])
    assert not used, (
        f"run_shadow_log references order-submission symbols {sorted(used)} — it must "
        "stay read-only (history + free feeds only, never trades)"
    )
