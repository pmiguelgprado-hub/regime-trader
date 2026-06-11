"""Regression: attributes used on `client` in main.py must exist on AlpacaClient.

2026-06-09/10: run_rebalance --execute called client.get_orders(), which does not
exist on AlpacaClient (real method: get_order_history). The nightly launchd run
computed targets, then crashed with AttributeError before submitting a single
order — three sessions with zero trades. Dry-run never reaches the call, so the
test suite stayed green. This test walks main.py's AST so any future dead-wired
client call fails in CI instead of at 22:30.
"""
import ast
from pathlib import Path

from broker.alpaca_client import AlpacaClient

MAIN = Path(__file__).resolve().parent.parent / "main.py"


def _client_attrs_used() -> set[str]:
    tree = ast.parse(MAIN.read_text())
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "client"
    }


def test_main_client_attrs_exist_on_alpaca_client() -> None:
    used = _client_attrs_used()
    assert used, "expected main.py to reference client.<attr> somewhere"
    missing = sorted(a for a in used if not hasattr(AlpacaClient, a))
    assert not missing, (
        f"main.py uses AlpacaClient attributes that don't exist: {missing}"
    )
