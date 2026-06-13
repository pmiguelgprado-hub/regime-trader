#!/usr/bin/env python
"""Emergency flatten PREVIEW (gap 2 — kill-switch drill, dry-run only).

Lists current paper positions and the market sell orders that WOULD flatten the
book — submitting NOTHING. Use during a quarterly kill-switch drill, or as the
first step of a real manual flatten (then submit via the Alpaca console or the
gated --risk-check ladder; this tool deliberately has no order-submission path).

    python scripts/flatten_preview.py

Reuses core.risk_monitor.plan_flatten_orders so the preview matches exactly what
the intraday flatten tier would do.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from broker.alpaca_client import AlpacaClient, AlpacaConfig
    from core.risk_monitor import plan_flatten_orders
    from main import load_credentials

    creds = load_credentials()
    paper = str(creds.get("paper", "true")).lower() != "false"
    client = AlpacaClient(AlpacaConfig(creds["api_key"], creds["secret_key"], paper=paper))
    client.connect()
    acct = client.get_account()
    held = {p["symbol"]: int(float(p["qty"])) for p in client.get_positions()}
    orders = plan_flatten_orders(held)

    print(f"=== FLATTEN PREVIEW ({'PAPER' if paper else 'LIVE'}) — NO ORDERS SUBMITTED ===")
    print(f"account equity: ${float(acct['equity']):,.2f}")
    print(f"positions held: {len(held)}")
    for o in orders:
        print(f"  {o.get('side', 'sell').upper():4} {o['symbol']:6} x{o['qty']}")
    print(f"\n{len(orders)} sell order(s) would flatten the book.")
    print("To execute: Alpaca console, or the gated --risk-check flatten tier "
          "(requires --execute AND risk_monitor.allow_orders=true).")


if __name__ == "__main__":
    main()
