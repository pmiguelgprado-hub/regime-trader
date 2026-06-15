# Runbook — Activate a 2nd paper account for new sleeves (T5.4)

**Status: rails built, activation GATED on Pablo (needs a real Alpaca paper account).**

## Why

The frozen baseline owns paper account 1; the challenger and quality sleeves run
there as DRY-RUN + synthetic NAV because two books executing on one account would
fight over positions. To run a new sleeve *executing* in parallel — clean
attribution, real fills — it needs its own account. Trigger: activating the first
T2 sleeve for real execution (quality is first in line).

## Rails already in place

- `main.load_credentials(account="sleeve")` reads `ALPACA_API_KEY_SLEEVE` /
  `ALPACA_SECRET_KEY_SLEEVE` / `ALPACA_PAPER_SLEEVE`, falling back to the main
  account's vars when absent (so everything works unchanged until you create it).
- Each sleeve already writes its OWN snapshot + `*_nav` track column + plist; only
  the broker account is shared today.

## Activation steps (Pablo)

1. Create a second Alpaca **paper** account; generate its API key/secret.
2. Add to `.env` (never commit):
   ```
   ALPACA_API_KEY_SLEEVE=...
   ALPACA_SECRET_KEY_SLEEVE=...
   ALPACA_PAPER_SLEEVE=true
   ```
3. Point the chosen sleeve's run at the sleeve account — e.g. in the sleeve's
   `run_rebalance` path, call `load_credentials("sleeve")` instead of
   `load_credentials()`, and enable `--execute` (the quality sleeve currently
   force-dry-runs; lift that guard only for the sleeve account).
4. Verify in isolation: confirm account 1 (baseline/challenger/hedge) positions are
   untouched and the sleeve trades only its own targets in account 2.
5. The gate evidence keeps flowing: the sleeve's `*_nav` column now reflects REAL
   fills in account 2 instead of synthetic mark-to-market — note the switch date in
   its prereg (the series changes basis at that point).

## Guardrail

Until step 4 verification passes, the new account stays observe/paper. Never enable
execution on account 1 for a non-frozen sleeve (it would fight the baseline). The
frozen books' attribution must stay byte-clean in account 1.
