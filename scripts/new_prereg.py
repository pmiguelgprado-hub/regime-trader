#!/usr/bin/env python
"""Pre-registration generator + automatic ledger charge (T4.2).

Emits a standardized frozen-prereg skeleton (section structure mined from the
existing preregs: challenger 2026-06-05, hedge 2026-06-12) and registers the
hypothesis in the trials ledger (T4.1) in the same breath — so it is impossible
to start a forward gate without charging its configs to the DSR budget.

Usage:
    python scripts/new_prereg.py --slug quality-edgar --family quality \\
        --hypothesis "EDGAR-PIT quality+momentum sleeve beats EW S&P500 net" \\
        --n-configs 2 [--universe "S&P 500"] [--cadence monthly]

The generated doc is a SKELETON: every <TODO> must be filled and the doc
committed BEFORE the first forward observation. The freeze date is the commit
date of the completed doc, not the generation date.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import research_ledger as rl  # noqa: E402

TEMPLATE = """---
type: prereg
status: draft          # -> frozen cuando todos los <TODO> estén resueltos y commiteados
tags: [regime-trader, prereg, {family}]
created: {date}
related: ["[[2026-06-12-improvement-roadmap]]"]
---

# Pre-registro — {slug} ({family})

**Trial id (ledger):** `{trial_id}` · **configs cargadas:** {n_configs} ·
**n_trials familia '{family}' tras este cargo:** {family_trials}

> CONGELADO al commit de este doc con status: frozen. Después de eso, cualquier
> cambio = enmienda nueva con su propio cargo en el ledger.

## 0. Qué se está validando

Hipótesis: {hypothesis}

<TODO: mecanismo económico — por qué esto debería existir y quién está al otro
lado del trade. Sin mecanismo plausible, no se congela.>

## 1. Knobs CONGELADOS (sin barrido)

| Knob | Valor | Justificación |
|---|---|---|
| Universo | {universe} | <TODO> |
| Cadencia | {cadence} | <TODO> |
| <TODO resto de knobs> | | |

Variantes preregistradas: {n_configs} (cargadas arriba; ninguna variante
adicional sin enmienda + cargo nuevo).

## 2. Datos y medición

<TODO: fuente de datos (gratis — invariante del programa), point-in-time-ness,
serie NAV diaria (patrón track-record: columna/CSV propio, append-only),
aislamiento del libro (snapshot propio `book_snapshot_<sleeve>.json`).>

## 3. Benchmarks (mismo motor de costes)

<TODO: contra qué se compara, net-of-cost, investable.>

## 4. Criterios de aceptación (TODOS deben cumplirse; falla si falla cualquiera)

1. Ventana forward: ≥12 meses paper.
2. <TODO: umbral Sharpe / exceso vs benchmark>
3. **DSR > 0.5** con n_trials = {n_configs} (este prereg) — verificar contra el
   ledger en la adjudicación, no contra la memoria.
4. **PBO < 0.5** (CSCV, `backtest/performance.py::pbo_cscv`) cuando aplique
   backtest de soporte.
5. <TODO: maxDD / criterios operativos>

## 5. Modos de fallo (falsación explícita)

<TODO: qué resultado mata la idea de forma definitiva — enumerar ANTES de mirar
los datos forward. Blocklist actual: R1 timer, rotación vía B,
shorts-por-régimen, hmm_prob deploy directo.>

## 6. Expectativa honesta

<TODO: prior realista y por qué; qué dirían los escépticos.>

## 7. Reproducción (loci de código)

<TODO: módulos/flags/plists que implementan esto; commit SHA al congelar.>
"""


def create(slug: str, family: str, hypothesis: str, n_configs: int,
           universe: str, cadence: str,
           ledger_path: str = rl.LEDGER_PATH,
           out_dir: str = "docs/analysis") -> Path:
    """Generate the skeleton, charge the ledger, return the doc path."""
    date = datetime.now(timezone.utc).date().isoformat()
    out = Path(out_dir) / f"{date}-{slug}-prereg.md"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing prereg: {out}")
    row = rl.register(ledger_path, family=family, hypothesis=hypothesis,
                      n_configs=n_configs, prereg=str(out))
    doc = TEMPLATE.format(slug=slug, family=family, hypothesis=hypothesis,
                          n_configs=n_configs, universe=universe, cadence=cadence,
                          date=date, trial_id=row["id"],
                          family_trials=rl.n_trials(ledger_path, family=family))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True, help="kebab-case name for the doc")
    ap.add_argument("--family", required=True,
                    help="trial family (momentum/sentiment/quality/regime/crypto/meta)")
    ap.add_argument("--hypothesis", required=True, help="one-line hypothesis")
    ap.add_argument("--n-configs", type=int, required=True, dest="n_configs",
                    help="configurations charged to the ledger by this freeze")
    ap.add_argument("--universe", default="<TODO>")
    ap.add_argument("--cadence", default="<TODO>")
    args = ap.parse_args()
    out = create(args.slug, args.family, args.hypothesis, args.n_configs,
                 args.universe, args.cadence)
    print(f"prereg skeleton: {out}")
    print(f"family '{args.family}' n_trials now: {rl.n_trials(family=args.family)}")
    print("fill every <TODO>, set status: frozen, commit — THEN start the clock.")


if __name__ == "__main__":
    main()
