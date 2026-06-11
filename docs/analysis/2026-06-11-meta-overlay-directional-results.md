---
type: analysis
status: final
tags: [regime-trader, meta-overlay, fuzzy, hmm_prob, directional-eval, falsification]
created: 2026-06-11
related: ["[[2026-06-11-meta-overlay-triage]]", "[[2026-06-05-deployed-book-prereg]]", "[[2026-06-05-challenger-directional-results]]"]
---

# Eval direccional capa fuzzy (hmm_prob) — 2026-06-11

Comando: `.venv/bin/python -m backtest.run_challenger_eval --limit 100 --start 2015-01-01`
(N=100 supervivientes S&P 500, sesgo supervivencia → SOLO direccional, no gate).
12 variantes por el MISMO motor de costes; log completo en `tmp/eval_hmm_prob.log`.

## Resultado (extracto)

| estrategia | tot_ret | CAGR | Sharpe | maxDD | DSR |
|---|---|---|---|---|---|
| baseline_raw_hmm | 352.2% | 21.9% | 0.78 | −26.3% | 0.92 |
| **raw_hmm_prob** | **356.1%** | **22.0%** | **0.79** | **−26.0%** | 0.92 |
| raw_crashonly (DESPLEGADO) | 467.3% | 25.5% | **0.86** | −30.7% | 0.94 |
| raw_none (control) | 592.2% | 28.9% | 0.85 | −44.5% | 0.94 |
| SPY_hold | 195.1% | 15.2% | 0.59 | −33.7% | — |

PBO (CSCV, 12 variantes): 0.38 (<0.5 → el ganador in-sample generaliza razonablemente).

## Lectura

1. **La capa fuzzy hace lo que predijo la matemática:** `hmm_prob` domina a `hmm`
   en las TRES métricas (retorno, Sharpe, drawdown) en su comparación like-for-like.
   Quitar el cliff del argmax mejora el overlay hmm sin ningún parámetro nuevo.
   La mejora es **marginal** (+0.01 Sharpe, +0.3pp DD) — coherente con que el
   posterior filtrado ya es confiado (prob≈1) la mayoría de barras; la masa gris
   solo existe en las transiciones, que es exactamente donde recorta daño.
2. **NO desbanca al book desplegado:** `crash_only` (0.86) sigue ganando a toda la
   familia hmm (0.78–0.79) en Sharpe, igual que en la eval N=200 del 2026-06-05.
   El de-risk quirúrgico solo-pánico retiene más upside que la interpolación
   continua de la banda media — el coste de la suavidad es recortar también
   subidas buenas.
3. **Decisión:** el book desplegado NO cambia (`overlay: crash_only`). `hmm_prob`
   queda como representante superior de la familia hmm para futuras
   comparaciones y como fuente del **hazard** (señal ortogonal que alimenta el
   tail-hedge de opciones y el dashboard — esa señal no depende de qué overlay
   escala el gross).
4. Sin sweep: una sola variante añadida, pre-anunciada en el triage memo; entra
   en el set de trials del DSR/PBO como corresponde.
