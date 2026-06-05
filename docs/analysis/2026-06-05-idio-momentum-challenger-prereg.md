---
type: analysis
status: frozen
tags: [regime-trader, cross-sectional, via-c, challenger, pre-registration, gate, residual-momentum, vol-target, forward-paper]
created: 2026-06-05
related: ["[[2026-06-04-cross-sectional-prereg]]", "[[2026-06-04-stock-picking-feasibility]]", "[[2026-06-03-oos-validation]]"]
---

# Pre-registro CONGELADO — Challenger Vía C: momentum idiosincrático + volatility targeting

> **Criterio fijado ANTES de acumular track record. No se mueve.** Misma disciplina que falsó
> los tres positivos previos (timing, re-entrada, rotación). El baseline congelado
> ([[2026-06-04-cross-sectional-prereg]]) **sigue corriendo intacto** como control limpio; este
> challenger es un libro **paralelo** con su propio gate. Mover el poste = sobreajuste.
> **Dinero real BLOQUEADO** hasta que este gate pase en forward-paper.

## 0. Qué se está validando

El predictor baseline (momentum 12-1 crudo) es **el factor académico desnudo** — literalmente
el baseline de los tutoriales retail. La literatura ofrece dos mejoras price-only de evidencia
fuerte que el challenger introduce:

1. **Momentum idiosincrático / residual** (Blitz-Huij-Martens 2011; Chaves 2016;
   Blitz-Hanauer-Vidojevic 2017). Rankear por el residuo del modelo de mercado (beta de
   mercado estimada en ventana larga, residuo puntuado en sub-ventana 12-1 reciente),
   estandarizado por su vol residual. Evidencia: ~2x Sharpe del momentum convencional, riesgo
   de crash ≈ eliminado, replicado en 21 países incl. Japón. En producción en Robeco y AQR.
2. **Volatility targeting** (Daniel-Moskowitz 2016; Barroso-Santa-Clara 2015). Escalar el
   gross a vol objetivo constante. Evidencia: duplica el Sharpe del momentum estático, los
   crashes son parcialmente predecibles (panic states). Reutiliza `vol_target_scale` ya
   existente.

**Hipótesis a falsar:** el book challenger bate al baseline **y** al índice en términos
riesgo-ajustados y neto de costes. Nula = no aporta sobre el baseline ni sobre tener el índice.

## 1. Knobs CONGELADOS (sin barrido)

| Knob | Valor | Fuente |
|---|---|---|
| Señal | momentum **residual** 12-1 (lookback 252, skip 21, est_window 504) | `residual_momentum_score` |
| Selección | top decil (`frac=0.10`), equal-weight, sector-cap 0.30 | heredado del baseline |
| Caps | `max_single=0.15`, `max_concurrent=50` | risk caps existentes |
| Overlay | `vol_target` (target_vol=0.12, vol_window=126, cap=1.0, floor=0.0) | `vol_target_scale` |
| Rebalanceo | mensual (selección **y** gross fijados en el rebalanceo) | `make_book_weights_challenger` |
| Costes | slippage por turnover + `credit_cash_rf=True` | motor emparejado |

**Variantes pre-registradas = 2** (enfocado, para no inflar multiple-testing):
- **V1** = residual + overlay `vol_target` (la candidata principal).
- **V2** = residual + overlay `none` (ranker desnudo — aísla si el overlay aporta).

Los modos `hmm` y `both` se computan en el eval **direccional** solo para atribución, NO son
variantes candidatas del gate (no cuentan en n_trials del forward-paper). Ningún knob se sweepea.

## 2. Datos y medición

- **Backtest histórico = SOLO DIRECCIONAL.** Universo = constituyentes actuales del S&P 500 →
  sesgo de supervivencia. `backtest/run_challenger_eval.py` produce la tabla direccional con
  DSR y PBO. Sirve para **descartar temprano** (si no bate al baseline ni en datos sesgados,
  fuera), NO para declarar edge.
- **Track record = forward-only**, empieza el día del despliegue paper, OOS por construcción.
- **Ventana mínima de evaluación: ≥ 12 meses (~250 sesiones)** antes de juzgar el gate.

## 3. Benchmarks (mismo motor de costes)

1. **Baseline congelado** (momentum 12-1 + overlay HMM) — el control directo.
2. **S&P 500 equal-weight (RSP)** — si no bate al EW del propio universo, no hay edge de selección.
3. **SPY (cap-weight)** — el índice a batir.

## 4. Criterios de aceptación (TODOS deben cumplirse; falla si falla cualquiera)

1. **Sharpe(V1) > Sharpe(baseline)** Y **Sharpe(V1) > Sharpe(EW-S&P500)** Y **> Sharpe(SPY)**.
2. **maxDD(V1) < maxDD** de los tres benchmarks (menos profundo).
3. **Neto de costes** (slippage + cash-credit activos).
4. **DSR > 0.5** con n_trials = 2 (las dos variantes pre-registradas).
5. **PBO < 0.5** (CSCV sobre el conjunto de variantes).
6. **El overlay aporta incrementalmente:** V1 (vol_target) bate a V2 (none) en Sharpe O maxDD.
   Si no, desplegar el ranker desnudo y descartar el overlay.

## 5. Modos de fallo (falsación explícita)

- V1 ≤ baseline o ≤ EW-S&P500 riesgo-ajustado → el residual no aporta sobre el momentum crudo
  ni sobre el índice → descartar.
- Overlay vol_target resta (V1 < V2) → quitar el overlay.
- DSR ≤ 0.5 o PBO ≥ 0.5 → el Sharpe no es robusto a multiple-testing / la selección sobreajusta.
- Track record < 12 meses → no se lee (ruido).

## 6. Expectativa honesta

Techo realista retail forward-paper: mejora riesgo-ajustada **modesta** sobre el baseline, *si*
pasa. NO es un batidor de mercado: factores públicos hace años, decaimiento post-publicación,
los costes muerden, la competencia es institucional. Precedente del proyecto: 3/3 avenidas
previas falsadas. **Tratar como hipótesis a falsar, no como mejora garantizada.**

## 7. Reproducción (loci de código)

```
señal residual:   core/cross_sectional_ranking.py::residual_momentum_score / rank_universe_residual
weight_fn:        core/cross_sectional_ranking.py::make_book_weights_challenger
overlay vol:      core/asset_rotation.py::vol_target_scale (reutilizado)
overlay hmm:      core/asset_rotation.py::regime_gross_scale (atribución)
eval direccional: backtest/run_challenger_eval.py
métricas:         backtest/performance.py (deflated_sharpe_ratio, pbo_cscv)
tests:            tests/test_challenger.py
live (paper, GATED): main.py::run_rebalance --challenger
```

**Estado:** pipeline construido + gate congelado. Edge DESCONOCIDO hasta ≥12 meses de paper.
Dinero real BLOQUEADO.
