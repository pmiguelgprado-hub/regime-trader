---
type: prereg
status: draft          # -> frozen cuando todos los <TODO> estén resueltos y commiteados
tags: [regime-trader, prereg, sentiment]
created: 2026-06-14
related: ["[[2026-06-12-improvement-roadmap]]"]
---

# Pre-registro — news-sentiment (sentiment)

**Trial id (ledger):** `20260614-c4e4b0` · **configs cargadas:** 2 ·
**n_trials familia 'sentiment' tras este cargo:** 2

> CONGELADO al commit de este doc con status: frozen. Después de eso, cualquier
> cambio = enmienda nueva con su propio cargo en el ledger.

## 0. Qué se está validando

Hipótesis: Cross-sectional news-sentiment factor (Benzinga via Alpaca, free) ranks S&P 500 names; tested standalone and as a 12-1 momentum conditioner, vol-targeted, beats EW-S&P500 net over 12mo forward paper

<TODO: mecanismo económico — por qué esto debería existir y quién está al otro
lado del trade. Sin mecanismo plausible, no se congela.>

## 1. Knobs CONGELADOS (sin barrido)

| Knob | Valor | Justificación |
|---|---|---|
| Universo | S&P 500 (PIT snapshots, T5.3) | <TODO> |
| Cadencia | monthly rebalance, daily gross overlay | <TODO> |
| <TODO resto de knobs> | | |

Variantes preregistradas: 2 (cargadas arriba; ninguna variante
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
3. **DSR > 0.5** con n_trials = 2 (este prereg) — verificar contra el
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
