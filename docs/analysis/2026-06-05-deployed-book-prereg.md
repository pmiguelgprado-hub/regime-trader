---
type: analysis
status: frozen
tags: [regime-trader, cross-sectional, via-c, deployed-book, vol-target, daily-cadence, pre-registration, gate, forward-paper]
created: 2026-06-05
related: ["[[2026-06-05-challenger-directional-results]]", "[[2026-06-04-cross-sectional-prereg]]", "[[2026-06-05-idio-momentum-challenger-prereg]]"]
---

# Pre-registro CONGELADO — Libro DESPLEGADO: raw momentum + overlay HMM, cadencia diaria

> **Decisión de Pablo (2026-06-05): desplegar la mejor opción en la cuenta Alpaca paper
> ACTUAL (sin segunda cuenta), con movimientos diarios y el bot atento en todo momento.**
> Supersede el pre-registro baseline `2026-06-04-cross-sectional-prereg` (overlay HMM,
> mensual) para el libro vivo. Reloj forward reiniciado (apenas había arrancado: ~2 días,
> primer auto-execute real era 2026-07-01). **Dinero real BLOQUEADO.**

## 0. El libro desplegado

- **Alfa = momentum 12-1 crudo** (Jegadeesh-Titman). Ganó retorno **y** Sharpe en el eval
  direccional y es robusto a cualquier tamaño de universo (el residual necesita el
  universo completo; ver §3). `momentum_score` / `rank_universe`.
- **Overlay de riesgo = HMM regime** (`regime_gross_scale` vía `_overlay_gross`): escala el
  gross por el régimen de volatilidad (full en risk-on, ~50% en risk-off). Elegido porque
  el objetivo de Pablo es **maximizar return + Sharpe** con control de riesgo, y `hmm` es el
  punto de la frontera que lo hace (Sharpe 0.87 ≈ el máximo sin overlay 0.93, maxDD -27% vs
  -40%). Combinar overlays ("both") de-riesga el doble → MENOR return+Sharpe (no es el
  objetivo). El vol-target queda disponible como `overlay: vol_target` (1 línea) si se
  prioriza drawdown. Sigue atento a diario: el régimen HMM se recalcula cada día hábil.
- **Cadencia diaria, dos escalas de tiempo** (lo que hace al bot "atento en todo momento"
  sin desvirtuar una señal lenta):
  - **Selección**: re-rankeo del momentum **solo en el primer run de cada mes nuevo**.
  - **Riesgo**: cada día hábil (L-V, 22:30 tras el cierre US) se recalcula el régimen +
    el gross vol-target y se re-escala la exposición a la vol de hoy (des-riesga en picos,
    re-riesga en calma). `book_targets_fixed_selection` + memo por mes en `run_rebalance`.

## 1. Por qué esta configuración (eval direccional, N=200, net-of-cost, PBO 0.26)

Frontera Sharpe↔drawdown sobre el momentum crudo (PBO 0.26 = el ranking generaliza):

| variante | ret. total | Sharpe | maxDD |
|---|---:|---:|---:|
| raw_none (sin overlay) | 814.6% | **0.93** | -39.9% |
| **raw_hmm (DESPLEGADO)** | 475.2% | **0.87** | -26.6% |
| raw_vol_target | 253.6% | 0.82 | -22.6% |
| raw_both (hmm × vol) | 179.0% | 0.75 | **-15.9%** |
| SPY | 229.9% | 0.59 | -33.7% |
| EW-S&P500 | 265.8% | 0.66 | -35.8% |

Los overlays cambian Sharpe por drawdown de forma **monótona**: menos overlay = más
return+Sharpe pero más drawdown. **Combinar hmm+vol_target ("both") de-riesga el doble →
el MENOR return+Sharpe** (0.75/179%), no el mayor — error común. Objetivo de Pablo =
maximizar return+Sharpe **con** control de riesgo → `hmm` es el punto óptimo: Sharpe 0.87
(≈ el máximo sin overlay 0.93), maxDD -26.6% (muy por debajo del -40% de `none`), bate
SPY/EW en Sharpe **y** drawdown. **Trade-off: vs `none` cuesta 0.06 de Sharpe y compra
13 pts de drawdown (-40%→-27%).**

**Cambiar de overlay = una línea** (`config/settings.yaml::cross_sectional.overlay`):
`none` (máx Sharpe/retorno, máx DD) · `hmm` (desplegado) · `vol_target` (menos DD, menos
return) · `both` (mín DD, mín return).

## 2. Knobs CONGELADOS (sin barrido)

| Knob | Valor |
|---|---|
| Señal | momentum 12-1 (lookback 252, skip 21) |
| Selección | top decil (0.10), equal-weight, sector-cap 0.30, max_concurrent 50, max_single 0.15 |
| Overlay | `hmm` (risk_on_gross 1.0, risk_off_gross 0.5); vol-target knobs quedan en config para el cambio de 1 línea |
| Cadencia | diaria L-V; re-rank selección 1er run de cada mes; re-escala gross diario |
| Costes | slippage por turnover + `credit_cash_rf` |

## 3. Por qué raw y no residual en el libro vivo

El residual (idiosincrático) es competitivo solo con el universo COMPLETO (N=503:
resid_none Sharpe 0.83 ≈ raw 0.87); a N=40/200 pierde claramente. El raw es robusto a
cualquier N. El libro vivo arranca operando el universo completo, pero el raw evita el
riesgo de que la cola del residual no aguante. El residual + vol-target queda construido y
testeado como **challenger** ([[2026-06-05-idio-momentum-challenger-prereg]]) por si se
abre una 2ª cuenta para correrlo en paralelo. El backtest NO adjudica raw-vs-residual
(sesgo de supervivencia + long-only + toro-beta); decide el forward paper.

## 4. Gate (forward paper ≥12 meses)

Batir, neto de costes, en Sharpe **y** maxDD a: SPY (cap-weight) + RSP (EW-S&P500). DSR>0,
y el overlay debe aportar (si `vol_target` no mejora maxDD sobre `none` en vivo, revertir a
`none`/`hmm`). Histórico = solo direccional (sesgo supervivencia). **Real money BLOCKED.**

## 5. Honestidad

Backtest sesgado, no es prueba de edge. 3/3 avenidas previas del proyecto falsadas. El
raw_none gana Sharpe+retorno en el backtest pero con -40% de drawdown (peor que el índice);
el vol-target sacrifica algo de Sharpe por un drawdown muy inferior — esa es la apuesta de
"equilibrio". El edge real se desconoce hasta ≥12 meses de paper.
