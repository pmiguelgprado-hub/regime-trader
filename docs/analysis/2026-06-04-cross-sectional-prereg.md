---
type: analysis
status: frozen
tags: [regime-trader, cross-sectional, via-c, pre-registration, gate, momentum, forward-paper]
created: 2026-06-04
related: ["[[2026-06-04-stock-picking-feasibility]]", "[[2026-06-04-markov-edge-redesign]]", "[[2026-06-04-rotation-results]]", "[[2026-06-03-oos-validation]]"]
---

# Pre-registro CONGELADO — Vía C: book cross-sectional (predictor de retorno + overlay HMM)

> **Este criterio se fija ANTES de acumular track record y no se mueve.** Es la misma
> disciplina que cazó los tres falsos positivos previos (timing, re-entrada, rotación). Si
> el book no pasa el gate tal y como está escrito aquí, **no se despliega dinero real** —
> punto. Mover el poste después = la trampa de sobreajuste. **Dinero real BLOQUEADO.**

## 0. Qué se está validando

El **predictor de retorno es cross-sectional** (momentum 12-1 sobre los constituyentes del
S&P 500 = la señal de alfa) y el **HMM es el overlay de riesgo** (escala la exposición bruta
total por régimen de volatilidad). Arquitectura, evidencia experta y razonamiento en
[[2026-06-04-stock-picking-feasibility]]. v1 = reglas (sin entrenar); el predictor ML
(Gu-Kelly-Xiu) es un v2 con su propio pre-registro y datos de pago.

**Hipótesis a falsar:** el book bate al índice en términos **riesgo-ajustados y neto de
costes**. Hipótesis nula = no aporta sobre tener el índice (igual que rotación restó valor
vs la cesta estática).

## 1. Knobs CONGELADOS (sin barrido)

| Knob | Valor | Fuente |
|---|---|---|
| Señal | momentum 12-1 (lookback 252, skip 21) | Jegadeesh-Titman; `config/settings.yaml::cross_sectional` |
| Selección | top decil (`top_fraction=0.10`), equal-weight | fijo |
| Caps | `max_single=0.15`, `max_concurrent=50` | risk caps existentes |
| Overlay HMM | `risk_on_gross=1.0`, `risk_off_gross=0.5` (interp. en banda media) | `regime_gross_scale` |
| Rebalanceo | mensual | `make_book_weights` (memo por año-mes) |
| Costes | slippage por turnover + `credit_cash_rf=True` (cash ocioso al rf) | motor emparejado |

**Ningún knob se sweepea.** Cualquier barrido posterior es multiple-testing → invalida el
gate (y exigiría DSR con n_trials real).

## 2. Datos y medición: SOLO forward paper

- **Backtest histórico sobre constituyentes de HOY = NO cuenta** (sesgo de supervivencia,
  [[2026-06-03-oos-validation]]). Sirve solo como *smoke de fontanería* (pesos sanos,
  turnover plausible, no crashea), nunca como evidencia de edge.
- **El track record empieza el día del despliegue paper** y es OOS limpio por construcción.
- **Longitud mínima antes de evaluar el gate: ≥ 12 meses (~250 barras diarias).** Evaluar
  antes = leer ruido (la lección R-4: un Sharpe de pocas observaciones no se sostiene).

## 3. Benchmarks (universe-aware, los DOS)

1. **S&P 500 equal-weight** — el control decisivo que faltó en la vía B: si el book no bate
   ni a holdear los mismos nombres a peso igual, el momentum + overlay no aportan.
2. **SPY (cap-weight)** — el índice que se quiere batir.

Ambos medidos con el **mismo motor de costes** (slippage + cash-credit) que el book.

## 4. Criterio de aceptación (congelado)

El book debe, sobre la ventana forward (≥12 meses), cumplir **TODO**:

1. **Sharpe(book) > Sharpe(EW-S&P500)** Y **Sharpe(book) > Sharpe(SPY)**.
2. **maxDD(book) menos negativo** que el de **ambos** benchmarks.
3. **Neto de costes** (slippage + cash-credit activados — medir net-of-cost o no medir).
4. **DSR > 0** (Deflated Sharpe; corrige por el nº de configuraciones — aquí n_trials=1 si
   de verdad no se sweepeó).
5. **El overlay HMM añade valor incremental:** book con overlay (`use_overlay=True`) bate al
   ranker desnudo (`use_overlay=False`) en Sharpe **o** maxDD. Si no, se despliega el ranker
   desnudo y se descarta el overlay (no se arrastra maquinaria que no paga — la rotación ya
   enseñó que des-riesgar por régimen puede restar).

**Falla cualquiera → FAIL → no hay dinero real.**

## 5. Qué lo falsaría (explícito)

- El book empata o pierde vs EW-S&P500 risk-adjusted (el momentum no supera el coste de
  rotar vs holdear el índice equiponderado) — la firma de "sin edge" que vimos en rotación.
- El overlay HMM resta (de-risk net-negativo), igual que el floor-sweep de
  [[2026-06-03-oos-validation]].
- DSR ≤ 0 (el Sharpe no sobrevive a la corrección por búsqueda).
- Track record < 12 meses: **no se evalúa** (no se adelanta el veredicto).

## 6. Expectativa honesta

Techo realista de un picker retail forward-paper: **mejora riesgo-ajustada modesta**, no un
destroza-mercados — y solo *si* pasa el gate. Las primas de factor decayeron post-2003, el
momentum tiene crashes (2009), y los costes muerden ([[2026-06-04-stock-picking-feasibility]]
§3). Un combo HMM+NN+Black-Litterman *publicado* aun así dio IR −0.1 vs S&P. Tratar como
**hipótesis a falsar**, no como cura.

## 7. Reproducir / operar

```bash
# Plan de rebalanceo (dry-run; no envía órdenes):
.venv/bin/python main.py --rebalance
# Smoke de fontanería del backtest (sesgo de supervivencia — NO es edge):
#   usar Backtester.run_portfolio(frames, weight_fn=make_book_weights(...))
```

Código: `core/cross_sectional_ranking.py` (ranker + book weights + plan),
`core/asset_rotation.py::regime_gross_scale` (overlay), `Backtester.run_portfolio(weight_fn=)`
(backtest), `main.run_rebalance` (paper, dry-run; live GATED). Métricas/benchmarks:
`backtest/performance.py` (Sharpe/maxDD/DSR), `backtest/benchmarks.py` (EW/SPY).
Tests: `tests/test_cross_sectional.py`, `tests/test_constituents.py`.

**Estado:** pipeline desplegable en dry-run + gate congelado. Edge DESCONOCIDO hasta ≥12
meses de paper. **Dinero real BLOQUEADO.**
