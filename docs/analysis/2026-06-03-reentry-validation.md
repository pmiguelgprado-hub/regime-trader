---
type: analysis
status: active
tags: [regime-trader, halt-reentry, risk-adjusted, oos-validation, falsification]
created: 2026-06-03
related: ["[[2026-06-03-oos-validation]]", "[[2026-06-03-halt-reentry-risk-adjusted-design]]"]
---

# Validación del fix de re-entrada del halt (objetivo riesgo-ajustado)

Continuación de [[2026-06-03-oos-validation]] (que falsó el edge en retorno bruto) y
del spec [[2026-06-03-halt-reentry-risk-adjusted-design]]. Objetivo reformulado
(decisión de usuario): **Sharpe > buy&hold Y maxDD < buy&hold** en un holdout de
5 ETFs no vistos, tuneando solo en SPY. Se arregló el destructor nº1 medido — el
latch del halt — re-habilitando por normalización de vol (K barras calmas) en vez
de por recuperar el pico.

## TL;DR — el fix funciona mecánicamente, pero el thesis riesgo-ajustado NO sobrevive

El fix del latch **funciona**: en SPY el Sharpe sube de −0.16 a 0.40 y el % halted
cae de 54.7 % a 4.4 %. Y **recorta el drawdown** frente a buy&hold en 4 de 5 ETFs.
**PERO el Sharpe queda por debajo de buy&hold en los 5 ETFs del holdout (0/5 pasan
el gate).** El des-riesgo sacrifica más retorno del que ahorra en riesgo → Sharpe
peor que simplemente mantener. **El overlay de régimen no tiene skill riesgo-ajustado
fuera de muestra.** Por protocolo, **no se re-tunea**. Dinero real sigue BLOQUEADO.

El +0.04 de Sharpe en SPY (0.40 vs bh 0.36) que parecía un win era **in-sample y
dentro del ruido ±8 % de R-4** — no generalizó a un solo activo no visto.

## Método

- **Tuning SOLO en SPY 2004-24** (set de desarrollo): grid `peak_reentry_calm_bars`
  K ∈ {0,3,5,10} × `halt_floor_mult` ∈ {0, 0.25}. Métrica de selección = Sharpe.
- **Holdout congelado:** la combo ganadora (K=3, floor=0.25) se corre **una vez** en
  los 5 ETFs nunca tocados (QQQ/IWM/EFA/EEM/TLT). Sin re-tuneo tras ver el resultado.
- Cada fase en **un solo proceso** (R-4: comparaciones intra-proceso exactas; absolutos
  ±~8 % entre procesos). El gate compara strat vs bh dentro de la misma corrida.
- Implementación: `core/risk_manager.py` (re-entrada por calma) + `backtest/backtester.py`
  (`_calm_flag` = `vol_rank < 0.67`) + `backtest/oos_validation.py` (gate). TDD, 191 tests.

## Tuning en SPY (dev)

| K | floor | Sharpe | maxDD | retorno | % halted |
|---|---|---|---|---|---|
| 0 | 0.0 | −2.72 | −11.3 % | −11.3 % | 98.5 % |
| 0 | 0.25 | −0.16 | −21.1 % | 61.0 % | 54.7 % |
| **3** | **0.25** | **0.40** | −36.4 % | 382.2 % | 4.4 % |
| 3 | 0.0 | 0.37 | −36.4 % | 348.8 % | 4.6 % |
| 5 | 0.0 | 0.36 | −36.4 % | 333.2 % | 5.8 % |
| 5 | 0.25 | 0.36 | −42.4 % | 330.6 % | 5.5 % |
| 10 | 0.0 | 0.38 | −30.0 % | 348.7 % | 7.7 % |
| 10 | 0.25 | 0.38 | −32.7 % | 348.4 % | 8.0 % |

SPY buy&hold Sharpe (ref) = **0.36**. El fix del latch convierte el K=0 patológico
(−2.72 / 98.5 % halted) en Sharpe ~0.36-0.40 con halted ~4-8 %. La mejora mecánica es
enorme, pero el Sharpe aterriza **en** bh, no por encima. La combo "ganadora"
(0.40 vs 0.36) gana por un margen menor que el ruido de proceso.

## Holdout — 5 ETFs no vistos (K=3, floor=0.25 congelado)

| Activo | Sharpe strat | Sharpe bh | maxDD strat | maxDD bh | ret strat | ret bh | PASA |
|---|---|---|---|---|---|---|---|
| QQQ | 0.41 | 0.54 | −37.8 % | −53.4 % | 422 % | 1028 % | No |
| IWM | 0.14 | 0.23 | −42.1 % | −57.4 % | 151 % | 242 % | No |
| EFA | −0.16 | 0.02 | −48.1 % | −61.0 % | 17 % | 55 % | No |
| EEM | −0.21 | 0.02 | −55.5 % | −66.4 % | −21 % | 18 % | No |
| TLT | −0.24 | −0.02 | −49.6 % | −48.4 % | 7 % | 66 % | No |

**0/5 pasan** (`Sharpe > bh` Y `maxDD < bh`). El maxDD es mejor que bh en 4/5 (TLT
la excepción), pero **el Sharpe es peor que bh en los 5**. El recorte de drawdown es
real, pero el coste en retorno lo supera → peor retorno por unidad de riesgo.

## Interpretación

1. **El fix del latch es real y valioso *mecánicamente*** — elimina la trampa del
   halt permanente (Sharpe SPY −0.16→0.40, halted 54.7 %→4.4 %) y recorta el drawdown
   frente a bh casi siempre. Como *reductor de drawdown* sí hace algo.
2. **Pero no hay skill riesgo-ajustado.** En activos no vistos el Sharpe es
   consistentemente **menor** que buy&hold. Des-riesgar por régimen sacrifica más
   retorno del que ahorra en volatilidad/drawdown. Esto es coherente con el test de
   barajado preliminar de [[2026-06-03-oos-validation]] (skill del overlay marginal)
   y con el sweep del floor (retorno monótono con exposición).
3. **El "win" de SPY (0.40>0.36) era in-sample.** El holdout es justo para esto: el
   protocolo de no-contaminación cazó un falso positivo que el tuning habría vendido
   como éxito. (Lección repetida del proyecto: no concluir de un activo/periodo.)

## Veredicto

- **Thesis riesgo-ajustado FALSADO en el holdout (0/5).** El bot, con el overlay de
  régimen, **no bate a buy&hold en Sharpe** fuera de muestra. **Dinero real BLOQUEADO.**
- **No re-tunear** (decisión previa al holdout, respetada). Buscar una combo que pase
  estos 5 sería in-sample y deshonesto.
- **El fix del latch se conserva como código** (opt-in por `peak_reentry_calm_bars`,
  default 0 = legacy). Estrictamente mejor que el latch roto; útil si en el futuro se
  adopta un objetivo de **solo-reducción-de-drawdown** (ver abajo). No se cambia el
  default sin decisión explícita — no cambia el veredicto "no desplegable".

## Qué queda sobre la mesa (honesto)

El bot **sí** reduce drawdown frente a buy&hold de forma consistente (4/5, márgenes
de 10-20 pp). Lo que NO hace es batir el Sharpe. Dos caminos honestos:

- **(a) Aceptar otro objetivo:** "exposición tipo-mercado con caídas más suaves, a
  cambio de menor retorno". Es un producto legítimo (perfil conservador) pero **NO es
  "batir al mercado"** — y su Sharpe es peor, así que ni siquiera es "mejor retorno
  ajustado al riesgo". Difícil de justificar frente a, p.ej., 60/40 o simplemente
  menos exposición a buy&hold.
- **(b) Aceptar que el overlay de régimen carece de señal predictiva** suficiente. Para
  batir a bh en Sharpe haría falta una fuente de alpha real (selección, factor, señal
  con poder predictivo), que este HMM de régimen de volatilidad **no ha demostrado**.
  Construir eso = proyecto nuevo, no un fix.

Recomendación: **parar de optimizar este overlay** para batir al mercado. La evidencia
(OOS retorno + OOS riesgo-ajustado + barajado) converge en lo mismo: no hay edge
desplegable. Si Pablo quiere seguir, que sea con un objetivo explícito y medible
distinto, no "superar al mercado siempre".

## Lecciones

- **El holdout hizo su trabajo:** cazó un falso positivo in-sample (SPY 0.40>0.36) que
  el tuning habría vendido como éxito. El protocolo de no-contaminación es lo que
  separa "thorough-looking" de "decisivo".
- **Arreglar un bug ≠ crear edge.** El fix del latch era correcto y necesario, pero
  quitar una patología solo te devuelve al nivel del benchmark, no por encima.
- **Reducir drawdown ≠ mejor Sharpe.** Si el recorte de retorno supera el de riesgo,
  el Sharpe empeora aunque el drawdown mejore. Medir ambos.
