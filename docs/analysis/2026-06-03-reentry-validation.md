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

## TL;DR — el fix funciona; el skill riesgo-ajustado es marginal y estrecho (1/5), NO desplegable

El fix del latch **funciona**: en SPY el Sharpe sube de −0.16 a 0.40 y el % halted
cae de 54.7 % a 4.4 %. Y **recorta el drawdown** frente a buy&hold en los 5 ETFs.

Con la medición **justa** (acreditando el cash ocioso al tipo sin riesgo — ver §R-2,
crítico para comparar Sharpe contra un buy&hold 100 % invertido), el holdout da
**1/5: solo QQQ pasa** `Sharpe>bh ∧ maxDD<bh` (0.56 vs 0.54). SPY (dev) también pasa
(0.40 vs 0.36). PERO QQQ está ~0.9 correlacionado con SPY (el activo de tuning) → ese
único pass es **el eco correlacionado del dev, no una confirmación independiente**; y
su margen (+0.02) está **dentro del ruido ±8 % de R-4** (no distinguible de cero). Los
**4 activos de verdad independientes** — IWM/EFA/EEM/TLT — **fallan todos**. Lectura
limpia: **sin skill generalizable**. **1/5 (y frágil) ≪ listón ≥4/5 → NO desplegable.
Dinero real sigue BLOQUEADO.** Por protocolo, **no se re-tunea**.

**Nota de medición (corrección importante):** la primera corrida del holdout dio 0/5,
pero estaba **sesgada** — el backtester no acreditaba interés al cash ocioso, penalizando
el Sharpe de una estrategia des-riesgada (que mantiene cash) con el hurdle rf que se
resta a ambos. Al corregirlo (commit del cash-credit), QQQ pasa de fallar (0.41) a pasar
(0.56) y el veredicto correcto es **1/5, no 0/5**. (El advisor cazó este confound antes
de fijar la conclusión fuerte.)

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

## Holdout — 5 ETFs no vistos (K=3, floor=0.25 congelado, cash acreditado a rf)

| Activo | Sharpe strat | Sharpe bh | maxDD strat | maxDD bh | ret strat | ret bh | PASA |
|---|---|---|---|---|---|---|---|
| (SPY dev) | 0.40 | 0.36 | −37.8 % | −54.7 % | 382 % | 431 % | (sí) |
| QQQ | 0.56 | 0.54 | −32.9 % | −53.4 % | 669 % | 1028 % | **Sí** |
| IWM | 0.16 | 0.23 | −36.7 % | −57.4 % | 168 % | 242 % | No |
| EFA | −0.11 | 0.02 | −44.1 % | −61.0 % | 33 % | 55 % | No |
| EEM | −0.17 | 0.02 | −55.2 % | −66.4 % | −9 % | 18 % | No |
| TLT | −0.20 | −0.02 | −45.4 % | −48.4 % | 15 % | 66 % | No |

**1/5 holdout pasa** (`Sharpe > bh` Y `maxDD < bh`): solo QQQ. **El maxDD es mejor que
bh en los 5** (el recorte de drawdown es real y universal), pero el Sharpe solo bate a
bh en large-cap US (SPY dev, QQQ). En small-cap (IWM), internacional (EFA/EEM) y bonos
(TLT) el Sharpe queda por debajo: el coste en retorno supera el ahorro de riesgo.

(La primera corrida, sin acreditar cash, daba 0/5 — sesgo de medición; ver TL;DR + §R-2.)

## Interpretación

1. **El fix del latch es real y valioso *mecánicamente*** — elimina la trampa del
   halt permanente (Sharpe SPY −0.16→0.40, halted 54.7 %→4.4 %) y recorta el drawdown
   frente a bh en todos los activos.
2. **El único "pass" del holdout es el correlato del set de tuning — no es una
   confirmación independiente.** QQQ está ~0.9 correlacionado en retornos diarios con
   SPY (mismos mega-caps, misma estructura de régimen de vol). "Pasa en SPY (dev) y QQQ"
   = **una misma apuesta apareciendo dos veces en forma correlacionada**, no dos pruebas.
   Los **4 activos genuinamente independientes del set de tuning** — IWM (small-cap),
   EFA (intl desarrollado), EEM (emergentes), TLT (bonos, ~0 corr) — **fallaron todos.**
   Esa es la firma de **ausencia de skill generalizable**, no de "skill estrecho".
3. **Y ese único pass está dentro del propio ruido R-4.** QQQ +0.02 (0.56 vs 0.54) y
   SPY-dev +0.04 son los márgenes **más finos**, no los más holgados: con ±~8 % de
   deriva entre procesos en los números dependientes del HMM (bh es fijo, la estrategia
   no), el Sharpe de QQQ podría caer ~0.48-0.64 en otra corrida. Un +0.02 **no es
   distinguible de cero**. Coherente con el barajado preliminar (skill marginal) y el
   sweep del floor (retorno ~monótono con exposición) de [[2026-06-03-oos-validation]].
4. **El retorno siempre queda muy por debajo de bh** (QQQ 669 % vs 1028 %), incluso
   donde pasa el gate — porque pasa por *menor drawdown*, no por más retorno.

## Veredicto

- **NO desplegable: 1/5 en el holdout** (objetivo ≥4/5), y ese 1 es el correlato del
  set de tuning (QQQ≈SPY ~0.9) dentro del ruido R-4, mientras los 4 activos
  independientes fallan → **sin skill riesgo-ajustado generalizable**. **Dinero real
  BLOQUEADO.**
- **No re-tunear** (decisión previa al holdout, respetada). Buscar una combo que pase
  más de estos 5 sería in-sample.
- **El fix del latch + el cash-credit se conservan como código** (`peak_reentry_calm_bars`
  default 0 = legacy; `credit_cash_rf` default False = legacy). Ambos son mejoras
  correctas; no se cambia el default sin decisión explícita y no cambian el veredicto.

## Qué queda sobre la mesa (honesto)

El bot **sí** reduce drawdown frente a buy&hold de forma consistente (5/5, márgenes
10-20 pp) y bate el Sharpe de bh en large-cap US (SPY, QQQ). Lo que NO hace es
generalizar: 1/5 en el holdout. Caminos honestos:

- **(a) Restringir a large-cap US (SPY/QQQ) — pista débil, probablemente un espejismo.**
  SPY y QQQ son el mismo activo en la práctica (~0.9 corr): "pasa en los dos" es una
  apuesta, no dos, y por +0.02-0.04 Sharpe dentro del ruido R-4. No venderlo como lead
  prometedor. Si se persigue, el **mínimo** es un holdout **temporal** estricto (los 5
  activos ya están quemados) — y aun así se persigue una señal a nivel de ruido. El
  retorno seguiría muy por debajo de bh (QQQ 669 % vs 1028 %).
- **(b) Aceptar otro objetivo:** "exposición con caídas más suaves a cambio de menor
  retorno" (5/5 en drawdown). Producto conservador legítimo, pero **NO "batir al
  mercado"**, y solo mejora Sharpe en 1/5 → difícil de justificar frente a 60/40 o a
  simplemente reducir exposición a bh.
- **(c) Para batir a bh en Sharpe de forma robusta** haría falta alpha real (selección,
  factor, señal predictiva) que este HMM de régimen de volatilidad **no ha demostrado
  fuera de large-cap US**. Eso = proyecto nuevo, no un fix.

Recomendación: **no desplegar.** El único pass (QQQ) es el eco correlacionado de SPY
dentro del ruido R-4, y los 4 activos independientes fallan → no hay skill generalizable.
Si Pablo quiere seguir pese a todo, large-cap US (a) es la pista — pero **débil y
probablemente espejismo**, exige un holdout temporal independiente, y persigue una señal
a nivel de ruido. No perseguir small-cap/intl/bonos: ahí no hay nada.

## Lecciones

- **Verifica la métrica antes de fijar la conclusión fuerte.** El holdout daba 0/5,
  pero el backtester no acreditaba el cash ocioso → penalizaba el Sharpe de una
  estrategia des-riesgada con el hurdle rf sobre cash al 0 %. Corregido = 1/5. La
  conclusión fuerte ("0/5, sin skill, parar de optimizar") era un artefacto de medición.
  (El advisor lo cazó antes de commitear el claim — lección [[feedback-test-the-requirement]]:
  el mismo patrón que el bug del halt y el C2 MtM.)
- **El holdout hizo su trabajo igualmente:** el "win" de SPY dev (0.40>0.36, margen <ruido)
  no se sostuvo como regla general (1/5). El protocolo de no-contaminación separó
  "thorough-looking" de "decisivo".
- **Arreglar un bug ≠ crear edge robusto.** El fix del latch era correcto y necesario,
  pero quitar la patología te devuelve ~al nivel del benchmark, con skill solo en un
  nicho (large-cap US), no transversal.
- **Reducir drawdown ≠ mejor Sharpe.** El recorte de drawdown fue universal (5/5), pero
  solo se tradujo en mejor Sharpe en 2 activos. Si el coste en retorno supera el ahorro
  de riesgo, el Sharpe empeora aunque el drawdown mejore. Medir ambos — y acreditar el
  cash, o el Sharpe del que des-riesga sale artificialmente penalizado.
