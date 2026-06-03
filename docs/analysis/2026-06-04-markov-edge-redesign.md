---
type: analysis
status: active
tags: [regime-trader, edge, hmm, strategy-redesign, markov, overfitting]
created: 2026-06-04
related: ["[[2026-06-02-why-no-edge]]", "[[2026-06-03-oos-validation]]", "[[2026-06-03-reentry-validation]]", "[[2026-06-02-optimization-and-roadmap]]"]
---

# Estrategia para batir al mercado con cadenas de Markov — diagnóstico honesto y rediseño

> **Punto de partida sin rodeos.** El handoff que motivó esta pregunta
> (`2026-06-03-improvements-handoff.md`) afirma un *edge competitivo provisional*
> (52.8 % / Sharpe 1.22 en SPY 2019-24). **Ese handoff es previo a la falsación.**
> La validación OOS del mismo proyecto ([[2026-06-03-oos-validation]], walk-forward
> 2004-24, 6 ETFs) demostró que **ese edge no existe fuera de muestra**. Este
> documento parte de ese hecho, no lo discute. **Dinero real sigue BLOQUEADO.**

---

## 0. Respuesta directa: ¿ejecutar el HMM cada 2 horas en vez de a diario?

**No resuelve el problema y degrada tu mejor herramienta de validación.** El timeframe
no es el eje del fallo. El diagnóstico OOS prueba que el alfa es **cero**, no que el
muestreo sea "demasiado lento". Cambiar daily → 2h:

| Eje | Efecto de pasar a 2h |
|---|---|
| **Alfa** | Misma lógica regime-vol-timing, grano más fino → sigue modulando beta, **no crea señal direccional**. |
| **Costes** | ~3-4 barras/día (sesión US 6.5 h) → rebalanceos intradía → turnover↑ → slippage (`_slippage_rate`, base + premium-vol) se come el retorno. |
| **Ruido** | Retornos intradía tienen peor señal/ruido (microestructura, gaps). El HMM sobre features precio-vol captará **aún más persistencia-de-volatilidad**, que es justo lo que no paga. |
| **Validación** | yfinance da **~60 días** de intradía. El walk-forward 2004-24 que cazó el no-edge es **inviable a 2h con datos gratis** → pierdes la falsación de 20 años. |
| **Overfitting** | Más barras + más decisiones = más grados de libertad para sobreajustar. |

**Único caso donde 2h tendría sentido:** si la *fuente de alfa* fuera intradía
(microestructura, mean-reversion intradía). Eso es **otra estrategia**, no este HMM, y
ahí compites contra HFT con datos y ejecución que tú no tienes → edge retail ≈ 0.

**Veredicto: mantener daily.** Optimizar el timeframe es pulir el eje equivocado.

---

## 1. Diagnóstico del problema actual

**No es un bug.** El bug del halt-latch (peso 0 latcheado → bot parado >50 %) ya se
arregló ([[2026-06-02-why-no-edge]]). Lo que queda es **estructural**.

**Causa raíz — con evidencia, no opinión.** El barrido de `halt_floor_mult` en SPY
2004-24 (de [[2026-06-03-oos-validation]]):

| floor | retorno | Sharpe | % halted |
|---|---|---|---|
| 0.0 | −11.3 % | −2.72 | 98.5 % |
| 0.25 | 73.3 % | −0.10 | 53.2 % |
| 0.5 | 180.6 % | 0.18 | 30.9 % |
| 1.0 (halt off) | 420.3 % | 0.40 | 19.5 % |
| **buy & hold** | **430.8 %** | — | — |

El retorno y el Sharpe **escalan monótonamente con la exposición**. Cuanto menos
des-riesgas, más ganas, hasta converger en buy&hold. **Esto demuestra alfa = 0: la
estrategia solo controla beta.** El des-riesgo del halt es net-negativo (recorta
retorno sin comprar Sharpe). Y en los 6 ETFs (2004-24) **pierde contra buy&hold en los
6**, Sharpe negativo en 5/6.

**Por qué un HMM precio-vol no genera alfa direccional.** El modelo detecta
*persistencia de régimen de volatilidad* — algo real y autocorrelado. Pero los
**retornos** de un índice son casi un random walk: no hay estructura de transición que
los prediga. El HMM es excelente modelando lo que **no paga** (vol) y ciego ante lo que
sí pagaría (signo del retorno futuro).

**El benchmark es casi imposible.** Long-only sobre UN índice con drift alcista fuerte.
El regime-timing históricamente **reduce drawdown**, rara vez **bate el retorno total**.
Pedirle a un timer de un solo activo que supere a buy&hold es pedirle que haga lo que la
literatura dice que casi nunca consigue.

**Las "victorias" en crisis son exposición, no detección.** En 7 crisis históricas el
bot bate a buy&hold — porque está **halted 73-100 %** (fuera del mercado). Prueba:
China 2015, 87 % halted, **pierde igual** (−7.9 % vs −7.0 % bh). No es que detecte la
crisis; es que está apagado. El sweep lo confirma: todo es un único efecto exposición.

---

## 2. Por qué el vídeo de referencia parece funcionar y tu bot no

Pregunta legítima: si un tutorial muestra un bot de Markov que bate al mercado, ¿es
falso? **Probablemente no falso, pero sí engañoso** — de la forma más común que existe.
El backtest del vídeo casi seguro tiene uno o varios de estos (los mismos que inflaron
tu 52.8 %):

1. **In-sample, sin walk-forward OOS** — ajusta y mide en el mismo periodo. Tu 52.8 %
   era exactamente esto: UN activo, UN periodo alcista.
2. **Sin costes** — cero slippage/comisiones. Mata cualquier estrategia con turnover.
3. **Cherry-pick de activo + periodo** — SPY en mercado alcista. Nadie enseña el ciclo
   2004-24 multi-activo que hiciste tú.
4. **Look-ahead leakage** — Viterbi en vez de forward-filter, scaler ajustado con datos
   futuros. Sutil, invisible en un vídeo.
5. **Beta disfrazado de alfa** — justo lo que tu barrido de floor desenmascaró.

**Incentivo de contenido + sesgo de supervivencia.** "Batí al mercado con Markov + IA"
da clicks; un walk-forward de 20 años con costes no. Ves los 100 vídeos con backtest
bonito, no los 10.000 bots que quebraron en silencio.

**Markov SÍ se usa en fondos reales** — pero para **gestión de riesgo y asignación
consciente de régimen**, como UNA señal entre muchas, con datos propietarios y ejecución
cara. La literatura sobre regime-timing como **alfa retail standalone long-only** es de
mixta a negativa. EMH semifuerte: señal pública + simple + derivada de precio se arbitra
y desaparece. Si un bot nivel-tutorial batiera al mercado robustamente, no sobreviviría
al contacto con el mercado.

**"La IA lo hace funcionar ahora" = marketing.** La IA mejora la *detección de régimen*
(lagging), no la *predicción de retorno* (casi random walk en índices).

**La prueba la tienes en casa.** Reprodujiste el número bonito (52.8 %) **y luego hiciste
lo que el vídeo no hace**: OOS multi-activo, walk-forward, costes, shuffle test. Se
desplomó. Eso no es que tu bot sea peor — es que **tu rigor cazó lo que la falta de
rigor del vídeo esconde**. El del vídeo no es más rentable; está menos validado.

---

## 3. Propuesta de modelo basado en Markov (3 vías)

> **Aviso de método.** No voy a sustituir un sobre-promesa (52.8 %) por otra. Lo que
> sigue son **hipótesis de investigación con su propio riesgo de overfitting**, no la
> cura. Cada una debe pasar el gate de §5 antes de creérsela.

**Recomendación: vía (B) + reorientar el objetivo a riesgo-ajustado.**

### (A) Mismo activo, objetivo riesgo-ajustado
Mantener el universo equity-index pero medir contra **Calmar / Sharpe / MAR**, no
retorno total. *Honestidad:* tu propio holdout ([[2026-06-03-reentry-validation]]) dice
que aquí el margen es **marginal y dentro de ruido** (QQQ +0.02 Sharpe, y QQQ es corr 0.9
con SPY). Techo bajo. No es donde está el potencial.

### (B) RECOMENDADA — cambiar el universo: rotación cross-asset dirigida por régimen
El HMM detecta el régimen de mercado **global** (risk-on / risk-off) y decide **QUÉ
activo** de una cesta diversificada (equities / bonos / oro / cash), no **cuánto** de UNO.
Aquí los modelos de régimen históricamente **ganan su sitio**: el valor viene de la
diversificación + rotación entre activos descorrelacionados, no de cronometrar un solo
índice. *Honestidad:* batir un **60/40 estático** o **risk-parity** OOS sigue siendo
difícil → es una hipótesis a validar, no una victoria asegurada.

### (C) Régimen como overlay sobre una fuente de alfa independiente
El HMM = **gate de riesgo**, no la señal. La señal viene de una fuente con edge probado
(momentum cross-sectional, carry). Solo aporta si la base ya funciona; si no, multiplicas
ruido por ruido.

### Probabilidades de transición — lo pediste, y el código actual las IGNORA
Hoy `StrategyOrchestrator.update_regime_infos` (`core/regime_strategies.py`) **bucketea
por el estado filtrado actual** (rank de vol) y **tira la matriz de transición**.
Propuesta concreta:

- Usar **P(estado_{t+1} | estado_t)** del HMM → tilt media-varianza ponderado por la
  **distribución del próximo estado** y la **duración esperada** del régimen actual
  (1 / (1 − p_ii)).
- *Honestidad imprescindible:* sobre features **precio-vol**, la estructura de transición
  codifica **persistencia de volatilidad, no predicción de retorno** → esto **refina el
  sizing**, no fabrica alfa donde el sweep mostró cero.
- Para que las transiciones predigan **retorno** habría que cambiar las **observaciones**:
  meter **features exógenos macro** con contenido predictivo real — pendiente de la curva
  de tipos, spreads de crédito (HY-IG), term-structure del VIX (contango/backwardation).
  Esos tienen señal forward; los features de precio son casi todos vol autocorrelada.

---

## 4. Estrategia de trading derivada

- **Mantener long-only / no-short.** El short ya falló en walk-forward (V-recovery rápida,
  el HMM llega 2-3 días tarde).
- **Sustituir el halt binario por vol-targeting continuo.** En vez de "apagar" en
  drawdown (lo que originó el permanent-flat trap), escalar el peso para mantener la
  **vol de cartera objetivo constante**: `w_t = vol_target / vol_estimada_t`, capado.
  Es la versión principista, **no latchea**, y degrada con gracia. Sizing tipo
  **fracción-Kelly capada**, nunca Kelly pleno.
- **Vía (B):** peso por activo = f(prob. régimen risk-on, vol esperada del activo), con
  los caps de concentración que ya existen en `validate_signal` (15 %/nombre,
  1 % equity/trade de riesgo, leverage ≤ 1.25x).
- **Coste-realismo OBLIGATORIO antes de cualquier claim:** activar `credit_cash_rf=True`
  (acreditar cash ocioso al risk-free) **y** cobrar financiación del leverage (R-2). El
  confound de no acreditar cash ya falseó un resultado 0/5 → 1/5 una vez
  ([[2026-06-03-reentry-validation]]). Medir net-of-cost o no medir.

---

## 5. Validación y métricas — el rigor actual es un ACTIVO

El walk-forward + look-ahead guards (`assert_no_lookahead`, forward-filter, no Viterbi
live) + shuffle test + cash-credit **es lo que cazó el no-edge**. El sistema funcionó. La
mejora **no es más validación sobre la misma estrategia** — es aplicar ese mismo rigor a
una **señal genuinamente distinta** (vía B/C, features exógenos).

- **Gate pre-registrado, congelado antes de mirar el holdout:** ≥ **4/5 activos
  independientes** (no correlatos de SPY) baten a buy&hold en **riesgo-ajustado**
  (Sharpe mayor **y** maxDD menos negativo). El 1/5 actual no pasa.
- **Deflated Sharpe Ratio (DSR)** — corrige el Sharpe por el número de configuraciones
  probadas. El barrido de floor + cualquier barrido de params **es un multiple-testing
  que infla el falso descubrimiento**; sin DSR, repites la trampa.
- **PBO (Probability of Backtest Overfitting)** y **CPCV (Combinatorial Purged
  Cross-Validation, López de Prado)** — miden directamente cuánto se degrada el ranking
  de configuraciones fuera de muestra.
- **Ampliar el shuffle/permutation test:** n ≫ 3, varias semillas (hoy n=3, contaminado
  por el no-determinismo R-4).
- **Benchmarks de honestidad** (a sumar en `backtest/performance.py`, que ya tiene bh,
  SMA200, random×100): **60/40 estático** y **risk-parity** para la vía (B). Si no bates
  un 60/40 que no piensa, no tienes nada.

---

## 6. Recomendaciones prácticas de implementación

**No habilitar dinero real.** Condiciones que tendrían que ser ciertas para revisarlo:

> ≥ 4/5 activos independientes baten a buy&hold en riesgo-ajustado OOS
> **Y** DSR > 0 **Y** PBO bajo **Y** coste-realismo activado (cash-credit + leverage-cost).

**Orden de trabajo sugerido:**
1. Reorientar objetivo a riesgo-ajustado; activar `credit_cash_rf=True` + leverage-cost.
2. Prototipo vía (B): rotación cross-asset (equities/bonos/oro/cash) con vol-targeting.
3. Gate pre-registrado + DSR / PBO / CPCV; shuffle ampliado.
4. **Solo si pasa** → paper supervisado. Nunca antes.

**Evitar overfitting (resumen operativo):**
- Límite duro de grados de libertad (pocos params, justificados).
- Walk-forward only; **holdout intocado** hasta el final.
- **Pre-registrar el gate** antes de ver resultados.
- DSR / PBO como guardia permanente, no como adorno.
- Mantener **daily** (no 2h) — §0.

---

## TL;DR

El bot no bate al mercado porque **no tiene alfa, solo modula beta** (probado por el
barrido de floor monótono). No es un bug ni un problema de timeframe — bajar a 2h
empeora costes y mata tu validación de 20 años. El vídeo de referencia "funciona"
porque casi seguro es in-sample, sin costes y cherry-picked; tu rigor desenmascaró lo
que el suyo esconde. Camino con sentido: **rotación cross-asset dirigida por régimen +
objetivo riesgo-ajustado + features macro exógenos**, validado con un gate
pre-registrado y DSR/PBO. Tratarlo como **hipótesis**, no como cura. **Dinero real
BLOQUEADO** hasta pasar el gate.
