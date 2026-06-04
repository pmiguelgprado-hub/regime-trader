---
type: analysis
status: active
tags: [regime-trader, stock-picking, cross-sectional, momentum, factor, via-c, feasibility, hmm-overlay]
created: 2026-06-04
related: ["[[2026-06-04-markov-edge-redesign]]", "[[2026-06-04-rotation-results]]", "[[2026-06-03-oos-validation]]", "[[2026-06-03-reentry-validation]]"]
---

# Stock-picking dentro del S&P 500 — estudio de viabilidad honesto

> **Punto de partida.** Pablo pregunta si el bot, en vez de hacer *timing* del SP500
> (cuándo comprar/vender el índice), puede decidir **qué empresas** del SP500 comprar y
> vender, para batir al índice. Este documento responde como análisis de viabilidad
> *decision-grade*: ¿merece la pena construirlo, con qué arquitectura, qué datos, y bajo
> qué criterio de aceptación? **No promete batir al mercado** — eso repetiría la trampa de
> sobreajuste que el proyecto ya superó. **Dinero real sigue BLOQUEADO.**

---

## 0. Contexto: por qué esta pregunta llega ahora

El motor actual está **exhaustivamente falsado** como generador de alfa. Tres vías,
tres fracasos OOS limpios (ver docs enlazados):

| Vía | Qué se probó | Resultado OOS |
|---|---|---|
| Timing 1 activo | HMM modula peso en SPY | ✗ floor-sweep monótono = solo beta; 6/6 ETFs pierden vs buy&hold |
| Re-entrada halt | Soltar halt tras K barras calmas | ✗ 1/5 holdout, y ese 1 (QQQ) es eco correlacionado 0.9 con SPY |
| Rotación cross-asset (vía B) | HMM elige activo de cesta | ✗ pierde vs cesta estática EW **y** risk-parity en los 3 periodos; R-4 0.37–0.49 no reproducible |

Conclusión transversal del proyecto: **el HMM es un clasificador de volatilidad, no un
predictor de retorno.** Es bueno detectando *régimen de riesgo* (autocorrelado, real) y
ciego ante el *signo del retorno futuro* (casi random walk en un índice).

El instinto de "picking" de Pablo **no compite** con esto: **es exactamente la vía C** de
[[2026-06-04-markov-edge-redesign]] §3 — *régimen como overlay sobre una fuente de alfa
independiente*. El picking aporta la señal de retorno que el HMM nunca tuvo; el HMM hace
lo único que sabe hacer (gate de riesgo). Por eso esta dirección es la **única no
falsada** que queda, junto con los features macro exógenos.

---

## 1. El problema honesto: alfa transversal ≠ timing

Batir al índice ponderado por capitalización exige **alfa transversal**: predecir qué
acciones rendirán **mejor que otras** (retornos *relativos*). Es un problema
**distinto y más difícil** que el timing de un solo activo, y de los más difíciles de
las finanzas cuantitativas:

- Timing pregunta "¿dentro o fuera del mercado?" → 1 decisión binaria/escalar.
- Picking pregunta "¿cuáles de ~500 nombres, en qué peso, rebalanceando cuándo?" → cientos
  de decisiones acopladas, con su correspondiente explosión de grados de libertad (=
  munición para sobreajustar).

**Declaración sin rodeos:** esto es un **proyecto de investigación nuevo, multi-mes/año**,
no un ajuste del bot actual. Reutilizará infraestructura (backtester, métricas, caps de
cartera, gate), pero la **señal** es nueva y hay que construirla y falsarla desde cero con
el mismo rigor que mató a las tres vías anteriores.

---

## 2. Arquitectura coherente (vía C): picking = alfa, HMM = overlay de riesgo

```
                 ┌─────────────────────────────┐
   precios SP500 │  RANKER TRANSVERSAL (alfa)   │  long top-decil, EW
   (constituyentes)→ momentum / value / quality │ ────────────┐
                 │  → score por nombre, rank    │             │
                 └─────────────────────────────┘             ▼
                                                      ┌──────────────────┐
   SPY (proxy)   ┌─────────────────────────────┐      │  CARTERA final   │
   ───────────── │  HMM (overlay de RIESGO)     │ ───► │  peso_i = rank_i │
                 │  régimen vol global →        │ gross│  × gross_régimen │
                 │  escala EXPOSICIÓN BRUTA     │      │  (caps existentes)│
                 │  (risk-on 100% / risk-off ↓) │      └──────────────────┘
                 └─────────────────────────────┘
```

- El **ranker** decide *qué* y *en qué peso relativo* (la apuesta de alfa).
- El **HMM** decide *cuánta exposición bruta total* según el régimen de vol del mercado —
  sube en risk-on, desescala en risk-off. **No** elige nombres ni predice sus retornos.
- Esto le da al HMM el trabajo que su evidencia respalda (gate de riesgo) y le quita el que
  falló (predicción de retorno). Encaja con todo lo aprendido en el proyecto.

**Importante (lección de la vía B):** el overlay de régimen, por sí solo, **resta** valor
si el coste de desescalar supera lo que ahorra. Por eso el HMM aquí es *secundario y
capado*: si el ranker no tiene alfa, multiplicar por un overlay no la crea. **La base debe
funcionar primero.** El overlay se valida como mejora *incremental* sobre el ranker
desnudo, nunca como la fuente del edge.

---

## 3. Fuentes de alfa documentadas — y el aviso del trader de 30 años

Factores transversales con literatura sólida (históricamente batieron al índice
cap-weighted):

| Factor | Señal | Referencia |
|---|---|---|
| **Momentum** | retorno 12m−1m, comprar ganadores | Jegadeesh & Titman 1993 |
| **Value** | barato por fundamentales (P/B, P/E, EV/EBITDA) | Fama & French 1992 |
| **Quality** | rentabilidad/solidez (ROE, baja deuda, márgenes) | Novy-Marx 2013 |
| **Low-vol** | baja volatilidad → mejor riesgo-ajustado | Frazzini & Pedersen 2014 |

**El aviso honesto (no es barra libre):**
- **Decaimiento post-publicación.** Buena parte de las primas se erosionó tras hacerse
  públicas (~2003+): el mercado las arbitra. El backtest de los 80-90 sobreestima lo que
  cobrarás hoy.
- **Momentum tiene crashes violentos.** 2009: el factor momentum se desplomó (los
  "perdedores" rebotaron). Una estrategia momentum desnuda tiene cola izquierda fea — de
  ahí que el overlay de régimen del HMM *podría* ganarse su sitio amortiguando esos
  crashes (hipótesis a validar, no promesa).
- **Costes te comen el margen retail.** Rebalanceo mensual de ~50 nombres = turnover alto.
  Net-of-cost (slippage + spread + comisión), el edge fino se evapora. Compites contra
  AQR, RenTec, Two Sigma — que tienen datos, ejecución y costes que tú no tienes.
- **No es alfa garantizada; es una hipótesis con base académica.** Sigue necesitando pasar
  el gate (§5) o es otro 52.8% in-sample esperando a ser falsado.

---

## 4. El muro de los datos (decisivo) — y la vía elegida

Backtestear picking **honestamente** sobre histórico exige:

1. **Constituyentes point-in-time del SP500** — qué empresas estaban *en el índice en cada
   fecha pasada*. Sin esto, elegir hoy NVDA/MSFT para un backtest de 2004 es *hindsight*:
   en 2004 habrías comprado GE, Citi, Nokia, Lehman.
2. **Precios libres de sesgo de supervivencia** — incluyendo empresas *deslistadas/quebradas*.
   Usar solo supervivientes infla el retorno: estás midiendo "las que sobrevivieron",
   precisamente el sesgo que [[2026-06-03-oos-validation]] advierte.

**Alpaca NO proporciona nada de esto.** Da precios de los nombres *actuales*. Construir el
backtest con membresía actual reproduce el sesgo de supervivencia letal → cualquier número
sería ficción optimista.

### Dos caminos (Pablo eligió el primero)

| Camino | Cómo | Pros | Contras |
|---|---|---|---|
| **★ Forward paper-only** (ELEGIDO) | Operar los ~500 constituyentes de *hoy* en paper desde el día 1 | OOS limpio desde el minuto 1; **cero** sesgo de supervivencia; **coste 0** de datos; usa Alpaca tal cual | Tarda **meses** en acumular track record; no se puede backtestear el histórico ahora |
| Datos point-in-time de pago | Norgate / Sharadar / CRSP (membresía histórica + deslistados) | Backtest de 15-20 años *ya*; gate aplicable hoy | ~30-150 $/mes; aún así riesgo de overfit del histórico |

**Decisión registrada:** **forward paper-only.** Es el camino intelectualmente más limpio:
el track record que genere es OOS verdadero por construcción, inmune al sesgo de
supervivencia y al *p-hacking* del histórico. El precio es la paciencia (meses). Si en
algún momento se quiere acelerar la validación, se revisita el camino de pago — pero
documentado aquí, no como deuda oculta.

---

## 5. Gate pre-registrado (congelado ANTES de mirar resultados)

Mismo principio que cazó los tres falsos positivos previos: **el criterio se fija antes,
no se mueve después.**

**Benchmarks (universe-aware, los dos):**
- **SP500 equal-weight** (el control que faltó en la vía B: si no bates ni a holdear los
  mismos nombres a peso igual, el picking no aporta).
- **SPY cap-weight** (el índice que Pablo quiere batir).

**Criterio de aceptación:** el picker (con y sin overlay HMM) debe batir a **ambos** en
**riesgo-ajustado** — Sharpe **mayor** Y maxDD **menos negativo** — **neto de costes
realistas**: `credit_cash_rf=True` + slippage por turnover por nombre + comisión. *Medir
net-of-cost o no medir* (el confound del cash-credit ya falseó un 0/5→1/5,
[[2026-06-03-reentry-validation]]).

**Guardias anti-overfit obligatorias:**
- **DSR (Deflated Sharpe Ratio)** — corrige por el nº de factores/params probados. Probar
  4 factores × variantes = multiple-testing que infla el falso descubrimiento.
- **PBO (Probability of Backtest Overfitting)** + **CPCV** (López de Prado) — miden cuánto
  se degrada el ranking de configuraciones fuera de muestra.
- **Overlay HMM = mejora incremental.** Validar el ranker desnudo primero; el overlay solo
  "pasa" si añade Sharpe/recorta maxDD *sobre* el ranker, no como excusa del edge.

**Sin pasar el gate → sin paper supervisado prolongado → sin dinero real. Nunca antes.**

---

## 6. Spec del prototipo mínimo (NO se construye esta sesión)

Diseño de referencia para cuando Pablo dé el "go" (acotado en grados de libertad a
propósito):

- **Universo:** constituyentes actuales del SP500 (lista estática inicial; refrescar
  periódicamente en forward).
- **Rebalanceo:** mensual (equilibra señal vs turnover/coste).
- **Señal:** conjunto de factores **fijo y pequeño** (empezar con momentum 12-1; añadir
  quality solo si se justifica) → score → rank.
- **Construcción:** long-only top-decil **equal-weight** (el short ya falló en walk-forward,
  V-recoveries; no reintroducir). ~50 nombres.
- **Overlay HMM:** escala la exposición bruta total por régimen de vol (proxy SPY), capado.
- **Riesgo:** reutilizar caps de `core/portfolio.py` (max_single 15%, max_concurrent) y la
  lógica de `validate_signal`.
- **Coste/métrica:** reutilizar `backtest/performance.py` (DSR, benchmarks bh/SMA200/random,
  motor de coste emparejado) — ya existe el andamiaje.

Infra reutilizable confirmada: `core/portfolio.py`, `backtest/performance.py`,
`backtest/benchmarks.py`. La señal transversal (ranker) es lo único genuinamente nuevo.

---

## 7. Veredicto

- **¿Es posible?** Sí, conceptualmente — vía C es la arquitectura correcta y la única
  dirección no falsada. El picking le da al HMM el rol que su evidencia respalda.
- **¿Es fácil o garantizado?** No. Es el problema más difícil de las finanzas cuant,
  proyecto nuevo multi-mes, primas decaídas, costes que muerden, competencia institucional.
- **¿Techo realista?** Para un picker retail forward-paper: **mejora riesgo-ajustada
  modesta**, *si* sobrevive al gate — **no** un "destroza-mercados". Cualquier promesa
  mayor es marketing (la misma dinámica EMH / "por qué los vídeos engañan" de
  [[2026-06-04-markov-edge-redesign]] §2).
- **¿Recomendación?** Si Pablo quiere perseguirlo, **adelante con disciplina**: forward
  paper-only, gate pre-registrado, DSR/PBO, base (ranker) antes que overlay. Tratarlo como
  **hipótesis a falsar**, no como cura. El rigor del proyecto es el activo — aplicarlo a
  una señal genuinamente nueva es lo correcto; aflojarlo reproduce el 52.8% fantasma.

**Dinero real BLOQUEADO** hasta pasar el gate. Esta sesión entrega el estudio; la
construcción es una decisión de Pablo, no un hecho consumado.
