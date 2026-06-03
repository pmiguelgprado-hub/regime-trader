---
type: analysis
status: active
tags: [regime-trader, edge, oos-validation, backtest, falsification, reproducibility]
created: 2026-06-03
related: ["[[2026-06-02-why-no-edge]]", "[[2026-06-02-optimization-and-roadmap]]"]
---

# Validación fuera de muestra del edge (multi-activo + multi-periodo)

Continuación directa de [[2026-06-02-why-no-edge]]. Esa sesión arregló el bug del
halt (peso 0 latcheado) y midió **52.8 % / Sharpe 1.22 en SPY 2019-24**, declarando
"edge competitivo **provisional**, pendiente de validar fuera de muestra". Esta
sesión hace esa validación. Herramienta: `backtest/oos_validation.py` +
`scripts/run_oos_matrix.py` + `scripts/shuffle_test.py` (reproducibles).

## TL;DR — el "edge" NO sobrevive fuera de muestra

**El 52.8 %/1.22 era un artefacto de un activo y un periodo (SPY 2019-24, alcista).**
Sobre el ciclo completo (2004-2024) y en 6 ETFs, la estrategia **pierde contra
buy & hold en TODOS**, con Sharpe negativo en 5 de 6. El sweep del halt-floor
demuestra que el retorno **escala monótonamente con la exposición** (cuanto menos
des-riesgas, más ganas): **el des-riesgo del halt es net-negativo en exposición**
(no compra Sharpe, solo recorta retorno). **Dinero real sigue BLOQUEADO** — y ahora
con evidencia robusta, no de una sola medición.

Matiz (test de barajado, **preliminar**): el mapa régimen→estrategia *podría*
llevar algo de señal (en una corrida real batió a 3 barajados), pero la muestra es
pequeña (n=3) y el nivel está contaminado por R-4 (ver abajo); aun en el mejor caso
(halt desactivado) solo **iguala** a buy & hold. Si hay habilidad es pequeña y queda
anulada por el latch del halt en cualquier ajuste desplegable. **No tratar como
establecido** hasta más semillas + redraws (ver Recomendaciones).

## Método (falsación, no confirmación)

- **Walk-forward OOS real** por activo: una corrida larga (máx histórico ~2004-24),
  sin re-runs por crisis (el backtester necesita train 504 + warmup ~450 ≈ 4 años
  por fold). Se **trocea** la serie OOS resultante por ventanas de crisis con
  métricas independientes de la base (`total_return` = compuesto de los retornos
  del tramo; benchmarks recalculados por tramo). Tests en `tests/test_oos_validation.py`.
- **Solo ETFs** (SPY/QQQ/IWM/EFA/EEM/TLT) — nombres sueltos meten sesgo de
  supervivencia.
- **Sweep de `halt_floor_mult` ∈ {0, 0.25, 0.5, 1.0}** en SPY: aísla
  exposición-vs-habilidad. floor=1.0 ≈ halt desactivado (mult sizing = 1.0 en HALTED).
- **Test de barajado** (`Backtester.shuffle_regimes`, ya existía): a floor=1.0,
  mapa régimen→estrategia real vs permutado al azar. real≈barajado ⇒ sin habilidad.
- **Config CONGELADA** en los valores actuales antes de validar. Cero tuning previo
  (ver §Recomendaciones — cualquier número post-tuning sobre estos periodos sería
  in-sample, no OOS).

## Gate previo: el ancla reproduce

SPY 2019-24 `--compare`, dos corridas en el mismo proceso: **52.8352 % / Sharpe
1.216 / bh 69.90 % / sma 40.86 % / halted 0.9 %**, idénticas, y coincide con el
handoff. El HMM siembra `random_state=42`. El número del que partíamos es fiable.

## Resultados

### 1. Amplitud — 6 ETFs, ciclo completo 2004-2024 (floor 0.25 por defecto)

| Activo | Estrategia | Buy & Hold | Sharpe estr. |
|---|---|---|---|
| SPY | 73.3 % | 430.8 % | −0.10 |
| QQQ | 201.5 % | 1027.9 % | 0.23 |
| IWM | 47.4 % | 241.9 % | −0.19 |
| EFA | −9.3 % | 55.0 % | −0.89 |
| EEM | −2.3 % | 18.1 % | −0.50 |
| TLT | 38.7 % | 66.5 % | −0.32 |

**Infra-rendimiento masivo en los 6.** Sharpe negativo en 5/6. Sobre 20 años, SPY
rinde ~2.7 % CAGR (por debajo del 4.5 % sin riesgo → Sharpe negativo) frente al
~8.7 % de buy & hold.

### 2. Sweep del halt-floor (SPY, mismo proceso → comparables)

| floor | Retorno | Sharpe | Max DD | % halted |
|---|---|---|---|---|
| 0.0 | −11.3 % | −2.72 | −11.3 % | 98.5 % |
| 0.25 | 73.3 % | −0.10 | −21.1 % | 53.2 % |
| 0.5 | 180.6 % | 0.18 | −27.3 % | 30.9 % |
| 1.0 | 420.3 % | 0.40 | −40.4 % | 19.5 % |

(buy & hold = 430.8 % en todo el tramo.)

**Retorno Y Sharpe crecen monótonos con el floor.** Cuanto menos des-riesgas, mejor.
floor=1.0 (halt = no-op) ≈ buy & hold. Ojo al alcance: el sweep **mantiene fijo el
mapa de asignación real y solo varía el halt**, así que aísla **el coste de
exposición del halt** — NO dice nada sobre la habilidad del mapa de régimen (eso es
el test de barajado, §4). Lo que prueba: **el des-riesgo por halt resta retorno sin
comprar Sharpe.** Dos destructores de valor **distintos**:

- **floor 0.25 → 1.0: 73 % → 420 %.** Todo ese hueco es el **latch del halt por
  drawdown-de-pico.** Es la catástrofe. Incluso con el "fix" de floor 0.25, el bot
  sigue **halted el 53 % de 20 años** — tras una caída de pico, con solo 25 % de
  exposición el equity recupera lento hacia el pico → sigue >10 % bajo pico años →
  halted años → se pierde la recuperación. El fix `b5eb2fe` se validó solo en
  2019-24 (0.9 % halted, el periodo afortunado sin caída profunda); OOS el latch
  reaparece. **El "fix" era él mismo un artefacto de un periodo.**
- **floor 1.0 (halt off) ≈ bh:** el overlay de régimen *sin el halt* es ~neutral en
  exposición — peso muerto leve, no catástrofe.

### 3. Tramos de crisis — SPY floor 0.25

| Ventana | Estrat. | Buy&Hold | SMA200 | Sharpe | % halted | bate bh |
|---|---|---|---|---|---|---|
| full | 73.3 % | 430.8 % | 319.9 % | −0.10 | 53.2 % | No |
| gfc_2008 | −15.7 % | −37.7 % | −12.8 % | −1.56 | 85.1 % | Sí |
| euro_2011 | −0.4 % | −3.8 % | −14.2 % | −1.29 | 100 % | Sí |
| china_2015 | −7.9 % | −7.0 % | −7.5 % | −2.67 | 86.9 % | No |
| q4_2018 | −10.9 % | −13.5 % | −10.0 % | −3.23 | 27.0 % | Sí |
| covid_2020 | −10.6 % | −13.5 % | −17.2 % | −3.29 | 73.1 % | Sí |
| bear_2022 | −4.4 % | −17.7 % | −12.6 % | −2.39 | 100 % | Sí |

**"Bate a bh en las crisis" NO es habilidad de detección — es estar fuera del
mercado.** En las ventanas donde "protege", está halted 73-100 % del tiempo (euro
y 2022 el 100 %). Es exactamente el efecto de baja-exposición que el sweep prueba
que es net-negativo. **china_2015 lo confirma: 87 % halted y aun así pierde contra
bh** (−7.9 vs −7.0). La protección sigue al % halted, no a la detección. Los dos
hallazgos ("sin edge en ciclo completo" + "protege en crisis") son **un mismo
fenómeno**, no dos.

(En activos más volátiles — IWM/EFA/EEM — bate a bh en las 6 crisis por márgenes
grandes; mismo mecanismo: más tiempo fuera = menos pérdida en la caída, a costa de
perderse toda la recuperación posterior, que es donde se va el ciclo completo.)

### 4. Test de barajado — ¿el mapa de régimen lleva señal? (SPY, floor 1.0, mismo proceso)

| | Retorno | Sharpe |
|---|---|---|
| **real** (vol-rank) | **494.6 %** | **0.45** |
| barajado seed 1 | 259.3 % | 0.25 |
| barajado seed 2 | 223.9 % | 0.22 |
| barajado seed 3 | 356.6 % | 0.33 |

**real > los 3 barajados** dentro de ese proceso (comparación exacta intra-proceso).
*Sugiere* que el mapa régimen→estrategia lleva información, **pero el resultado es
preliminar, no concluyente:**

- **n=3 barajados** (rango 224-357 %, dispersión 133 pp) — muestra pequeña.
- **real=494.6 % fue la lectura MÁS ALTA de floor 1.0 de toda la sesión.** Las otras
  corridas floor 1.0 a config idéntica dieron 420.3 / 421.7 / 432.3 / 441.2 % (ver
  R-4). El proceso del barajado sacó el techo del rango. El margen "~210 pp" mezcla
  un draw real afortunado con una muestra barajada amplia.
- La **estabilidad del margen entre procesos no está medida** (lo tengo una vez).

Lo robusto: incluso el mejor real a floor 1.0 solo **iguala** a buy & hold (~430-494 %).
Si hay habilidad de mapa es pequeña y, en cualquier caso, se la come entera el latch
del halt en cualquier floor desplegable. Para firmarlo: muchas semillas de barajado
+ 2-3 redraws de real entre procesos, comparando el **margen** (no el nivel).

## Hallazgo colateral (R-4): no-determinismo entre procesos en horizontes largos

Las corridas largas (2004-24, 38 folds) **no reproducen entre procesos**: floor 1.0
midió 420.3 / 421.7 / 432.3 / 441.2 / 494.6 % en distintas invocaciones (~±5-15 %).
**No es threading** (single-thread BLAS sigue variando) **ni hash seed**
(`PYTHONHASHSEED=0` sigue variando). Dentro de **un** proceso es determinista (el
gate da bit-idéntico; 2019-24 corto reproduce entre procesos). Causa probable:
no-determinismo FP del entorno (Accelerate/vecLib en macOS, alineación de memoria
vía ASLR) **amplificado por la selección de reinicios** (`if ll > best_ll`) que
voltea el ganador en folds límite a lo largo de 38 folds → camino de régimen
divergente.

**No invalida ningún veredicto:** los huecos vs bh (100-800 pp) y la monotonía del
sweep superan con creces ±15 %, y **cada tabla corrió en un solo proceso** (sweep,
barajado, amplitud), así que las comparaciones internas son exactas. Pero significa
que (a) un número de backtest individual a 20 años lleva ±~5-15 %, y (b) **cada
`--run-once` en vivo es un proceso nuevo → puede seleccionar un modelo distinto.**
Acción: investigar estabilidad de la selección de reinicios (¿desempate
determinista? ¿fijar el mejor modelo por BIC con tolerancia?); pinnear hilos +
`PYTHONHASHSEED` en deploy no basta.

## Veredicto

- **R1 re-actualizado: SIN edge desplegable.** La estrategia pierde contra buy &
  hold en los 6 ETFs sobre el ciclo completo, con Sharpe ≤ bh siempre y negativo en
  5/6. El 52.8 %/1.22 era un artefacto de SPY 2019-24. **Dinero real BLOQUEADO**,
  ahora con base robusta (6 activos × 4 floors × 6 crisis, no una medición).
- **Causa dominante: el latch del halt por drawdown-de-pico** (73 %→420 % al
  soltarlo). El fix de floor de la sesión anterior solo lo mitigó y se validó en el
  único periodo donde no mordía.
- **El overlay de régimen *quizá* tenga algo de habilidad** (en una corrida batió al
  barajado, pero n=3 + contaminado por R-4 → preliminar) y su techo (halt off) solo
  iguala a bh.

## Recomendaciones (orden)

1. **NO tunear sobre estos periodos.** El roadmap item 2 (separar bear/crash,
   overlay SMA, re-entrada) es **tuning**: cualquier número que produzca sobre
   2004-24 sería in-sample. Reservar un holdout (p.ej. 2004-2016 desarrollo /
   2017-2024 validación final) antes de tocar la asignación.
2. **Arreglar el latch del halt como problema nº1**, no el overlay — es la
   catástrofe medida (73→420 %), independiente de si el mapa tiene skill. Mecánica
   candidata: re-habilitar el halt por enfriamiento/normalización de vol en vez de
   "esperar a recuperar el pico con exposición recortada" (que se auto-bloquea).
   TDD: test del requisito (re-entra tras X días normales / vol < umbral), no de un
   umbral cómodo. **Antes de invertir en rediseñar la re-entrada del overlay**,
   confirmar que el mapa de régimen lleva señal de verdad: el barajado de §4 es
   preliminar (n=3, contaminado por R-4) — correr más semillas + redraws de real y
   medir la estabilidad del **margen**. Si el margen no aguanta, el overlay es peso
   muerto y no merece tuning; el techo realista en cualquier caso es *acercarse* a
   bh con menos drawdown, no batirlo.
3. **Resolver R-4** antes de cualquier claim de precisión o de fiar el modelo en
   vivo (cada `--run-once` puede elegir modelo distinto).
4. Replantear la tesis honestamente: "des-riesgar por régimen" recorta drawdown pero
   sacrifica más retorno del que salva en el ciclo. Si el objetivo es Sharpe > bh,
   la evidencia actual no lo respalda; si es *menor drawdown a costa de retorno*,
   decir eso explícitamente y medir contra ese objetivo, no contra retorno bruto.

## Lecciones

- **Validar OOS antes de declarar edge.** El 52.8 % "competitivo provisional" no
  sobrevivió a otro activo ni a otro periodo. La intuición del handoff ("es UN
  activo, UN periodo") era correcta y decisiva.
- **Falsar, no confirmar.** El sweep del floor y el barajado (tests de
  discriminación que el advisor forzó) separaron *exposición* de *habilidad* — sin
  ellos, la matriz habría reproducido "bate a SMA200, < bh" y se habría llamado
  "confirmado provisional". Thorough-looking ≠ decisivo.
- **"Bate a bh en crisis" puede ser estar fuera del mercado, no skill.** Mira el
  %halted antes de atribuir habilidad.
- **Un fix validado en un periodo es un artefacto de ese periodo** (el halt-floor de
  la sesión previa: 0.9 % halted en 2019-24, 53 % OOS).
- **No-determinismo entre procesos** descubierto solo porque dos corridas de
  "misma config" discreparon — verifica que tu medición reproduce antes de comparar
  números entre corridas.
