---
type: handoff
status: active
tags: [regime-trader, closeout, oos-validation, verdict, decision]
created: 2026-06-03
related: ["[[2026-06-03-oos-validation]]", "[[2026-06-03-reentry-validation]]", "[[2026-06-03-halt-reentry-risk-adjusted-design]]"]
---

# Cierre de sesión — validación OOS + fix de re-entrada (2026-06-03)

Cierre de la línea de trabajo "¿tiene edge el bot?". No duplica los análisis; apunta
a ellos. Rama `feat/high-priority-improvements`, 192 tests verdes, árbol limpio,
commits `0785a08..21cc22e`. **Ningún cambio en el path de dinero real** (todas las
mejoras son opt-in con default legacy; el bot desplegado sigue siendo SPY-only,
hard-halt, sin tocar).

## El arco de la sesión (qué se aprendió)

1. **El "edge" del handoff anterior era un artefacto.** El 52.8 %/Sharpe 1.22 (SPY
   2019-24) no sobrevivió fuera de muestra. Validación en 6 ETFs 2004-24: pierde contra
   buy&hold en los 6, Sharpe negativo en 5/6. El sweep del halt-floor mostró retorno
   monótono con la exposición = "pura exposición, no skill". Detalle:
   [[2026-06-03-oos-validation]].
2. **Reformulación honesta del objetivo** (imposible batir bh en retorno bruto
   des-riesgando): pasar a **Sharpe > bh Y maxDD < bh** en holdout, tuneando solo en SPY.
   Spec: [[2026-06-03-halt-reentry-risk-adjusted-design]].
3. **Se arregló el destructor nº1 medido** (el latch del halt): re-entrada por
   normalización de vol (K barras calmas) en vez de recuperar pico. Funciona mecánicamente
   (SPY Sharpe −0.16→0.40, halted 54.7 %→4.4 %; drawdown recortado vs bh en 5/5).
4. **El holdout falló el gate.** Con cash acreditado a rf (confound de medición que cazó
   el advisor): **1/5**, y ese 1 (QQQ) es el correlato del activo de tuning (SPY ~0.9),
   dentro del ruido R-4; los 4 activos independientes (IWM/EFA/EEM/TLT) fallan → **sin
   skill riesgo-ajustado generalizable**. Detalle: [[2026-06-03-reentry-validation]].
5. **Hallazgo colateral R-4:** no-determinismo entre procesos en horizontes largos
   (no threading, no hash; argmax de selección de reinicios sobre 38 folds amplifica
   ruido FP del entorno). Cada `--run-once` vivo puede elegir modelo distinto.

## Veredicto vs el objetivo del vídeo de referencia

**El bot NO cumple el objetivo implícito del vídeo** ("bot de regímenes que genera
ganancias / protege en caídas y bate al mercado"). Razones, con evidencia:

- **El vídeo nunca demostró una ventaja medida.** Enseña la maquinaria (detección de
  régimen, dashboard, equity que sube en un mercado alcista) — pero subir en un toro no
  es batir a buy&hold. Nuestro backtest a 6 activos / 20 años es la prueba rigurosa que
  el vídeo nunca hizo, y el resultado es: no hay edge.
- **¿Genera ganancias?** En un mercado alcista sí sube (está largo), pero **menos que
  tener el índice** — des-riesga y se pierde upside. No "corrige pérdidas" mejor de lo
  que las corrige simplemente tener menos exposición o un fondo índice.
- **¿Protege en caídas?** Sí, recorta drawdown (5/5 vs bh), pero el coste en retorno
  supera el ahorro de riesgo → Sharpe peor en 4/5. La protección es "estar fuera del
  mercado", no habilidad de detección (en las crisis donde "gana" está 73-100 % halted;
  china_2015: 87 % halted y aun así pierde).

**Conclusión: no está "todo correcto".** Como herramienta para *batir o seguir al
mercado*, un fondo índice es superior y más barato. Dinero real BLOQUEADO. Tiene valor
como proyecto de aprendizaje (infra sólida, 192 tests, pipeline honesto) — no como
generador de alpha.

## Sobre extenderlo a acciones individuales (Alphabet/Amazon/MSFT/Nvidia…)

**No es óptimo — no hacerlo.** Razones:

- Es un **problema distinto y más difícil**: el bot hace *timing de régimen de
  volatilidad sobre un activo*, no *selección transversal* (qué acción comprar/vender
  para batir al índice = predecir retornos relativos = alpha, lo más difícil de
  finanzas). El HMM no tiene esa maquinaria.
- **Sesgo de supervivencia letal:** elegir hoy NVDA/MSFT para un backtest histórico es
  hindsight (en 2004 habrías elegido GE/Citi/Nokia). Haría falta membresía point-in-time
  del índice.
- **SpaceX no cotiza** → no invertible.
- Y construir selección sobre un motor **sin edge probado** = construir sobre arena.

Si algún día se persigue, sería un **proyecto nuevo** (ranker momentum/factor con datos
point-in-time), no una extensión de este bot.

## Estado del código (sin tocar el path real)

- **Mejoras añadidas, todas opt-in, default legacy** (el bot desplegado no cambia):
  `peak_reentry_calm_bars` (re-entrada del halt; default 0 = legacy), `credit_cash_rf`
  (cash a rf para Sharpe justo; default False), `SliceMetrics` bh_sharpe/bh_mdd.
- **Herramienta de validación reproducible:** `backtest/oos_validation.py` (+ tests),
  `scripts/{run_oos_matrix,shuffle_test,tune_reentry,validate_reentry,validate_reentry_cashcredit}.py`.
- **El bot en vivo:** sigue SPY-only, `--run-once` diario (launchd L-V 22:15), hard-halt.
  Los otros ETFs fueron **solo backtest** (datos offline yfinance) — nunca operados,
  nunca en cartera. El dashboard muestra SPY porque es lo único que opera.

## Abierto (no bloqueante; decisiones de Pablo)

- **¿Parar esta línea?** Recomendado: sí, como búsqueda de alpha. Documentado.
- Si se quiere seguir con objetivo "solo-reducción-de-drawdown" (no batir): definir
  métrica honesta y validar con **holdout temporal** (los 5 ETFs ya están quemados).
- Push a GitHub (sin `gh`/creds — necesita Pablo) si se quiere respaldar la rama.
- S-1 (sesión live supervisada) y demás roadmap: irrelevantes si no se fondea.
- R-2/R-3/R-4 documentados como deudas si el proyecto se reactiva.

## Lecciones de la sesión (ya en memoria)

- Validar OOS antes de declarar edge; un fix validado en un periodo es artefacto de
  ese periodo.
- Falsar, no confirmar (sweep floor + barajado + holdout = tests de discriminación).
- "Bate a bh en crisis" puede ser estar fuera del mercado, no skill — mira el %halted.
- Verifica la métrica antes del claim fuerte (el cash-credit: 0/5→1/5).
- El holdout solo cuenta si el activo es independiente del de tuning (QQQ≈SPY no suma).
- advisor antes de declarar done (cazó: contaminación del edge, overclaim del barajado,
  confound del cash-credit, el correlato QQQ-SPY). Ver [[feedback-test-the-requirement]].
