---
type: analysis
status: active
tags: [regime-trader, optimization, learning, multi-asset, roadmap]
created: 2026-06-02
related: ["[[2026-06-02-why-no-edge]]", "[[improvement-review]]", "[[s1-live-verification]]"]
---

# Optimización, aprendizaje diario y diseño multi-activo (E-1)

Cubre tres preguntas de Pablo: (1) cómo funciona la **retroalimentación diaria y el
aprendizaje**, (2) **cómo funcionaría el multi-activo (E-1)**, (3) qué más cambiar
para optimizar el bot. Estado tras la sesión 2026-06-02 (halt-fix + A-1/A-3/A-4/S-2).

## 1. Cómo funciona la retroalimentación diaria y el aprendizaje

**Cadencia:** launchd lanza `--run-once` L-V 22:15 → un proceso nuevo cada día →
`TradingSystem.run_cycle`:

1. **Arranque:** carga el modelo champion (`hmm_SPY.pkl` / registro A-4). Si está
   viejo por fichero (`_needs_retrain`, 7 d) → reentrena y guarda.
2. **`seed_buffers`** — rellena el buffer con histórico reciente de Alpaca (la barra
   diaria recién cerrada).
3. **`maybe_retrain`** (A-1, **opt-in** `hmm.auto_retrain`) — si el modelo en memoria
   supera `max_age_days`, reentrena desde el buffer y pasa **dos puertas** antes de
   adoptarse (A-4): convergencia + champion-challenger (log-likelihood en holdout ≥
   champion). Si promueve: `update_regime_infos` (re-cablea el orquestador) +
   `registry.save_version`+`promote` (versionado + rollback). Si no: conserva el
   champion + alerta.
4. **`process_symbol`** por símbolo: features → HMM filtrado (forward, sin
   look-ahead) → régimen + estabilidad/flicker → estrategia (alocación por
   vol-rank) → veto de riesgo (`validate_signal`) → orden **delta** (compra/vende
   solo la diferencia al objetivo) con stop bracket.
5. **Riesgo MtM (C2):** `_update_risk_posture` mete la equity (realizada + no
   realizada) en el breaker cada barra → halt si cruza umbrales → liquida (C6).
   Con el **halt-floor** (fix de hoy) un halt mantiene un 25 % mínimo en el backtest
   para no congelarse; en vivo el halt sigue siendo duro (cierra todo).
6. **`update_trailing_stops`** + **`save_state`** (snapshot para dashboard/recuperación).

**Qué "aprende" hoy (automático):**
- Reentrenamiento del HMM (por edad; por **drift** = pendiente de cablear A-3→A-1).
- Selección de nº de regímenes por BIC en cada fit.
- Gate champion-challenger + rollback (A-4).

**Qué NO retroalimenta aún (gaps):**
- El **P&L realizado no ajusta parámetros** (A-5 pendiente): `min_confidence`,
  multiplicadores y umbrales son fijos. No hay bucle "rendimiento por régimen →
  recalibrar".
- Los **detectores de drift (A-3)** existen pero no disparan retrain todavía
  (falta persistir la distribución de features de entrenamiento como referencia).
- `auto_retrain` está **OFF por defecto**; en el path desplegado el refresh diario
  lo da el arranque (`_needs_retrain`), no el lazo.

## 2. E-1 — cómo funcionaría el multi-activo (S&P)

**Hoy:** sleeve de **un símbolo**. El backtester opera solo el primero; el lazo vivo
itera todos pero los caps multi-nombre (`max_single_position` 15 %, `max_concurrent`
5, sector, correlación) **no están validados** (M4/M5), y la correlación está inerte
(sin `price_history`).

**Diseño propuesto (cómo operaría con N activos del S&P):**

- **Régimen de mercado compartido (recomendado):** el HMM detecta el régimen de
  *volatilidad de mercado* sobre un índice (SPY). Ese régimen fija el **presupuesto
  de exposición bruta** de la cartera (p.ej. low-vol→95 %·1.25x, high-vol→60 %). Es
  coherente con la tesis del proyecto (el edge es evitar drawdowns por vol, no
  predecir cada acción) y reutiliza todo lo construido.
- **Selección/peso por activo:** dentro del presupuesto, repartir entre los N
  símbolos. Opciones de peso: (a) **equal-weight** (simple, robusto); (b)
  **vol-parity** (cada activo aporta riesgo similar — pesa menos a los más
  volátiles); (c) **trend-weighted** (sobrepondera los que están sobre su SMA200).
- **Capas de riesgo (ya existen, ahora SÍ activas):** `max_single_position` limita
  cada nombre; `max_concurrent` limita cuántos a la vez; **correlación** (alimentar
  `price_history` con retornos) recorta cuando dos posiciones van demasiado juntas;
  sector cap si se mapea sector.
- **Equity de cartera:** suma de sleeves; el breaker MtM actúa sobre la equity
  agregada (un halt reduce toda la cartera al floor, no símbolo a símbolo).
- **Rebalanceo:** por símbolo, gate de delta (ya implementado en C4) — compra/vende
  solo la diferencia al peso objetivo de cada nombre.

**Día a día con N activos:** cada `--run-once` calcula el régimen de mercado →
presupuesto bruto → pesos objetivo por símbolo (selección + caps) → para cada
símbolo, orden delta hacia su objetivo con su stop. El snapshot/dashboard pasa a
mostrar la cartera.

**Forks de diseño que necesitan tu decisión (cambian la implementación):**
- **Régimen:** compartido de mercado (recomendado, reusa el stack) vs **HMM por
  símbolo** (más potente, N modelos, N× coste y validación).
- **Peso:** equal-weight (recomendado para v1) vs vol-parity vs trend-weighted.
- **Universo:** subconjunto líquido (p.ej. 10-30 nombres) vs S&P 500 completo
  (necesita datos + coste + los caps de concurrencia mandan).

**Importante (del análisis de edge):** multi-activo **diversifica** (baja
volatilidad de cartera) pero **no crea edge** por sí mismo — con el mismo régimen y
de-risking, los nombres van correlacionados. El valor es robustez, no rentabilidad.

## 3. Qué más cambiar para optimizar (priorizado, post halt-fix)

**Validación del edge (lo primero ahora que el halt está arreglado):**
1. **Re-validar fuera de SPY/2019-24.** El 52.8 %/Sharpe 1.22 es **un activo, un
   periodo** (bull). Correr otros periodos (incluir 2008/2015/2018) y otros activos;
   confirmar walk-forward OOS estricto. Sin esto, "tiene edge" sigue siendo
   provisional para dinero real.
2. **Cerrar el hueco vs buy&hold (52.8 % vs 69.9 %)** — ahora como hipótesis a
   testear (ya no contaminadas): diferenciar bear (mantener) vs crash (cortar);
   re-entrada más rápida tras vol; overlay de tendencia (SMA200) como filtro de
   asignación; revisar el floor de 60 % mínimo.

**Aprendizaje (cerrar el bucle):**
3. **Cablear drift→retrain (A-3→A-1):** persistir la distribución de features de
   entrenamiento; disparar retrain cuando PSI/entropía crucen umbral (hoy solo edad).
4. **A-5 — feedback de P&L:** registrar rendimiento realizado por régimen y
   compararlo con el esperado; informe de degradación; recalibrar `min_confidence`.
5. **A-4 follow-up:** gate por **métricas de trading** (Sharpe/maxDD en backtest
   reciente) además del log-likelihood, y **holdout purgado**.

**Realismo / seguridad / operación:**
6. **R-2/R-3:** coste de financiación del leverage + unificar proveedor de datos
   (entrenar desde Alpaca, no yfinance) para que train y live compartan distribución.
7. **Guard de orden abierta** (footgun documentado): impedir doble orden si
   `--run-once` corre 2× antes de llenar.
8. **S-1:** ejecutar la sesión de verificación en vivo (runbook) — desbloquea
   confianza en C2/C5/C6 reales.
9. **Alinear halt backtest vs live:** hoy el backtest usa floor 25 % y live cierra
   todo; decidir si live también debe mantener un floor (cambia el perfil de riesgo
   real) o si el backtest debe reflejar el cierre duro.

**Veredicto de madurez:** mecánica sólida y testeada (172 tests), edge **provisional
y competitivo** (pendiente validación multi-periodo), **NO listo para dinero real**
hasta (1) validar edge fuera de muestra y (2) ≥1 mes de paper limpio + S-1.
