---
type: analysis
status: active
tags: [regime-trader, review, trading, hmm, mejoras]
created: 2026-06-02
updated: 2026-06-02
related: ["[[go-live-review]]", "[[2026-06-01-senior-audit]]"]
---

# regime-trader — Revisión de mejoras (5 dimensiones)

Revisión basada en el **código implementado** y en la **conformidad vídeo↔código** ya registrada
en `docs/audit/2026-06-01-senior-audit.md` §8 (el vídeo —"build a fully automated trading bot with
Claude Code", ~34 min— se pasó por `/watch` en una sesión previa). Estado de partida:
**PAPER-READY**, 136 tests, desplegado en paper (launchd diario), **sin edge probado** (backtest
SPY 2019–24: 6.9 % vs 69.9 % buy&hold) → bloqueado para dinero real.

Esta revisión reencuadra y amplía los hallazgos de los audits previos bajo cinco dimensiones de
sistema y rellena la peor cubierta: **aprendizaje continuo** (evolución del modelo). Cada mejora
se acota a algo observable en el código o en §8; **no se proponen funcionalidades inventadas**. Lo
ya reparado en §10 del audit se marca explícitamente para no recomendarlo de nuevo.

---

## Resumen ejecutivo — mayor impacto

| # | Mejora | Dim. | Prioridad | Por qué |
|---|---|---|---|---|
| 1 | **Propagar el reentrenamiento al lazo vivo** (chequeo de retrain in-loop + `update_regime_infos` tras refit) | Aprendizaje | Alta | `_needs_retrain` solo se evalúa en el arranque (`main.py:838/936`): un proceso largo nunca refresca el modelo; y un refit que no re-cablee el orquestador deja el mapa vol-rank obsoleto. |
| 2 | **Gate champion-challenger + registro versionado al reentrenar** | Aprendizaje | Alta | Hoy el modelo reentrenado se usa directo, sin validar que no degrada. Sin rollback. |
| 3 | **Verificar en mercado abierto el feed del breaker + stream de fills** | Seguridad | Alta | C2/C5/C6 reparados y unit-tested, pero nunca ejercitados en una sesión real (riesgo vivo nº1 residual). |
| 4 | **Detección de drift que dispara retrain por evento** (PSI de features, entropía de la posterior, BIC) | Aprendizaje | Alta | El retrain actual solo mira la edad del fichero; el modelo puede degradarse entre ciclos sin que nada lo note. |
| 5 | **Validar caps multi-activo + realismo de fills** | Escalab./Realismo | Alta | El backtest es de un solo símbolo y con slippage plano; sostienen la credibilidad del backtest y el escalado a cartera. |

> **Corrección (verificado contra código):** una versión previa de este resumen listaba como #1
> "alinear etiquetas de régimen entre refits (match húngaro)". **No es un bug en este código** y se
> ha retirado: `HMMEngine._assign_labels` re-deriva las etiquetas por *mean-return* en cada `fit`, y
> `StrategyOrchestrator.update_regime_infos` (`regime_strategies.py:374`) reconstruye el mapa
> régimen→estrategia por *rank de volatilidad* en cada refit; la decisión de alocación
> (`generate_signals:421`) no depende de ningún mapeo posicional persistente, así que una permutación
> de estados no desincroniza nada (el orquestador "ignora las etiquetas" por diseño). El riesgo real
> de esa familia es la **propagación del retrain** (ítem A-1), no el etiquetado.

> **Meta-bloqueo (fuera de estas 5 dimensiones):** la estrategia **no tiene edge probado** (R1 del
> audit). Eso bloquea *dinero real* y es trabajo de *investigación de estrategia*, no de sistema.
> Ninguna mejora de abajo lo resuelve; todas mejoran el vehículo, no la señal.

---

## 1. Realismo

Acercar el sistema a condiciones de mercado reales (datos, slippage, comisiones, latencia).

### R-1 · Modelo de fills simplista — **Prioridad Alta** — ✅ PARCIALMENTE IMPLEMENTADO (2026-06-02)
- **Gap:** el backtester cargaba **5 bps planos** sobre el turnover (`slippage_pct: 0.0005`),
  constante e independiente de la volatilidad, del tamaño de la orden y del régimen. En alta
  volatilidad el slippage real es muy superior → el backtest sobreestimaba el retorno neto.
- **Implementado (componente de volatilidad):** nuevo `Backtester._slippage_rate(base, vol_coeff,
  atr_pct)` = `base + vol_coeff·ATR%`, cableado en el rebalanceo (`backtester.py`) leyendo el ATR
  ya disponible en `sig.metadata["atr"]`. Config `slippage_vol_coeff` (`settings.yaml`,
  **default 0.0 = comportamiento legado**, opt-in para no romper el headline 6.9 %). Tests:
  `tests/test_backtest.py` — rate floor, premio por volatilidad, wiring real, y compat coeff=0.
- **Pendiente (componente de tamaño/ADV):** impacto de mercado proporcional a la participación
  sobre el volumen medio (`impact ≈ k·qty/ADV`). El turnover del backtester es una fracción de
  peso, no acciones; requiere mapear peso→acciones con el volumen de la barra. Follow-up.

### R-2 · Sin comisiones, fees ni coste de financiación — **Prioridad Media**
- **Gap:** equities en Alpaca son sin comisión (correcto omitirla), pero el backtester no modela
  **fees SEC/TAF** en ventas ni —más importante— el **coste de carry del leverage 1.25x**
  (`low_vol_leverage: 1.25`). El régimen low-vol apalancado aparece gratis; en real paga interés de
  margen, lo que erosiona justo el régimen que más contribuye al retorno.
- **Solución:** añadir `financing_cost_annual` al backtester, aplicado diariamente sobre el exceso
  de exposición por encima de 1.0x:
  ```python
  daily_carry = max(0, gross_exposure - 1.0) * (financing_cost_annual / 252)
  equity *= (1 - daily_carry)
  ```

### R-3 · Split de proveedor de datos (yfinance vs Alpaca) — **Prioridad Media**
- **Gap:** el backtest carga histórico de **yfinance** (`data/market_data.py:34 load_ohlcv`); el
  lazo vivo consume **Alpaca** (`MarketData` sobre `AlpacaClient`). Ajustes de dividendos/splits,
  husos y redondeos difieren entre proveedores → el HMM entrena sobre una distribución y opera
  sobre otra ligeramente distinta.
- **Solución:** backfill de entrenamiento desde Alpaca (misma fuente que vive), o documentar y
  testear la equivalencia de barras entre ambos proveedores antes de confiar en el modelo en vivo.

### R-4 · Solo daily; intraday nunca validado — **Prioridad Media**
- **Gap:** `timeframe: 1Day` está **LOCKED** (`settings.yaml`), decisión más coherente que el vídeo
  (que por defecto corría el lazo a 5-min sin re-validar, §8 hallazgo 2). Pero significa que la vía
  intradía es hoy una vía muerta: HMM, regímenes de volatilidad y breakers están todos calibrados
  en diario.
- **Solución:** si se quiere intradía, re-ejecutar la validación completa (walk-forward + stress)
  a 5-min antes de habilitarlo. Mientras no exista, mantener el lock y no presentar intradía como
  capacidad disponible.

### R-5 · Sin riesgo de gap/overnight ni latencia — **Prioridad Baja**
- **Gap:** con barras diarias, la ejecución real ocurre al open siguiente, pero el backtest asume
  fill al precio de la barra de decisión. No se modela el gap open-vs-cierre previo ni el retardo
  de decisión.
- **Solución:** modelar el fill al open del día siguiente (con su gap) en lugar del close de la
  señal; cuantifica el coste de actuar con un día de retardo.

---

## 2. Seguridad

Controles de riesgo, validaciones, gestión de errores y protección del capital.

### S-1 · Feed del breaker MtM y stream de fills sin verificar en mercado abierto — **Prioridad Alta**
- **Gap:** C2/C5/C6 (breaker de drawdown sobre P&L real, suscripción de fills, liquidación en halt)
  están **reparados y unit-tested** (§10 del audit), pero el README advierte que siguen
  *unverified* hasta la primera sesión con mercado abierto: que llegue un fill real por el stream y
  que el breaker reciba equity mark-to-market por barra.
- **Solución:** una sesión paper supervisada con checklist de aceptación:
  1. llega un fill real al stream → 2. se calcula P&L realizado → 3. el `CircuitBreaker` recibe la
  equity por barra → 4. al cruzar `daily_dd_halt: 0.03` / `max_dd_from_peak: 0.10` se dispara el
  halt → 5. `close_all_positions` aplana. Registrar evidencia de cada paso.

### S-2 · Sin reconexión del stream — **Prioridad Alta**
- **Gap (= H5 del audit):** una caída del WebSocket deja al bot **ciego sin avisar**; no hay
  reconexión ni watchdog.
- **Solución:** reconexión con backoff exponencial + **watchdog de heartbeat** sobre el timestamp
  de la última barra; si no llega barra en N minutos de sesión → alerta y modo seguro (no operar).
  ```python
  while running:
      try:
          await stream.run()
      except StreamError:
          await asyncio.sleep(backoff); backoff = min(backoff * 2, 60)
      # watchdog aparte:
      if now() - last_bar_ts > timedelta(minutes=N) and market_open():
          alerts.warn("stream_stale"); enter_safe_mode()
  ```

### S-3 · Chequeo de correlación inerte en vivo — **Prioridad Media**
- **Gap (= M4):** `PortfolioState.price_history` arranca vacío en vivo → el límite de correlación
  se salta en silencio, dando falsa sensación de protección de cartera.
- **Solución:** alimentar series de retornos al tracker para activar el cap, o desactivarlo
  explícitamente y documentarlo (mientras sea single-symbol, irrelevante; al escalar a cartera,
  imprescindible — ver E-1).

### S-4 · Kill-switch externo — **Prioridad Media**
- **Gap:** la única parada es por drawdown interno. Si el lazo se cuelga o se comporta mal, no hay
  forma de detenerlo desde fuera del proceso.
- **Solución:** parada manual fuera de proceso —flag file que el lazo comprueba cada barra, o
  endpoint— que liquide y bloquee independientemente del estado interno.

### S-5 · Reconciliación de posiciones en cada arranque — **Prioridad Baja**
- **Gap:** `state_snapshot.json` no persiste posiciones; la recuperación depende de
  `tracker.sync_on_startup()` reconciliando contra el broker.
- **Solución:** documentar que el broker es la fuente de verdad al reiniciar y **testear el path de
  discrepancia** (estado local ≠ broker) — hoy es el camino crítico de recuperación y no está
  ejercitado.

---

## 3. Escalabilidad

Arquitectura, modularidad y capacidad de ampliar activos, timeframes o estrategias.

### E-1 · Caps multi-activo nunca backtesteados — **Prioridad Alta**
- **Gap (= M5):** el backtester es un *sleeve* de **un único símbolo** (opera el primero de la
  lista). El lazo vivo itera todos los `broker.symbols`, pero los caps de riesgo
  (`max_single_position: 0.15`, `max_concurrent: 5`, sector, correlación) asumen una cartera
  multi-nombre que **nunca se ha backtesteado**. El default actual (`symbols: [SPY]`) es prudente.
- **Solución:** o un backtester multi-activo con asignación a nivel de cartera, o mantener live
  restringido a un símbolo hasta validar. No habilitar el universo ampliado (comentado en
  `settings.yaml`) sin backtest multi-activo.

### E-2 · Features O(n²) por barra — **Prioridad Media**
- **Gap (= H3):** cada barra recomputa el conjunto completo de features sobre todo el buffer. El
  buffer ya está acotado (`tests/test_buffer_bounded.py`), pero el coste por barra crece con la
  ventana.
- **Solución:** cálculo incremental/rolling (z-scores y EMAs actualizables en O(1) por barra) en
  lugar de recomputar la ventana entera.

### E-3 · Una sola estrategia hard-codeada — **Prioridad Media**
- **Gap:** `core/regime_strategies.py` codifica un mapeo fijo régimen-de-volatilidad→alocación. No
  hay forma de añadir o comparar estrategias sin tocar el orquestador.
- **Solución:** interfaz `Strategy` (protocol) con `target_allocation(regime, features) -> weight`,
  seleccionable por config. Habilita A/B (champion-challenger de estrategia) y nuevas estrategias
  sin reescribir el lazo.

### E-4 · Registro de modelos por símbolo/timeframe — **Prioridad Media**
- **Gap:** un único pickle `models/hmm_SPY.pkl`. Para N símbolos × M timeframes no escala y no hay
  trazabilidad de versiones.
- **Solución:** registro versionado `models/<symbol>/<tf>/hmm_<hash>.pkl` + metadata (BIC, fecha,
  ventana de entrenamiento). Es prerequisito de E-1 y de las mejoras A-4/A-2 de la dimensión 5.

### E-5 · Lazo secuencial — **Prioridad Baja**
- **Gap:** `process_symbol` evalúa símbolos en serie.
- **Solución:** con multi-activo, evaluar en paralelo (`asyncio.gather`) respetando
  `max_daily_trades: 20` y el orden de prioridad por señal.

---

## 4. Estabilidad

Robustez, manejo de excepciones, monitorización y recuperación ante fallos.

### T-1 · Cómputo pesado dentro del handler async — **Prioridad Alta**
- **Gap (= H5):** el refit del HMM y el recompute de features corren **dentro del callback de
  barra**; pueden bloquear el event loop y hacer perder barras o fills mientras se procesan.
- **Solución:** mover el trabajo CPU-bound a un executor (`loop.run_in_executor`) o a una cola de
  trabajo, dejando el handler de stream ligero.

### T-2 · Estado durable mínimo — **Prioridad Media**
- **Gap (= M7/#11):** `state_snapshot.json` restaura `equity_peak` y señales recientes, no las
  posiciones abiertas ni el estado del breaker.
- **Solución:** persistir un snapshot completo (posiciones, stops, estado del `CircuitBreaker`) con
  escritura atómica (write-temp + rename), y arranque idempotente que reconcilia contra broker.

### T-3 · Dashboard no es un attach en vivo + fragilidad de API — **Prioridad Media**
- **Gap (= M7):** `monitoring/streamlit_app.py` / `dashboard.py` renderizan `state_snapshot.json`,
  no la instancia viva (sin IPC entre procesos). Además, en esta revisión se reprodujo un fallo
  real de API en el path de refresh vivo: `st.sidebar.caption(...)` llamado dentro de una función
  decorada con `@st.fragment` → `StreamlitAPIException` (`streamlit_app.py:253`).
- **Solución aplicada (este cambio):** usar el context manager en lugar de la forma de atributo:
  ```python
  @st.fragment(run_every=interval)
  def _live():
      render(symbol, toggles)
      with st.sidebar:
          st.caption(f"Last refresh: {datetime.now():%H:%M:%S}")
  ```
- **Solución pendiente:** estado compartido que el dashboard sondee (o métricas push) para reflejar
  la instancia viva, no el último snapshot diario.

### T-4 · Taxonomía de errores + escalado de alertas — **Prioridad Media**
- **Gap:** `monitoring/alerts.py` está rate-limited (bien), pero no clasifica errores (transitorio
  de broker vs fatal vs problema de datos) ni define rutas de escalado.
- **Solución:** clasificar errores y enrutar (warn → halt según severidad); el watchdog de S-2/T-2
  alimenta este canal.

### T-5 · Supervisión de proceso — **Prioridad Baja**
- **Gap:** launchd corre un `runonce` diario; sin auto-restart ante un crash intradía.
- **Solución:** política de reinicio supervisado + alerta al reiniciar.

---

## 5. Aprendizaje continuo (evolución del modelo)

Mecanismos para que el HMM incorpore información nueva, se reentrene y actualice parámetros con
datos recientes. **Dimensión peor cubierta hoy** — el modelo es esencialmente un pickle estático.

### A-1 · Retrain solo al arranque + sin propagar al lazo vivo — **Prioridad Alta** — ✅ IMPLEMENTADO (2026-06-02)
- **Implementado:** `TradingSystem.install_model(new_hmm)` reemplaza el motor **y** llama a
  `orchestrator.update_regime_infos` (cierra el riesgo de mapa obsoleto); `retrain_from_buffer`
  refit sobre el buffer vivo con guarda de datos insuficientes; `maybe_retrain` dispara por edad
  del modelo en memoria (`hmm.max_age_days`), cableado en `run_cycle`. Tests:
  `tests/test_live_retrain.py` (propagación, refit+propagación, guarda, dispara/no-dispara por edad).
- **Pendiente:** disparo por **drift** (A-3 ya provee PSI/entropía); requiere persistir la
  distribución de features de entrenamiento como referencia. Follow-up.

- **Gap:** `main.py:89 _needs_retrain` decide reentrenar comparando el `mtime` del pickle con
  `HMM_MAX_AGE_DAYS` (~7 días), y solo se evalúa en el **startup** del lazo (`main.py:838,936`),
  nunca a mitad de sesión. Un proceso de larga duración (el lazo vivo) **nunca refresca su modelo
  en memoria**, por mucho que envejezca. Y si se añade un retrain in-loop, hay un segundo riesgo:
  el orquestador en vivo mantiene su propio `regime_to_strategy`/`vol_rank` construido al arrancar;
  si el refit no llama a `StrategyOrchestrator.update_regime_infos(...)` (`regime_strategies.py:374`)
  con los nuevos `regime_info`, el mapa **queda obsoleto** (apunta a los estados del modelo viejo).
- **Solución:** job de reentrenamiento programado (launchd/cron) + comprobación periódica dentro
  del lazo, con disparadores por evento además de por edad (ver A-3). **El test crítico** del fix
  es que tras un retrain in-loop el lazo (a) reemplace el `HMMEngine` que usa y (b) **invoque
  `update_regime_infos`** con los nuevos `regime_info` — verificable sin broker.

### A-2 · ~~Alineación de etiquetas entre refits~~ — **Retirado (no es un bug aquí)**
- **Verificado contra código:** `HMMEngine._assign_labels` (`hmm_engine.py:359`) re-deriva las
  etiquetas por *mean-return* en cada `fit`, y `StrategyOrchestrator.update_regime_infos`
  reconstruye el mapa régimen→estrategia por *rank de volatilidad* en cada refit. La alocación
  (`generate_signals:421`) se decide por `regime_state.state_id` contra ese mapa reconstruido, sin
  ningún mapeo posicional persistente. Una permutación de estados entre refits **no desincroniza
  nada** (el orquestador ignora las etiquetas por diseño). Forzar un match húngaro al modelo previo
  sería, en el mejor caso, un no-op para el trading; en el peor, anularía el etiquetado correcto por
  mean-return. **El riesgo real de esta familia vive en A-1 (propagación del retrain).**

### A-3 · Sin detección de drift — **Prioridad Alta**
- **Gap:** no hay vigilancia de drift de datos ni de concepto; el modelo puede degradarse entre
  ciclos de 7 días sin que nada lo detecte.
- **Solución:** monitorizar (a) **PSI** de la distribución de cada feature vs la ventana de
  entrenamiento, (b) **entropía de la posterior** del régimen (incertidumbre creciente), (c) BIC
  del modelo vivo. Si cruzan umbral → disparar retrain fuera de ciclo + alerta.
  ```python
  def psi(expected, actual, bins=10):
      e = np.histogram(expected, bins)[0] / len(expected) + 1e-6
      a = np.histogram(actual,   bins)[0] / len(actual)   + 1e-6
      return float(np.sum((a - e) * np.log(a / e)))
  if psi(train_feat, live_feat) > 0.25:          # 0.25 = drift significativo
      trigger_retrain("feature_drift")
  ```

### A-4 · Sin gate de validación al promover modelo (champion-challenger) — **Prioridad Alta**
- **Gap:** hoy el modelo reentrenado se guarda y se usa directamente (`main.py:701 hmm.save`). No
  hay validación de que el nuevo modelo no degrade, ni rollback al anterior.
- **Solución:** el modelo reentrenado entra como **challenger**; pasa walk-forward + stress sobre
  la ventana reciente y solo **reemplaza al champion** si no degrada métricas clave (Sharpe, max
  DD, agreement de régimen). Si no pasa, se conserva el anterior y se alerta. El registro versionado
  (E-4) habilita el rollback.
  ```python
  challenger = train(recent_window)
  if validate(challenger).sharpe >= champion_metrics.sharpe * 0.95:
      registry.promote(challenger)               # versionado + rollback disponible
  else:
      alerts.warn("challenger_rejected"); keep(champion)
  ```

### A-5 · Sin bucle de feedback de P&L vivo — **Prioridad Media**
- **Gap:** el P&L realizado (una vez S-1 lo cablee de extremo a extremo) no retroalimenta nada: ni
  reajuste de umbrales, ni atribución por régimen, ni revisión de parámetros.
- **Solución:** registrar performance por régimen en vivo y compararla con la esperada del backtest
  → informe periódico de degradación. Es el precursor natural del trabajo sobre el edge (R1).

### A-6 · Umbrales estáticos — **Prioridad Baja**
- **Gap:** `min_confidence: 0.55` y los multiplicadores de tamaño son fijos en `settings.yaml`.
- **Solución:** tras acumular histórico vivo, calibrar el umbral de confianza por su rendimiento
  realizado por régimen (no antes — sin datos vivos sería sobreajuste).

---

## Tabla-resumen priorizada

| Dim. | ID | Mejora | Prioridad | Esfuerzo |
|---|---|---|---|---|
| Aprendizaje | A-1 | Retrain in-loop + propagación (`update_regime_infos`) | Alta | Medio |
| Aprendizaje | A-4 | Champion-challenger + registro versionado | Alta | Alto |
| Seguridad | S-1 | Verificar breaker/fills en mercado abierto | Alta | Bajo |
| Aprendizaje | A-3 | Detección de drift → retrain por evento | Alta | Medio |
| Seguridad | S-2 | Reconexión de stream + watchdog | Alta | Medio |
| Escalab. | E-1 | Validar/backtestear caps multi-activo | Alta | Alto |
| Realismo | R-1 | Modelo de fills realista (vol+tamaño) | Alta | Medio |
| Estab. | T-1 | Cómputo pesado fuera del handler async | Alta | Medio |
| Aprendizaje | A-2 | ~~Alineación de etiquetas~~ — retirado (no-bug) | — | — |
| Realismo | R-2 | Comisiones/fees/coste de financiación | Media | Bajo |
| Realismo | R-3 | Unificar proveedor de datos | Media | Medio |
| Realismo | R-4 | Re-validar antes de habilitar intradía | Media | Alto |
| Seguridad | S-3 | Activar/documentar cap de correlación | Media | Bajo |
| Seguridad | S-4 | Kill-switch externo | Media | Bajo |
| Escalab. | E-2 | Features incrementales (O(1)/barra) | Media | Medio |
| Escalab. | E-3 | Interfaz de estrategia enchufable | Media | Medio |
| Escalab. | E-4 | Registro de modelos versionado | Media | Medio |
| Estab. | T-2 | Estado durable completo + atómico | Media | Medio |
| Estab. | T-3 | Dashboard attach en vivo (fix API ya aplicado) | Media | Medio |
| Estab. | T-4 | Taxonomía de errores + escalado | Media | Medio |
| Aprendizaje | A-5 | Feedback de P&L vivo por régimen | Media | Medio |
| Realismo | R-5 | Riesgo de gap/overnight + latencia | Baja | Bajo |
| Seguridad | S-5 | Test del path de reconciliación | Baja | Bajo |
| Escalab. | E-5 | Lazo paralelo multi-activo | Baja | Bajo |
| Estab. | T-5 | Supervisión de proceso + auto-restart | Baja | Bajo |
| Aprendizaje | A-6 | Calibrar umbrales con histórico vivo | Baja | Bajo |
