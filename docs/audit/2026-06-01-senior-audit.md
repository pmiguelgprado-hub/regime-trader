# regime-trader — Auditoría técnica senior (paper-trading readiness)

_Fecha: 2026-06-01. Revisor: auditor técnico senior (sistemas de trading algorítmico). Fuente primaria: el código real del repo. Tests: 102 passed (py3.14). No se modificó nada del sistema; este documento es aditivo._

> **✅ ACTUALIZACIÓN 2026-06-02 — REPARADO, AHORA PAPER-READY.** C1–C6 + H2/M1/M3/M6 + H3/H4 cableados y testeados (102→132 tests); dashboard Streamlit; verificado contra cuenta Alpaca **paper** real (conexión + cuenta + entitlement datos H6 + round-trip orden submit/cancel sin posición). Detalle en **§10**. El veredicto original (abajo) queda como registro histórico del estado pre-reparación. **Paper-ready ≠ rentable:** R1 (sin edge) sigue bloqueando dinero real; correr paper ≥1 mes primero.
>
> **Veredicto ORIGINAL (2026-06-01, pre-reparación): NO APTO PARA PAPER TRADING en su estado actual.**
> El backtest, la capa de riesgo unitaria y los adaptadores de broker (contra un SDK mockeado) están bien construidos. Pero el **lazo en vivo (`main.py` → `run_live` → `run_stream`) está estructuralmente incompleto**: arranca con buffer frío, no coloca stops, no rebalancea (solo acumula), y **los circuit breakers de drawdown —la característica de seguridad de cabecera— no tienen ningún caller en vivo, así que no funcionan**. Esto último va más allá de lo que reconoce el propio `go-live-review.md`.

El proyecto incluye un `docs/go-live-review.md` honesto y de alta calidad escrito por el autor. Esta auditoría lo verifica de forma independiente contra el código, **confirma** sus hallazgos #1–#3, y añade tres bloqueos críticos que su premisa daba por funcionando (C1, C2, C5) y corrige la premisa de uno (C3 vs su #4).

---

## 1. Resumen ejecutivo

1. **El lazo en vivo nunca rellena el buffer con histórico (C1).** `run_live` (`main.py:502`) construye el sistema y llama a `run_stream`, pero `system.buffers` arranca vacío. `build_features` necesita ~450 barras de warmup (z-score 252 + SMA200) → en vivo **no emite ninguna señal durante ~450 barras** (con `timeframe: 1Day` eso son ~2 años). El bot sería un no-op silencioso al arrancar.
2. **Los circuit breakers de drawdown están inertes en vivo (C2).** `update_drawdown_state` solo lo llama el backtester (`backtester.py:199`); `CircuitBreaker.update` solo lo llama `apply_fill_to_risk`, que **no tiene ningún caller**. En vivo, `RiskManager.state` se queda en `NORMAL` para siempre y el breaker nunca recibe P&L. Los halts diario/semanal/peak —lo que vende el README como protección— **no se ejecutan**.
3. **No se coloca ningún stop en vivo (C3).** `process_symbol` ejecuta vía `submit_signal` (orden límite simple), **nunca** `submit_bracket_order`. El `stop_loss` que calcula la estrategia es puramente informativo. `update_trailing_stops` es no-op porque `stop_order_id` nunca se rellena. Posiciones sin protección de stop.
4. **El lazo en vivo solo acumula; no rebalancea ni reduce (C4, = go-live #3).** `submit_signal` compra `approved_shares` **cada barra** sin mirar la posición actual y sin rama de venta. En barras repetidas del mismo signo, sobre-acumula hasta el techo de exposición; y no puede reducir cuando baja el target.
5. **El stream de fills nunca se suscribe (C5).** `subscribe_fills` está definido pero **no se llama** en `run_live`. Sin él: no hay P&L realizado en vivo, no se etiqueta régimen en los fills, `advance_bar`/holding-period no avanza, y —encadenado con C2— el breaker nunca se alimenta.
6. **Nada liquida en un halt (C6).** `close_all_positions` está definido pero sin caller; y la estrategia es *always-long* (alocación mínima 0.60/0.30, nunca 0) → ni siquiera por régimen puede aplanar. Solo el backtester sabe salir a 0 (`must_exit`).
7. **La estrategia no tiene edge probado (research).** Backtest SPY 2019–2024: **6.9% vs 69.9% buy-and-hold**, y pierde contra alocación aleatoria. La metodología del backtest es correcta (walk-forward sin leakage, verificado), así que el resultado es de fiar: esto **bloquea dinero real**, no paper. Paper es gratis y es donde se cazan C1–C6.
8. **La documentación del CLI no arranca como está escrita (H2).** README muestra `python main.py backtest --symbols ...` y `python main.py --mode backtest`, pero el parser real usa flags `--backtest` (`main.py:697`). Ninguna de las dos formas documentadas en "Usage"/"Backtest CLI" funciona.
9. **Lo que SÍ está bien:** feature engineering estrictamente causal (forward algorithm en HMM, z-scores trailing, guard `assert_no_lookahead`); `validate_signal` con veto real y bien testeado; backtester walk-forward con refit por fold sin leakage; adaptadores de broker con parsing/backoff/reconcile testeados contra mock. La arquitectura por capas es limpia. El problema no es la calidad de las piezas: es que **el cableado del lazo en vivo está a medias** y las piezas de seguridad no están conectadas.
10. **Bloqueos principales para paper:** C1 (buffer frío), C3 (sin stop), C4 (sobre-acumulación), C2+C5+C6 (seguridad de drawdown inerte + sin liquidación). Hasta resolver C1, C3, C4 y la tripleta de seguridad, **paper trading sería peligroso e inútil a la vez** (no opera al principio; luego opera sin red).

---

## 2. Mapa del proyecto

### Módulos
| Capa | Archivo | Estado |
|---|---|---|
| Entry/orquestación | `main.py` (`TradingSystem`, modos CLI) | Backtest/dry-run/train OK. **Live a medias.** |
| Regime engine | `core/hmm_engine.py` | Sólido. Forward filtering, BIC, save/load. |
| Estrategia | `core/regime_strategies.py` | Coherente. Always-long por vol-rank. |
| Riesgo | `core/risk_manager.py` | `validate_signal` sólido; breakers **no cableados en vivo**. |
| Señales | `core/signal_generator.py` | **Esqueleto muerto** ("no logic implemented yet"). |
| Features | `data/feature_engineering.py` | Causal, correcto. |
| Datos | `data/market_data.py` | yfinance (BT) + stream Alpaca (live, sin cobertura). |
| Broker | `broker/alpaca_client.py`, `order_executor.py`, `position_tracker.py` | Unidades OK contra mock; varias rutas sin caller en vivo. |
| Backtest | `backtest/backtester.py`, `performance.py`, `stress_test.py` | Bien. Walk-forward, allocation-based, single-asset. |
| Monitoring | `monitoring/logger.py`, `alerts.py`, `dashboard.py` | Unidades OK. Dashboard no es attach en vivo. |

### Flujo que *debería* ser, extremo a extremo (live)
```
arranque: connect → get_account → (backfill histórico al buffer) → train/load HMM
          → sync_on_startup → subscribe_fills(stream)
por barra: ingest_bar → build_features → predict_regime_filtered (filtered)
          → flicker/stability → generate_signals (target alloc)
          → [calcular delta vs posición actual] → validate_signal (veto/sizing)
          → si |delta|≥umbral: submit (BRACKET con stop) buy si delta>0 / sell si delta<0
          → update_trailing_stops
por fill: on_fill → realized P&L → breaker.update → si HALT: close_all_positions
rollover: reset_daily/weekly de breaker + contadores
```

### Lo que el código hace HOY (live)
```
arranque: connect → get_account → train/load HMM → sync_on_startup
          → run_stream   (NO backfill, NO subscribe_fills)
por barra: ingest_bar → build_features (vacío ~450 barras) → ... 
          → generate_signals → validate_signal
          → submit_signal (límite simple, SIN stop, compra full cada barra)
por fill: (nadie escucha)  → breaker nunca se alimenta → state siempre NORMAL
halt: imposible (breaker inerte); aunque ocurriera, nadie liquida
```

### Inconsistencias entre módulos
- **Backtest vs live divergen en la mecánica de trading.** El backtester es *allocation/delta-based* con salida a 0 (`backtester.py:212-230`); el live es *acumulación bruta* sin delta ni venta. El backtest **no representa** el comportamiento del live → cualquier resultado de backtest es inválido como predicción del paper.
- **`signal_generator.py`** define `SignalGenerator`/`TradeSignal` que el pipeline real (`StrategyOrchestrator` + `validate_signal`) no usa. Parece requerido y no lo es.
- **Riesgo en vivo desconectado de su propio motor:** `validate_signal` consulta `breaker.check()` y `circuit_breaker_status`, pero nada actualiza esos estados en vivo.
- **CLI documentado ≠ CLI real** (H2).
- **`requirements.txt`** lista `alpaca-trade-api` (legacy, deprecado) *y* `alpaca-py`; el código usa `alpaca-py`. Dependencia muerta + versiones sin pinear.

---

## 3. Hallazgos

### C1 — Buffer en vivo nunca se rellena con histórico
- **Severidad:** CRÍTICA · **Módulo:** `main.py` (`run_live`/`run_stream`)
- **Problema:** El buffer rodante arranca vacío y solo crece barra a barra desde el stream; `build_features` devuelve vacío hasta acumular ~450 barras de warmup.
- **Evidencia:** `run_live` (`main.py:502-579`) no precarga `system.buffers`. `run_stream.on_bar` (`main.py:310-318`) hace `ingest_bar` de una fila. `build_features` dropea warmup (z-score 252 + SMA200; `feature_engineering.py:306-322`). Compárese con `run_dry_run`, que sí siembra `sys_.buffers[symbol] = ohlcv.iloc[:i+1]` (`main.py:466`).
- **Riesgo práctico:** Con `1Day`, el bot no emite señales durante ~2 años. No-op silencioso; el operador cree que "funciona".
- **Cómo arreglarlo:** En `run_live`, antes de `run_stream`, hacer `md.get_history(symbol, lookback_bars=N)` (con N ≥ `zscore_window`+200, p.ej. 500) y `system.ingest_bar`/sembrar el buffer por símbolo. Reusar `data.market_data.load_ohlcv`/`get_history`.
- **Validación:** Test que arranca el sistema con buffer sembrado de 500 barras y verifica que la **primera** barra en vivo ya produce features no vacíos y un signal. Dry-run que confirme `process_symbol` devuelve señal en la barra 1.

### C2 — Circuit breakers de drawdown inertes en vivo
- **Severidad:** CRÍTICA · **Módulo:** `core/risk_manager.py` + `main.py`
- **Problema:** En vivo nunca se actualiza la posture de riesgo ni el breaker latching. `RiskManager.state` queda `NORMAL` siempre; los halts diario/semanal/peak no pueden dispararse.
- **Evidencia:** `grep update_drawdown_state` → único caller `backtester.py:199` (no hay caller en `main.py`). `CircuitBreaker.update` → único caller `position_tracker.apply_fill_to_risk:177`, que **no tiene caller**. En `process_symbol`, `circuit_breaker_status` se rellena con `self.risk.state` (`main.py:264`), que nunca cambia en vivo.
- **Riesgo práctico:** La protección frente a drawdowns/gaps —el argumento de seguridad del README— **no existe en ejecución**. El bot seguiría operando a través de un -10% sin reducir ni parar.
- **Cómo arreglarlo:** (a) Cablear fills → `apply_fill_to_risk(fill, portfolio_state, risk.breaker)` en el handler de fills (ver C5). (b) En cada rollover / barra, recomputar drawdown desde equity del broker (`tracker.refresh().equity`) y/o llamar a `update_drawdown_state`. (c) En `validate_signal`/lazo, respetar `breaker.check()` y, en HALT, no abrir y liquidar (C6). Configurar `risk.lock_file` en vivo (hoy `null` → lock persistente deshabilitado, `settings.yaml`).
- **Validación:** Test de integración del lazo en vivo simulado (sin broker real) que inyecta una serie de fills perdedores y verifica transición `NORMAL→REDUCED→HALTED`, que `validate_signal` empieza a rechazar, y que se dispara liquidación.

### C3 — No se coloca ningún stop protector en vivo
- **Severidad:** CRÍTICA · **Módulo:** `broker/order_executor.py` + `main.py`
- **Problema:** El live entra con orden límite simple sin stop adjunto. El stop calculado por la estrategia nunca llega al broker.
- **Evidencia:** `process_symbol` → `self.executor.submit_signal(...)` (`main.py:220`). `submit_signal` → `submit_order(... OrderType.LIMIT ...)` sin stop (`order_executor.py:132-177`). `submit_bracket_order` (que sí adjunta `StopLossRequest`) **no tiene caller** fuera de tests. `update_trailing_stops` requiere `pos.stop_order_id`, que nunca se rellena (`main.py:247`).
- **Nota sobre el go-live #4:** el doc del autor propone "tras `submit_bracket_order`, capturar el id de la pata de stop". Pero `submit_bracket_order` **no se ejecuta nunca**; la premisa del fix es falsa. El problema raíz no es "los trailing stops no disparan", es que **no hay ningún stop colocado**.
- **Riesgo práctico:** Posiciones completamente expuestas a un gap/crash. Sin red dura.
- **Cómo arreglarlo:** En el envío en vivo usar `submit_bracket_order` (ya valida `stop_loss>0`). Tras el fill, recuperar las patas hijas (`get_order_history`/legs) y guardar el id del stop en `Position.stop_order_id` para habilitar `modify_stop`/trailing.
- **Validación:** Test (mock SDK) que verifica que una señal aprobada genera una `MarketOrderRequest` con `order_class=BRACKET` y `stop_price` correcto; y que `stop_order_id` queda poblado y `update_trailing_stops` puede tighten.

### C4 — Sobre-acumulación: sin gate de delta/rebalance ni rama de venta (= go-live #3)
- **Severidad:** CRÍTICA · **Módulo:** `main.py` (`process_symbol`) + `core/regime_strategies.py`
- **Problema:** `generate_signals` emite una alocación-objetivo completa cada barra; `submit_signal` compra `approved_shares` cada vez sin comparar con la posición actual. No hay venta para reducir.
- **Evidencia:** `process_symbol` (`main.py:212-225`) no calcula delta vs `tracker`. `position_size` dimensiona por *peso objetivo* (hasta 15% concentración) **por orden, ignorando lo ya tenido** (`risk_manager.py:403-411`). El backtester sí hace `delta = target - held_weight` y vende/sale (`backtester.py:212-230`).
- **Riesgo práctico:** En barras repetidas del mismo régimen, acumula ~15% por barra hasta que `check_exposure` recorta cerca del techo (~1.0x equity, ver H4) — varias veces la posición pretendida (~13–15%). Y no reduce cuando el target baja de 0.95→0.60. Comportamiento opuesto al backtest.
- **Cómo arreglarlo:** Portar la lógica del backtester: calcular `current_weight` desde `tracker`, `delta = target_weight − current_weight`, operar solo si `|delta| ≥ rebalance_threshold` o `must_exit`; `delta>0` → buy, `delta<0` → **sell**. Dimensionar la orden sobre `delta`, no sobre el target absoluto.
- **Validación:** Test que alimenta el **mismo** signal 5 barras seguidas con `tracker` no-vacío y verifica que tras alcanzar el target no se emiten más compras; y que al bajar el target se emite una venta del tamaño correcto. (El dry-run actual no lo caza porque usa `positions=[]` fresco cada barra — ese es justo el punto ciego.)

### C5 — El stream de fills nunca se suscribe en vivo
- **Severidad:** CRÍTICA · **Módulo:** `broker/position_tracker.py` + `main.py`
- **Problema:** `run_live` no arranca `subscribe_fills`, así que no entran eventos de fill: ni P&L realizado, ni etiqueta de régimen, ni alimentación del breaker (C2).
- **Evidencia:** `grep subscribe_fills` → solo definición (`position_tracker.py:240`); ningún caller en `main.py`. El tracker en vivo solo se actualiza por polling en `_portfolio_state` (`main.py:256-258` `tracker.refresh()`).
- **Riesgo práctico:** Estado de posiciones/P&L reconstruido solo por polling; sin disparo de breaker; `advance_bar` (holding-period) nunca avanza.
- **Cómo arreglarlo:** En `run_live`, arrancar el stream de trade-updates (en hilo/loop async) con `regime_provider=lambda: system.last_regime...` y enrutar a `apply_fill_to_risk` (para cerrar C2). Manejar reconexión.
- **Validación:** Test del handler `on_fill`/`apply_fill_to_risk` (ya unitable) + un test de integración que confirme que un fill perdedor mueve el breaker.

### C6 — Nada liquida en un halt; la estrategia no puede aplanar
- **Severidad:** CRÍTICA (encadenada con C2) · **Módulo:** `broker/order_executor.py` + `core/regime_strategies.py`
- **Problema:** Aunque el breaker llegara a HALT, ningún código llama a `close_all_positions`. Además la estrategia es always-long con alocación mínima 0.60 (high-vol) / 0.30 efectivo bajo incertidumbre → nunca emite target 0; solo `target_size_multiplier()=0` (vía HALT, inerte en vivo) produciría salida.
- **Evidencia:** `grep close_all_positions` → solo definición/uso interno, sin caller del lazo. Alocaciones mínimas en `regime_strategies.py` (0.60 high-vol; `_apply_uncertainty` halve). `must_exit` solo existe en el backtester.
- **Riesgo práctico:** No hay mecanismo de "salir todo" ante un evento; el sistema solo sabe estar largo.
- **Cómo arreglarlo:** En el lazo, si `breaker.state is HALTED` (tras C2) → `executor.close_all_positions()` + dejar de abrir. Considerar un target 0 explícito en defensivo extremo.
- **Validación:** Test que fuerza HALT y verifica llamada a `close_all_positions` y cese de nuevas órdenes.

### R1 — La estrategia no tiene edge probado (research, gateo de dinero real)
- **Severidad:** ALTA (para dinero real; no bloquea *paper*) · **Módulo:** estrategia/backtest
- **Problema:** Rinde 6.9% vs 69.9% buy-and-hold (SPY 2019–2024) y pierde contra aleatorio.
- **Evidencia:** `docs/go-live-review.md` #1; metodología de backtest verificada como correcta (walk-forward, refit por fold `backtester.py:171`, filtered inference, sin fit-on-full → resultado fiable). Artefactos en `backtest_output/SPY/`.
- **Riesgo práctico:** Aunque se arregle todo lo operativo, no hay razón para financiarla.
- **Cómo arreglarlo:** Re-scopear a research, o iterar la tesis (de-riskea en vol y se pierde los rebotes). No es un fix de código.
- **Validación:** Backtest con benchmarks (`--compare`) batiendo buy-and-hold ajustado por riesgo de forma robusta antes de pensar en capital real.

### H2 — CLI documentado no ejecutable
- **Severidad:** ALTA (trazabilidad/operación) · **Módulo:** `README.md`
- **Problema:** README "Usage" usa `--mode backtest`/`--mode live` y "Backtest CLI" usa `python main.py backtest ...`; el parser real son flags `--backtest`, `--live`, etc.
- **Evidencia:** `parse_args` (`main.py:694-713`) define `--backtest/--stress-test/--train-only/--dry-run/--dashboard/--live`. No existe `--mode` ni subcomando posicional.
- **Cómo arreglarlo:** Corregir README a los flags reales (el `go-live-review.md` ya los lista bien).
- **Validación:** Copiar/pegar cada comando del README y que arranque.

### H3 — Buffer no acotado + recomputo O(n²) de features por barra
- **Severidad:** ALTA (robustez operativa) · **Módulo:** `main.py`/`feature_engineering`
- **Problema:** `ingest_bar` hace `pd.concat` indefinido; `process_symbol` recomputa `build_features` sobre **todo** el buffer cada barra. Coste y memoria crecen sin límite en una sesión larga.
- **Evidencia:** `ingest_bar` (`main.py:167-175`); `build_features(buf)` (`main.py:189`).
- **Cómo arreglarlo:** Recortar el buffer a una ventana rodante (p.ej. `tail(zscore_window+250)`) tras cada ingest; o computar features incrementalmente.
- **Validación:** Test que tras 10.000 barras el buffer está acotado y el tiempo por barra es ~constante.

### H4 — La leverage configurada es inalcanzable (cap efectivo a 1.0x)
- **Severidad:** ALTA (lógica) · **Módulo:** `core/risk_manager.py`
- **Problema:** `check_exposure` rechaza si `leverage > max_exposure*max_leverage` = 0.80×1.25 = **1.0**. El `max_leverage` 1.25 nunca llega a vincular; `low_vol_leverage: 1.25` es código muerto en la práctica. La fórmula `max_exposure*max_leverage` es dimensionalmente rara (mezcla fracción de exposición con múltiplo de leverage).
- **Evidencia:** `check_exposure` (`risk_manager.py:436-441`).
- **Cómo arreglarlo:** Definir explícitamente el techo de exposición bruta (¿`max_exposure` como fracción? ¿`max_leverage` como múltiplo?) y no su producto. Decidir si 1.25x es alcanzable o eliminar la config de leverage.
- **Validación:** Test que verifica que con leverage 1.25 y exposición pretendida, el cap aplicado coincide con la intención de diseño documentada.

### H5 — Cómputo pesado dentro del handler async + sin reconexión
- **Severidad:** ALTA (robustez en vivo) · **Módulo:** `main.py`/`data/market_data.py`
- **Problema:** El callback de barra ejecuta todo el pipeline (incl. inferencia HMM) síncronamente dentro del handler async del WebSocket → bloquea el event loop. No hay lógica de reconexión ni manejo de desconexión del stream.
- **Evidencia:** `subscribe_bars._on_bar` (`market_data.py:244-250`) llama `callback(...)` que es `run_stream.on_bar` (`main.py:310-318`) → `process_symbol` (inferencia). `stream.run()` bloqueante; sin try/reconnect.
- **Cómo arreglarlo:** Desacoplar recepción de cómputo (cola + worker), o ejecutar el pipeline en executor. Añadir reconexión con backoff y heartbeat.
- **Validación:** Prueba de resiliencia: matar/restaurar el stream y verificar reanudación; medir que un fill/quote no se pierde durante un cómputo largo.

### H6 — Entitlement de datos en tiempo real (verificar)
- **Severidad:** ALTA (operativa, no de código) · **Módulo:** datos
- **Problema:** El plan de Alpaca debe devolver barras en tiempo real para el timeframe elegido; el tier gratuito ha sido históricamente retrasado/limitado.
- **Evidencia:** `go-live-review.md` #5. No verificable desde el código.
- **Cómo arreglarlo:** Confirmar entitlement antes de fiarse de señales intradía.

### M1 — `core/signal_generator.py` es un esqueleto muerto
- **Severidad:** MEDIA · **Evidencia:** docstring "Skeleton only — no logic implemented yet" (`signal_generator.py:6`); no se importa en el pipeline en vivo. **Fix:** borrarlo o implementarlo; no dejarlo aparentando ser parte del flujo.

### M2 — `min_confidence` de riesgo no cableado
- **Severidad:** MEDIA · **Evidencia:** `config_min_conf` lee `getattr(self.config, "min_confidence", 0.55)` (`risk_manager.py:687`) pero `RiskConfig` no tiene ese campo ni existe `risk.min_confidence` en `settings.yaml`. Siempre 0.55. **Fix:** añadir el campo o documentar que es constante; quitar el `getattr` engañoso.

### M3 — Mismatch de timeframe (daily-calibrado vs stream) (= go-live #2)
- **Severidad:** MEDIA/ALTA · **Evidencia:** todo calibrado en daily (`min_train_bars=504`, vol regimes), `settings.yaml timeframe: 1Day`. **Fix/decisión:** correr en daily (coherente con backtest) o re-validar todo a 5-min. No mezclar.

### M4 — Chequeo de correlación inerte en vivo
- **Severidad:** MEDIA · **Evidencia:** `_max_correlation` requiere `portfolio_state.price_history`, que en vivo va vacío (`main.py:_portfolio_state` no lo rellena). Skip silencioso. **Fix:** alimentar series de retornos o documentar que los límites de correlación están inactivos hasta cablearlos.

### M5 — Caps multi-activo nunca backtesteados
- **Severidad:** MEDIA · **Evidencia:** backtester opera solo el primer símbolo; el live itera todos `broker.symbols`; `max_single_position`/`max_concurrent`/sector/correlación asumen cartera multi-nombre no validada. **Fix:** o backtest multi-activo, o restringir live a un símbolo.

### M6 — Dependencias sin pinear + SDK legacy muerto
- **Severidad:** MEDIA · **Evidencia:** `requirements.txt` con rangos abiertos y `alpaca-trade-api` (deprecado) junto a `alpaca-py` (el que se usa). **Fix:** pinear versiones; eliminar `alpaca-trade-api`.

### M7 — Dashboard no es attach en vivo / snapshot mínimo (= go-live #10, #11)
- **Severidad:** MEDIA/BAJA · **Evidencia:** `run_dashboard` renderiza el último `state_snapshot.json`; `save_state` no persiste posiciones abiertas. Aceptable (recuperación vía `sync_on_startup`) pero conviene saberlo.

### L1 — Deriva de estado del README
- **Severidad:** BAJA · Mantener el README alineado con `go-live-review.md` (que es la fuente honesta).

---

## 4. Gaps para paper trading

### a) Imprescindible antes de paper (sin esto, paper es peligroso o no-op)
- **C1** backfill del buffer en vivo.
- **C4** gate de delta/rebalance con rama de venta.
- **C3** colocar stop real (bracket) + poblar `stop_order_id`.
- **C2 + C5 + C6** cablear fills → breaker → halt → liquidación; configurar `risk.lock_file`.
- **M3** decidir y fijar timeframe coherente con la calibración (daily recomendado).
- **H2** CLI/README correctos (operación reproducible).

### b) Recomendable antes de paper
- **H3** buffer acotado / features incrementales.
- **H5** desacoplar cómputo del handler async + reconexión del stream.
- **H6** verificar entitlement de datos en tiempo real.
- **H4** definir bien el techo de exposición/leverage.
- **M4** activar (o documentar como inactivo) el chequeo de correlación.
- Test de integración del lazo en vivo simulado end-to-end (sin broker real).

### c) Mejoras posteriores (antes de dinero real)
- **R1** edge de la estrategia (bloquea financiación, no paper).
- **M5** validar caps multi-activo o restringir a un símbolo.
- **M1** borrar/implementar `signal_generator.py`.
- **M6/M2/M7/L1** higiene de dependencias, config y docs.

---

## 5. Plan de reparación por fases

> Principio: cambios mínimos, trazables, verificables. Cada fase deja el sistema **más correcto y testeado**, no necesariamente "listo para paper" hasta cerrar el slice de seguridad. **"Done" = verificado por test/dry-run, no "operé en paper".**

### Fase 0 — Higiene y reproducibilidad (riesgo casi nulo)
- **Objetivo:** que lo documentado funcione y el repo sea fiable.
- **Archivos:** `README.md`, `requirements.txt`, (opcional) borrar `core/signal_generator.py`.
- **Cambios:** corregir CLI del README a flags reales (H2); pinear versiones y quitar `alpaca-trade-api` (M6); eliminar el esqueleto muerto (M1).
- **Tests:** `pytest` sigue verde; smoke de cada comando del README (`--backtest`, `--dry-run`, `--train-only`, `--dashboard`).
- **Done:** todos los comandos del README arrancan; suite verde.

### Fase 1 — Backfill del buffer en vivo (C1) [propuesta detallada abajo]
- **Objetivo:** que el lazo en vivo produzca señal desde la primera barra.
- **Archivos:** `main.py` (`run_live`), tests.
- **Cambios:** sembrar `system.buffers[symbol]` con `lookback_bars≥N` antes de `run_stream`.
- **Tests:** test que con buffer sembrado, `process_symbol` devuelve señal en barra 1.
- **Done:** dry-run/te st muestra señal inmediata; **no** habilita paper aún (C3/C4 pendientes).

### Fase 2 — Gate de delta/rebalance + venta (C4)
- **Objetivo:** que el live rebalancee como el backtest, sin sobre-acumular.
- **Archivos:** `main.py` (`process_symbol`), `broker/order_executor.py` (reusar buy/sell), tests.
- **Cambios:** calcular `current_weight` desde `tracker`, `delta=target−current`, operar solo si `|delta|≥rebalance_threshold`/`must_exit`, dimensionar sobre `delta`, sell si `delta<0`.
- **Tests:** mismo signal repetido no re-compra tras alcanzar target; baja de target genera venta.
- **Done:** test de acumulación pasa con `tracker` no-vacío.

### Fase 3 — Stop real en vivo (C3)
- **Objetivo:** toda entrada con stop colocado en el broker.
- **Archivos:** `main.py` (envío), `broker/order_executor.py` (`submit_bracket_order`, captura de leg id), `broker/position_tracker.py`.
- **Cambios:** usar bracket en vivo; recuperar pata stop y poblar `stop_order_id`; habilitar trailing.
- **Tests:** bracket con `stop_price` correcto; `stop_order_id` poblado; `update_trailing_stops` tighten.
- **Done:** tests verdes; dry-run muestra intención de stop.

### Fase 4 — Slice de seguridad: fills → breaker → halt → liquidación (C5+C2+C6)
- **Objetivo:** que la protección de drawdown funcione de verdad en vivo.
- **Archivos:** `main.py` (`run_live`, lazo/rollover), `broker/position_tracker.py`, `config/settings.yaml` (`risk.lock_file`).
- **Cambios:** suscribir fills → `apply_fill_to_risk`; recomputar drawdown desde equity del broker; en HALT cesar aperturas y `close_all_positions`; fijar `lock_file`.
- **Tests:** integración con fills perdedores simulados → `NORMAL→REDUCED→HALTED`, rechazo de señales, liquidación disparada; lock file escrito/respetado.
- **Done:** test de seguridad pasa. **Con Fases 1–4 + M3, recién entonces el sistema es candidato a paper.**

### Fase 5 — Robustez operativa (H3, H5, H6, H4, M4)
- **Objetivo:** que aguante una sesión real (memoria, desconexión, datos).
- **Archivos:** `main.py`, `data/market_data.py`, `core/risk_manager.py`.
- **Cambios:** buffer acotado; desacoplar cómputo del socket + reconexión; verificar entitlement; arreglar techo de exposición; cablear/documentar correlación.
- **Tests:** resiliencia de stream; buffer acotado; cap de exposición coincide con diseño.
- **Done:** sesión simulada larga estable.

### Fase 6 — Edge (R1) [research, antes de dinero real]
- Iterar/re-scopear la estrategia; `--compare` batiendo buy-and-hold ajustado por riesgo de forma robusta.

---

## 6. Validación final — checklist de aceptación "listo para paper"

**Funcionales**
- [ ] Primera barra en vivo produce features+signal (C1).
- [ ] Todos los comandos del README arrancan (H2).
- [ ] HMM entrena/`save`/`load` round-trip OK (`--train-only`).

**Riesgo**
- [ ] Serie de fills perdedores dispara `REDUCED`→`HALTED` en vivo (C2/C5).
- [ ] En HALT se cesan aperturas y se liquida (C6).
- [ ] Toda entrada lleva stop colocado en el broker y `stop_order_id` poblado (C3).
- [ ] `lock_file` se escribe en peak-DD y bloquea reanudación (C2).

**Integración**
- [ ] Repetir el mismo target N barras no sobre-acumula; bajada de target vende (C4).
- [ ] Reconexión del stream sin pérdida de barras/fills (H5).
- [ ] `tracker.reconcile` cuadra contra el broker tras reinicio.

**Operativas**
- [ ] Buffer acotado en sesión larga; tiempo/barra constante (H3).
- [ ] Timeframe coherente entre calibración, config y stream (M3).
- [ ] Entitlement de datos en tiempo real confirmado (H6).
- [ ] Logs/alerts/dashboard reflejan el estado real durante una sesión multi-día de paper.

**Edge (para dinero real, no para paper)**
- [ ] Backtest batiendo buy-and-hold ajustado por riesgo de forma robusta (R1).

---

## 7. Código incompleto — parches propuestos (no aplicados)

### C1 — sembrar el buffer en `run_live` (parche mínimo)
**Archivo:** `main.py`, dentro de `run_live`, tras construir `system` y antes de `system.run_stream(md)`.
```python
# --- BACKFILL: seed rolling buffers so features are ready on bar 1 ---
warmup = int(config.get("hmm", {}).get("min_train_bars", 504)) + 260  # z-score(252)+SMA200 margin
for sym in symbols:
    hist = md.get_history(sym, lookback_bars=warmup, timeframe=timeframe)  # OHLCV DataFrame
    if hist is None or hist.empty:
        tlog.log(tlog.main, "backfill_warn", f"no history for {sym}", level="WARNING")
        continue
    system.buffers[sym] = hist
tlog.log(tlog.main, "backfill_done", f"seeded {len(symbols)} buffers (~{warmup} bars)")
```
**Por qué:** sin sembrar el buffer, `build_features` devuelve vacío ~450 barras y el bot no opera. Verificar la firma real de `get_history`/`get_history_multi` en `data/market_data.py` (líneas 141/160) y ajustar args. Es aditivo, no toca la lógica de decisión.

### H2 — corregir CLI en README
**Archivo:** `README.md` secciones "Backtest CLI" y "Usage": reemplazar `python main.py backtest --symbols ...` y `python main.py --mode backtest|live` por los flags reales (`--backtest`, `--live`, ...) tal como ya documenta `docs/go-live-review.md`.
**Por qué:** los comandos publicados no ejecutan; rompe reproducibilidad y confianza.

### M1 — eliminar esqueleto muerto
**Archivo:** borrar `core/signal_generator.py` (y cualquier import). **Por qué:** declara un pipeline que no existe ("no logic implemented yet") y confunde el mapa mental del sistema.

---

## 8. Conformidad vídeo vs código (pasada `/watch`)

_Vídeo: "build a fully automated trading bot with Claude Code" (~34 min, captions). Cotejo de las 8 fases prescritas contra lo construido._

**Veredicto de conformidad:** el build **implementa fielmente todas las fases del vídeo**, y en varios puntos es **más riguroso/honesto** que el tutorial (timeframe diario coherente, `go-live-review.md` honesto, desviación 252→504 documentada). **Los bloqueos críticos C1–C6 NO son errores del build respecto al vídeo: son huecos que el propio vídeo nunca especifica.** El paso más peligroso (Fase 7, main loop) es justo el que el vídeo despacha en ~40 s.

### Fase a fase
| Fase (vídeo) | Prescrito | Construido | Veredicto |
|---|---|---|---|
| Scaffolding | proyecto `regime-trader`, 31 files/8 dirs | idéntico | ✅ |
| Brain (HMM) [03:52,13:16] | clasificador de volatilidad, probar 3–7 regímenes y auto-elegir, ordenar por mean return, **forward algorithm (no `model.predict`, no look-ahead)**, ~2 años **diarios**, stability 3 barras, flicker >4/20→incertidumbre | exacto | ✅ Fiel |
| Allocation [16:29] | low-vol 95% @1.25x, mid si trend, reduce; vol-rank orchestrator, incertidumbre | exacto | ✅ Fiel |
| Backtest [18:34] | walk-forward, **IS 252d**, OOS ~6m, slippage, métricas por régimen+confianza, benchmarks buy-hold/200-SMA/random, stress crashes | implementado; **IS 252→504** (HMM lo exige) | ✅ con desviación documentada |
| Risk [21:39] (frame 52) | veto absoluto; CB 2%→½, 3%→close all, 5%/sem→½, 7%→close all, 10% peak→halt+lock manual; 1% riesgo, 15% single, 30% sector, corr | `RiskConfig` coincide **exacto** | ✅ a nivel unitario; ⚠️ inerte en vivo (C2/C5/C6) |
| Broker (Alpaca) [24:44] | client/order_executor/position_tracker/market_data; trade de prueba NVDA real verificado en Alpaca [28:37] | adaptadores OK contra **mock**; **el trade real nunca se ejecutó** | ⚠️ paso de verificación omitido |
| Main loop (Fase 7) [29:36] | startup→main loop por barra (**default 5-min**)→shutdown→error handling | `TradingSystem` parcial | ❌ aquí viven C1–C6 |
| Monitoring/Dashboard [30:45] | logging+alerts; **dashboard Streamlit** web (`localhost:8501`, frames 7/74) | `monitoring/` rich **terminal**; sin streamlit | ⚠️ divergencia real |

### Hallazgos de la conformidad
1. **El vídeo es la fuente de C1–C6, no el build.** La Fase 7 ([29:36-30:18]) es un prompt "glue everything" de ~40 s que **nunca especifica**: backfill del buffer, gate delta/rebalance, rama de venta, stop adjunto a la entrada, ni el cableado fills→breaker→halt→liquidación. El build implementó lo que se pidió; lo que se pidió estaba incompleto en el punto más crítico. (El vídeo incluso dice "una vez cableado esto, técnicamente no necesitas el dashboard" — despacha el lazo en vivo.)
2. **El mismatch de timeframe nace del vídeo.** Entrena en **diario** [14:39] pero por defecto corre el lazo en **5-min** [29:58]. El build eligió diario (`timeframe: 1Day`) — **más coherente** que el vídeo. M3/go-live #2 traza al tutorial.
3. **La seguridad más enfatizada es la más rota.** El vídeo vende los circuit breakers como "más importantes que los HMM" y dice que "cazan drawdowns sobre P&L real" (frame 52). En el lazo construido **nada les pasa P&L real** (C2/C5) → la pieza estrella es la más inerte en vivo.
4. **Divergencias build vs vídeo concretas:**
   - **Dashboard:** vídeo = app **Streamlit** web (`streamlit run`, `localhost:8501`, frames 7 y 74); build = dashboard **rich** de terminal que renderiza `state_snapshot.json`, sin dependencia streamlit, sin attach en vivo (M7). El dashboard vistoso del intro **no es lo que produce el repo**.
   - **Nº de tests:** vídeo termina en **134**; build en **102** (~32 menos). Verificar si falta cobertura o es distinta granularidad.
   - **IS window:** 252 (vídeo) → 504 (build, desviación necesaria documentada).
5. **El propio vídeo recomienda paper-first ≥1 mes** [25:21],[32:22] y revisar cada rebalanceo — alineado con esta auditoría. No contradice el veredicto: refuerza "paper primero, fondear al final".

**Implicación para el plan:** ninguna fase de reparación contradice el vídeo; **completan** lo que el tutorial dejó implícito en la Fase 7 y reponen el dashboard (decidir rich-terminal vs Streamlit como el vídeo).

---

## Nota de alcance
Auditoría hecha con **el código como fuente primaria**; cotejada además contra el vídeo (sección 8). Si quieres, hago una pasada de conformidad **vídeo vs código** para detectar fases del tutorial que quedaron a medias respecto a lo prometido. Falta contexto externo en: entitlement real de tu cuenta Alpaca (H6) y la intención de diseño exacta del techo de exposición (H4) — ambos requieren tu confirmación.

---

## 9. Diagnóstico del gap de tests (102 vs 134) — Fase 0 (2026-06-02)

_Encargo de Fase 0: **diagnosticar, no fabricar**. No se inventó ningún test para cuadrar el número._

### Datos (build actual, py3.14)
`pytest --co` → **102 tests**, repartidos por módulo:

| Archivo | Tests | Fase cubierta |
|---|---:|---|
| `test_hmm.py` | 12 | Brain (HMM) |
| `test_strategies.py` | 12 | Allocation |
| `test_orchestration.py` | 9 | Allocation/main pipeline |
| `test_risk_validate.py` | 18 | Risk (veto/sizing) |
| `test_risk.py` | 10 | Risk (breakers unitarios) |
| `test_backtest.py` | 11 | Backtest |
| `test_lookup_ahead.py` | 6 | Backtest (no-leakage) |
| `test_orders.py` | 9 | Broker (executor) |
| `test_broker.py` | 9 | Broker (client/tracker vs mock) |
| `test_monitoring.py` | 6 | Monitoring |
| **Total** | **102** | |

### Conclusión: granularidad/forma del tutorial, **no** cobertura perdida de una pieza testeada
1. **No hay lista canónica de los 134 del vídeo** que cotejar test a test. El número del vídeo es el contador acumulado mientras construye; no es un manifiesto. Cualquier "cuadre" exacto sería fabricado → no se hace.
2. **Las 8 fases tienen archivo de test propio con cobertura real.** El delta de ~32 no es "falta el test de un módulo que existe y está sin probar"; es:
   - **(a) Granularidad/conteo.** El vídeo suma iteraciones de test según itera; el build consolidó casos (p.ej. parametrizaciones que cuentan como 1).
   - **(b) Dashboard Streamlit no portado.** El vídeo construye un dashboard **Streamlit** (con sus tests); el build lo cambió por un dashboard **rich de terminal** (`test_monitoring.py` = solo 6). Los tests del dashboard web del tutorial no existen aquí porque el componente no existe aquí (ver §8, M7).
3. **El hueco de cobertura real NO son 32 tests sueltos: es el lazo en vivo.** `main.py` tiene **5 regiones `# pragma: no cover`** —`run_live`, `run_stream`, la rama de orden en vivo, `_portfolio_state` de broker, `update_trailing_stops`— y son **exactamente** donde viven C1–C6. Ningún test unitario extra sobre módulos ya probados cierra ese hueco; lo cierran los **tests de integración del lazo simulado** que añaden las fases de reparación (F1+).

**Implicación:** el gap 102↔134 es ruido de conteo + el dashboard no portado. La señal de cobertura que importa es la del lazo en vivo (pragma-no-cover = C1–C6), y se ataca con las fases de reparación, no fabricando tests para igualar 134.

### Estado Fase 0 (2026-06-02)
- **H2** README → flags reales (`--backtest`/`--live`/…); todos los comandos casan con `--help`. ✅
- **M6** `requirements.txt` pineado al venv test-green; fuera `alpaca-trade-api` (legacy) + `websocket-client`/`schedule` (no importados, no instalados). ✅
- **M1** `core/signal_generator.py` borrado (esqueleto muerto, 0 imports). Conteo de tests sin cambio (102 → confirma que estaba muerto). ✅
- **Gap tests** diagnosticado (esta §9), sin fabricación. ✅
- `pytest`: **102 passed** tras los cambios.

---

## 10. Estado reparación F1–F5 + dashboard (2026-06-02)

Patrón en todas las fases (advisor-confirmado): el núcleo testeable se extrae/usa con TDD; el call-site en `run_live`/`run_stream` queda `# pragma: no cover` (no testeable sin broker). **`pytest`: 102 → 129 passed** tras toda la tanda. Commits atómicos por hallazgo.

| Fase | Hallazgo | Fix (núcleo testeado) | Commit |
|---|---|---|---|
| F1 | **C1** buffer frío | `TradingSystem.seed_buffers` siembra `min_train_bars+260` por símbolo antes del stream | `0ee8393` |
| F2 | **C4** sobre-acumulación | `_rebalance_order(target/held, weights, threshold)` → delta en **acciones**; buy/sell/hold; `must_exit` liquida | `1b78ac0` |
| F3 | **C3** sin stop | entradas live vía `submit_bracket_order`; `_stop_leg_id` captura la pata stop → `stop_order_id` (late-attach async) | `e6a8a8a` |
| F4 | **C5+C6** | stream de fills (daemon thread)→`on_fill` (tracking); HALT→`close_all_positions`; `lock_file` live | `9323251` |
| F4b | **C2** (corregido) | `_update_risk_posture` alimenta el breaker con **equity MtM por barra** (realized+unrealized) → halta en drawdown aunque NO haya fills (caso buy-and-hold). `on_fill` quedó solo tracking (evita doble conteo). El primer intento solo cableó la pata de pérdidas realizadas (fills); pasaba por un test que usaba sell-fills, no el path que importa | `bbc5e50` |
| M3 | timeframe | `settings.yaml timeframe: 1Day` LOCKED (HMM/regímenes/breakers daily-calibrados) | `9323251` |
| H3 | buffer O(n²) | `ingest_bar` recorta a `_buffer_cap` (=seed depth) | `ab75dc0` |
| Dashboard | M7 + LOCK usuario | Streamlit web (`streamlit run monitoring/streamlit_app.py`) + capa de datos pura testeada; verificado booteando (health ok) | `82e81a8` |

### Pendientes — NO código autónomo (requieren decisión/credenciales del usuario)
- **R1 (edge):** la estrategia sigue sin edge (6.9% vs 69.9% buy-hold). Bloquea **dinero real**, no paper. No es fix de código → investigación.
- **H6 (entitlement):** que la cuenta Alpaca devuelva barras en tiempo real al timeframe elegido. No verificable desde el código; requiere la cuenta.
- **H4 (techo leverage):** `check_exposure` capa efectiva a 1.0x (`max_exposure*max_leverage`), `low_vol_leverage 1.25` muerto. La **intención de diseño** (¿1.25x alcanzable?) es decisión de riesgo del usuario — NO se tocó para no inventar política.
- **H5 (async + reconexión):** el pipeline corre síncrono en el handler del socket; los fills van en daemon thread sin reconexión con backoff. No testeable sin broker → diferido y marcado como scaffolding.
- **M4/M5 (multi-activo):** chequeo de correlación inerte en vivo (sin `price_history`) y caps multi-activo nunca backtesteados. **Lean: restringir live a 1 símbolo** hasta validar multi-activo; la correlación es moot en single-asset.

### Smoke real contra Alpaca paper (2026-06-02) — verificado
`tmp/smoke_alpaca.py`, `tmp/smoke_order.py`, `tmp/smoke_live_paths.py` (`PYTHONPATH=.`):
- Conexión + cuenta paper ($100k equity) ✅
- **H6 entitlement / C1 seed depth:** `get_history(SPY,1Day,764)` → **764 barras** reales (last close 758.54) → features listos en barra 1 en producción ✅ (no es no-op silencioso).
- **Path de entrada live real (bracket):** `submit_bracket_order` → submitted, order_id + **stop_leg id capturado** aun con mercado cerrado → C3 verificado contra Alpaca real, no solo mock ✅. Cancelado, sin posición.
- Round-trip orden simple submit/cancel ✅. `--train-only` (HMM→models/hmm_SPY.pkl) y `--dry-run` (20/20) ✅.
- **Sin verificar hasta la 1ª sesión con mercado abierto:** llegada de un fill real por el stream (`subscribe_fills`) y el feed del breaker MtM desde equity live (`tracker.refresh`). Ambos cableados + unit-testeados, no ejercitados en vivo.

### Veredicto actualizado → **PAPER-READY**
Con C1–C6 cableados/testeados, H4 resuelto, y los paths live verificados contra Alpaca paper, el sistema es **paper-ready** (default 1 símbolo SPY). **Sigue NO listo para dinero real:** R1 (sin edge) lo bloquea — correr paper ≥1 mes vigilando cada rebalanceo. "Paper-ready" ≠ "rentable". Pendientes no-código: R1 (research), H5 (reconexión), M4/M5 (multi-activo).
