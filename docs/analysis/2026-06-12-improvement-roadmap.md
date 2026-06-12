---
type: roadmap
status: active
tags: [regime-trader, roadmap, gates, jump-model, edgar, sentiment, research-factory, free-data]
created: 2026-06-12
related: ["[[2026-06-04-cross-sectional-prereg]]", "[[2026-06-05-deployed-book-prereg]]", "[[2026-06-05-idio-momentum-challenger-prereg]]", "[[2026-06-12-hedge-activation]]", "[[2026-06-11-alpha-engine-v2-architecture]]", "[[2026-06-04-ml-v2-deferral-decision]]"]
---

# Roadmap de mejoras — 2026-06-12

Revisión completa del sistema (exploración + deep research) → roadmap por tiers.
Decisiones de Pablo (2026-06-12): (1) este doc es el entregable, implementación en
sesiones siguientes; (2) **value/quality REABIERTO vía SEC EDGAR** (la decisión
no-datos-de-pago cerraba datos *pagados*; EDGAR companyfacts es gratis y PIT real);
(3) research factory: **rails primero, loop nocturno después**.

## 0. Invariantes (no negociables)

- **No datos de pago** (decisión 2026-06-12, [[2026-06-12-hedge-activation]] §3).
- **Preregs congelados intocables**: baseline ([[2026-06-04-cross-sectional-prereg]],
  [[2026-06-05-deployed-book-prereg]]), challenger
  ([[2026-06-05-idio-momentum-challenger-prereg]]), tail-hedge
  ([[2026-06-12-hedge-activation]]). Nada de lo que sigue muta knobs, señales ni
  cadencias de los libros en gate.
- **Alfa nueva = prereg nuevo + cargo en ledger n_trials** (ver T4.1). Sin excepción.
- **Timeframe 1Day lock (M3)** — scope: libro equity. Sleeve crypto define su propia
  cadencia en su prereg (asset class nueva, no creep del lock).
- **Patrón de aislamiento universal**: señal nueva → shadow-log sin órdenes, O libro
  separado con `book_snapshot_<sleeve>.json` + track CSV propio (patrón challenger).
  Los libros congelados jamás leen outputs de señales nuevas.

## 1. Hallazgo crítico (verificado en vivo, 2026-06-12)

**El gate del challenger se muere de hambre.** `track_record.csv` solo registra
book/spy/ew — no hay columna challenger — y `com.regimetrader.challenger.plist` NO
está cargado. Un gate forward de 12 meses sin NAV diario = gate auto-fallado por
inanición de datos. Es T0.1, máxima prioridad del programa.

Estado launchd verificado: runonce, rebalance y recordtrack SÍ cargados (exit 0);
challenger, riskcheck y dashboard NO. `track_record.csv` al día hasta 2026-06-11.

Nota R-4: `core/hmm_engine.py:299` ya siembra restarts como `random_state + i`
(determinista per-process). La banda Sharpe 0.37–0.49 entre procesos viene de otro
sitio — sospechosos: end-date de datos = "now", threading BLAS/OMP, orden de NaN
handling, tie-breaks entre restarts con log-likelihood casi igual. Ver T0.4.

## 2. Tier 0 — Operar los gates (urgente, cero alfa nueva)

| # | Qué | Dónde | Esfuerzo |
|---|---|---|---|
| T0.1 | **Feed challenger**: columna aditiva `challenger_nav` en `core/track_record.py::append_day` (append-only, jamás reescribir filas — la evidencia de gate es inmutable) + cargar plist challenger | `core/track_record.py`, `main.py --record-track`, `deploy/com.regimetrader.challenger.plist` | S |
| T0.2 | Cargar riskcheck plist **observe-only**; verificar por grep que ningún path de órdenes es alcanzable con `allow_orders=false` | `deploy/com.regimetrader.riskcheck.plist`, `core/risk_monitor.py` | S |
| T0.3 | Higiene git: commitear `core/quality_ranking.py`+test (hoy untracked), merge `feat/meta-overlay`→`main`, tag `gate-baseline-2026-06-12`, columna `code_sha` en track record. launchd ejecuta lo checked-out: cambiar de branch a mitad de gate = cambio silencioso de comportamiento en vivo | git + `core/track_record.py` | S |
| T0.4 | **Auditoría determinismo R-4**: pin `OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` en plists, congelar end-date como parámetro, hash de matriz de transición por run, `tests/test_determinism.py` (fit doble → posteriors idénticos). Política **pin-champion**: cargar campeón de `core/model_registry.py` en vez de refit diario; refit solo por trigger de drift (T3.3). Documentar como *enmienda operativa* (el gate mide NAV del libro, no internals del modelo); dual-log refit-vs-pinned 2 semanas para demostrar equivalencia | `core/hmm_engine.py`, `core/model_registry.py`, plists, `tmp/repro_check.py`→tests | M |
| T0.5 | **Telegram + heartbeat**: `_send_telegram` en `monitoring/alerts.py` (reusar `_is_rate_limited`; AIOS gateway ya tiene bot LIVE). Heartbeat: última fila de track_record >2 días hábiles → CRITICAL. Gates desatendidos 12 meses necesitan sistema nervioso | `monitoring/alerts.py`, `config/settings.yaml` | S |
| T0.6 | **Panel gate-countdown** en Streamlit: días transcurridos/restantes por gate (baseline 2026-06-04, challenger 2026-06-05, hedge 2026-06-12), DSR rolling sobre track record (reusar `backtest/performance.py::deflated_sharpe_ratio`), estado hedge + presupuesto consumido, contador de trials del ledger | `monitoring/streamlit_app.py`, `monitoring/dashboard_data.py` (nuevo `gate_status()`) | M |

## 3. Tier 1 — Regime engine v2 (solo shadow, cero impacto en órdenes)

El HMM como *timer* de retorno está falsado (R1, vía B). Lo que NO está falsado:
clasificación de régimen de volatilidad como overlay de riesgo. Tier 1 ataca la
calidad del clasificador, en shadow, con promoción solo vía libro nuevo preregistrado.

- **T1.1 — Statistical Jump Model challenger** (`core/jump_model.py`). Evidencia:
  arXiv 2402.05272 (Shu/Mulvey, *Downside Risk Reduction Using Regime-Switching
  Signals: A Statistical Jump Model Approach*) + arXiv 2406.09578: JMs dan regímenes
  más persistentes y menos flicker que HMM (en un estudio Sharpe 0.78 vs 0.51), y el
  flicker es exactamente nuestro dolor conocido (`alerts.py::flicker_exceeded`).
  Implementación: fit tipo k-means con penalización λ por salto de estado, mismo panel
  de features que consume el HMM. Generalizar `core/model_registry.py` a protocolo
  `RegimeEngine` (ABC con `fit/predict_proba/regime_labels`; hoy está HMM-typed).
  Shadow log `logs/shadow_regime.csv`: fecha, régimen+confianza HMM, régimen+confianza
  SJM, retorno realizado siguiente. Grid de λ se carga al ledger n_trials. Promoción
  = libro NUEVO preregistrado; jamás hot-swap en libros congelados. Esfuerzo: L.
  Depende de T0.4 (harness de reproducibilidad).
- **T1.2 — BOCPD** (Adams-MacKay, run-length posterior) en `core/changepoint.py`.
  Corroborador model-free del hazard de `core/meta_overlay.py` (que es model-internal).
  Mismo shadow CSV. Posible uso futuro como corroboración del trigger del hedge —
  solo vía enmienda preregistrada tras cerrar el gate actual. Esfuerzo: M. Paralelo.
- **T1.3 — Features macro gratis** en `data/macro_data.py`: ratio VIX/VIX3M (CBOE
  gratis; backwardation precedió el 100% de los drawdowns mayores 1990–2025; contango
  = sin señal) + FRED HY OAS, pendiente de curva, NFCI. Framing explícito:
  **confirmación de riesgo, NO timing de retorno** (eso está falsado). Guardrail
  duro: estas features JAMÁS entran al panel del HMM campeón (sería mutación
  silenciosa del gate) — solo shadow fit + assert diario de hash del modelo campeón.
  Esfuerzo: M. Paralelo.
- **T1.4 — Informe shadow mensual** auto-generado: matriz de acuerdo entre motores,
  conteo de flicker, counterfactual de downside (qué habría hecho el libro campeón
  bajo labels SJM, vía `backtest/backtester.py` con labels inyectados), DSR.
  Output `docs/analysis/YYYY-MM-shadow-regime-report.md`. Esfuerzo: S tras T1.1–T1.3.

## 4. Tier 2 — Sleeves de alfa nueva (datos gratis, prereg + libro propio cada una)

Cada sleeve replica el patrón challenger: prereg propio congelado antes de arrancar,
snapshot propio, track propio, cargo en n_trials, flag CLI propio. Ninguna toca los
libros congelados.

### T2.1 — EDGAR value/quality (REABIERTO — máxima convicción, el código ya existe)

- **Qué**: `data/edgar_data.py` sustituye el stub SimFin; `core/fundamental_features.py`
  y `core/quality_ranking.py` (ya construidos y testeados) quedan sin cambios aguas
  arriba. SEC companyfacts API: gratis, sin key, 10 req/s → 503 nombres en <2 min,
  cache en `data/cache/edgar/`.
- **PIT real**: timestamp = filing date (`filed`/`accn`). Regla anti-restatement:
  **first-filed-only** (el valor publicado primero es el que existía en t). Cadena de
  fallback de tags XBRL: `Revenues` → `SalesRevenueNet` →
  `RevenueFromContractWithCustomerExcludingAssessedTax` → extensión documentada.
  Alineación fiscal vía `fy/fp/frame`.
- **Honestidad de backtest**: histórico solo con constituyentes actuales → todo
  output etiquetado **SURVIVORSHIP-BIASED** hasta T5.3. Evidencia primaria = forward
  paper ≥12mo con prereg nuevo (`docs/analysis/2026-0X-quality-edgar-prereg.md`,
  umbrales DSR/PBO del template challenger).
- Esfuerzo: M. Depende de T0.3 (commitear quality files primero).

### T2.2 — Factor de sentimiento de noticias (Benzinga vía Alpaca, gratis)

- **Qué**: Alpaca News API — gratis, histórico Benzinga desde 2015, 200 calls/min →
  factor **backtesteable Y forward**, sin pagar. `data/news_data.py` +
  `core/sentiment_factor.py`. Agregado diario por ticker → factor cross-sectional;
  testear standalone y como condicionador del momentum 12-1.
- **Scorer**: FinBERT para backtest (reproducible, versión pineada); worker Ollama
  local (qwen3.5:9b, gratis/privado — sinergia AIOS) como scorer challenger.
  Re-score del histórico con modelo nuevo = trial nuevo en el ledger.
- **PIT**: solo headlines con timestamp anterior al close del día señalado.
- Esfuerzo: L. Paralelo a T2.1.

### T2.3 — Crypto TS-momentum (Alpha Engine v2 fase 3, ya diseñada)

- BTC/ETH, Alpaca crypto gratis, 24/7, ≤10% NAV, vol-target a vol equity. Reusar
  maquinaria TS-mom y `vol_target_scale` de `core/asset_rotation.py` (módulo
  archivado como avenida de rotación, pero sus funciones son sólidas). Adaptar
  `broker/order_executor.py` (crypto: sin brackets/shorts). Prereg define su propia
  cadencia (M3 = scope equity). Prioridad menor que T2.1/T2.2 (prior más débil;
  beneficio = descorrelación). Esfuerzo: M.

## 5. Tier 3 — Meta/ML sobre exhaust propio (datos gratis por construcción)

- **T3.1 — Meta-labeling** (López de Prado): etiquetado triple-barrier sobre fills
  propios + modelo secundario (sklearn ya es dep) que filtra/dimensiona señales del
  primario — cambia recall por precision, sube Sharpe/F1. Umbral para siquiera
  entrenar: **≥200 round-trips cerrados** (~12–18 meses al ritmo actual). Acción
  AHORA: construir el pipeline de etiquetado (`core/meta_labeling.py` +
  `logs/trade_labels.parquet`) para que las labels se acumulen durante los gates.
  El modelo, cuando haya muestra. Sizing-only sobre versión nueva preregistrada.
- **T3.2 — Estudio conformal/uncertainty gross-scaling**: intervalos de predicción
  sobre posterior de régimen → escalar gross con incertidumbre alta. Solo estudio
  sobre shadow logs. Comparación honesta contra el fuzzy posterior existente
  (`meta_overlay.py`) — puede que ya capture lo mismo; falsar antes de construir.
- **T3.3 — Drift→retrain cableado**: `core/drift.py` (PSI/entropía, A-3) → retrain
  candidato → `model_registry.save_version` como challenger (A-4) → comparación en
  holdout → **humano promueve** (sin auto-promote). Coherente con pin-champion T0.4.

## 6. Tier 4 — Research factory (rails ahora, loop después)

La ambición real del programa: industrializar el ciclo hipótesis→falsación que hoy
es artesanal. Literatura: Chain-of-Alpha (arXiv 2508.06312), QuantaAlpha, AlphaAgent
(KDD 2025 — exploración regularizada contra alpha decay). El riesgo central que esos
papers ignoran y nosotros no: minería masiva de factores explota n_trials y colapsa
el DSR. Por eso el ledger va PRIMERO.

- **T4.1 — Ledger global de hipótesis/trials** `research/registry.jsonl` (append-only):
  id, fecha, familia (momentum/sentiment/quality/regime), n_configs probadas,
  veredicto, link a prereg. Alimenta `n_trials` de
  `performance.py::deflated_sharpe_ratio` de forma auditable (hoy se pasa ad hoc).
  Conteo por familia completo; cross-family con effective-trials documentado.
  **Precede a toda research nueva.** Esfuerzo: S.
- **T4.2 — Generador de preregs** `scripts/new_prereg.py`: emite doc estandarizado
  (hipótesis, universo, cadencia, métricas de gate con umbrales DSR/PBO, trials
  cargados, fecha de congelación, criterios de falsación) desde template minado de
  los preregs existentes; auto-registra en T4.1. Esfuerzo: S.
- **T4.3 — Motor CPCV** `backtest/cpcv.py`: combinatorial purged CV con purge+embargo
  → distribución de Sharpe por paths. El walk-forward 504/126/126 actual es un solo
  path = veredictos de alta varianza. Complementa `pbo_cscv` existente
  (`performance.py:125`). Esfuerzo: M.
- **T4.4 — Loop nocturno agéntico** (GATED — decisión Pablo: manual unas semanas,
  plist después): Claude Code headless toma hipótesis del backlog → formaliza factor
  → implementa en sandbox `research/candidates/<id>/` (read-only sobre `core/` y
  `data/`, jamás escribe estado del trader) → corre CPCV+DSR/PBO con trials cargados
  al ledger → veredicto en `research/vault/<id>.md` → digest Telegram → **humano
  adjudica promoción** → prereg vía T4.2 → libro forward vía patrón T2. Rails duros:
  presupuesto N hipótesis/semana (disciplina de trials > throughput), blocklist de
  ideas falsadas (R1, rotación B, shorts-por-régimen, hmm_prob deploy) chequeada
  antes de cada run. Esfuerzo: L. Depende de T4.1+T4.2+T4.3.
- **T4.5 — Postmortem mensual LLM**: track records + logs de régimen + fills +
  alertas → narrativa + anomalías → `docs/analysis/YYYY-MM-postmortem.md`. Barato,
  cero riesgo de gate. Esfuerzo: S.

## 7. Tier 5 — Ejecución / infra

- **T5.1 — Slippage vol-aware**: estimar coeficiente desde fills paper propios
  (decision price vs fill por régimen de ATR%), setear `slippage_vol_coeff`
  (`config/settings.yaml:171`, hoy 0.0; hook ya codificado en
  `backtest/backtester.py:229`). Solo afecta backtests de research — los gates se
  juzgan sobre fills reales. Jamás re-correr backtests que ya son evidencia de gate.
- **T5.2 — Caps de correlación/sector en vivo** (M4/M5 inertes): promover
  `tmp/verify_sector_cap.py` a test + assert en rebalance; **log-only durante la
  ventana de gate** (armarlos cambiaría pesos en vivo a mitad de gate); armar en
  renovación de libro.
- **T5.3 — Universo point-in-time**: snapshot mensual
  `data/universe/YYYY-MM-constituents.csv` (parse de la tabla de cambios de
  Wikipedia; caveat documentado) — **empezar YA**: PIT perfecto hacia delante aunque
  el histórico sea imperfecto. `data/constituents.py` gana param `as_of`. Mata el
  survivorship de los backtests cross-sectional (crítico para T2.1/T2.2).
- **T5.4 — 2ª cuenta paper Alpaca**: trigger = activación de la primera sleeve T2.
  Sleeves nuevas en cuenta 2; campeón/challenger/hedge intocados en cuenta 1
  (aislamiento de atribución limpio). Multi-cuenta en `config/credentials` +
  `broker/alpaca_client.py`.

## 8. Gaps elite-shop (lo que faltaría incluso con todo lo anterior)

1. **Ledger de experimentos con lineage**: cada run de backtest/shadow → JSONL con
   config hash + code SHA + data hash + métricas. Sin esto, en 12 meses no puedes
   *probar* qué produjo la evidencia del gate. Se pliega en T4.1.
2. **Kill-switch drill trimestral**: simular caída de API Alpaca, día de datos malos,
   orden runaway; verificar ladder + alertas + runbook de flatten manual.
   `docs/runbooks/kill-switch-drill.md`. Paper = drills gratis.
3. **Análisis de capacidad**: %ADV por posición en cada rebalance (fills de fantasía
   en colas small-cap del libro cross-sectional). Log en order_executor + informe mensual.
4. **Atribución condicional a régimen sobre track record VIVO** mensual (reusar
   `performance.py::regime_breakdown`, hoy solo backtests). Alimenta T4.5.
5. **Centinela de calidad de datos diario**: splits/dividendos anómalos, precios
   stale, NaN rate del panel → alerta. Un día de datos malos durante el gate =
   evidencia corrupta. Plug en `--run-once`.
6. **Inmutabilidad de evidencia**: hash-chain o commit nocturno de
   `track_record*.csv` + snapshots a dir/branch de evidencia. Tamper-evidence del
   auto-red-teamer contra su yo-futuro.

## 9. Secuencia prioritaria y dependencias

**Top 5:**
1. **T0.1 + T0.3** — feed del challenger + merge/tag/code_sha. Integridad de gates;
   todo lo demás es irrelevante si los gates se mueren de hambre o el código deriva.
2. **T0.4** — determinismo + pin-champion. Sistema no reproducible = infalsable = no-sistema.
3. **T0.5 + gap 6** — Telegram heartbeat + hash de evidencia. 12 meses desatendidos.
4. **T4.1 + T4.2** — ledger + generador de preregs ANTES de cualquier research nueva.
5. **T2.1 EDGAR** — mejor EV/esfuerzo (código existe, dato gratis, diseño ya gated);
   su reloj de 12 meses es el recurso escaso: cuanto antes arranque, antes adjudica.

**Tracks paralelos**: (A) ops T0+gaps · (B) shadow T1 · (C) rails T4.1–T4.3.
**Cadenas**: T0.4→T1.1→T1.4 · T0.3→T2.1 · T4.1+T4.2+T4.3→T4.4 · T0.4→T3.3 ·
fills→T3.1-modelo · activación T2→T5.4.

## 10. Flags de riesgo sobre gates congelados

- Features nuevas en el panel del HMM campeón = mutación silenciosa del gate.
  Aislamiento: shadow fit, artefactos de modelo separados, assert diario de hash del campeón.
- Pin-champion (T0.4) = enmienda operativa, no de estrategia; dual-log 2 semanas.
- Slippage (T5.1) = parámetro de backtest; evidencia de gate son fills reales.
- Caps (T5.2) = log-only hasta renovación de libro.
- Factory sin T4.1 = el DSR de TODO el programa queda inválido. Ledger primero.

## Apéndice — fuentes del research (2026-06-12)

- Jump models: arXiv 2402.05272 (Shu/Mulvey/Kolm), arXiv 2406.09578, JM+MPC (MDPI
  Mathematics 13(17):2837, 2025), SSRN 4556048.
- Alpaca News API: docs.alpaca.markets/docs/historical-news-data — gratis, Benzinga
  desde 2015, 200 calls/min free tier.
- SEC EDGAR: sec.gov/search-filings/edgar-application-programming-interfaces —
  companyfacts JSON, sin key, 10 req/s; caveats restatements/tags/fiscal.
- Meta-labeling: López de Prado (2018), PMR *Meta-Labeling: Calibration and Position
  Sizing*, Hudson & Thames triple-barrier evidence.
- LLM alpha mining: Chain-of-Alpha (arXiv 2508.06312), AlphaAgent (KDD 2025),
  QuantaAlpha (arXiv 2602.07085) — y su anti-patrón: trials sin ledger.
- VIX term structure: CBOE term structure (gratis), Macrosynergy — backwardation
  precede drawdowns; contango sin señal de timing.
