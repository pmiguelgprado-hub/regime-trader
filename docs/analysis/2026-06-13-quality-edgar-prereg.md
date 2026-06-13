---
type: prereg
status: frozen         # CONGELADO 2026-06-13 (despliegue automático Pablo "/goal pasalo ya al bot real")
tags: [regime-trader, prereg, quality, edgar, free-data]
created: 2026-06-13
frozen: 2026-06-13
related: ["[[2026-06-12-improvement-roadmap]]", "[[2026-06-05-idio-momentum-challenger-prereg]]", "[[2026-06-04-cross-sectional-prereg]]"]
---

# Pre-registro — quality-edgar (quality)

**Trial id (ledger):** `20260613-930e9f` · **configs cargadas:** 2 ·
**n_trials familia 'quality' tras este cargo:** 2

> **CONGELADO 2026-06-13.** Desplegado en paper bajo el mandato de Pablo
> ("pásalo ya al bot real"). Defaults de los knobs resueltos por el agente y
> EXPUESTOS para veto (§1, §8) — dinero real BLOQUEADO, así que Pablo puede
> revertir/enmendar knobs en la próxima sesión con ~cero datos perdidos. Tras el
> congelado, cualquier cambio = enmienda nueva con su propio cargo en el ledger.
> El reloj de 12 meses empieza HOY (primera fila `quality_nav` en track_record).

## 0. Qué se está validando

**Hipótesis:** una sleeve de calidad PIT vía EDGAR (gross profitability Novy-Marx +
bajo apalancamiento) combinada con el libro de momentum 12-1 ya en vivo,
vol-targeted, bate a EW-S&P500 y SPY netos de coste en Sharpe **y** maxDD sobre 12
meses de paper forward.

**Mecanismo económico.** Calidad y momentum son el par complementario clásico (AQR
*Quality Minus Junk*; Asness-Frazzini-Pedersen; Novy-Marx 2013 *The Other Side of
Value*): la calidad (rentabilidad bruta / activos) predice retorno cross-sectional
con baja correlación al momentum de precio, así que combinarlas sube el Sharpe de
cartera vía descorrelación — la **única** palanca estructural de Sharpe que el
review honesto 2026-06-11 avaló y que **no** es apalancamiento. El otro lado del
trade: inversores que sobre-extrapolan crecimiento y sub-precian rentabilidad
aburrida y persistente.

**Qué NO se afirma:** ningún edge de backtest histórico (el universo es
survivorship-biased hasta T5.3). Evidencia primaria = forward paper.

## 1. Knobs CONGELADOS (sin barrido)

| Knob | Valor | Justificación / locus |
|---|---|---|
| Señal calidad | composite z: gross_profitability + (−leverage) | Novy-Marx; `core/fundamental_features.py::quality_features` |
| Combinación | `quality_momentum` (average-of-ranks con 12-1) | AQR value+momentum; `quality_ranking.combined_rank` |
| Momentum | 12-1 (lookback 252, skip 21) | idéntico al baseline; `cross_sectional` |
| Selección | top decil (`frac=0.10`) | igual que baseline |
| Overlay riesgo | `vol_target` a 12% anual (vol_window 126) | único overlay que sobrevivió validación |
| Pesos | equal-weight, cap por nombre 0.15 | igual que baseline |
| Cap sector GICS | 0.30 | `quality_ranking.make_book_weights_quality` |
| Cadencia | rebalance mensual, overlay gross diario | M3 lock equity |
| Fundamentales | EDGAR companyfacts, anual (fp=FY), first-filed-only | `data/edgar_data.py` |

**Variantes preregistradas = 2** (cargadas en ledger): **(A)** `combine="quality"`
(calidad sola) y **(B)** `combine="quality_momentum"` (default). **DESPLEGADA = B**
(una sola cuenta paper → un solo libro/snapshot vivo; A es la alternativa
preregistrada NO corrida en paralelo hasta T5.4). n_trials=2 es deliberadamente
**conservador**: B se eligió sobre A por teoría (AQR value+momentum), no por datos,
y aun así el DSR corrige por 2 (no nos damos crédito por la elección). Ningún otro
knob se sweepea; cualquier barrido posterior = multiple-testing → enmienda + cargo.

## 2. Datos y medición

- **Fuente:** SEC EDGAR companyfacts (gratis, sin key, ~10 req/s; `data/edgar_data.py`),
  cache en `data/cache/edgar/`. Sustituye el stub SimFin sin cambiar nada aguas abajo
  (mismo `company_block` shape; verificado en vivo AAPL/MSFT/JPM/XOM/NVDA 2026-06-13).
- **PIT real:** `Publish Date = filed` (fecha de presentación, no fiscal `end`).
  Anti-restatement **first-filed-only** por (fy, fp). Cadena de fallback de tags XBRL
  documentada (`Revenues`→`SalesRevenueNet`→`RevenueFromContract...`; se fusionan,
  no se elige el primero — bug real de migración de tags de Apple, cazado y testeado).
- **Serie NAV diaria:** patrón track-record — columna/CSV propio del libro quality,
  append-only, sembrado al equity del día 1 (ver §7). Aislamiento: `book_snapshot_quality.json`.
- **Net of cost:** fills reales del paper (cuenta 2, T5.4) o synthetic mark-to-market
  del snapshot (igual que el challenger) hasta que la cuenta 2 esté activa.

## 3. Benchmarks (mismo motor de costes)

EW-S&P500 (RSP, investable, neto) y SPY (cap-weight). Buy-and-hold, sembrados al
equity día-1 del libro (idéntico a `core/track_record.py`).

## 4. Criterios de aceptación (TODOS deben cumplirse; falla si falla cualquiera)

1. Ventana forward: ≥12 meses paper.
2. Bate a EW-S&P500 **y** SPY en Sharpe neto **y** maxDD.
3. **DSR > 0.5** con n_trials = 2 (este prereg) — verificar contra el ledger en la
   adjudicación, no contra la memoria (`performance.deflated_sharpe_ratio`).
4. **PBO < 0.5** (CSCV, `performance.pbo_cscv`) si se aporta backtest de soporte
   (etiquetado SURVIVORSHIP-BIASED hasta T5.3).
5. maxDD no peor que el libro baseline en el mismo periodo.

## 5. Modos de fallo (falsación explícita)

- La sleeve no bate a EW-S&P500 neto (la calidad no aporta sobre equal-weight).
- DSR ≤ 0.5 (el Sharpe no sobrevive a la corrección por las 2 variantes).
- maxDD peor que baseline (la calidad no añade defensa, solo beta).
- Cobertura de fundamentales < 60% del universo (demasiados None → la sleeve es de
  facto un sub-universo sesgado, no la S&P 500).
- Blocklist heredada (no re-explorar): R1 timer, rotación vía B, shorts-por-régimen,
  hmm_prob deploy directo.

## 6. Expectativa honesta

Prior moderado-positivo: calidad+momentum es de los pocos pares con evidencia
out-of-sample robusta y mecanismo claro. Pero (a) sobre EW-S&P500 ya diversificado
el excess Sharpe esperado es modesto, (b) financieros/energía no reportan GrossProfit
(→ None → caen del factor calidad; la sleeve sobre-pondera de facto tech/industria —
sesgo sectorial a vigilar con el cap 0.30), (c) 12 meses es muestra corta para
distinguir skill de suerte (de ahí DSR/PBO). Los escépticos dirán que es beta de
calidad disfrazada; el cap sector y la comparación maxDD lo testean.

## 7. Reproducción (loci de código)

- Datos: `data/edgar_data.py` (`ticker_to_cik`, `company_facts`, `to_company_block`).
- Features: `core/fundamental_features.py` (sin cambios — drop-in).
- Construcción: `core/quality_ranking.py::make_book_weights_quality` (sin cambios).
- **CABLEADO (2026-06-13):** `main.run_rebalance --quality` (flag, patrón `--challenger`,
  fuerza dry-run — jamás `--execute` en la cuenta compartida), `book_snapshot_quality.json`,
  columna `quality_nav` en `core/track_record.py::append_day` (síntesis mark-to-market en
  `run_record_track`), `deploy/com.regimetrader.quality.plist` (diario L-V 22:40),
  `data/edgar_data.py::load_blocks` (universo, cache `data/cache/edgar/`).
- Commit SHA al congelar: ver git (commit con `status: frozen`); tag opcional `gate-quality-2026-06-13`.

## 8. FREEZE CHECKLIST — RESUELTA (despliegue agente 2026-06-13)

- [x] Knobs §1 comprometidos (defaults del agente, EXPUESTOS para veto de Pablo).
- [x] Variantes: A=`quality`, B=`quality_momentum`; **B desplegada** (1 cuenta), n_trials=2 conservador.
- [x] Umbrales §4: DSR>0.5, PBO<0.5, cobertura≥60%.
- [x] Libro cableado + verificado live: rerank full-universe **501/503 cobertura EDGAR
      (99.6% ≫ 60%)**, 48 nombres, gross 0.54, snapshot propio, baseline intacto,
      `quality_nav` sintetizado en track_record (verificado +0.06% una fila).
- [x] Cuenta: **synthetic mark-to-market** (forzado por restricción — sin 2ª cuenta;
      2ª cuenta = T5.4, trigger = primera sleeve activa, que es ESTA).
- [x] Congelado con `status: frozen` + cargo confirmado en ledger (quality=2).

**Veto disponible para Pablo (sin coste — dinero real bloqueado):** cambiar variante
desplegada (A↔B), overlay, umbrales, o pausar el plist. Cualquier cambio post-hoy =
enmienda + cargo nuevo en el ledger.
