# Alpha Engine v2 — arquitectura value/fundamentals, long/short y crypto (2026-06-11)

**Estado: DISEÑO. Nada de esto toca los books congelados.** El baseline (raw momentum +
crash_only) y el challenger (residual momentum + vol-target) corren su gate forward de
≥12 meses con knobs congelados. Mutarlos a mitad de gate invalida la pre-registración.
v2 es una pista de investigación paralela con su propio prereg, paper-only.

## 0. Qué pidió el mandato y qué resiste el contraste con la evidencia

El mandato (sesión 2026-06-11) pide: value investing por PER, long/short con
apalancamiento, crypto con arbitraje y "caza de exploits", loop de 15 minutos con
modelos OpenRouter, todo orientado a maximizar Sharpe. Evaluación honesta, punto por
punto:

| Petición | Veredicto | Razón |
|---|---|---|
| Value screen (PER vs sector + solvencia) | ✅ viable, cadencia MENSUAL | Factor documentado (HML, Fama-French; QMJ, Asness). Los fundamentales cambian por trimestre: un screen value a 15 min es incoherente — no existe información nueva que explotar entre informes |
| Long/short beta-neutral | ✅ viable en paper | Alpaca paper soporta shorts; el challenger ya estima betas market-model (est_window 504) que se reutilizan para neutralizar |
| Apalancamiento | ⚠️ gated | `gross_cap: 1.0` está congelado en ambos books. v2 L/S usa gross 2× (100/100) con net ~0 — eso no es apalancamiento direccional |
| Crypto momentum | ⚠️ sleeve pequeño, TS-momentum semanal | Evidencia académica de XS/TS momentum en crypto existe (SSRN 4322637; AUT con costes reales; Springer 2025 vol-managed) pero los costes y el slippage se comen gran parte. Solo BTC/ETH líquidos |
| Arbitraje crypto / "exploits" de mercado | ❌ rechazado | Arbitraje cross-exchange retail muere por fees + latencia + riesgo de custodia. "Detección de exploits/ineficiencias" sin definición falsable = la receta exacta del overfitting que ya falsamos 3 veces (R1, vía B, floor-sweep) |
| Loop 15 min para comprar/vender intradía | ❌ como alfa, ✅ como riesgo | `timeframe: 1Day` está LOCKED (M3): HMM, vol-regimes y breakers calibrados a diario. El HMM demostró ser clasificador de vol, no predictor de retorno. El loop de 15 min construido hoy (`--risk-check`) hace lo único defendible: vigilar drawdown intradía y cortar exposición en pánico (alert → derisk → flatten) |
| OpenRouter/LLM en el loop | ⚠️ solo anotación | R-4: no-determinismo entre procesos. Un LLM jamás gatea órdenes; puede anotar alertas (clasificar titulares al disparo de un derisk) como contexto humano |
| Maximizar Sharpe | ✅ reformulado | La vía real al Sharpe de cartera no es apalancar un sleeve: es combinar sleeves poco correlacionados (momentum equity + value L/S + crypto TS-mom) + vol-targeting, que es el único overlay que sobrevivió validación propia (Barroso; Daniel-Moskowitz) |

## 1. Restricción vinculante: datos point-in-time

El value screen necesita fundamentales **point-in-time** (lo que se sabía en la fecha,
no lo restateado). Fuentes gratis (yfinance) traen restatements + survivorship → un
backtest value con ellas queda sesgado al alza y no es evidencia. Es la misma decisión
de datos de pago ya pendiente para el predictor ML v2 (memo 2026-06-04). Opciones:

- **Sin datos PIT** (gratis): el screen solo puede evaluarse **forward** — se publica el
  book en paper hoy y se mide 12 meses. Lento pero limpio. Coste 0.
- **Con datos PIT** (SimFin+ ~€35/mes, Sharadar/Nasdaq Data Link ~$49/mes, EODHD):
  backtest histórico honesto posible antes del paper.

**Decisión de Pablo, fase 0.** Sin ella no se escribe `fundamental_screen.py`: módulo
sin fuente de datos = cableado muerto (lección del audit playbook — hoy mismo un
`get_orders` muerto tuvo al bot 3 sesiones sin operar).

## 2. Módulo FundamentalScreen (diseño, no implementado)

```
core/fundamental_screen.py
  @dataclass Fundamentals: pe, debt_to_equity, current_ratio, interest_coverage,
                           fcf_yield, roa, accruals, shares_out_change  # por símbolo, PIT
  def sector_relative_value(fundamentals, sector_map) -> dict[str, float]
      # z-score del earnings yield (1/PE) vs mediana GICS del sector (reusa el
      # sector_map de data/constituents.py que ya alimenta max_sector_fraction).
      # Earnings yield, no PE: maneja earnings negativos sin descontinuidad.
  def solvency_filter(fundamentals) -> set[str]
      # Piotroski F-score >= 5 + Altman Z > 1.8 + interest_coverage > 2.
      # El value sin filtro de salud compra value traps — el cross-check que pide
      # el mandato es exactamente lo que Piotroski (2000) formalizó.
  def value_scores(...) -> dict[str, float]   # mismo contrato que momentum_scores
```

Contrato idéntico a `cross_sectional_ranking.momentum_scores` → entra directo en
`Backtester.run_portfolio(weight_fn=)` y en el pipeline de `run_rebalance` sin tocar
ejecución. Cadencia: re-rank **mensual** con datos del último trimestre publicado;
los runs diarios solo re-escalan gross (mismo patrón que el book actual).

## 3. Framework Long/Short (diseño)

- **Construcción**: rank por score combinado (value z + momentum z — la combinación
  value+momentum es el par clásico con correlación negativa, AQR "Value and Momentum
  Everywhere"); long decil superior, short decil inferior, equal-weight por pata.
- **Neutralidad**: beta de mercado por nombre con el market-model del challenger
  (`est_window: 504`); escalar la pata corta para β_long ≈ β_short → net beta ~0.
- **Gross**: 2× NAV (100/100), `gross_cap` propio de v2; el vol-target overlay
  (reusa `vol_target_scale`) escala el gross total a 12% anualizado.
- **Plomería de órdenes**: `plan_rebalance_orders` ya difiere target vs held con
  signos — falta permitir `target_shares` negativos y mapear delta<0 sobre posición 0
  a sell-to-open (Alpaca lo trata como short si el nombre es shortable). Guard nuevo:
  excluir nombres hard-to-borrow (campo `shortable` del asset).
- **Riesgos específicos cortos**: squeeze (cap por nombre 1% + stop de pérdida por
  posición corta), crowding, y el momentum crash — la pata corta es donde el factor
  explota en rebotes post-crisis (Daniel-Moskowitz 2016); el vol-targeting existe
  precisamente para eso.

## 4. Sleeve crypto (diseño, fase 3)

- Alpaca soporta crypto en paper (24/7, sin PDT). Sleeve **separado** con NAV asignado
  pequeño (≤10% del book v2), nunca mezclado con el book equity.
- Señal: **time-series momentum semanal** sobre BTC/ETH (señal binaria sobre retorno
  12-semanas, cash si negativo). XS momentum necesita ≥20 nombres líquidos y la cola
  de liquidez en crypto es venenosa — el paper de AUT con costes reales muestra que
  los retornos brutos se comprimen fuerte tras fees/slippage.
- Vol-target al sleeve (la vol de BTC es 3-5× equity; sin esto domina el riesgo total).
- Arbitraje y "anomalías de volumen": fuera. No falsable con infra retail.

## 5. Por qué esto puede subir el Sharpe de la cartera (y qué lo decidirá)

Sharpe de cartera = f(Sharpe de cada sleeve, correlaciones). Momentum equity (book
actual), value L/S (correlación histórica negativa con momentum) y crypto TS-mom
(correlación baja con ambos) es la única palanca estructural disponible que no es
apalancamiento. Cada sleeve lleva su gate pre-registrado (DSR/PBO como el actual,
`pbo_cscv` ya está en el repo). **Nada se promociona por backtest bonito: solo por
forward paper ≥12 meses.** La evidencia propia del proyecto (3 avenidas falsadas)
es la razón de este estándar.

## 6. Roadmap gated

| Fase | Entregable | Gate de entrada |
|---|---|---|
| 0 | Decisión datos PIT (Pablo): gratis-forward-only vs SimFin/Sharadar | — |
| 1 | `fundamental_screen.py` + dry-run book value mensual + prereg | Fase 0 |
| 2 | Soporte shorts en plomería (targets negativos, shortable guard) + book L/S paper (3ª cuenta Alpaca) + prereg | Fase 1 + verificación supervisada |
| 3 | Sleeve crypto TS-mom paper + prereg | Fase 2 estable |
| — | Risk monitor 15-min: **hecho hoy** (observe mode; `allow_orders` = enmienda de gate documentada) | — |

## 7. Construido hoy (2026-06-11), verificado

1. Fix `get_orders` → `get_order_history` (main.py:1301) + test AST anti-cableado-muerto
   (`tests/test_client_wiring.py`). Rebalance ejecutado real: 55 órdenes PAPER, 60/60 FILLED.
2. `core/risk_monitor.py` + `main.py --risk-check` + `deploy/com.regimetrader.riskcheck.plist`
   (NO cargado — gated). 13 tests nuevos. Verificado contra mercado abierto: `+2.14% → ok`.
3. `deploy/com.regimetrader.dashboard.plist` (NO cargado); dashboard relanzado, HTTP 200.

## Referencias

- Piotroski (2000) — F-score; Asness et al. — Quality Minus Junk; AQR — Value and
  Momentum Everywhere; Daniel & Moskowitz (2016) — Momentum Crashes; Barroso &
  Santa-Clara (2015) — momentum vol-managed.
- Crypto: SSRN 4322637 (XS momentum crypto); AUT working paper TS/XS momentum con
  costes reales; Springer s11408-025-00474-9 (vol-managed crypto momentum, 2025).
- Internas: `2026-06-04-stock-picking-feasibility.md`, `2026-06-04-cross-sectional-prereg.md`,
  `2026-06-05-deployed-book-prereg.md`, `2026-06-05-idio-momentum-challenger-prereg.md`,
  `2026-06-03-oos-validation.md`.
