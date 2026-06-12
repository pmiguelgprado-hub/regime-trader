---
type: analysis
status: frozen
tags: [regime-trader, meta-overlay, fuzzy, options, kelly, execution, triage, pre-registration]
created: 2026-06-11
related: ["[[2026-06-03-oos-validation]]", "[[2026-06-04-markov-edge-redesign]]", "[[2026-06-05-deployed-book-prereg]]", "[[2026-06-11-alpha-engine-v2-architecture]]"]
---

# Triage del mandato "Cognitive Overlay & Quant Strategy Innovator" (2026-06-11)

> Mandato: capa de "metacognición y estrategias exóticas" sobre el HMM — regímenes fuzzy,
> opciones no lineales, cortos sintéticos/pares, order flow, Kelly modificado, ejecución
> antifrágil, dashboard. Este memo contrasta CADA pieza con la evidencia acumulada del
> proyecto ANTES de escribir código. Lo que sobrevive se construye; lo que contradice
> evidencia falsada se rechaza con cita; lo que necesita su propio gauntlet queda gated.

## 0. La restricción que el mandato ignora

El mandato asume "modelos de clasificación de regímenes estables → explotarlos
lucrativamente". La evidencia del proyecto dice otra cosa:

- **3/3 avenidas de alfa-por-régimen falsadas OOS** (timing single-asset, re-entrada,
  rotación cross-asset): el HMM es un **clasificador de volatilidad**, no un predictor de
  retorno ([[2026-06-03-oos-validation]], [[2026-06-04-rotation-results]]).
- El rol validado del HMM es **overlay de riesgo** sobre un alfa independiente (momentum
  cross-sectional, vía C). El book desplegado ni siquiera usa el overlay hmm: `crash_only`
  lo batió direccionalmente (Sharpe 0.93 vs 0.87, [[2026-06-05-deployed-book-prereg]]).
- Dinero real BLOQUEADO tras gate forward ≥12 meses pre-registrado. Nada de lo de hoy lo
  desbloquea.

**Consecuencia:** toda pieza del mandato que use $S_t$ como señal de *retorno* hereda una
premisa falsada → se rechaza o se re-formula como acción de *riesgo*. Toda pieza que sea
una familia de alfa nueva (pares, carry de opciones, order flow) es un programa de
investigación propio con su prereg y su gauntlet — no "una capa" que se atornilla.

## 1. Veredictos

| # | Propuesta del mandato | Veredicto | Por qué |
|---|---|---|---|
| 1a | Regímenes híbridos / fuzzy (superar lag de transición) | **BUILD** | El lag actual es un artefacto del argmax: `weight_fn(ts, vol_rank)` recibe el rank del estado MAP y descarta el posterior filtrado completo que `HMMEngine.predict_regime_proba` ya emite. `E[vol_rank | posterior]` es continuo, causal, 0 parámetros nuevos. No es alfa nuevo: es quitar una discretización a un overlay ya validado como rol. |
| 1b | Señal de transición inminente | **BUILD** | `P(tier alto en t+1) = (π_t A)·1_high` con la transmat ya aprendida. Causal, 0 parámetros. Alimenta hedge de opciones y dashboard. NO predice dirección — predice riesgo de cambio de régimen de vol, que es lo único que el HMM sabe hacer. |
| 2a | Tail-hedge con put spreads al detectar transición a alta vol | **BUILD (dry-run) + GATED (submit)** | Consistente con HMM=clasificador de vol: comprar protección es acción de RIESGO, no de alfa. Riesgo definido (debit spread), presupuesto de prima duro. Sin datos históricos de chains no hay backtest honesto → forward-paper only, igual que vía C. Submit real de órdenes gated tras revisión de Pablo (cuenta paper necesita nivel de opciones activado). |
| 2b | Iron condors dinámicos / strangles asimétricos (vender vol en calma) | **GATED (diseño, OFF)** | Short-vol = cola pesada; recoger primas funciona hasta que un gap se lleva años de carry. Sin chains históricas no es backtesteable; el "régimen calmado" del HMM no está validado como predictor de vol FUTURA realizada vs implícita (eso es el variance risk premium — programa propio). Builder de condor incluido en el módulo pero `enabled: false` y sin path de submit. |
| 2c | Cortos sintéticos en régimen bajista | **REJECT** | Es timing direccional por régimen — exactamente la vía falsada 2026-06-03 con 6 ETFs/20 años. Cambiar "reducir gross" por "ponerse corto" multiplica la apuesta sobre la misma premisa muerta. El libro ya tiene de-risk continuo (crash_only/vol_target). |
| 2d | Pairs trading de alta velocidad | **REJECT (HF) / DEFER (lento)** | "Alta velocidad" con feed IEX gratuito (~3% del tape), sin colocation y con latencia retail = expectativa negativa tras costes. Pares lentos por cointegración = familia de alfa nueva → pertenece al pipeline Alpha Engine v2 ([[2026-06-11-alpha-engine-v2-architecture]] §3 long/short), con su propio prereg. No es una capa del HMM. |
| 2e | Order flow → "micro-tendencias institucionales" | **REJECT (datos)** | VPIN (Easley/López de Prado/O'Hara 2012) y OFI (Cont/Kukanov/Stoikov 2014) requieren tape completo (SIP, $99+/mes) o L2. Con IEX gratuito la señal es ruido muestreado. Misma decisión pendiente que ML v2: presupuesto de datos primero ([[2026-06-04-ml-v2-deferral-decision]]). Si Pablo paga datos, se diseña con su prereg. |
| 3a | Kelly modificado anti-correlación, capital fluye al cambiar régimen | **BUILD (lib) / NO cableado a libros** | Kelly multi-activo `f*=Σ⁻¹μ` con μ̂ de estrategias SIN edge validado = apalancar ruido de estimación (Michaud). Lo defendible hoy: presupuestador de riesgo ERC (equal risk contribution) con shrinkage Ledoit-Wolf y Σ mezclada por posterior de régimen (las correlaciones "ocultas" SON las del régimen de pánico). Kelly fraccional ≤0.25 implementado pero OFF, gated a sleeves con gate forward pasado. NO se cablea a los libros live: reasignar capital mid-gate cambia la estrategia pre-registrada. |
| 3b | Ejecución antifrágil: órdenes ocultas, escalonado | **BUILD parcial** | Alpaca NO soporta iceberg/hidden orders (hecho, no opinión) → esa mitad es inimplementable. Slicing TWAP con límites marketables + escalada a market sí: útil en días de transición, inofensivo el resto. OFF por defecto; el book mensual de large-caps a tamaño paper apenas mueve spread. |
| 4 | Dashboard estilo AIOS + curva de cartera Alpaca | **BUILD** | Sin riesgo de estrategia. `get_portfolio_history` verificado en alpaca-py 0.43.4 (venv real). Data layer puro + vista, patrón ya establecido. Gauge de hazard de transición incluido. |

## 2. Matemática de la capa fuzzy (lo que se construye)

Posterior filtrado (ya existente, causal): $\pi_t(s) = P(S_t=s \mid y_{1:t})$.

- **Vol-rank esperado** (sustituye al rank del argmax):
  $\widetilde{vr}_t = \sum_s \pi_t(s)\, vr(s)$, con $vr(s)$ el rank estático por
  `expected_volatility` que ya computa `StrategyOrchestrator`. Estimador puntual
  Bayes-óptimo bajo pérdida cuadrática; elimina el cliff del argmax y el lag del filtro de
  estabilidad para el *sizing* (la confirmación de 3 barras sigue gobernando lo discreto).
  `regime_gross_scale` ya interpola linealmente → con entrada continua el gross es continuo.
- **Hazard de transición a tier alto**:
  $h_t = \sum_s \pi_t(s) \sum_{s' \in \mathcal{H}} A_{ss'} = (\pi_t A)\cdot \mathbf{1}_{\mathcal{H}}$,
  $\mathcal{H} = \{s: vr(s) \ge $ `HIGH_VOL_MIN`$\}$. Incluye la masa ya en $\mathcal{H}$
  (estar en pánico = hazard alto, correcto para un trigger de hedge).
- **Entropía predictiva normalizada**: $H(\pi_t A)/\log K$ — incertidumbre del régimen de
  mañana, métrica de dashboard y de "uncertainty mode".

Cero parámetros ajustados nuevos → nada que barrer → nada que sobreajustar. Modo de
overlay nuevo: `hmm_prob` (idéntico a `hmm` pero alimentado con $\widetilde{vr}_t$).

## 3. Pre-registro v1 del tail-hedge de opciones (congelado antes de cualquier submit)

- **Trigger:** $h_t \ge 0.35$ dos cierres consecutivos, estando el book con gross > 0.6.
- **Estructura:** put debit spread sobre el proxy (SPY): long put ~4% OTM, short put ~10%
  OTM, vencimiento 30–60 DTE (el mensual más cercano al centro).
- **Presupuesto:** prima neta ≤ 25 bp del equity por trimestre, ≤ 100 bp/año. Máx 1
  estructura viva. Sin rolls automáticos en v1 (expira o se cierra al volver $h_t < 0.20$
  durante 5 cierres).
- **Métrica del gate (forward paper ≥ 2 ciclos de hedge):** maxDD del book con hedge vs
  sin hedge, y coste anual realizado vs presupuesto. Si el drag supera el ahorro de DD →
  se apaga. El condor short-vol queda fuera de v1.
- Estos números son defaults de practicante (drag típico de programas sistemáticos de put
  spreads 50–150 bp/año), elegidos UNA vez, sin sweep. Cambiarlos = enmienda documentada.
- **ENMIENDA 2026-06-12 (pre-activación, antes de todo track record):** `budget_quarter_bp`
  25→50 — el smoke real cazó que 25 bp ($245 en cuenta de $98k) no compraba ni 1 spread
  (debit real $466.5). Activado en paper con allow_orders. Ver
  [[2026-06-12-hedge-activation]].

## 4. Qué NO cambia hoy

- Book desplegado (`overlay: crash_only`) intacto. `hmm_prob` queda disponible para eval
  direccional y para el challenger, no se despliega.
- Ningún plist nuevo cargado, ningún submit de opciones, ningún flujo de capital
  inter-libro. Dinero real sigue BLOQUEADO.
