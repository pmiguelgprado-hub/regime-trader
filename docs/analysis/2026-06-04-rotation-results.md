---
type: analysis
status: active
tags: [regime-trader, rotation, cross-asset, validation, falsification, no-edge, r4]
created: 2026-06-04
related: ["[[2026-06-04-rotation-prereg]]", "[[2026-06-04-markov-edge-redesign]]", "[[2026-06-03-oos-validation]]"]
---

# Resultados — Vía B: rotación cross-asset dirigida por régimen

> Ejecuta el test pre-registrado de [[2026-06-04-rotation-prereg]]. **Veredicto: FAIL.**
> La rotación por régimen **no** bate a risk-parity ni a un simple *hold* equal-weight
> de la misma cesta. Bate al 60/40 clásico, pero solo por composición de cesta (tiene
> GLD/QQQ que el 60/40 no), no por timing. **Dinero real BLOQUEADO.** Resultado negativo
> robusto — exactamente lo que el pre-registro estaba diseñado para detectar.

## TL;DR

| Periodo | rotación | 60/40 | risk-parity | **cesta EW estática** |
|---|---|---|---|---|
| full 2007-24 | **0.42** / 0.33 | 0.36 / 0.28 | 0.59 / 0.43 | 0.50 / 0.38 |
| p1 2007-15 | 0.36 / 0.46 | 0.31 / 0.26 | 0.52 / 0.51 | 0.37 / 0.32 |
| p2 2016-24 | 0.48 / 0.36 | 0.42 / 0.33 | 0.65 / 0.47 | 0.63 / 0.46 |

(Sharpe / Calmar, neto de costes emparejados, `credit_cash_rf=True`. Span OOS
2007-10..2024-12, 4330 barras.)

**El gate exige batir a TODOS los benchmarks en Sharpe Y Calmar, en los 3 periodos.**
La rotación pierde contra risk-parity y contra la cesta estática en los 3 → **FAIL**.

## El hallazgo central: el régimen RESTA valor vs holdear la cesta

Una cesta equal-weight de los mismos 4 activos (SPY/QQQ/TLT/GLD), **sin régimen, sin
timing, sin vol-target**, rinde Sharpe 0.50 / Calmar 0.38. La rotación inteligente por
régimen rinde **0.42 / 0.33** — peor. El HMM no añade asignación útil; el coste de
des-riesgar en los regímenes turbulentos supera lo que ahorra, igual que el floor-sweep
de [[2026-06-03-oos-validation]] mostró para un solo activo. El patrón es idéntico:
**el HMM modula exposición, no genera alfa.**

El único benchmark que la rotación bate (60/40) es el más débil: el 60/40 clásico no
tiene oro ni Nasdaq. La cesta de la rotación sí → ese "win" es **composición de cesta
montada en el bull de bonos+oro 2008-24**, no detección de régimen.

## Dos controles que confirman "exposición, no skill"

### 1. Test de permutación de tiers (skill test)
`scripts/run_rotation_skill_test.py` reusa una sola detección de régimen y prueba las 6
permutaciones del mapa tier→allocation. La real (identidad: calma→equities) rankea **#1
de 6** (Sharpe 0.43 vs mejor-otra 0.36) — el régimen lleva *algo* de señal direccional,
consistente con [[2026-06-04-markov-edge-redesign]]. **Pero las 6 permutaciones pierden
contra la cesta estática (0.50).** La señal existe pero es demasiado pequeña para superar
el coste de des-riesgar vs quedarse en la cesta. No es desplegable.

### 2. Banda de robustez R-4 (8 procesos)
`scripts/rotation_r4_band.py`, N=8 procesos separados (R-4 = no-determinismo entre
procesos, documentado):

- Rotación Sharpe: **mediana 0.427, rango 0.373-0.490, σ 0.039**.
- **0/8 baten** la cesta estática (0.504). **0/8 baten** risk-parity (0.587).
- `plain == identity-perm` en las 8 → el swing es **R-4 puro, no un bug** de `run_rotation`.

**El swing 0.37-0.49 sobre ruido FP entre procesos descalifica el despliegue por sí solo**,
con independencia de la cuestión del edge: un Sharpe que no es reproducible no es operable.

## Bug cazado: proxy equivocado infló un PASS falso

La **primera** corrida de validación dio rotación Sharpe 0.70 → "PASS". Era un bug:
`run_rotation(frames, ROT)` sin proxy explícito usaba `next(iter(frames))` = `'GLD'` (primera
clave ordenada alfabéticamente), detectando régimen sobre **oro**, no sobre SPY como manda
el pre-registro. Con el proxy correcto (SPY), la rotación da 0.42 (= mediana de la banda
R-4) y **FAIL**. El control de robustez cazó el falso positivo antes de que llegara al
veredicto — el valor de los controles, no goalpost-moving. Fix: `proxy="SPY"` explícito +
benchmark `static_basket` añadido al gate.

## Honestidad de método

- El gate pre-registrado se evaluó tal cual se escribió. El `static_basket` (cesta EW) se
  añadió como benchmark decisivo: si la rotación no bate ni a holdear su propia cesta, el
  régimen no aporta — es el control que faltaba en el pre-registro.
- **Lección para el pre-registro:** debió exigir **robustez multi-semilla R-4** desde el
  inicio (el 0.70 de una corrida no se sostuvo en 8). Conecta con R-4 en
  [[2026-06-03-oos-validation]]. Verificar la métrica/proxy antes de fijar el claim —
  mismo patrón que el halt-bug y el cash-credit ([[feedback-test-the-requirement]]).
- No se persiguen variantes para rescatar el edge (chase de la señal #1 = sobreajuste a
  ruido). El resultado negativo se reporta y se cierra.

## Qué significa y qué NO

- **NO desplegable.** Ni edge robusto (pierde vs cesta estática y risk-parity) ni
  reproducible (R-4 0.37-0.49). **Dinero real BLOQUEADO.**
- **NO** es "la rotación cross-asset es imposible" en general — es que *esta* versión
  (mapa teórico de 0 parámetros sobre features precio-vol, proxy SPY) no supera un hold
  estático. Una versión con **features exógenos macro** (curva de tipos, spreads de
  crédito, VIX term-structure — [[2026-06-04-markov-edge-redesign]] §3) sigue siendo la
  hipótesis no falsada, pero con su propio riesgo de overfit y sin garantías.
- El régimen lleva señal direccional **marginal** (#1 de 6 perms), insuficiente para pagar
  el coste de des-riesgar. Consistente con todo el proyecto: **el HMM es clasificador de
  volatilidad, no predictor de retorno.**

## Reproducir

```bash
.venv/bin/python scripts/run_rotation_validation.py    # gate pre-registrado (FAIL)
.venv/bin/python scripts/run_rotation_skill_test.py    # permutaciones + cesta estática
for i in $(seq 1 8); do .venv/bin/python scripts/rotation_r4_band.py; done  # banda R-4
```

Artefactos: `tmp/rotation_validation.json`, `tmp/rotation_skill_test.json`,
`tmp/r4_band.log`. Tests: `tests/test_asset_rotation.py`, `test_rotation_backtest.py`,
`test_benchmarks.py`, `test_dsr.py` (34 nuevos, verdes).
