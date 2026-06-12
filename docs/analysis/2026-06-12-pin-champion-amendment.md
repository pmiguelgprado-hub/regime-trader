---
type: amendment
status: active
tags: [regime-trader, gate, determinism, pin-champion, R-4]
created: 2026-06-12
related: ["[[2026-06-12-improvement-roadmap]]", "[[2026-06-04-cross-sectional-prereg]]", "[[2026-06-05-deployed-book-prereg]]"]
---

# Enmienda operativa: pin-champion + determinismo (T0.4)

**Tipo: enmienda OPERATIVA, no de estrategia.** El gate congelado mide el NAV del
libro (Sharpe/maxDD/DSR contra SPY y EW a 12 meses); no preregistra internals del
modelo. Esta enmienda no toca señales, knobs de construcción, cadencias ni umbrales.
Cambia *cómo se mantiene reproducible* el artefacto de modelo que ya estaba en vivo.

## Causa raíz R-4 (resuelta, con evidencia)

R-4: procesos idénticos daban Sharpe en banda 0.37–0.49. Dos fuentes verificadas:

1. **BLAS multihilo** (la dominante). Test directo 2026-06-13: dos fits HMM
   idénticos (misma semilla, mismos datos, mismo proceso) divergen a ~5e-13 con
   Accelerate/OpenBLAS multihilo — las reducciones en coma flotante no tienen orden
   determinista. EM amplifica esa semilla de error en iteraciones, y con restarts
   casi empatados en log-likelihood el ganador cambia → modelo distinto → banda de
   Sharpe. Con `OMP/OPENBLAS/MKL/VECLIB_*_THREADS=1` los fits son **bit-idénticos**
   (`tests/test_determinism.py`, 6 tests).
2. **Refit semanal con end-date = "now"**: `--run-once` re-entrenaba por edad
   (`max_age_days: 7`) sobre datos hasta el momento del run → cada semana un modelo
   distinto por construcción, sin registro de qué modelo produjo qué decisión.

## Qué cambia

| Antes | Después |
|---|---|
| `--run-once` re-entrena si pickle >7 días | Carga el **campeón del registry** (`models/SPY/`), jamás refit por edad |
| `--rebalance` carga `models/hmm_SPY.pkl` suelto | Ídem: campeón del registry |
| Sin rastro de qué modelo corrió | `champion_hash` (SHA-256/16 de la matriz de transición) logueado en cada run; `code_sha` en cada fila del track record |
| Swap silencioso posible | Assert diario: hash cargado vs hash registrado en promoción → alerta CRITICAL si difiere |
| BLAS multihilo (no determinista) | 1 hilo en los 6 plists + `main.py` + `conftest.py` (setdefault, override posible) |

**Campeón fijado 2026-06-13 (bootstrap):** versión `20260612T222305_ba627b`,
transition hash `8c807d916bf62af5`, 5 regímenes — el pickle que ya estaba en vivo
(3.05 días de edad, no requirió retrain). Rollback: `ModelRegistry.rollback("SPY")`.

## Dual-log refit-vs-pinned (2 semanas)

Mientras la regla antigua habría re-entrenado (campeón >7 días), `--run-once` ajusta
un **shadow desechable** (mismos hiperparámetros, datos frescos) y apendea una fila a
`logs/shadow_refit.csv`: fecha, hash campeón/shadow, régimen y confianza de ambos
sobre la última barra, `agree`. El shadow no se guarda, no toca el registry, no
genera órdenes.

**Adjudicación (~2026-06-27):**
- Acuerdo alto (≥90% de días `agree`) → el pin es inocuo; apagar `hmm.dual_log_refit`.
- Desacuerdo sostenido → territorio del trigger de drift (T3.3); humano decide
  retrain + promoción vía registry (champion-challenger A-4 ya implementado).

Promoción manual hasta T3.3. `max_age_days` queda solo como pacer del dual-log.

## Congelación de end-date

Plegada en el pin: sin refit automático no hay end-date rodante. Retrains manuales
quedan versionados en el registry (timestamp UTC en el id de versión) y solo entran
en vivo por promoción explícita.

## Archivos

- `core/hmm_engine.py::transition_hash` · `core/model_registry.py::promote/champion_hash`
- `main.py::load_pinned_champion` + wiring en `run_once`/`run_rebalance` + env pins
- `core/shadow_refit.py` (dual-log) · `config/settings.yaml::hmm.dual_log_refit`
- `deploy/*.plist` (EnvironmentVariables, 6/6) — recargados 2026-06-13
- Tests: `test_determinism.py` (6), `test_shadow_refit.py` (3), `test_model_registry.py` (+4)

## Lo que NO cambia

Señales, universo, cadencia mensual/diaria, overlay, umbrales de riesgo, métricas y
ventana del gate, y el modelo concreto que estaba decidiendo (se pina, no se sustituye).
