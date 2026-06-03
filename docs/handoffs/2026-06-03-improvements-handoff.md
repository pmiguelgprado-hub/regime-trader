# Handoff — regime-trader · Optimización post-edge-fix

_Creado 2026-06-03. Foco siguiente sesión: **validar edge + completar optimización**. Ejecutar en `~/AIOS/projects/regime-trader/`. Rama: `feat/high-priority-improvements` (pusheada a GitHub `pmiguelgprado-hub/regime-trader`)._

## Prompt para pegar en la terminal nueva
> Lee este handoff + `docs/analysis/2026-06-02-why-no-edge.md` + `docs/analysis/2026-06-02-optimization-and-roadmap.md`. Estás en la rama `feat/high-priority-improvements` (180 tests verdes). El bot ya tiene edge competitivo tras arreglar el bug del halt (52.8% Sharpe 1.22 en SPY 2019-24). Tu trabajo: (1) **re-validar el edge fuera de muestra** (otros periodos/activos) — es lo primero porque el 52.8% es UN activo/UN periodo; (2) seguir la lista priorizada de optimización del roadmap. TDD estricto (test que falla→fix→verde), commits atómicos, advisor antes de declarar done. Caveman mode global activo. No habilites dinero real. Enséñame diff + `pytest` por cambio.

## 1. Goal
El proyecto pasó de "PAPER-READY pero sin edge (bloqueado dinero real)" a **"edge competitivo provisional"** tras descubrir y arreglar que el cortacircuitos dejaba al bot parado >50% del tiempo. La meta ahora: **confirmar que el edge es real fuera de muestra** y cerrar los gaps de optimización. "Done" del edge = backtests limpios en varios periodos/activos con walk-forward OOS, NO "operé en paper".

## 2. State of play (2026-06-03)
- **180 tests verdes** (`.venv/bin/python -m pytest -q`, py3.14). Rama `feat/high-priority-improvements` pusheada (`origin` = `https://github.com/pmiguelgprado-hub/regime-trader.git`).
- **EL HALLAZGO de la sesión:** el veredicto "sin edge" (6.9% vs 69.9%) era un **BUG**, no la estrategia. El halt por drawdown-de-pico ponía peso 0 → equity congelado bajo el pico → drawdown-de-pico se quedaba >10% para siempre → bot **parado 53% del backtest, 1 sola transición** (nunca se reactivaba), perdiéndose la recuperación. Fix (`halt_floor_mult=0.25`): en halt mantiene 25% mínimo → equity recupera → breaker se levanta. **Re-medición: 6.9%→52.8%, Sharpe -0.12→1.22, parado 53%→0.9%.** Detalle: `docs/analysis/2026-06-02-why-no-edge.md`.
- **Strategy vs benchmarks (SPY 2019-24, limpio):** strategy 52.8% / Sharpe 1.22; buy&hold 69.9% / 1.45; SMA200 40.9% / 1.00; random 27.3%. **Bate SMA200 y random; aún <buy&hold en bull puro** (esperable para des-riesgo).

## 3. Hecho esta sesión (commits 7e075e7..9514506)
- **Streamlit fix** (7e075e7): sidebar dentro de `@st.fragment` → fragment dedicado. Verificado con `AppTest`.
- **R-1** (7fe2186): slippage escalado por volatilidad (`_slippage_rate`), opt-in `slippage_vol_coeff` (default 0).
- **A-3** (98ce214): `core/drift.py` — PSI + entropía posterior + predicado. NO cableado al lazo aún.
- **A-1** (5831401, 81b425f): `install_model` (propaga a orquestador vía `update_regime_infos` — el bug real), `retrain_from_buffer` (guardas), `maybe_retrain` (opt-in `hmm.auto_retrain` OFF por defecto).
- **halt-floor** (b5eb2fe): EL fix. `RiskConfig.halt_floor_mult`; `target_size_multiplier` devuelve floor en HALTED (solo afecta sizing del backtester; vivo conserva hard-halt).
- **A-4** (7a1c711): champion-challenger gate (`mean_log_likelihood` en holdout) + `core/model_registry.py` (versionado + rollback).
- **S-2** (6922be6): `broker/stream_supervisor.py` (reconexión backoff + watchdog stale).
- **S-1** (a2b9c4c): runbook `docs/runbooks/s1-live-verification.md` (sesión en vivo — NO automatizable).
- **E-1 v1** (9514506): `core/portfolio.py` + `Backtester.run_portfolio` (multi-activo BACKTEST, equal-weight capado, régimen compartido).

## 4. Pendiente — orden recomendado
1. **Re-validar edge OOS (PRIORIDAD 1).** El 52.8% es SPY/2019-24. Correr otros periodos (incluir 2008/2015/2018/bear) y otros activos. Confirmar walk-forward sin leakage. Sin esto el edge sigue provisional para dinero real.
2. **Cerrar hueco vs buy&hold** (hipótesis ya testables, NO contaminadas): diferenciar bear (mantener) vs crash (cortar); re-entrada más rápida tras vol; overlay tendencia SMA200; revisar floor 60%.
3. **E-1 v2:** cablear `portfolio_target_weights` al lazo VIVO (`process_symbol` por cartera), activar correlación (alimentar `price_history`), pesos vol-parity/trend.
4. **Aprendizaje:** cablear drift→retrain (A-3→A-1, persistir distribución de features de entrenamiento como referencia); A-5 (feedback de P&L por régimen → recalibrar `min_confidence`); A-4 follow-up (gate por Sharpe/maxDD + holdout purgado).
5. **Realismo/ops:** R-2 (coste financiación leverage), R-3 (unificar datos train=Alpaca no yfinance), guard de orden abierta (footgun: `--run-once` 2×/día duplica).
6. **Decisión:** alinear halt backtest (floor 25%) vs live (cierra todo) — ¿live también floor? cambia perfil de riesgo real.
7. **S-1:** ejecutar sesión en vivo supervisada (runbook) — solo Pablo.

## 5. Cómo correr
```bash
cd ~/AIOS/projects/regime-trader
.venv/bin/python -m pytest -q                                   # 180 tests
.venv/bin/python main.py --backtest --compare --symbols SPY --start 2019-01-01 --end 2024-12-31
.venv/bin/python -c "from backtest.backtester import *; ..."    # run_portfolio para multi-activo
```
Despliegue real = `--run-once` (launchd L-V 22:15, proceso nuevo/día). NO `--live` (run_stream = cadencia minuto, estrategia es diaria).

## 6. Ficheros clave
- `main.py` `TradingSystem` (lazo vivo: `process_symbol`, `run_cycle`, `install_model`/`retrain_from_buffer`/`maybe_retrain`).
- `core/risk_manager.py` (`halt_floor_mult`, `target_size_multiplier`, `update_drawdown_state` no-latching).
- `backtest/backtester.py` (`run` single-asset, `run_portfolio` multi, `_slippage_rate`).
- `core/{drift,portfolio,model_registry}.py` (nuevos).
- `config/settings.yaml` (flags: `slippage_vol_coeff`, `auto_retrain`, `challenger_tol`, `halt_floor_mult`, `max_reconnects`).

## 7. Lecciones (no repetir)
- **No concluir de una medición rota.** "Sin edge" era el bug del halt. Antes de juzgar la estrategia, verifica que la medición no está contaminada.
- **Verifica el CLAIM, no solo que la cita existe** (A-2 era falso — labels se re-derivan cada fit).
- **Test el requisito, no el camino cómodo** (halt-recovery: el test diferencial floor-vs-0, no un umbral arbitrario).
- **advisor antes de declarar done** (cazó: A-2 falso, auto-promote sin gate, la contaminación del edge).
- Lecciones previas: [[feedback-test-the-requirement]], [[feedback-streamlit-fidelity]], [[audit-playbook-ai-built]].
