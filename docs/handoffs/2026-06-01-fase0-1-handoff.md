# Handoff — regime-trader · Fase 0 + Fase 1

_Creado 2026-06-01. Foco siguiente sesión: **implement**. Ejecutar en `~/AIOS/projects/regime-trader/`._

## Prompt para pegar en la terminal nueva
> Lee primero `docs/audit/2026-06-01-senior-audit.md` (auditoría senior; §5 plan por fases, §7 parches) y este handoff. Ejecuta **Fase 0** y luego **Fase 1** con TDD (test que falla → fix → verde), commits atómicos por cambio, sin refactors grandes. No toques el lazo en vivo más allá de C1. No habilites paper trading. Caveman mode global activo. Para cada fase: enséñame el diff y la salida de `pytest` antes de pasar a la siguiente.

## 1. Goal
Dejar el repo reproducible (Fase 0) y que el lazo en vivo emita señal desde la barra 1 (Fase 1, bug **C1**). Ninguna de las dos habilita paper — eso exige también C2/C3/C4 (fases posteriores). "Done" = verificado por test/dry-run, no "operé en paper".

## 2. State of play
- **Auditoría completa y verificada** → `docs/audit/2026-06-01-senior-audit.md`. Veredicto: NO APTO PARA PAPER. 102 tests pasan (py3.14, `.venv`).
- **Backtest/risk-unit/broker-vs-mock**: sólidos. **Lazo en vivo**: incompleto (C1–C6).
- **Conformidad vídeo↔código**: §8 del audit. Build fiel a las 8 fases; C1–C6 son huecos del propio vídeo (Fase 7 hand-wavy).
- **Pendiente esta tanda**: Fase 0 + Fase 1. Nada empezado aún.

### Fase 0 — Higiene (riesgo ~nulo). Detalle en audit §5/§7.
1. **H2** README CLI: cambiar `--mode backtest`/`main.py backtest …` por flags reales (`--backtest`,`--live`,`--dry-run`,`--train-only`,`--dashboard`,`--stress-test`). Fuente correcta: `main.py:694-713` y `docs/go-live-review.md`.
2. **M6** `requirements.txt`: pinear versiones; borrar `alpaca-trade-api` (legacy; el código usa `alpaca-py`).
3. **Gap tests 102 vs 134** (el vídeo acaba en 134, build en 102): **diagnosticar, no fabricar**. `pytest --co -q | wc -l`, comparar por módulo; reportar si es cobertura perdida o granularidad distinta. No inventar tests para cuadrar el número.
4. **M1** (opcional, bajo riesgo): borrar `core/signal_generator.py` (esqueleto muerto, sin import en el pipeline).
- **Done F0**: comandos del README arrancan; `pytest` verde; gap de tests documentado en el audit.

### Fase 1 — Backfill buffer en vivo (C1). Parche base en audit §7.
- En `main.py::run_live`, antes de `system.run_stream(md)`, sembrar `system.buffers[sym]` con histórico (`warmup ≥ min_train_bars + 260`). **Verificar la firma real** de `get_history`/`get_history_multi` (`data/market_data.py:141/160`) antes de escribir.
- **TDD**: test que falla hoy → con buffer sembrado de ~500 barras, `process_symbol` devuelve señal en la barra 1 (hoy `[]` por buffer frío).
- **Done F1**: test verde + dry-run muestra señal inmediata. **NO habilita paper.**

## 3. Open decisions
- **Dashboard (BLOQUEADO por el usuario): Streamlit web fiel al vídeo** (`streamlit run`, `localhost:8501`), reemplazando el `monitoring/dashboard.py` rich-terminal actual. Ref visual: vídeo §8 del audit (frames 7/74), panels = regímenes aprendidos, risk controls (Daily/Weekly/Peak DD, leverage, exposure, circuit-brk), signal feed, price+regime overlay, volumen/confianza. **NO se ejecuta en F0/F1** — es fase propia (añadir `streamlit` a requirements, leer estado de `state_snapshot.json`/IPC). Capturado aquí como requisito firme; planificar tras la fase de seguridad.
- **Gap 102-vs-134 tests**: lean = solo diagnosticar en F0; fabricar tests faltantes = fase posterior si el diagnóstico revela cobertura perdida.
- **Timeframe daily vs 5-min (M3)**: decisión necesaria **antes de paper**, no en F0/F1. Lean = daily (coherente con calibración).

## 4. Skills to use
- `superpowers:test-driven-development` — Fase 1 (test rojo → fix → verde) y cualquier test nuevo.
- `superpowers:verification-before-completion` — correr `pytest` y pegar salida antes de declarar done.
- `superpowers:systematic-debugging` — diagnóstico del gap de tests.
- (Fase dashboard, después) Streamlit nativo; no usar `frontend-slides`/`taste` (son para HTML/slides, no Streamlit).

## 5. Artifacts
- Auditoría (fuente de verdad, plan + parches): `docs/audit/2026-06-01-senior-audit.md`
- Review honesta del autor: `docs/go-live-review.md`
- CLI real: `main.py:694-713` · Buffer/lazo: `main.py:167-228, 502-579`
- Backfill a usar: `data/market_data.py:141,160` · Backtest delta (modelo a portar luego): `backtest/backtester.py:212-230`
- Memoria proyecto: `~/.claude/projects/-Users-pablomiguelgonzalezprado-AIOS/memory/project-regime-trader.md`
