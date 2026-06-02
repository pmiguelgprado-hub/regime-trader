---
type: runbook
status: active
tags: [regime-trader, live, verification, circuit-breaker, S-1]
created: 2026-06-02
related: ["[[go-live-review]]", "[[improvement-review]]"]
---

# S-1 — Verificación en mercado abierto (sesión supervisada)

**Por qué existe:** C2/C5/C6 (stream de fills → P&L → breaker MtM → halt →
liquidación) están cableados y **unit-tested**, pero nunca ejercitados con el
mercado abierto y un fill real. Esto NO lo puede hacer un test ni el agente:
requiere una sesión de paper supervisada por Pablo. Este runbook es el checklist.

> Requisito: cuenta **paper** Alpaca con claves en `.env`. NO usar live.
> Mantén `broker.paper_trading: true`.

## Preparación (mercado cerrado)
1. `cp .env.example .env` (si no existe) + claves paper. Verifica:
   `.venv/bin/python -m pytest -q` → todo verde.
2. `.venv/bin/python main.py --train-only --symbols SPY` → confirma
   `models/hmm_SPY.pkl` (o `models/SPY/` si registro A-4 activo).
3. Smoke sin órdenes: `.venv/bin/python main.py --dry-run --symbols SPY` →
   debe emitir decisión (approved_shares > 0).
4. Logs en vivo en otra terminal: `tail -f logs/*.log`.

## Durante la sesión (mercado ABIERTO) — checklist C5→C2→C6
Lanza UN ciclo: `.venv/bin/python main.py --run-once --symbols SPY`. Verifica, en
orden, en logs + en el order history de Alpaca:

- [ ] **C5 — llega un fill real al stream.** La orden bracket entra (NEW→FILLED al
      open). Log `signal_submitted_*` + el tracker registra la posición/precio medio.
- [ ] **P&L realizado** aparece en el tracker tras el fill (no 0 tras llenarse).
- [ ] **C2 — el breaker recibe equity por barra.** Log de `_update_risk_posture`:
      `risk_state` refleja la equity MtM (realizada + no realizada), no se queda
      en NORMAL artificialmente.
- [ ] **Stop protector colocado (C3).** El bracket tiene pierna STOP con id
      capturado (`stop_order_id` poblado); `update_trailing_stops` ya no es no-op.
- [ ] **C6 — halt liquida.** (Opcional, forzar con umbral bajo en config de prueba:
      bajar `risk.max_dd_from_peak` temporalmente) → al cruzar, `close_all_positions`
      aplana TODO y se levanta alerta `circuit_breaker_halt`.

## Criterios de aborto (parar y revisar)
- Sobre-acumulación: el bot compra el target completo repetidamente (fallo C4 — no
  debería; ya reparado). Verifica que en barras repetidas NO re-compra.
- Órdenes duplicadas: ejecutar `--run-once` más de 1×/día antes de que llene
  (footgun documentado). Programa launchd 1×/día.
- El breaker no recibe equity (risk_state clavado en NORMAL pese a caída) → C2 roto.

## Post-sesión
- Revisa `state_snapshot.json` + dashboard (`streamlit run monitoring/streamlit_app.py`).
- Registra resultados aquí (fecha, qué se verificó, capturas del order history).
- Tras ≥1 mes de paper limpio + edge validado → recién considerar dinero real
  (sigue bloqueado hasta entonces).

## Estado
- [ ] Pendiente de primera sesión supervisada (a fecha 2026-06-02).
