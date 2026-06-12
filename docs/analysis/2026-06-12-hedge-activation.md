---
type: analysis
status: frozen
tags: [regime-trader, options, tail-hedge, activation, amendment, no-paid-data]
created: 2026-06-12
related: ["[[2026-06-11-meta-overlay-triage]]", "[[2026-06-11-meta-overlay-directional-results]]", "[[2026-06-04-ml-v2-deferral-decision]]"]
---

# Activación tail-hedge (paper) + decisión datos de pago — 2026-06-12

## 1. Activación (orden de Pablo, 2026-06-12)

`options_hedge.enabled: true` + `allow_orders: true`. El plist nocturno corre
`--rebalance --execute`, así que un trigger real ENVÍA el MLEG en la cuenta paper.

Verificación previa (evidencia, no supuestos):

- **Cuenta paper: options_trading_level 3** (multi-leg permitido), status ACTIVE,
  options_buying_power $55,490 — consultado vía API real.
- **Smoke end-to-end dry-run contra la chain real de SPY** (spot 738.15):
  selección correcta long 709P / short 664P (= 4.0%/10.0% OTM exactos), expiry
  2026-07-24 (42 DTE, dentro de 30–60), net debit real 4.665 desde quotes mid.
  State temporal — el estado live no se contaminó.

## 2. Enmienda pre-activación: `budget_quarter_bp` 25 → 50

El smoke cazó un fallo de calibración del prereg: 25 bp de $97,999 = $245 de
headroom trimestral, pero UN spread cuesta $466.5 (debit 4.665 × 100). Con el
cap original el overlay no podía actuar nunca — un hedge que jamás compra no es
conservador, es decorativo.

- Enmienda hecha ANTES de iniciar el track record del hedge (se activa hoy):
  no contamina ningún forward acumulado. El prereg queda congelado desde hoy
  con 50 bp/trimestre.
- 50 bp = exactamente 1 estructura/trimestre a precios actuales; el cap anual
  (100 bp, sin cambio) sigue siendo el límite duro a ~2 estructuras/año —
  dentro del drag de referencia de programas sistemáticos de put spreads
  (50–150 bp/año).
- `max_structures: 1` sin cambio. Métrica del gate sin cambio (maxDD con vs sin
  hedge + coste realizado vs presupuesto, ≥2 ciclos de hedge).

## 3. Decisión: NO datos de pago (Pablo, 2026-06-12)

Regla: si una vía exige datos de pago y no existe equivalente gratuito honesto,
**no se implementa**. Consecuencias, vía por vía:

| Vía | ¿Fuente gratuita honesta? | Decisión |
|---|---|---|
| Tail-hedge opciones (forward) | SÍ — Alpaca da chains + quotes live gratis; lo que no hay gratis es histórico de chains, por eso es forward-only sin backtest | ACTIVADO hoy |
| Order flow / VPIN / OFI | NO — exigen SIP completo ($99+/mes) o L2; IEX gratuito ≈3% del tape = ruido muestreado | CERRADO (no implementar) |
| Value PIT / FundamentalScreen (Alpha Engine v2 fase value) | NO — fundamentales point-in-time sin supervivencia son de pago (Sharadar/CRSP); SimFin gratuito no es PIT-limpio para backtest honesto | CERRADO como backtest; solo viable como forward-paper puro si algún día se prioriza |
| ML v2 (panel de entrenamiento) | NO — mismo muro PIT | CERRADO (la deferral de [[2026-06-04-ml-v2-deferral-decision]] pasa de "pendiente de decisión" a decidida: no) |
| Pares lentos por cointegración | PARCIAL — precios diarios gratis bastan para investigar, PERO es programa de alfa nuevo con su propio prereg, no bloqueado por datos | Queda en backlog Alpha Engine v2 (no por datos — por prioridad) |

Cierra la "decisión datos pendiente de Pablo" que arrastrábamos desde 2026-06-04.
