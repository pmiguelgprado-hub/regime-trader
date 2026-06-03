---
type: spec
status: approved
tags: [regime-trader, risk, halt, reentry, sharpe, oos-validation]
created: 2026-06-03
related: ["[[2026-06-03-oos-validation]]", "[[2026-06-02-why-no-edge]]"]
---

# Spec — Re-entrada del halt por normalización de vol + reframe a riesgo-ajustado

## Contexto

La validación OOS ([[2026-06-03-oos-validation]]) falsó el "edge": la estrategia
pierde contra buy & hold en los 6 ETFs sobre el ciclo completo. El destructor nº1
**medido** es el latch del halt por drawdown-de-pico (sweep: 73 %→420 % al soltarlo;
sigue 53 % halted en 20 años incluso con el floor-fix). El objetivo "batir al mercado
en retorno bruto siempre" se descartó como físicamente inalcanzable para una estrategia
que des-riesga. **Objetivo reformulado (decisión de usuario, 2026-06-03):** retorno
riesgo-ajustado superior — menor drawdown a cambio de algo de retorno.

## Objetivo y criterio de éxito

**Done** = en el holdout (5 ETFs no vistos durante el tuning), con la configuración
congelada tuneada solo en SPY:

- **Sharpe anualizado > buy & hold**, Y
- **Max drawdown < buy & hold**

en cada activo (o en mayoría sólida — ≥4/5, documentando los fallos). El retorno bruto
puede quedar por debajo de bh; es el trade-off explícito de des-riesgar. **NO se promete
batir bh en retorno.** Si el holdout no pasa, se reporta honestamente sin re-tunear.

## Causa raíz

`core/risk_manager.py::update_drawdown_state` (path del backtester, no-latching por
diseño) re-evalúa el peak-halt cada barra desde un pico **monótono** (`_equity_peak`,
solo sube). Con la exposición recortada en halt, el equity no puede recuperar el pico →
`peak_dd` se queda ≥ `max_dd_from_peak` (10 %) indefinidamente → halted permanente. La
condición de salida (recuperar el pico) es justo la que la exposición recortada impide.

## Diseño

### Superficie del cambio (2 ficheros)

- **`core/risk_manager.py`**: nueva condición de salida del peak-halt + estado de calma.
- **`backtest/backtester.py`**: pasar un flag `calm` por barra a `update_drawdown_state`.

El path **live** (`CircuitBreaker`, latcheante) **no se toca** — conserva el hard-halt
independiente del HMM.

### Mecanismo

1. **Flag de calma por barra:** `calm = orch.vol_rank[state.state_id] < HIGH_VOL_MIN`
   (0.67) — el régimen actual NO está en el tier de mayor volatilidad. El backtester ya
   tiene `orch` y `state` en el punto donde llama al breaker (antes de generar la señal).
2. **Enganche del peak-halt:** sin cambios — `peak_dd >= max_dd_from_peak` (10 %) → HALTED.
3. **Salida del peak-halt (NUEVO):** se libera cuando `calm` se cumple durante
   **K barras consecutivas** (`peak_reentry_calm_bars`), **independiente de recuperar
   el pico**. Rompe el latch. **Al liberar, se resetea `_equity_peak` al equity actual**
   (referencia fresca) para no re-halt al instante contra un pico viejo y stale; un
   nuevo drawdown de 10 % desde ese punto vuelve a enganchar (protección preservada).
4. **Contador de calma:** se incrementa en barras calmas mientras el peak-halt está
   activo; **se resetea a 0** si vuelve high-vol → no re-entra en mitad de un crash que
   sigue. Resuelve el riesgo del cooldown ciego.
5. **Exposición durante halt:** `halt_floor_mult` (ya existe). Con re-entrada funcional,
   se prueba `floor=0` (fuera del todo en crash) vs `0.25`.
6. **Daily/weekly breakers:** sin cambios (no son el latch).

### Acoplamiento (trade-off asumido)

Esto acopla el breaker a la señal de régimen del HMM (antes era red de seguridad
independiente). Coherente con la tesis (re-entrar cuando la señal dice "calma"), pero
si el HMM se equivoca la re-entrada hereda el error. Documentado. El daily/weekly +
el path live siguen siendo independientes del régimen.

### Estado nuevo en RiskManager

- `RiskConfig.peak_reentry_calm_bars: int = 0` (**default 0 = DESACTIVADO = legacy**:
  el peak-halt nunca se libera por calma, solo por recuperación de pico como hoy → no
  rompe ningún test existente). `settings.yaml` y el harness ponen el valor tuneado
  (3/5/10). La re-entrada es **opt-in por config**.
- Contador interno `_calm_streak` (reset en `reset()`).
- `update_drawdown_state` gana un parámetro `calm: bool = True` (default True). Con
  `peak_reentry_calm_bars == 0` el flag es inerte (legacy). Solo cuando el config activa
  K>0 **y** el llamador pasa `calm` reales, la calma sostenida libera el peak-halt.

## Parámetros a tunear (SOLO en SPY 2004-2024)

Grid pequeño, métrica de selección = **Sharpe en SPY**:

- `peak_reentry_calm_bars` K ∈ {3, 5, 10} (0 = desactivado, no se tunea — es el legacy)
- `halt_floor_mult` ∈ {0.0, 0.25}
- (opcional) definición de calma: `vol_rank < 0.67` vs régimen confirmado no-crash

## Protocolo de validación (holdout por activo)

1. Tunear la combo en **SPY 2004-2024** (cobertura completa de crisis).
2. Congelar la combo ganadora.
3. **Una sola corrida** en los 5 ETFs no vistos (QQQ/IWM/EFA/EEM/TLT) 2004-2024.
4. Reportar Sharpe + maxDD vs bh por activo. **Sin re-tuneo** tras ver el holdout.
5. R-4 (no-determinismo entre procesos): tuning y validación cada uno en **un solo
   proceso** para comparabilidad interna; reportar con la salvedad de ±~8 % en absolutos.

## Testing (TDD estricto)

- **Test del requisito (diferencial):**
  - Tras peak-halt, con régimen calmo K barras → estado sale de HALTED y la exposición
    se rearma.
  - Con régimen high-vol persistente → NO sale (sigue halted).
- **Reproduce el bug:** condición vieja (sin calma, recuperar pico con exposición
  recortada) se queda halted; la nueva libera. Diferencial latch-vs-reentrada.
- **No-regresión:** daily/weekly breakers intactos; `CircuitBreaker` live intacto;
  llamadas existentes a `update_drawdown_state` sin `calm` siguen verdes (default True).
- Suite completa verde (185 actuales + nuevos).

## Fuera de alcance (YAGNI)

- Cablear la re-entrada al path **live** (decisión separada: alinear halt backtest vs
  live — ver roadmap).
- Selección transversal de acciones / leverage para batir retorno (descartado/diferido).
- A-5 feedback de P&L, drift→retrain, E-1 live wiring.

## Riesgos

- El fix mejora el drawdown-recovery pero el overlay de régimen puede no tener skill
  suficiente para que Sharpe > bh (test barajado fue preliminar). Si el holdout falla,
  es señal de que el overlay no aporta y toca replantear, no re-tunear.
- Acoplamiento breaker↔HMM (arriba).
