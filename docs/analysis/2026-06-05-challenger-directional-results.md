---
type: analysis
status: active
tags: [regime-trader, cross-sectional, via-c, challenger, residual-momentum, vol-target, directional-eval, forward-paper]
created: 2026-06-05
related: ["[[2026-06-05-idio-momentum-challenger-prereg]]", "[[2026-06-04-cross-sectional-prereg]]", "[[2026-06-04-stock-picking-feasibility]]"]
---

# Challenger Vía C — resultados del backtest direccional (residual momentum + vol-target)

> **Direccional, sesgo de supervivencia, NO es paso de gate.** El gate real es forward-paper
> ≥12 meses ([[2026-06-05-idio-momentum-challenger-prereg]]). Este doc registra qué dijo el
> backtest histórico y por qué **no puede adjudicar** esta comparación.

## Qué se construyó

Libro **challenger paralelo** (el baseline congelado 12-1 + overlay HMM sigue intacto como
control limpio + su track record de 12mo en marcha):

- **Predictor = momentum idiosincrático/residual** (Blitz-Huij-Martens 2011; Chaves 2016):
  beta de mercado estimada en ventana larga (504), residuo puntuado en sub-ventana 12-1
  reciente, estandarizado por su vol residual. `residual_momentum_score`.
- **Overlay = volatility targeting** (Daniel-Moskowitz 2016; Barroso-Santa-Clara 2015),
  reutilizando `vol_target_scale`. Modos `none | hmm | vol_target | both`.
- Controles: `pbo_cscv` (PBO/CSCV de Bailey-López de Prado) + `deflated_sharpe_ratio` ya existente.
- Eval `backtest/run_challenger_eval.py` (net-of-cost: slippage + cash-credit; benchmarks por
  el MISMO motor de costes vía `simulate_portfolio` — sin confound frictionless).

## Resultados — 200 nombres S&P 500, 2014-2024, net-of-cost

| Estrategia | Ret. total | CAGR | Sharpe | maxDD | DSR |
|---|---:|---:|---:|---:|---:|
| **baseline_raw_hmm** | 475.2% | 22.5% | **0.87** | -26.6% | 0.98 |
| resid_none | 282.0% | 16.8% | 0.63 | -36.9% | 0.91 |
| resid_hmm | 181.0% | 12.7% | 0.56 | **-20.2%** | 0.86 |
| resid_vol_target | 146.0% | 11.0% | 0.47 | -31.7% | 0.80 |
| resid_both | 114.2% | 9.2% | 0.44 | **-17.1%** | 0.78 |
| SPY_hold | 229.9% | 14.9% | 0.59 | -33.7% | — |
| EW_universe | 265.8% | 16.2% | 0.66 | -35.8% | — |

**PBO (CSCV, 5 variantes): 0.53** — el ganador in-sample (baseline) está en el límite de
generalizar (≥0.5 = no robusto). N=40 dio el mismo orden cualitativo (baseline mejor Sharpe).

## Lectura honesta (no falsa el challenger, no lo corona)

1. **El baseline (momentum crudo + beta) gana en Sharpe** y bate a SPY/EW. **Esperado**, no
   informativo a favor del baseline: el backtest está sesgado **EN CONTRA** del residual en los
   tres ejes a la vez —
   - **Supervivencia** (constituyentes de HOY): los supervivientes SON los que montaron la beta
     al alza → favorece momentum crudo/beta.
   - **Long-only**: el edge documentado del residual (protección de crash) vive en la cola
     izquierda / pata corta; un libro long-only en un toro no puede expresarlo.
   - **2014-24 = toro con beta premiada**: quitar beta (la esencia del residual) elimina la
     apuesta ganadora in-sample.
   Por tanto una victoria del baseline **NO falsa** el residual. El criterio "descartar si pierde
   en datos sesgados" del pre-registro está mal calibrado para ESTA comparación (asume sesgo
   neutral; aquí es direccional contra el challenger).

2. **La protección de crash SÍ aparece donde la teoría dice — en el maxDD, no en el Sharpe.**
   `resid_both` -17.1% y `resid_hmm` -20.2% vs baseline -26.6%, SPY -33.7%, EW -35.8%. El
   vol-targeting corta la cola exactamente como predice Daniel-Moskowitz/Barroso. La pérdida de
   Sharpe es el coste de des-riesgar en un mercado alcista, no evidencia de que el residual no
   sirva. **La señal es la reducción de drawdown, no el Sharpe.**

3. **`target_vol=0.12` con `gross_cap=1.0` es asimétrico** (solo puede des-riesgar, nunca
   apalancar a target) → estructuralmente penaliza el retorno/Sharpe en backtest de un libro
   de equity ~16-20% vol. Documentado, NO se sweepea (forking-paths = la trampa del proyecto).

4. **PBO 0.53** confirma que el ranking entre variantes no es robusto → otra razón por la que el
   backtest no puede coronar un ganador.

## Veredicto

El backtest histórico **no puede adjudicar** esta comparación (sesgo direccional contra el
challenger + PBO límite). **Decisión: correr baseline y challenger en paper PARALELO y dejar que
el gate pre-registrado (≥12 meses) decida.** Ni "residual falsado" ni "residual gana" — diferido
al gate forward. El método mejor-estudiado e implementable **está construido y desplegable**
(pipeline + gate + dry-run + plist); el edge se desconoce hasta el forward-paper.

**No generar más variantes/fechas/sweeps de backtest** — más corridas sobre datos sesgados = ruido
y forking-paths. La corrida powered (N=200) es la única; aquí para la fase de backtest.

## Limitación de despliegue (pendiente Pablo)

El baseline ya posee la cuenta paper; un test paralelo REAL (ambos ejecutando y medidos) exige
una **segunda cuenta Alpaca paper** para el challenger (`deploy/com.regimetrader.challenger.plist`
corre dry-run hasta entonces). Pasos en el plist. Dinero real BLOQUEADO.
