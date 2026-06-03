---
type: analysis
status: active
tags: [regime-trader, rotation, cross-asset, pre-registration, overfitting, gate]
created: 2026-06-04
related: ["[[2026-06-04-markov-edge-redesign]]", "[[2026-06-03-oos-validation]]"]
---

# Pre-registro — Vía B: rotación cross-asset dirigida por régimen

> **Documento de pre-registro.** Escrito y commiteado **ANTES** de correr ningún
> backtest de rotación. Congela hipótesis, knobs y criterio de aprobación para que un
> resultado positivo no se pueda fabricar moviendo parámetros tras ver los números
> (el patrón que falseó el floor-sweep y el cash-credit, [[2026-06-03-oos-validation]]).
> **Un resultado negativo es un desenlace válido y entregable.** No se ajusta para ganar.

## Hipótesis (una, falsable)

> El régimen de mercado detectado por el HMM (sobre SPY) puede dirigir una rotación
> entre sleeves descorrelacionados (equities / bonos / oro / cash) que **bate a un 60/40
> estático y a risk-parity en Sharpe Y Calmar**, neto de costes, **fuera de muestra**.

Nula (H0): la rotación **no** bate a 60/40 ni a risk-parity en riesgo-ajustado OOS →
el régimen no aporta valor de asignación; un mix estático que no piensa es superior.

## Por qué esto puede funcionar donde el timing de 1 índice no

[[2026-06-04-markov-edge-redesign]] §1: sobre UN índice long-only, el regime-timing solo
modula beta (alfa=0, probado por el sweep monótono). Aquí el grado de libertad es
**distinto**: rotar entre activos *descorrelacionados* cuyo orden de rentabilidad cambia
con el régimen (equities en risk-on, bonos/oro en risk-off). El valor vendría de la
**diversificación condicionada al régimen**, no de cronometrar un activo. Es la aplicación
donde los modelos de régimen históricamente ganan su sitio — pero batir un 60/40 OOS sigue
siendo difícil; por eso es hipótesis, no promesa.

## Knobs CONGELADOS (elegidos por prior económico, NO por barrido)

> Regla dura: **cero sweep.** Si pruebo N configuraciones y me quedo con la mejor, eso es
> multiple-testing y `n_trials` del DSR debe contarlo. Estos valores se fijan por teoría y
> no se tocan tras ver resultados.

| Knob | Valor congelado | Prior |
|---|---|---|
| Proxy de régimen | SPY | índice de referencia del mercado US |
| Sleeve equities | SPY + QQQ (50/50) | risk-on |
| Sleeve defensivo | TLT + GLD (50/50) | bonos largos + oro = refugio risk-off |
| Cash | rf 4.5 % anual | colchón risk-off |
| **Tier risk-on** (vol_rank ≤ 0.33) | 100 % equities | régimen calmo → máxima exposición a riesgo |
| **Tier mid** (0.33 < rank < 0.67) | 60 % equities / 40 % defensivo | transición → balanceado |
| **Tier risk-off** (vol_rank ≥ 0.67) | 0 % equities / 60 % defensivo / 40 % cash | estrés → fuera de equities |
| Vol-target | 10 % anual, ventana trailing 20 barras | normaliza riesgo, sustituye el halt binario |
| Cap de gross | 1.0 (sin leverage) | honestidad: el leverage infló el 52.8 % artefacto |
| Floor de gross | 0.0 | vol-target puede desexponer del todo en pánico |
| Slippage | 5 bps sobre turnover | igual que el resto del proyecto |
| Walk-forward | train 504 / test 126 / step 126 | default del backtester (refit por fold) |

Los tiers reutilizan `StrategyOrchestrator.vol_rank` (terciles ya existentes:
`LOW_VOL_MAX=0.33`, `HIGH_VOL_MIN=0.67`). **Se toma el tier limpio del proxy, NO el peso
ajustado por halt** — para no re-importar la patología del halt-latch ni doblar el riesgo
(el vol-target del libro de rotación es su única capa de riesgo).

## Simetría de costes (crítica — confound que ya flipó 0/5→1/5)

La rotación mantiene un sleeve de **cash explícito** en risk-off → el cash **debe** rentar
rf (`credit_cash_rf=True`), o el risk-off parece artificialmente malo. Y los benchmarks
(60/40, risk-parity) **pagan el MISMO slippage y reciben el MISMO trato de cash a rf**. La
rotación rebalancea más que un 60/40 estático; si el benchmark fuera sin fricción, la
comparación estaría amañada a mi favor. **Mismo modelo de coste exacto en los tres.**

## Gate de aprobación (PRE-REGISTRADO)

La rotación **PASA** solo si **TODO** lo siguiente es cierto, neto de costes emparejados:

1. **Riesgo-ajustado full-span (2007-2024 OOS):** Sharpe_rot > Sharpe_6040 **Y**
   Calmar_rot > Calmar_6040 **Y** Sharpe_rot > Sharpe_riskparity **Y** Calmar_rot >
   Calmar_riskparity.
2. **Generalización temporal:** lo anterior se cumple en **AMBOS** sub-periodos
   (2007-2015 y 2016-2024) por separado — no solo en el agregado.
3. **Deflación:** DSR > 0 con `n_trials` = nº de configuraciones probadas (=1 si no hay
   sweep, como manda este pre-registro).

Cualquier fallo → **H0 no se rechaza** → vía B no desplegable como búsqueda de alfa.
**Dinero real permanece BLOQUEADO** en cualquier caso hasta paper supervisado posterior.

## Qué NO se hace en este MVP

- No PBO/CPCV (sirven con modelos *ajustados*; aquí el mapa es de cero parámetros → DSR
  basta). Se añaden solo si el read simple muestra algo que merezca deflación más dura.
- No mapa aprendido (per-regime best asset) — alto riesgo de overfit; solo si el mapa
  teórico muestra promesa, y entonces con holdout reservado.
- No leverage. No 2h (ver [[2026-06-04-markov-edge-redesign]] §0).
