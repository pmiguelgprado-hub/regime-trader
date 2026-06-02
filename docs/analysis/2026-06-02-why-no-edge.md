---
type: analysis
status: active
tags: [regime-trader, edge, backtest, circuit-breaker, strategy]
created: 2026-06-02
updated: 2026-06-02
related: ["[[improvement-review]]", "[[go-live-review]]", "[[2026-06-01-senior-audit]]"]
---

# Por qué el bot no ganaba (era un bug, no falta de edge)

Investigación con evidencia: backtest real SPY 2019-01-01 → 2024-12-31
(`python main.py --backtest --compare`), artefactos en `backtest_output/SPY/`.

## ✅ DESENLACE (2026-06-02): arreglado y re-medido

El "sin edge" era el **bug del halt**, no la estrategia. Tras añadir el suelo
mínimo de asignación en halt (`halt_floor_mult=0.25`, no ir a 0 %):

| | Antes (roto) | **Después del fix** | buy&hold | SMA200 | random |
|---|---|---|---|---|---|
| Retorno | 6.92 % | **52.84 %** | 69.90 % | 40.86 % | 27.30 % |
| Sharpe | −0.12 | **1.22** | 1.45 | 1.00 | 0.59 |
| Max DD | −10.28 % | −10.33 % | −9.97 % | −9.06 % | — |
| % `halted` | 53–75 % | **0.9 %** | — | — | — |

**La estrategia SÍ tiene edge:** bate al filtro de tendencia (SMA200) y a la
aleatoria, con drawdown controlado (no es apalancamiento temerario). Solo queda
por debajo de buy & hold en retorno bruto — esperable para una estrategia que se
des-arriesga, en un periodo puramente alcista; en un mercado más volátil/lateral
el des-riesgo debería pagar. **Veredicto R1 actualizado: de "pierde contra todo"
a "edge competitivo, aún por debajo de buy&hold en bull, necesita validación más
amplia (otros periodos/activos) antes de dinero real".**

Las hipótesis de abajo quedan **FALSADAS por la re-medición**: "SMA200 bate al
HMM" → falso (52.8 % > 40.9 %); "vol-timing no tiene edge" → falso (Sharpe 1.22).
Eran artefactos de la corrida rota, como se advirtió. El resto del documento
queda como registro de la investigación.

---

## Resultado de partida

| Estrategia | Retorno total | Sharpe | Max DD |
|---|---|---|---|
| **nuestra** | **6.92 %** | **−0.12** | −10.28 % |
| buy & hold | 69.90 % | 1.45 | −9.97 % |
| **SMA200 (tendencia tonta)** | **40.86 %** | **1.00** | −9.06 % |
| aleatoria (100×) | 27.30 % ± 12.49 % | 0.59 | — |

La estrategia pierde contra todo. **PERO** (corrección importante): estas cifras
están **contaminadas** por el bug del halt de abajo — la estrategia estuvo fuera
del mercado el 53 % del tiempo. **No se puede concluir "sin edge" de una medición
rota.** El veredicto histórico "R1: sin edge, bloquea dinero real" queda
**EN SUSPENSO** hasta una re-medición limpia. Lo que sigue separa el hallazgo
sólido (el bug) de las hipótesis pendientes de re-test.

## Causa dominante: el cortacircuitos se queda permanentemente fuera

Del `regime_history.csv` (555 barras OOS):

- **53.3 % de las barras el bot está `halted` y plano (peso 0).** Más de la mitad
  del periodo, fuera del mercado.
- **Solo hay 1 transición a `halted`.** Salta **una vez** (una caída de −10 %
  desde máximos al principio) y **nunca se reactiva** en 5 años.
- En las barras `halted` el activo rindió **+0.1246 %/barra**, MÁS que en las
  normales (+0.0795 %). Se quedó plano justo en el mejor tramo y se perdió toda
  la recuperación.

**La trampa (mecánica exacta):** `max_dd_from_peak = 0.10` → al caer 10 % desde
el pico, halt → peso 0 → con el capital plano, el equity **no puede volver a
crecer** hacia el pico anterior → la caída-desde-máximo se queda >10 % para
siempre → el halt no se levanta nunca. El backtester no tiene un humano que
re-habilite (el diseño del vídeo dice "halt + lock **manual**"), así que en una
corrida desatendida de años el bot se apaga a la mitad y se queda en caja.

Esto **por sí solo** explica casi todo el hueco 6.9 % vs 69.9 %.

## Hipótesis secundaria (CONTAMINADA — re-testar tras el arreglo)

⚠️ Todo lo de abajo se mide sobre la corrida rota (53 % en `halted`): las barras
por régimen son una mezcla de halted/activo, y el SMA200 **no pasa por el
cortacircuitos** mientras la estrategia sí. Comparar una estrategia tullida con un
benchmark sano no prueba "la tendencia tiene edge y la volatilidad no". Son
**hipótesis a re-verificar tras la re-medición limpia**, no conclusiones.

Desglose por régimen (contribución al retorno):

| Régimen | % tiempo | Contrib. | Sharpe |
|---|---|---|---|
| crash | 11.4 % | **+8.48 %** | +2.29 |
| bull | 25.8 % | +3.11 % | +0.15 |
| euphoria | 16.8 % | +3.03 % | +0.43 |
| **bear** | 24.5 % | **−7.22 %** | **−1.74** |
| neutral | 21.6 % | 0.00 % | (todo en `halted`) |

- Reducir en **crash** SÍ aporta (+8.48 %, Sharpe 2.29) — la tesis funciona en
  caídas profundas y lentas.
- Reducir en **bear** (caídas suaves que rebotan) **destruye** (−7.22 %): vende
  el suelo y se pierde el rebote. El código mismo admite que detecta las
  V-recoveries "2-3 días tarde".
- La estrategia trata bear y crash casi igual (ambos ~60 %), cuando deberían ser
  opuestos.

## Por qué en el vídeo "funcionaba"

1. **El vídeo nunca demostró una ventaja medida.** Enseña la *maquinaria*
   (detección de regímenes, dashboard, equity que sube) — pero en un mercado
   alcista CUALQUIER estrategia long sube. Subir ≠ batir a buy & hold. El vídeo
   no muestra un benchmark riguroso a 5 años (audit §8: la Fase 7 es un prompt
   de ~40 s y "una vez cableado, no necesitas el dashboard").
2. **El cortacircuitos es de bloqueo manual.** En vivo un humano lo re-activa; en
   un backtest desatendido se queda bloqueado para siempre. El vídeo opera en
   corto/en vivo, donde la trampa no se manifiesta.
3. **Ajustes distintos:** el vídeo entrena diario pero corre el lazo a 5-min, y
   usa IS=252 (nosotros 504 porque el HMM lo exige). Nunca validó ninguno con
   rigor. Nuestro backtest es **más honesto** que el del vídeo, y por eso
   destapa la ausencia de edge.

**Resumen:** el vídeo vende un sistema que *parece* funcionar porque nunca lo
somete a la prueba que nosotros sí hacemos. Nuestro código es más riguroso; el
resultado feo es el correcto.

## Plan

**Paso 0 — arreglar la trampa del halt y RE-MEDIR (antes de cualquier conclusión).**
El halt por caída-desde-pico necesita una vía de recuperación: re-habilitar tras
enfriamiento/condiciones normalizadas, o no ir a 0 % absoluto (sonda mínima de
re-entrada), o re-entrar cuando el precio recupera la tendencia. Memory línea 44:
el breaker del backtester se diseñó **no-latching** → el halt permanente es un
**bug contra la intención**, no comportamiento esperado. Arreglar → re-correr
`--backtest --compare` → leer la cifra limpia. Barato y decisivo.

**Solo entonces** evaluar (hipótesis pendientes, NO conclusiones):
1. ¿Sigue habiendo infra-rendimiento tras la re-medición limpia? (re-mide R1).
2. ¿Asignación guiada por tendencia (`precio>SMA200`) bate a vol-timing? Re-test
   con ambos pasando por el mismo cortacircuitos.
3. Separar bear de crash (re-mirar el desglose por régimen ya limpio).
4. Re-entrada más rápida tras picos de vol (cazar la V-recovery).
5. ¿El gating por confianza aporta algo?

## Sobre E-1 (multi-activo SP500) y el edge

Multi-activo **no crea ventaja** por sí mismo: con la misma lógica de halt +
de-risking aplicada a varios nombres del S&P se obtiene infra-rendimiento
correlacionado (todos planos a la vez tras la misma caída de mercado). Diversificar
puede bajar la volatilidad de cartera, pero **los arreglos #1 y #2 importan más
que añadir activos**. Construir E-1 por capacidad está bien; esperar que arregle
la rentabilidad, no.
