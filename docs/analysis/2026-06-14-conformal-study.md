---
type: study
status: open
tags: [regime-trader, conformal, uncertainty, shadow, T3.2]
created: 2026-06-14
related: ["[[2026-06-12-improvement-roadmap]]", "[[overlay-fuzzy-posterior-vs-argmax]]"]
---

# Estudio conformal — ¿añade algo sobre el fuzzy posterior? (T3.2)

**Tesis a FALSAR antes de construir nada:** los intervalos de predicción conformes
sobre el clasificador de régimen dan una medida de incertidumbre con cobertura
garantizada; *podrían* servir para escalar el gross hacia abajo cuando la llamada de
régimen es incierta. Pero el libro YA computa `predictive_entropy_norm`
(`core/meta_overlay.py`) como proxy de incertidumbre. **Hipótesis nula: el ancho del
intervalo conformal y la entropía predictiva capturan lo mismo → conformal no aporta.**

## Por qué estudio, no despliegue

- Regla del programa: falsar antes de construir. Una segunda señal de incertidumbre
  correlacionada al 0.9 con la que ya existe es complejidad sin edge.
- Datos: el estudio corre sobre los shadow logs (`logs/shadow_regime.csv`,
  `logs/shadow_macro.csv`), que apenas arrancaron (2026-06-12). Necesita ≥3-6 meses
  de acumulación antes de tener potencia. No se construye nada hasta entonces.

## Diseño del test (cuando haya datos)

1. **Score de no-conformidad:** sobre la posterior filtrada del HMM campeón, usar
   `1 - max(posterior)` (margen) o el error de clasificación 1-paso como residuo.
   Calibrar `split_conformal_quantile` (`core/conformal.py`) en ventana held-out.
2. **Señal A (existente):** `predictive_entropy_norm` por barra.
   **Señal B (conformal):** ancho del intervalo / tamaño del conjunto de predicción.
3. **Falsación:**
   - Correlación A↔B. Si |ρ| > 0.85 → **conformal redundante, se descarta** (H0 no
     rechazada).
   - Si ρ moderado: ¿B predice mejor el downside realizado al día siguiente que A?
     (regresión del retorno|incertidumbre, o reducción de maxDD de un overlay
     gross-scaling guiado por B vs por A, en backtest sesgado + forward shadow).
   - Cobertura empírica del intervalo conformal ≈ nominal (1-α) como sanity check
     del propio método (`coverage`).
4. **Promoción (solo si B bate a A de forma robusta):** prereg nuevo + cargo en el
   ledger; gross-scaling por incertidumbre = enmienda al libro, jamás hot-swap.

## Guardrail

Igual que el resto de Tier 1/3: shadow only. Nada de esto toca el panel del HMM
campeón ni el path de órdenes hasta que (a) el test rechace H0 y (b) un prereg nuevo
lo congele. Si H0 no se rechaza, el entregable es la nota negativa — y eso es un
resultado válido que ahorra complejidad.

## Estado

- `core/conformal.py` + tests construidos (la maquinaria está lista).
- Test pendiente de datos (acumulación shadow ≥3-6 meses). Reevaluar ~2026-09.
