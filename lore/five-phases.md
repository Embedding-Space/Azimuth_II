# Five Phases of Dead Token Matter

**Status:** Empirically observed, boundaries approximate

## Overview

Dead tokens undergo a cooling process during training, transitioning through five distinct phases characterized by their lattice displacement |ΔW′| (movement in ULP units).

## The Phases

| Phase | |ΔW′| Range | Character |
|-------|-----------|-----------|
| **Classical Gas** | > 100 ULP | Violent, continuous motion. Tokens moving many quantization cells per step. Early training chaos. |
| **Quantum** | 10–100 ULP | Still energetic but quantization becoming relevant. Discrete hops visible. |
| **Thermal** | 1–10 ULP | Tokens hopping a few cells at a time. The "interesting middle" where dynamics are heterogeneous. |
| **Stumbling** | 0.1–1 ULP | Single-cell hops. Tokens barely moving, testing the lattice boundaries. Last gasp before freeze. |
| **Fimbulwinter** | < 0.1 ULP | Frozen solid. No detectable motion. The long silence. |

## Physical Intuition

Think of cooling from hot gas to solid ice:

- **Classical Gas:** Molecules flying freely, colliding violently
- **Quantum:** Motion still energetic but wavelike nature emerging
- **Thermal:** Brownian motion, jostling neighbors
- **Stumbling:** Molecules finding lattice positions, occasional hops
- **Fimbulwinter:** Crystal locked in place

The lattice (bfloat16 quantization grid) is always there, but only becomes *constraining* when motion drops below ~1 ULP.

## Key Insight: Stumbling

The stumbling phase (0.1–1 ULP) is where individual tokens "decide" to freeze. Some fight longer than others. The heatmap shows purple speckles—tokens moving exactly 1 cell while neighbors are already frozen.

This is not noise. It's the signature of the phase transition at the individual token level.

## Boundary Notes

The boundaries (100, 10, 1, 0.1) are round numbers for convenience. The actual physics is continuous, but these bins capture qualitatively different behavior.

## Visualization

Best seen in:
- Phase timeline heatmaps (tokens × time, colored by phase)
- Fraction-in-phase plots (stacked area over time)
- Individual token trajectories through phase space

## Related

- [fimbulwinter-onset.md](fimbulwinter-onset.md) — When the final freeze happens
- `box_4/notebooks/analysis/phase_transitions.ipynb` — Where this was first visualized

---

*Framework developed: November 25, 2025*
*Based on Thimble 8 lattice displacement analysis*
