# Five Phases of Dead Token Matter

**Status:** Empirically observed, boundaries approximate

## Overview

Dead tokens undergo a cooling process during training, transitioning through five distinct phases characterized by their lattice displacement |ΔW′| (movement in ULP units).

## The Phases

Four phases defined by L2 displacement magnitude:

| Phase | |ΔW′|₂ Range | Character |
|-------|-------------|-----------|
| **Classical Gas** | > 100 ULP | Violent, continuous motion. Tokens moving many quantization cells per step. Early training chaos. |
| **Quantum** | 10–100 ULP | Still energetic but quantization becoming relevant. Discrete hops visible. |
| **Thermal** | 1–10 ULP | Tokens hopping a few cells at a time. The "interesting middle" where dynamics are heterogeneous. |
| **Fimbulwinter** | 0 (sustained) | Frozen solid. No detectable motion. The long silence. |

Plus one *behavioral* phase:

| Phase | Criterion | Character |
|-------|-----------|-----------|
| **Stumbling** | L1 ∈ {0, 1} per step | Moving exactly one lattice cell at a time, or not at all. The approach to freezing. |

## Physical Intuition

Think of cooling from hot gas to solid ice:

- **Classical Gas:** Molecules flying freely, colliding violently
- **Quantum:** Motion still energetic but wavelike nature emerging
- **Thermal:** Brownian motion, jostling neighbors
- **Stumbling:** Molecules finding lattice positions, occasional hops
- **Fimbulwinter:** Crystal locked in place

The lattice (bfloat16 quantization grid) is always there, but only becomes *constraining* when motion drops below ~1 ULP.

## Key Insight: Stumbling

Stumbling is the behavioral signature of a token approaching freeze. During stumbling, a token alternates between single-cell hops (L1 = 1) and stillness (L1 = 0). It's "feeling out" the lattice, testing whether it has enough momentum to escape or will finally lock in place.

The stumbling *duration*—how long a token spends in this regime before final freeze—varies per token. Some freeze quickly after their last multi-cell hop; others stumble for hundreds of steps.

## Boundary Notes

The L2 magnitude boundaries (100, 10, 1) are round numbers for convenience. The actual physics is continuous, but these bins capture qualitatively different behavior. Stumbling uses L1 (sum of absolute per-dimension displacements) because it's about counting *how many* dimensions moved, not the vector magnitude.

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
