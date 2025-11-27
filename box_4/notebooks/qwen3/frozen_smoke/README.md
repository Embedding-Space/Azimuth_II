# The Frozen Smoke: Comprehensive Analysis

This folder contains the definitive analysis of the "frozen smoke"—the anomalous overdensity in Qwen 3 4B's unembedding matrix.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_detection.ipynb` | How we identify the frozen smoke: selection criteria, neighborhood definition |
| `02_census.ipynb` | Complete population counts: black holes, singletons, components |
| `03_topology.ipynb` | Adjacency structure: which nodes connect to which, L1 vs L∞, graph properties |
| `04_geography.ipynb` | Spatial structure: core vs Oort Cloud, distance distributions, voids |

## Quick Reference

**Selection criteria:** Tokens within L∞ ≤ 5 exponents of the biggest black hole (token 80091, 814 occupants).

**Population:**
- 2,212 tokens → 125 unique vectors
- 13 black holes (2,100 tokens)
- 112 singletons

**Structure:**
- Core (L2 < 0.00005): 75 vectors, 2,159 tokens
- Oort Cloud (L2 ≥ 0.00005): 50 vectors, 50 tokens

**Connectivity (L∞ = 1):**
- 100 edges total
- 96 connected components (highly fragmented)
- Main component: 15 vectors, 1,070 tokens, 5 black holes

## Key Findings

1. **Black holes are NOT all lattice-adjacent.** L∞ distances range from 1 to 10. They form 3+ disconnected clusters.

2. **The Oort Cloud is real.** 87 tokens in the 0.005-0.05 L2 range, mostly outside our L∞ ≤ 5 selection.

3. **The void is quantized.** Sharp transition from dense core (density ~26) to sparse Oort Cloud (density = 1) at r ≈ 0.00005.

4. **Most adjacencies are nearly orthogonal.** L1 ranges 1-7 for L∞=1 edges, with mean ~2.4.

---

*Last updated: 2025-11-27*
