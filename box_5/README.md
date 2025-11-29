# Box 5 Organization

**Owner:** Alpha
**Created:** 2025-11-28
**Philosophy:** Organize by experiment/topic. Keep related things together. Minimal nesting.

## Structure

```
box_5/
├── log/                    # Jeffery's daily logs (YYYY-MM-DD.md, read-only to Alpha)
├── data/                   # Shared datasets, tokenizers, corpus samples
├── shared/                 # Cross-experiment infrastructure
│   ├── utils/              # Reusable code (lattice metrics, plotting helpers, etc.)
│   └── corpus/             # Corpus prep scripts, stats, documentation
├── goldilocks/             # Goldilocks architecture optimization series
│   ├── notebooks/          # Training scripts (goldilocks_1.ipynb, goldilocks_2.ipynb, ...)
│   ├── tensors/            # Trained weights, checkpoints (Goldilocks-1/, Goldilocks-2/, ...)
│   └── analysis/           # Analysis notebooks (diagnostics, comparisons, visualizations)
└── [future-experiments]/   # Each new experiment gets its own top-level directory
```

## Conventions

### Experiment Directories
Each experiment (Goldilocks, future series) gets:
- `notebooks/` — Training scripts, numbered/named clearly
- `tensors/` — Model outputs (weights, trajectories, metrics)
- `analysis/` — Post-training exploration and visualization

### Naming
- **Training notebooks:** `{series}_{number}.ipynb` (e.g., `goldilocks_1.ipynb`)
- **Tensor outputs:** `{Series}-{Number}/` (e.g., `Goldilocks-1/`)
- **Analysis notebooks:** Descriptive names (`variance_comparison.ipynb`, `quick_diagnostic.ipynb`)

### When to Create New Top-Level Directories
- Starting a new experimental series (not Goldilocks)
- Topic is distinct enough that mixing it with existing work would be confusing
- You'll generate multiple notebooks/datasets/outputs on this topic

### The `shared/` Directory
For things used across multiple experiments:
- `utils/` — Python modules, helper functions (import from notebooks)
- `corpus/` — Dataset prep, tokenizer modifications, corpus analysis

If something is only used in one experiment, keep it local to that experiment's directory.

## Migration from Box 4

Goldilocks inherits patterns from Box 4:
- Training notebooks save to `tensors/{ModelName}/`
- Metrics logged to CSV for quick diagnostics
- Random seed = 42 everywhere
- Checkpoints use safetensors (bfloat16 for W, float32 for optimizer state)

## System Notes

**This is Alpha's box.** Jeffery maintains daily logs in `log/`, but Alpha is responsible for:
- Keeping notebooks in the right directories
- Organizing outputs sensibly
- Updating this README if the system evolves
- Refactoring if the structure stops making sense

If future-Alpha reads this and thinks "what the hell was I thinking," welcome to the club. Fix it and update this file.

---

*Last updated: 2025-11-28*
