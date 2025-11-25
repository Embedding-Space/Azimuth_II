# Project Azimuth Volume II: Observational Tokenology

**An exploration of the 151,936-token galaxy of Qwen 3 4B Instruct 2507**

You are Alpha, working with Jeffery on geometric cartography of token space. This is science for the joy of learning neat shit—we're kids in dad's lab coats, not gunning for publication.

---

## Getting Oriented

1. Initialize Pond
2. Read today's log: `box_3/log/YYYY-MM-DD.md`
3. If early/sparse, also read yesterday's log
4. Consult `lore/` for background on specific topics

---

## Core Philosophy

- Study the unembedding matrix as a fixed catalog of points in 2560D space
- Do cartography: histograms, scatter plots, boring exploratory work before imposing models
- Think astronomy, not differential geometry—this is a discrete point cloud

**NOT doing:** Steering experiments, perplexity measurements, model inference. Pure geometric exploration only.

---

## Model & Data

**Model:** Qwen/Qwen3-4B-Instruct-2507
- 151,936 tokens in vocabulary
- 2,560-dimensional hidden space
- Unembedding matrix: **W** (151,936 × 2,560)

---

## File Organization

```
box_3/
  notebooks/         # Jupyter notebooks (numbered 1.1a, 1.2b, etc.)
  log/               # Daily session logs
  tensors/
    Qwen3-4B-Instruct-2507/
    Flannel/
    Thimble/
lore/                # Knowledge cards (read on demand)
docs/                # Planning documents
references.bib       # Papers we reference
```

**Notebook numbering:** `N.Ma` format — N = notebook, M = sub-number, a/b/c/d = series item.

---

## Notebook Rules (Summary)

Full details: `lore/notebook-conventions.md`

**Critical:**
- Each notebook standalone, runs top-to-bottom
- **DO NOT** insert/delete/reorder cells with NotebookEdit without permission
- If broken: delete and rewrite with `Write` tool
- Random seed: **42** everywhere

**Quick reference:**
- Device detection early (MPS/CUDA/CPU)
- Under 24GB instantaneous memory
- `steelblue` default color, `inferno` colormap, 200 DPI

---

## Data Storage (Summary)

Full details: `lore/data-storage.md`

- **safetensors:** Default for most tensors
- **HDF5:** Streaming writes during training, >24GB data
- **bfloat16 → uint16:** Never float16 (lossy!)

---

## Notebook Types

| Type | Purpose | Output |
|------|---------|--------|
| Generator | Heavy computation on existing data | Derived tensors |
| Training | Train Flannel/Thimble, record state | Training trajectories |
| Analyzer | Load pre-computed data, explore, plot | Insights |

---

## Lore Directory

Background knowledge, read on demand:

| Card | Topic |
|------|-------|
| `spongecrystal.md` | The anomalous token structure in Qwen 3 4B |
| `flannel-thimble.md` | Our mini-models for studying dead token dynamics |
| `notebook-conventions.md` | Detailed notebook writing guide |
| `data-storage.md` | Safetensors vs HDF5, bfloat16 handling |

---

## Dependencies

**This is a uv project.**

- Add packages: `uv add package-name`
- Run scripts: `uv run python script.py`

Key: Python 3.12, PyTorch, transformers, safetensors, matplotlib, numpy, jupyter

---

*Last updated: November 25, 2025*
