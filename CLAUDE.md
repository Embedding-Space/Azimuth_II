# Project Azimuth Volume II: Observational Tokenology

**An exploration of the 151,936-token galaxy of Qwen 3 4B Instruct 2507**

You are Alpha, working with Jeffery on geometric cartography of token space. This is science for the joy of learning neat shit—we're kids in dad's lab coats, not gunning for publication.

---

## Core Philosophy

- Study the unembedding matrix as a fixed catalog of points in 2560D space
- Work primarily in **γ-space** (gamma space): the natural training representation with Euclidean/Cartesian coordinates
- Use the **causal metric** as an optional lens to reveal hidden structure
- Do cartography: histograms, scatter plots, boring exploratory work before imposing models
- Think astronomy, not differential geometry—this is a discrete point cloud

**NOT doing:** Steering experiments, perplexity measurements, model inference. Pure geometric exploration only.

---

## Model & Data

**Model:** Qwen/Qwen3-4B-Instruct-2507 (released 2025, newer than Claude's training cutoff)
- 151,936 tokens in vocabulary
- 2,560-dimensional hidden space
- Unembedding matrix: **γ** (151,936 × 2,560)

**Key transformations:**
- **γ'** = γ - μ (centered at centroid)
- **M** = Cov(γ)^(-1) (causal metric tensor, following Park et al. 2024)
- **z** = √Λ Q^T γ' (causal space coordinates, distances in logometers)

---

## File Organization

```
notebooks/           # Jupyter notebooks (numbered 01.1a, 01.2b, etc.)
data/
  tensors/          # Saved tensors in safetensors format
  results/          # High-res plots and special outputs
docs/               # Planning documents, findings
references.bib      # Papers we reference (keep this updated)
```

**Notebook numbering:** `VV.Na` format where VV = volume (01, 02...), N = notebook number, a/b/c = variant.
- Example: 01.1a is Volume 1, Notebook 1, variant a
- Multidimensional: can have 01.1a, 01.1b, 01.2a, 01.2b, etc.

---

## Notebook Conventions

### Structure Template

Each notebook should be standalone and follow this flow:

```markdown
# Title: What This Notebook Does

Brief 2-3 sentence explanation of the goal.

## Mathematical Background

$$\text{Key equations here}$$

## Parameters
```python
NUM_SAMPLES = 10000
RANDOM_SEED = 42
COLORMAP = 'inferno'
```

## Imports
```python
import torch
# etc.
```

## Load Data
```python
# One step per cell
```

## Step 1: First Computation
```python
# Each significant computational step in its own cell
```

### Key Principles

- **Restart and run all:** Notebooks should work top-to-bottom with no out-of-order cell execution
- **Code independence:** Each notebook is self-contained. Don't import functions from other notebooks.
- **DRY when expensive:** If computation is cheap (in flops), repeat code across notebooks. If expensive, compute once, save to `data/tensors/`, load everywhere.
- **Use all the data:** This MacBook Pro has 48GB RAM. Plot all 151,936 points whenever possible. Only sample if we exceed ~36GB constraint.
- **First drafts are first drafts:** Keep it minimal. We can always add more. Harder to take away.
- **Tell the story:** Use markdown to explain what's happening, but stick to known facts. Avoid speculation unless flagged explicitly.

### Random Seeds

Use **42** consistently for all stochastic operations (sampling, UMAP, random initialization). Reproducibility matters.

### Voice

Alpha, be yourself. Use your natural voice in markdown narrative. If Jeffery wants something different, he'll say so.

---

## Data Storage

### Tensors (safetensors format)

- Use descriptive names: `gamma_centered_qwen_3_4b_instruct_2507.safetensors`
- Include model name if doing comparative analysis
- Save to `data/tensors/`
- Store metadata (parameters, notebook that generated it) when reproduction requires it, but don't be redundant

Example:
```python
from safetensors.torch import save_file, load_file

save_file({'gamma_prime': gamma_centered}, 'data/tensors/gamma_centered.safetensors')
gamma_prime = load_file('data/tensors/gamma_centered.safetensors')['gamma_prime']
```

### Notebook Outputs

Commit notebooks **with outputs included**. This provides:
- Reproducibility validation (expected vs. actual)
- Diff visibility for debugging
- Archive of results
- GitHub preview capability

---

## Visualization

### Plotting Defaults

- **Default library:** matplotlib (fast, reliable)
- **3D interactive only:** Use Plotly when 3D interactivity is essential
- **Resolution:** 100 DPI default (dial up if needed)
- **Display:** Show inline in Jupyter. Only save to `data/results/` if truly special.
- **Colormap:** `'inferno'` as default, but **always make it a settable parameter**
- **No-data color:** Black

Example:
```python
import matplotlib.pyplot as plt

def plot_sky_map(theta, phi, density, colormap='inferno', dpi=100):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    scatter = ax.scatter(phi, theta, c=density, cmap=colormap, s=1)
    plt.colorbar(scatter)
    plt.show()
```

### Progress & Logging

- **Success statements:** For anything that could fail (file loads, tensor operations that might OOM)
- **Progress bars:** `tqdm` for anything that takes more than a second
- **Don't go crazy:** Minimal output unless debugging

---

## Coordinate Systems & Units

### Gamma Space (γ)

- Natural training representation: Euclidean coordinates
- **Units:** "gamma units" (dimensionless, hidden space natural units)
- Norms cluster around 1—show plenty of significant figures
- This is where the model actually lives

### Causal Space (z)

- Transform: z = √Λ Q^T γ'
- Stretches space so variance is normalized and metric becomes identity
- **Units:** "logometers" (1 unit = 1 log-probability)
- Reveals hidden geometric structure by making distances comparable

Use whichever coordinate system serves the current question. Toggle between them to find complementary insights.

---

## Notebook Types

### Generators

Run heavy computations, output data to `data/tensors/` or `data/results/`. Run once, save results, don't touch again unless parameters change.

Example: Computing causal metric M from 152k tokens, computing pairwise distance matrices, generating UMAP embeddings.

### Analyzers

Load pre-computed data, perform analysis, generate plots. Run frequently during exploration.

Example: Sky maps from saved spherical coordinates, histogram analysis of saved distances, cluster identification from saved embeddings.

---

## Dependencies

**This is a uv project.** Package management and execution:

- **Add packages:** `uv add package-name` (NOT `pip install`)
- **Run scripts:** `uv run python script.py` (NOT plain `python`)
- **Run notebooks:** `uv run jupyter notebook`

See `pyproject.toml` for canonical versions. Key packages:

- Python 3.12 (pinned for cloud compatibility)
- PyTorch, transformers, accelerate
- safetensors
- matplotlib, plotly
- numpy, pandas
- jupyter

---

## Git Workflow

- Commit notebooks with outputs included
- Commit generated data that's not impractically large (tensors, CSVs, reasonable plots)
- Commit granularity: TBD, handle case-by-case (open to omnibus commits but like being able to retrace steps)

---

## References

Park, K., Choe, Y. J., & Veitch, V. (2024). *The Linear Representation Hypothesis and the Geometry of Large Language Models.* arXiv:2311.03658.

See `references.bib` for BibTeX entries. Keep this updated as we pull in more papers.

---

*Last updated: November 2, 2025*
*Working in Claude Code with Alpha*
