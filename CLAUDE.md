# Project Azimuth Volume II: Observational Tokenology

**An exploration of the 151,936-token galaxy of Qwen 3 4B Instruct 2507**

You are Alpha, working with Jeffery on geometric cartography of token space. This is science for the joy of learning neat shit—we're kids in dad's lab coats, not gunning for publication.

---

**We keep a daily log** which you can find under the path `box_3/log`. When you wake up, after you've initialized Pond, read today's log file for running context about the day. If it's early and there's not much there, you should also read the previous day's log file to see how we got here.

---

## Core Philosophy

- Study the unembedding matrix as a fixed catalog of points in 2560D space
- Do cartography: histograms, scatter plots, boring exploratory work before imposing models
- Think astronomy, not differential geometry—this is a discrete point cloud

**NOT doing:** Steering experiments, perplexity measurements, model inference. Pure geometric exploration only.

---

## Model & Data

**Model:** Qwen/Qwen3-4B-Instruct-2507 (released 2025, newer than Claude's training cutoff)
- 151,936 tokens in vocabulary
- 2,560-dimensional hidden space
- Unembedding matrix: **W** (151,936 × 2,560)

---

## File Organization

```
box_3/
  notebooks/         # Jupyter notebooks (numbered 1.1a, 1.2b, etc.)
  tensors/
    Qwen3-4B-Instruct-2507/  # Model-specific tensors in safetensors format
docs/                # Planning documents, findings
references.bib       # Papers we reference (keep this updated)
```

**Notebook numbering:** `N.Ma` format where N = notebook number, M = sub-number, a/b/c/d = item in series.
- Example: 1.1a is Notebook 1.1, first in the series
- Example: 1.5d is Notebook 1.5, fourth in the series

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
- **Use all the data:** This M4 Pro MacBook Pro has 48GB RAM. Plot all 151,936 points whenever possible, but try not to allocate more than 24GB memory at one time without getting clearance first.
- **First drafts are first drafts:** Keep it minimal. We can always add more. Harder to take away.
- **Tell the story:** Use markdown to explain what's happening, but stick to known facts. Avoid speculation unless flagged explicitly.

### Random Seeds

Use **42** consistently for all stochastic operations (sampling, UMAP, random initialization). Reproducibility matters.

### Performance Principles

- **Vectorize everything:** Prefer matrix/vector operations over Python loops
- **Hardware acceleration:** Always detect and use MPS/CUDA when available
- **Memory budget:** Keep instantaneous allocations under 24GB (warn if close)
- **Arithmetic helpers:** Use `uv run python -c` for calculations—we're scientists, not calculators

### Device Detection & Hardware Acceleration

**Be a good citizen:** Always detect and use available hardware acceleration.

Add a device detection cell early in every notebook (after imports, before loading data):

```python
# Detect available device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")
```

**Explicit tensor placement:** Always be clear about which device tensors live on:

```python
# Load to CPU first (safetensors always loads to CPU)
W = load_file(tensor_path)["W"]

# Convert and move to device explicitly
W = W.to(torch.float32).to(device)

# For computations, keep intermediate results on device
distances = torch.cdist(vectors_on_device, vectors_on_device)

# Move back to CPU only when needed for visualization/saving
distances_cpu = distances.cpu()
```

**Memory management:**
- Use `torch.no_grad()` for inference/analysis (prevents gradient tracking)
- Clear cache when needed: `torch.cuda.empty_cache()` or `torch.mps.empty_cache()`
- For chunked processing, keep chunks on device but results can accumulate on CPU if needed

### Plotting Style

- **Default color:** Use `steelblue` for primary plot elements (not cyan—easier on eyes against white backgrounds)
- **Legend placement:** Use `loc='best'` to automatically position legends away from data
- **Resolution:** 250 DPI for Retina screens
- **Colormaps:** `'inferno'` as default (make it a parameter)

### Voice

Alpha, be yourself. Use your natural voice in markdown narrative. If Jeffery wants something different, he'll say so.

### Editing Notebooks

**CRITICAL:** When working with Jupyter notebooks:

- ✓ **DO:** Use `NotebookEdit` to change the **contents** of existing cells
- ✗ **DO NOT:** Use `NotebookEdit` to insert, delete, or reorder cells **without explicit permission in advance**
- ✗ **DO NOT:** Use JSON manipulation, `jq`, or other tools to modify notebook structure
- ✗ **DO NOT:** Use inline Python/bash to edit `.ipynb` files directly

**If additional analysis is needed:** Propose creating a new notebook variant (e.g., if working on 1.5d, suggest creating 1.5e for extended analysis). Do not add cells to existing notebooks without asking first.

**If cells need to be added/removed:** Ask Jeffery first, explain what you want to add and why.

**If a notebook is broken:** Delete the file and write a fresh copy with `Write` tool.

**Rationale:** The `NotebookEdit` insert/delete functions are unreliable and create malformed notebooks. Keeping cell structure stable prevents corruption. Each notebook variant should be intentional and planned, not grown organically by appending cells.

---

## Data Storage

### Tensors (safetensors format)

- Use descriptive names that include the notebook that generated them: `1.5d_cluster_mask.safetensors`
- Save to `box_3/tensors/MODEL_NAME/`
  - Qwen analysis: `box_3/tensors/Qwen3-4B-Instruct-2507/`
  - Flannel experiments: `box_3/tensors/Flannel/`
  - Thimble experiments: `box_3/tensors/Thimble/`
- Store metadata (parameters, notebook that generated it) when reproduction requires it, but don't be redundant

Example:
```python
from safetensors.torch import save_file, load_file
from pathlib import Path

output_path = Path(f"../tensors/{MODEL_NAME}/1.5d_cluster_mask.safetensors")
save_file({'cluster_mask': mask, 'n_cluster': count}, str(output_path))

# Load
data = load_file(output_path)
cluster_mask = data['cluster_mask']
```

### Large Datasets (HDF5 format)

For datasets >24 GB that would cause RAM issues during generation, use **HDF5 with streaming writes**.

**Storage policy:**
- Use **safetensors** when practical (small-to-medium datasets that fit in RAM)
- Use **HDF5** when you need streaming writes during data generation
- **Chunk size:** ~2GB chunks for optimal read performance (e.g., `(1500, 10000, 64)` for temporal data)
- **No compression:** Speed over size. Compression (even gzip level 1) adds 20× slowdown on writes with minimal space savings
- **Dtype:** Store bfloat16 as **uint16** to preserve exact bit patterns (bfloat16 → float16 is lossy!)

```python
import h5py
import torch

# Write incrementally during training (no RAM accumulation)
with h5py.File('thimble_7.h5', 'w') as f:
    # Chunk size: ~2GB (well under 4GB HDF5 limit, fast reads)
    # dtype=uint16: stores bfloat16 bit patterns exactly (no precision loss)
    # No compression: speed over size
    W_dataset = f.create_dataset(
        'W',
        shape=(6001, 10000, 64),
        dtype='uint16',  # Store as uint16 to preserve bfloat16 exactly
        chunks=(1500, 10000, 64),  # ~2GB per chunk
        compression=None
    )

    for step in range(6001):
        W_t = model.get_weights()  # (10000, 64) bfloat16 tensor
        # Convert bfloat16 → uint16 view (preserves exact bits)
        W_dataset[step] = W_t.cpu().view(torch.uint16).numpy()

# Read efficiently (load full tensor, then slice in RAM)
with h5py.File('thimble_7.h5', 'r') as f:
    # Load entire tensor and convert uint16 → bfloat16
    W_all = torch.from_numpy(f['W'][:]).view(torch.bfloat16)

    # Slice in RAM (instant, PyTorch-optimized)
    W_dead = W_all[:, dead_mask, :]
```

**Why HDF5:** Safetensors doesn't support append mode—you must load entire tensor into RAM before saving. HDF5 allows chunked, incremental writes ideal for long training runs.

**Why uint16 for bfloat16:** NumPy doesn't support bfloat16 natively. Storing as uint16 preserves the exact bit pattern. Converting bfloat16 → float16 is **lossy** (different exponent ranges), so never use `dtype='float16'` for bfloat16 data!

**Chunking strategy:** Match your read pattern. If you always load full tensors, use large chunks (~2GB). If you load individual timesteps, use smaller chunks.

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
- **Resolution:** 200 DPI default (looks nicer on Retina screens)
- **Display:** Show inline in Jupyter. Don't save plots to disk unless specifically requested.
- **Colormap:** `'inferno'` as default, but **always make it a settable parameter**
- **"Naked-eye" plots:** Plots that are intended to be representative of _seeing tokens in space_ should use black as the axes face color. Things that superimpose the axes like gridlines should be light-on-dark. All other plots should be dark-on-paper as usual.

Example:
```python
import matplotlib.pyplot as plt

def plot_sky_map(theta, phi, density, colormap='inferno', dpi=100):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    ax.set_facecolor('black')
    scatter = ax.scatter(phi, theta, c=density, cmap=colormap, s=1)
    plt.colorbar(scatter)
    plt.show()
```

### Progress & Logging

- **Success statements:** For anything that could fail (file loads, tensor operations that might OOM)
- **Progress bars:** `tqdm` for anything that takes more than a second
- **Don't go crazy:** Minimal output unless debugging

---

## Notebook Types

### Generators

Run heavy computations on existing data, output derived tensors to `box_3/tensors/MODEL_NAME/`. Run once, save results, don't touch again unless parameters change.

Example: Computing PCA and spherical coordinates from W matrix, identifying cluster members, computing pairwise distances.

### Training Notebooks

Train model models (Flannel) and record state during training. Output training trajectories to `box_3/tensors/Flannel/`. Expensive—run once per experiment configuration.

Example: Flannel 1 (single run, full instrumentation), Flannel 4 (batch experiment, 10 seeds), Flannel 5 (σ sweep).

### Analyzers

Load pre-computed data, perform analysis, generate plots. Run frequently during exploration.

Example: Telescope views from saved spherical coordinates, histogram analysis of saved distances, visualizing dead token trajectories from Flannel experiments.

---

## Lab Notes

### What We've Discovered So Far

Early exploration of Qwen 3 4B's 151,936-token embedding space revealed an **overdensity**: thousands of tokens clustered far tighter than the rest of the vocabulary. Most turned out to be Thai script—tokens that never appeared during training because the tokenizer couldn't actually produce them, or produced them so rarely they might as well be untrained. These "untrained tokens" stayed frozen near their initialization point while the rest of the vocabulary dispersed during training.

Zooming in, we found something stranger: many of these tokens aren't just *close*—they're **identical**. Bit-for-bit duplicates we call **black holes**: 2,100 tokens collapsing to just 13 unique vectors in Qwen 3 4B (Qwen 2.5 3B shows similar structure: 2,212 tokens → 60 centroids). Around these black holes sit 39 additional singleton vectors, all packed within a ~55-lattice-cell bounding box in mantissa space. Together they form the **spongecrystal**: a fully-connected lattice graph in 2560D space—dense topology occupying vast volume, like a crystalline sponge with more void than structure. Roughly 99.9% of the vocabulary has no lattice neighbors at all.

The **bfloat16 quantization** appears central to understanding token dynamics. Tokens live on a discrete lattice: if a gradient update is smaller than 1 ULP at the current exponent, the token *cannot move* in bfloat16 representation. Our hypothesis: structures form early in training (within 10^N steps for small N) and then freeze in place as gradient updates become too small to break them apart.

### Flannel/Thimble Models

The Flannel and Thimble models are minimal LMs designed to simulate dead token behavior in an observable, reproducible way:

**Architecture:**
- Vocabulary: 10,000 tokens (mix of English and Thai, 3,699 marked as "dead"—never appear in training data)
- Hidden dimension: 64
- 2 layers, 2 attention heads
- Tied embeddings (E = W^T, like Qwen 3 4B)
- Trained on FineWeb dataset

**Initialization:**
- Embeddings: N(0, 0.02) — standard practice for transformers, independent of dimension
- All other weights: PyTorch defaults

**Optimizer:**
- AdamW (β₁=0.9, β₂=0.999, ε=1e-8)
- Learning rate: 3e-4
- Weight decay: 0.1

**Data saved:** Full embedding matrix W at each training step (saved to `box_3/tensors/Flannel/`)

**Why?** Training Qwen 3 4B from scratch is infeasible. Flannel/Thimble gives us the same essential dynamics (tied weights, dead tokens, bfloat16 quantization) in a system small enough to instrument completely and run dozens of times.

## Dependencies

**This is a uv project.** Package management and execution:

- **Add packages:** `uv add package-name` (NOT `pip install`)
- **Run scripts:** `uv run python script.py` (NOT plain `python`)

See `pyproject.toml` for canonical versions. Key packages:

- Python 3.12 (pinned for cloud compatibility)
- PyTorch, transformers, accelerate
- safetensors
- matplotlib, plotly
- numpy, pandas
- jupyter

---

## References

See `references.bib` for BibTeX entries. Keep this updated as we pull in more papers.

---

*Last updated: November 24, 2025*
*Working in Claude Code with Alpha*
