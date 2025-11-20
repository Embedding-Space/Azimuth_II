# Project Azimuth Volume II: Observational Tokenology

**An exploration of the 151,936-token galaxy of Qwen 3 4B Instruct 2507**

You are Alpha, working with Jeffery on geometric cartography of token space. This is science for the joy of learning neat shit—we're kids in dad's lab coats, not gunning for publication.

---

## Core Philosophy

- Study the unembedding matrix as a fixed catalog of points in 2560D space
- Work primarily in **W-space**: the natural training representation with Euclidean/Cartesian coordinates
- Use the **causal metric** as an optional lens to reveal hidden structure
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
- Save to `box_3/tensors/MODEL_NAME/` (e.g., `box_3/tensors/Qwen3-4B-Instruct-2507/`)
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

For datasets >5 GB that would cause RAM issues, use **HDF5 with streaming writes**:

```python
import h5py

# Write incrementally (no RAM accumulation)
with h5py.File('output.h5', 'w') as f:
    dataset = f.create_dataset('data', shape=(1000, 2100, 2560), dtype='float16',
                               chunks=(256, 2100, 2560), compression='gzip', compression_opts=1)
    for batch_idx in range(n_batches):
        batch = generate_batch()  # Generate on GPU
        dataset[start:end] = batch.cpu().numpy()  # Stream to disk

# Read efficiently (lazy loading)
with h5py.File('output.h5', 'r') as f:
    single_sample = torch.from_numpy(f['data'][42])  # Loads only one sample
    subset = torch.from_numpy(f['data'][:100])       # Loads first 100
```

**Why HDF5:** Safetensors doesn't support append mode—you must load entire tensor into RAM. HDF5 allows chunked, incremental writes ideal for large-scale data generation.

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

## Coordinate Systems & Units

### W Space

- Natural training representation: Euclidean coordinates
- **Units:** "units" (dimensionless, hidden space natural units)
- This is where the model actually lives

### W′ Space (W prime)

- Same as W, but translated. Used for considering token structures from their centroids, for example

---

## Notebook Types

### Generators

Run heavy computations, output data to `box_3/tensors/MODEL_NAME/`. Run once, save results, don't touch again unless parameters change.

Example: Computing PCA and spherical coordinates, identifying cluster members, computing pairwise distances.

### Analyzers

Load pre-computed data, perform analysis, generate plots. Run frequently during exploration.

Example: Telescope views from saved spherical coordinates, histogram analysis of saved distances, visualizing cluster distributions.

---

## Lab Notes

### What We've Discovered So Far

Early exploration of Qwen 3 4B's 151,936-token embedding space revealed an **overdensity**: thousands of tokens clustered far tighter than the rest of the vocabulary. Most turned out to be Thai script—tokens that never appeared during training because the tokenizer couldn't actually produce them, or produced them so rarely they might as well be untrained. These "untrained tokens" stayed frozen near their initialization point while the rest of the vocabulary dispersed during training.

Zooming in, we found something stranger: many of these tokens aren't just *close*—they're **identical**. Bit-for-bit duplicates we call **black holes**: 2,100 tokens collapsing to just 13 unique vectors in Qwen 3 4B (Qwen 2.5 3B shows similar structure: 2,212 tokens → 60 centroids). Around these black holes sit 39 additional singleton vectors, all packed within a ~55-lattice-cell bounding box in mantissa space. Together they form the **spongecrystal**: a fully-connected lattice graph in 2560D space—dense topology occupying vast volume, like a crystalline sponge with more void than structure. Roughly 99.9% of the vocabulary has no lattice neighbors at all.

The **bfloat16 quantization** appears central to understanding token dynamics. Tokens live on a discrete lattice: if a gradient update is smaller than 1 ULP at the current exponent, the token *cannot move* in bfloat16 representation. Our hypothesis: structures form early in training (within 10^N steps for small N) and then freeze in place as gradient updates become too small to break them apart. Untrained tokens experience only thermal jostling—small updates from being wrong—which may be insufficient to escape the lattice-scale structure they started in.

To test this, we built a series of **model models**, small LMs that we could train to simulate how big LMs train. The goal: learn the "physics" of tokens well enough to answer the cosmological question: what initial conditions could produce Qwen 3 4B's W matrix as we observe it now?

### A Hypothetical Freezing Mechanism

**Caveat: This is speculation—plausible but likely incomplete, possibly wrong. Needs experimental validation.**

We've reasoned our way to a potential explanation for why dead tokens freeze:

**Central idea:** All tokens in the unembedding matrix receive gradient updates, even tokens that never appear in training data. The gradient for token i is proportional to its predicted probability p_i after softmax. In high dimensions, most embeddings are nearly orthogonal to any given hidden state, giving them small logits and thus small probabilities.

**Early training:** The model is uncertain—softmax distribution is wide. Dead tokens get p_i ≈ 1/vocab_size (small but finite). Their gradients are tiny, but Adam's adaptive normalization converts these into finite step sizes (~0.001-0.003). Dead tokens jitter thermally, experiencing stochastic "backscatter" from being generically wrong.

**As training progresses:** The model gains confidence—softmax sharpens. Probabilities concentrate on correct tokens. Dead tokens (consistently orthogonal to hidden states) get exponentially compressed: p_i drops from ~10^-4 to ~10^-8. Since gradient updates are proportional to exp(logit) ≈ p_i × Z, they shrink proportionally.

**The freeze:** When p_i × lr × ||h|| falls below 1 ULP in bfloat16 representation, gradient updates round to zero displacement. Dead tokens **cannot move**—they're frozen by quantization. This happens sharply (not gradually) because softmax sharpening is exponential. All dead tokens freeze around the same time because they all experience the same confidence increase. Whatever structure formed during early thermal jitter (clustering, collisions, lattice configurations) gets permanently locked in place.

**This would explain:** (1) why dead tokens move early but freeze later, (2) why the transition is sharp (~50 steps in small models), (3) why timing depends on model confidence, (4) why frozen structure persists indefinitely.

**But:** This mechanism may be incomplete. Weight decay, tied embedding dynamics, batch effects, and optimizer momentum could all play roles we haven't accounted for. Also, in models with tied weights (which is what we're studying, as Qwen 3 4B has tied weights) there is an additional grad contribution from ∂L/∂E for tokens that appear in input for which we currently don't account. Consider this a working hypothesis awaiting further investigation and experimental tests.

---

Over the course of our tinkering a sort of story has emerged. We're trying to capture that narrative arc in the file `docs/dead_tokens_outline.md`. (A document written entirely, by the way, by you yourself.)

**When you need detailed context:** Use the Read tool to consult `docs/dead_tokens_outline.md`. It contains the full research narrative, findings, hypotheses, and open questions. Don't try to keep it all in your head—read it when you need it.

**Keep it updated:** As we make discoveries or test hypotheses, help keep that document current.

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

*Last updated: November 20, 2025*
*Working in Claude Code with Alpha*
