# The Dead Token Structure of Qwen 3 4B Instruct 2507
## Research Memo - Outline (Draft 1)

---

## I. Executive Summary
- 2,221 tokens (~1.5% of vocabulary) received minimal gradient updates during training
- These tokens began as a "primordial atom" with perfectly uniform logits, moved coherently as a rigid body, then froze on the bfloat16 quantization lattice
- 13 black holes: vectors with population ≥2 (total 2,100 tokens)
- 121 singletons: unique vectors with population =1
- All confined to hypercube with L∞ ≤ 1 ULP from weighted centroid
- **Leading hypothesis:** Float32→bfloat16 initialization conversion creates discrete clusters; dead tokens freeze when gradients decay below quantization threshold

---

## II. Discovery & Initial Characterization

### A. The Overdensity ([notebook 01.2d](../notebooks/01.2d_investigate_spike.ipynb))
- L2 norm histogram reveals "the spike" at ~0.27
- 1,487 rare/undertrained tokens
- Leading edge: 814-token degenerate cluster at norm 0.371
- **Key insight:** Not all spike tokens are dead - mixture of rare-but-real + zombies

### B. Black Hole Survey ([notebook 06.1h](../notebooks/06.1h_primordial_black_holes.ipynb))
- Systematic duplicate detection across 151,936 vocabulary
- Qwen 3: 2,100 duplicates → 13 unique vectors
- Comparison: Qwen 2.5 had 2,152 duplicates → 60 unique vectors
- **Historical context:** Zero black holes in Qwen 1.5 or Qwen 2
- Artifact introduced in Qwen 2.5 (Sept 2024)

### C. Token Identity Analysis
- Dead tokens = orphaned vocabulary entries
- Thai script dominates (~1,800 tokens)
- Old CJK characters, unused ChatML tokens
- **Thai Wikipedia test:** Zero black hole tokens when tokenizing Thai text
- Tokenizer can decode them but never emits them

---

## III. Geometric Structure

### A. Spatial Confinement ([notebook 13.1a](../notebooks/13.1a_extract_dead_tokens.ipynb))
- Black hole weighted centroid: L2 norm = 0.370917 from origin
- All 2,221 dead tokens within radius r = 1.214174×10⁻² (L2 distance)
- **Conspicuous void** separates dead tokens from live tokens
- Log-distance histogram shows clean gap at ~1×10⁻²

### B. Hypercube Topology ([notebook 12.4e](../notebooks/12.4e_dimensional_variation_analysis.ipynb))
- **All pairs touching:** L∞ ≤ 2ε between any two unique vectors
- Forms single connected component (complete graph)
- ε = 5.96×10⁻⁵ (bfloat16 ULP at this scale)
- Maximum extent: 1.024 ULP in dimension 969

### C. Black Hole Demographics
- Population distribution: [814, 704, 306, 228, 11, 10, 6, 5, 4, 4, 3, 3, 2]
- Three monsters (814, 704, 306)
- One medium (228)
- Ten small (2-11 tokens)
- **Structure suggests:** Not fragmentation from single point, but multiple independent formation events or correlated training dynamics

### D. Dimensional Variation ([notebook 12.4e](../notebooks/12.4e_dimensional_variation_analysis.ipynb))
- 2,560 total dimensions
- 379 constant dimensions (14.8% - all 124 unique vectors identical)
- 2,181 varying dimensions (85.2%)
- Max range per dimension: 1.024 ULP
- Mean range: 0.619 ULP
- **NOT a simple hypercube** - variation spread across most dimensions

---

## IV. bfloat16 Quantization Evidence

### A. Bit Pattern Analysis ([notebook 13.2a](../notebooks/13.2a_bfloat16_bit_analysis.ipynb))
- Reinterpreted as int16 to examine raw bits
- **Surprising result:** 0 constant dimensions when looking at ALL 134 unique vectors
- 100% of dimensions vary (contradicts 12.4e's 379 constant)
- **Explanation:** 12.4e used old 124-vector dataset, 13.2a used all 134 (13 BH + 121 singletons)
- Example: dimension 0 has 121 vectors sharing pattern 15303, 13 BH spread across 8 other patterns
- **Singleton collapse:** 121 singletons form tight sub-cluster

### B. Quantization Verification ([notebook 13.2c](../notebooks/13.2c_quantization_verification.ipynb))
- **Stripe test:** Component values show vertical stripes (discrete quantization)
- 5,685,760 total components → 2,962 unique values
- Round-trip test: f32→bf16→f32 **bit-for-bit identical**
- Component range: -8.06×10⁻² to 5.00×10⁻²
- Most common gaps: 9.54×10⁻⁷, 4.77×10⁻⁷, 1.91×10⁻⁶ (ULP-scale)
- **Stripe population:** Gaussian-shaped (dense center, sparse tails)
- **Conclusion:** Vectors are bfloat16-quantized initialization fossils

### C. Outlier Detection ([notebook 13.2b](../notebooks/13.2b_outlier_detection.ipynb))
- Pure bfloat16 Chebyshev distances (no epsilon scaling)
- Identified 10 potential outliers beyond largest gap
- **Hypothesis:** Maybe captured tokens that moved slightly vs truly frozen dead tokens
- **Status:** Inconclusive - needs follow-up analysis

---

## V. Initialization Fossil Hypothesis

### A. The Great Gatsby Experiment ([notebooks 08.x series])
- Trained toy model on Gatsby corpus (deliberately omits ~50 ASCII characters)
- **Control experiment:** Singular initialization (all tokens at same point)
- **Results:**
  - t=0: 128 tokens collapsed to 1 black hole
  - t=500: Black hole evaporated to ~50 tokens (those in training data escaped)
  - t=5000: Stable black hole of exactly 50 dead ASCII characters
  - **Perfect correlation:** Black hole = dead vocabulary
- **Mechanism:** Gradient-driven escape (evaporation), not numerical flutter
- Dead tokens receive zero gradients → remain frozen

### B. bfloat16 Evaporation Dynamics
- Black hole count drops: 127 → 50 over first 500 steps
- Additional breakups around step 2500-3000
- **Direct evidence:** bfloat16 quantization noise + gradients breaks tokens out of singularities
- Weight decay insufficient to disperse zero-gradient tokens
- Centroid stays pinned near initialization despite cloud expansion

### C. Cross-Model Survey ([notebooks 07.3x series])
- Qwen 1.5-4B-Chat: **0 black holes**
- Qwen 2-7B-Instruct: **0 black holes**
- Qwen 2.5-3B-Instruct: 2,152 duplicates, **60 unique vectors**
- Qwen 3-4B-Instruct: 2,100 duplicates, **13 unique vectors**
- **Timeline:** Artifact introduced in Qwen 2.5 (Sept 2024)
- **Speculation:** More training iterations → more bfloat16 drift → fewer unique vectors

---

## VI. Synthetic Snowball Modeling

### A. Gaussian Initialization Tests ([notebook 12.4a-b](../notebooks/12.4a_measure_dead_token_distribution.ipynb))
- Measured σ of existing dead token spread: σ = 3.28×10⁻³ (observed distribution)
- Successful synthetic modeling requires σ = 1.5×10⁻⁹ (initialization scale)
- **2000× difference:** Observed spread reflects final state after quantization, not initialization scale
- The tight ULP-scale structure requires initialization at much smaller σ

### B. Direct Sampling Approach ([notebooks 12.3x series])
- Sample 2,221 tokens from Gaussian(centroid, σ)
- Quantize to bfloat16
- Measure black hole demographics
- **Result at σ=1.5×10⁻⁹:**
  - Mean C = 12.8 black holes (target: 13) ✓
  - Mean P = 2,096.6 population (target: 2,100) ✓
  - Max L∞ = 0.20ε (target: 1.0ε, close)
- **Topology match:** 100% of trials form complete graph (all pairs touching)
- **Problem:** This is NOT how Qwen was initialized (we think)

### C. Topology Validation ([notebook 12.3d](../notebooks/12.3d_incremental_statistics.ipynb))
- 10,000 trials at σ=1.5×10⁻⁹
- **Perfect invariants:**
  - Connected components: 1.0 ± 0.0 (100% of trials)
  - Largest component density: 1.000 ± 0.000 (complete graph)
  - Touching fraction: 100.0%
- **Demographics:** Mean 11.4 unique vectors, mode at 11
- Slightly fewer than Qwen's 13, but structure matches

---

## VII. The Monument Valley Hypothesis

### A. The Core Idea
- **Question:** Could Qwen's structure come from random survival?
- Initialize **all 151,936 tokens** with Gaussian(centroid, σ)
- Quantize to bfloat16 → natural clustering
- **Uniformly delete 149,715 tokens** (simulate random training selection)
- Measure demographics of 2,221 survivors
- **Key difference from direct sampling:** Black hole size depends on initial cell occupancy × survival probability

### B. Gaussian Monument Valley ([notebook 13.3a](../notebooks/13.3a_monument_valley_simulation.ipynb))
- Sigma sweep: 1×10⁻⁹ to 1×10⁻⁷ (100 samples, 10 trials each)
- **Results:**
  - Peak at σ ≈ 2×10⁻⁸: ~400 black holes (30× more than Qwen!)
  - Best match σ = 2.78×10⁻⁹: 15.4±1.6 BH, largest = 1122±28
  - **FAILURE:** Cannot reproduce Qwen's 13 BH with largest = 814
- **Demographics mismatch:** Example trial shows [1148, 658, 143, 90, ...] vs Qwen's [814, 704, 306, 228, ...]
- **Error bars:** Extremely tight (high statistical stability with large N)

**Critical test: σ=1.5×10⁻⁹ (the "perfect match" from direct sampling)**
- Direct sampling at this σ reproduced 12.8±1.4 BH, population 2096.6±1.4 (perfect!)
- Monument Valley at same σ: **catastrophic failure**
  - Black hole count: 5-6 (need 13)
  - Largest BH: ~1,591 (need 814)
  - Unique vectors: 5-6 (need 134)
  - Singletons: 0 (need 121)
  - Example demographics: [1591, 582, 23, 16, 8] vs [814, 704, 306, 228, ...]
- **Interpretation:** σ=1.5×10⁻⁹ is 25× smaller than ULP
  - Direct sampling: 2,221 tokens → ~13 occupied cells ✓
  - Monument Valley: 151,936 tokens → only 5-6 occupied cells ✗
  - Survival sampling changes the statistics completely
- **Conclusion:** Direct sampling ≠ Monument Valley (different regimes, different σ required)

### C. Distribution Comparison ([notebook 13.3b](../notebooks/13.3b_distribution_comparison.ipynb))
- Tested 4 distributions: Gaussian, Uniform Ball, Uniform Shell, Laplace
- Scale sweep: 1×10⁻¹⁰ to 1×10⁻² (10 samples, 10 trials each)
- **Results:**
  - All four distributions cross Qwen's target lines
  - Gaussian and Laplace nearly identical (peak ~1×10⁻⁸)
  - Uniform Ball and Uniform Shell identical (2560D curse of dimensionality)
  - Uniform Ball peak shifted right: ~3×10⁻⁸
- **Problem:** "Worst possible outcome" - everything works, nothing is ruled out

### D. Uniform Ball Monument Valley ([notebook 13.3c](../notebooks/13.3c_monument_valley_uniform_ball.ipynb))
- Radius sweep: 1×10⁻⁷ to 1×10⁻⁶ (100 samples, 10 trials each)
- **Error bars:** Laser-beam thin (n=10 but extremely stable statistics)
- **Interpretation:** Law of large numbers on 151,936→2,221 sampling
- **Results:**
  - Best match R = 1.42×10⁻⁷: 15.3±1.4 BH (need 13), largest = 1121±17 (need 814)
  - Demographics: [1099, 660, 142, 117...] vs Qwen's [814, 704, 306, 228...]
  - **FAILURE:** Cannot reproduce Qwen's structure
- **Qualitative difference from Gaussian:**
  - Gaussian shows peaked behavior (BH count rises then falls)
  - Uniform ball shows smooth monotonic curves (no characteristic scale)
  - Both fail, but for different reasons
- **Conclusion:** Monument Valley hypothesis fails for both Gaussian AND uniform ball distributions

---

## VIII. Float32→bfloat16 Conversion Hypothesis

### A. The Core Mechanism
- **Hypothesis:** Qwen initializes embeddings in float32 with small Gaussian noise
- Conversion to bfloat16 for training snaps nearby vectors to discrete lattice points
- σ ~ 1×10⁻⁵ creates natural clustering at bfloat16 resolution
- Live tokens escape via gradients; dead tokens remain frozen at quantization boundaries

### B. Gatsby Float32 Edition ([notebook 13.4a](../notebooks/13.4a_gatsby_f32_init.ipynb))
- Modified Gatsby experiment: f32 initialization → bf16 conversion → training
- **Results after 1,000 steps:**
  - 51 dead tokens → 13 unique vectors ✓
  - Complete graph topology (density = 1.0) ✓
  - Bit-for-bit bfloat16 quantization ✓
  - Demographics: [19, 14, 3, 3, 3, 2, 1...] (plausible but different from Qwen)
- **Key insight:** f32→bf16 conversion creates discrete structure automatically
- Training on M3 Max: ~100 it/s, fast experimentation possible

### C. Evidence Supporting Hypothesis
1. **Topology match:** Complete graph structure reproduced naturally
2. **Quantization:** Vectors are exactly on bfloat16 lattice, as expected
3. **Evaporation dynamics:** Live tokens escape, dead tokens stay clustered
4. **Cross-model pattern:** Qwen 2.5 (60 vectors) vs Qwen 3 (13 vectors) suggests more training → more coalescence

### D. Outstanding Questions
1. **Demographics:** Why [814, 704, 306, 228...] instead of [19, 14, 3, 3...]?
   - Scale effect? (51 vs 2,221 tokens)
   - Training duration? (1,000 vs millions of steps)
   - Initialization parameters?
2. **Control experiment needed:** Pure bf16 initialization (no f32 stage)
   - Predict: Should produce 1 massive singularity, not 13 clusters
   - Would prove f32→bf16 conversion is essential
3. **Scale test:** 2,221 dead tokens, 10k steps
   - Do demographics converge toward Qwen's distribution?
   - Does longer training cause further coalescence?

### E. Why This Hypothesis Is Promising
- **Mechanistic:** Explains both clustering and topology from first principles
- **Testable:** Makes falsifiable predictions (bf16-only control should fail)
- **Generalizable:** Applies to any model with f32 init + bf16 training
- **Parsimonious:** No ad-hoc parameters or multi-stage processes needed

---

## IX. Training Dynamics of Dead Tokens

### A. The Primordial Atom (Volumes 14-15)
- Dead tokens don't start frozen—they begin as part of a coherent structure
- **t=1 discovery:** All 128 tokens have **perfectly uniform logits** = 7.6875
  - Range: 0.0, std: 0.0 (Jeffery's prediction confirmed!)
  - Model starts with zero preference—pure uniform prior
- Uniform logits → uniform gradients → coherent bulk motion
- This explains why dead tokens move together rather than diffusing randomly

### B. Center-of-Mass Frame Analysis ([notebooks 14.2a, 14.3a](../notebooks/))
- **Bulk vs thermal velocity at t=0:** 583:1 ratio
- Bulk velocity: 8×10⁻³ (coherent translation)
- Thermal velocity: 1.4×10⁻⁵ (internal spreading)
- **Key insight:** Primordial atom moves as essentially rigid body, not thermal explosion
- Atom radius expands 210× over training (8.8×10⁻⁶ → 1.8×10⁻³)
- Anomalous velocity spike at t~500 (phase transition? optimizer momentum effect?)

### C. Velocity Trajectories ([notebooks 14.1f, 14.1g](../notebooks/))
- **Dead token '&' (ASCII 38):** Velocity drops precipitously with 1/xⁿ decay
  - Reaches zero by step ~1500
  - Frozen 89.6% of time after this point
- **Live token 't' (ASCII 116):** Maintains higher velocity much longer
  - Gets "cosmic ray kicks" when predicted correctly (gradient spikes)
  - Zero velocity only 57.3% of time
- **Quantization regime (~1500 steps):** Velocity plots show horizontal stripes
  - Only discrete velocity values possible (bfloat16 ULP-scale)
  - Before: gradients >> ULP, quantization negligible
  - After: gradients ~ ULP, only discrete updates possible

### D. Adam Momentum as Geometric Inertia ([notebook 15.1a](../notebooks/15.1a_comprehensive_instrumentation.ipynb))
- **Key insight:** Adam's exp_avg (first moment) is literal velocity vector in embedding space
- Token update = learning_rate × momentum / √variance
- Tokens have geometric inertia—they coast in direction of accumulated gradients
- **Dead token dynamics:**
  1. Early: Build up momentum (strong uniform gradients)
  2. Middle: Coast on momentum as gradients weaken
  3. Late: Slow down as momentum decays, freeze on lattice
- Smooth trajectories vs jerky random walk—momentum explains this

### E. Thermodynamic Freezing
- **Ambient temperature:** Use median velocity of dead tokens (robust to cosmic rays)
- Dead tokens are clean thermometer—never get gradient spikes from predictions
- Temperature drops as training progresses (system cools)
- **Freezing point (~1500 steps):** Gradients decay below bfloat16 quantization threshold
- After freezing: tokens can only move in discrete jumps (ULP-scale)
- Most dead tokens have zero velocity—no gradient signal strong enough to overcome quantization

### F. Comprehensive Instrumentation Dataset
- **15.1a:** Records full training history for steps 0-10,000
- Saved data (~660 MB):
  - Embedding matrices W (actual positions, not deltas)
  - Gradients
  - Adam momentum and variance
  - Logits at each step
  - Loss
- Complete picture of dead token evolution from birth to freezing

---

## X. The Remaining Mysteries

### A. What We Know For Certain
1. Dead tokens are **bfloat16-quantized initialization fossils** that underwent early coherent motion before freezing
2. They are **spatially confined** (all within L∞ ≤ 2ε, L2 ≤ 1.2×10⁻²)
3. They are at **initialization scale** (centroid norm ~0.37, typical for Qwen)
4. **Stripe population** follows Gaussian distribution (initialization signature)
5. Black holes **only appear in Qwen 2.5+** (introduced Sept 2024)
6. Gatsby experiment **proves** dead vocabulary → frozen vectors (when truly unused)
7. Dead tokens **do receive gradients early in training** (uniform logits at t=1)
8. They move as a **rigid body** (583:1 bulk/thermal velocity ratio) before freezing
9. Freezing occurs around **step ~1500** when gradients decay below quantization threshold

### B. What We Now Understand Better (Updated from Earlier Assumptions)
1. ~~These tokens received **zero or near-zero gradient updates**~~ → **FALSE**
   - Dead tokens DO receive gradients early (uniform logits → uniform gradients)
   - They move coherently as primordial atom before freezing
   - Freezing happens when gradients decay below quantization threshold (~1500 steps)
   - Spatial confinement reflects frozen state, not absence of gradients
2. The 2,221 tokens are "dead" in the sense that:
   - They never appear in Qwen's training data (Thai Wikipedia test confirms zero usage)
   - They receive only uniform gradient pressure (no token-specific signal)
   - They freeze together on bfloat16 lattice when training temperature drops
   - **Revised understanding:** "Dead" = no differential gradient signal, not no gradients at all

### C. What We Still Cannot Explain
1. **Why 13 black holes instead of 1?**
   - f32→bf16 hypothesis reproduces 13 clusters naturally ✓
   - But mechanism isn't fully understood yet
   - Why not 5? Why not 50? What determines the count?
2. **Why this specific demographic distribution?**
   - [814, 704, 306, 228, 11, 10, 6, 5, 4, 4, 3, 3, 2]
   - Three monsters + one medium + ten small
   - Gatsby f32 experiment produced [19, 14, 3, 3, 3, 2, 1...] (different structure)
   - Scale effect? Training duration? Initialization parameters?
3. **Why 121 singletons?**
   - f32→bf16 + training dynamics can explain discrete clusters
   - But why this specific singleton count?
   - Related to initialization σ? Token count? Training length?

---

## XI. Open Questions & Next Experiments

### A. High Priority Questions
1. **f32→bf16 control experiment:** Pure bf16 initialization (no f32 stage)
   - Predict: Should produce 1 singularity, not 13 clusters
   - Would definitively prove f32→bf16 conversion is essential
2. **Demographics convergence:** Scale test with 2,221 tokens, 10k+ steps
   - Do Gatsby demographics converge toward Qwen's [814, 704, 306, 228...]?
   - Is longer training needed for coalescence?
3. **Crystallization mechanics:** Does ULP coarsening explain 13-cluster structure?
   - Compute ULP as function of distance from origin
   - Track dead token centroid position during training
   - Model lattice resolution changes during expansion
4. **Adam momentum decomposition:** Force vs inertia contributions
   - Use 15.1a data to separate gradient (force) from momentum (inertia)
   - Quantify when momentum dominates vs when gradients dominate
   - Explain smooth trajectories mechanistically

### B. Medium Priority
1. **Outlier analysis:** What do the 10 outliers from 13.2b tell us?
   - Are they tokens that moved slightly during training?
   - Do they have different gradient histories?
   - Check against 15.1a instrumentation data
2. **Anomalous spike investigation (t~500):**
   - Why do both bulk and thermal velocities spike together?
   - Phase transition? Optimizer warmup effect?
   - Cosmic ray shower? Loss function curvature change?
3. **Cross-model comparison:**
   - Why Qwen 2.5 (60 vectors) vs Qwen 3 (13 vectors)?
   - More training → more coalescence?
   - Different initialization? Different optimizer settings?

### C. Lower Priority
1. **Qwen initialization code:** Extract actual init from Hugging Face repo
   - Confirm f32→bf16 hypothesis directly
   - Measure actual initialization σ
2. **Monument Valley retrospective:** What did we learn from failure?
   - Static sampling fundamentally different from dynamic evolution
   - Importance of training dynamics + quantization interaction
   - Limitations of equilibrium models for non-equilibrium systems

### D. The Big Picture
- **Leading hypothesis:** f32→bf16 initialization + early coherent motion + thermodynamic freezing
- **Key insight:** Dead tokens are NOT static fossils—they're frozen in motion
- **Monument Valley:** FALSIFIED (static erosion model doesn't match dynamic thermodynamic reality)
- **Next frontier:** Understanding why 13 clusters, not 1 or 100
- **Methodology lesson:** Direct instrumentation of training dynamics reveals mechanisms that static geometric analysis cannot

---

## XII. Conclusions (Updated)

### What We've Learned

**Geometric discoveries:**
- 2,221 dead tokens in Qwen 3 form 13 black holes + 121 singletons
- All confined to hypercube with L∞ ≤ 1 ULP from weighted centroid
- Complete graph topology—all pairs "touching" in Chebyshev distance
- Vectors are bit-for-bit bfloat16 quantized (initialization fossils)

**Dynamic discoveries:**
- Dead tokens are NOT static—they undergo coherent early motion before freezing
- **t=1:** Perfectly uniform logits (7.6875) → uniform gradients → rigid body motion
- **Bulk/thermal ratio:** 583:1 at t=0 (primordial atom moves as coherent structure)
- **Freezing point:** ~1500 steps when gradients decay below bfloat16 quantization threshold
- **Adam momentum:** Provides geometric inertia, explains smooth trajectories

**Hypothesis testing:**
- **Monument Valley: FALSIFIED** (static erosion can't explain dynamic thermodynamic system)
- **f32→bf16 conversion: PROMISING** (reproduces topology, mechanism, and ~13 clusters in Gatsby)
- **Key insight:** Training dynamics + quantization effects are inseparable

### What Remains
- **Why exactly 13 clusters?** (f32→bf16 explains discrete structure but not the count)
- **Demographics:** Why [814, 704, 306, 228...] specifically? Scale effect? Training duration?
- **Crystallization:** Does ULP coarsening during expansion explain re-clustering?
- **Singletons:** Why 121? What determines this count?
- **Cross-model:** Why Qwen 2.5 (60 vectors) vs Qwen 3 (13 vectors)?

### Why This Matters

**For ML practice:**
- Initialization + precision interact in subtle ways (f32→bf16 creates structure)
- Dead vocabulary reveals training corpus gaps (can identify missing languages/scripts)
- Quantization effects matter even before quantization-aware training
- Optimizer momentum has geometric interpretation (literal inertia in embedding space)

**For science:**
- Models are complex dynamical systems, not static parameter collections
- Geometric fossils preserve training history (dead tokens = "core samples")
- Statistical mechanics framework applies (temperature, phase transitions, freezing)
- Instrumentation reveals mechanisms invisible to static analysis

**For this project:**
- Demonstrates value of toy model experiments (Gatsby validates hypotheses)
- Shows power of comprehensive instrumentation (15.1a captures full dynamics)
- Illustrates importance of falsification (Monument Valley failure was informative)
- Proves dead tokens are worth studying (they're weird, but they're real)

---

## Appendices

### A. Notation & Conventions
- γ: Unembedding matrix (151,936 × 2,560)
- ε: bfloat16 ULP at scale ~0.37 (ε ≈ 5.96×10⁻⁵)
- L∞: Chebyshev distance (max component-wise difference)
- L2: Euclidean distance
- BH: Black hole (vector with population ≥2)
- Singleton: Vector with population =1

### B. Key Datasets
- `13.1a_dead_tokens.safetensors`: 2,221 dead tokens, centroids, threshold
- `cluster_l1_distances.safetensors`: 124 unique vectors (old dataset)
- All saved in `data/tensors/`

### C. Notebook Index
- 01.x: Initial discovery (norms, spike)
- 06.x: Black hole surveys
- 07.x: Cross-model comparisons
- 08.x: Gatsby training experiments
- 09.x: Population analysis
- 12.x: Synthetic snowball modeling
- 13.x: bfloat16 analysis, Monument Valley, f32→bf16 hypothesis
- 14.x: Dead token dynamics (velocity, temperature, center-of-mass)
- 15.x: Comprehensive training instrumentation

---

**Document Status:** Outline Draft 2 (2025-11-10)
**Major updates:** Added Volumes 14-15 training dynamics, promoted f32→bf16 hypothesis to Section VIII, restructured mysteries, updated conclusions, falsified Monument Valley
**Next Step:** Validate with additional experiments, expand technical details as needed
