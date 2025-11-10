# The Dead Token Structure of Qwen 3 4B Instruct 2507
## Research Memo - Outline (Draft 1)

---

## I. Executive Summary
- 2,221 tokens (~1.5% of vocabulary) never moved during training
- 13 black holes: vectors with population ≥2 (total 2,100 tokens)
- 121 singletons: unique vectors with population =1
- All confined to hypercube with L∞ ≤ 1 ULP from weighted centroid
- **Unsolved mystery:** Why 13 clusters instead of 1? Monument Valley hypothesis fails.

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
- **Structure suggests:** Multiple independent formation events, not fragmentation

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
- [Need to check: did we conclude anything from this?]

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
- Measured σ directly from Qwen's dead tokens: σ = 3.28×10⁻³
- Initial guess σ = 1.5×10⁻⁹ from earlier work
- **Measured σ is 2000× larger** than expected for ULP-scale structure
- Fast sweep (10,000 trials) at measured σ

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

## VIII. The Unsolved Mystery

### A. What We Know For Certain
1. Dead tokens are **bfloat16-quantized initialization fossils**
2. They are **spatially confined** (all within L∞ ≤ 2ε, L2 ≤ 1.2×10⁻²)
3. They are at **initialization scale** (centroid norm ~0.37, typical for Qwen)
4. **Stripe population** follows Gaussian distribution (initialization signature)
5. Black holes **only appear in Qwen 2.5+** (introduced Sept 2024)
6. Gatsby experiment **proves** dead vocabulary → frozen vectors (when truly unused)

### B. What We Assume (May Be Wrong!)
1. These tokens received **zero or near-zero gradient updates** during training
   - Thai Wikipedia test: zero tokens appear when tokenizing Thai
   - Spatial confinement suggests minimal movement
   - **Alternative:** Maybe received 1-10 updates in rare contexts?
   - **Testable:** Could correlated updates explain clustering?
2. The 2,221 tokens are truly "dead" (never appeared in pretraining)
   - vs "nearly dead" (appeared once or twice in rare contexts)
   - Small update counts could create correlated displacements
   - 704-token BH: all received same rare update?

### C. What We Cannot Explain
1. **Why 13 black holes instead of 1?**
   - If initialized at single point, should be 1 massive black hole
   - bfloat16 noise alone doesn't create 13 distinct clusters
   - Weight decay can't explain it (cluster is 0.166 units from origin, not at origin)
2. **Why this specific demographic distribution?**
   - [814, 704, 306, 228, 11, 10, 6, 5, 4, 4, 3, 3, 2]
   - Three monsters + one medium + ten small
   - Monument Valley (Gaussian) produces wrong distribution
   - Monument Valley (Uniform Ball) produces [results TBD]
3. **Why 121 singletons?**
   - Single-occupancy survivors from random deletion?
   - But demographics suggest structure, not randomness
4. **What initialization process creates this?**
   - Not pure Gaussian (Monument Valley fails)
   - Not uniform ball (preliminary results suggest failure)
   - Multi-stage? Non-uniform? Different distribution?

### C. Competing Hypotheses

#### Hypothesis 1: Multi-Stage Initialization
- Qwen initialized at single point
- Pre-training warmup with small learning rate
- Dead tokens diffuse via optimizer state updates (Adam momentum?)
- Freeze before main training
- **Problem:** Why 13 clusters? Why touching?

#### Hypothesis 2: Non-Uniform Survival
- Monument Valley assumes **uniform** deletion
- What if training data isn't uniformly random?
- Some dead tokens appear rarely, others never
- Rare appearances → partial updates → slightly different positions
- **Problem:** Dead tokens show zero movement in Gatsby experiment

#### Hypothesis 3: Float32 → bfloat16 Conversion (PROMISING)
- Initialize in float32 with Gaussian noise: `randvec + Gaussian(0, σ)`
- σ small enough (~1e-5) that conversion to bfloat16 creates discrete clusters
- **Tested in 13.4a (Gatsby Float32 Edition):**
  - 51 dead tokens → 13 unique vectors ✓
  - Complete graph topology (density = 1.0) ✓
  - bfloat16-quantized (bit-for-bit match) ✓
  - Demographics: [19, 14, 3, 3, 3, 2, 1...] (different from Qwen but plausible)
  - Only 1,000 training steps, ~100 it/s on M3 Max
- **Mechanism:** f32 init creates slight variation → bf16 conversion snaps to lattice → training lets live tokens escape → dead tokens frozen at quantization boundaries
- **Remaining questions:** Can we match Qwen's exact demographics? Need larger scale test (2,221 tokens)

#### Hypothesis 4: We're Missing Something Fundamental
- Initialization artifact we haven't considered
- Training dynamic we haven't measured
- Quantization effect we haven't understood

---

## IX. Open Questions & Next Experiments

### A. Immediate Questions
1. Does uniform ball Monument Valley match Qwen better than Gaussian?
2. What's the complete demographics distribution from 13.3c?
3. Can we find a distribution that reproduces [814, 704, 306, 228, ...]?
4. What do the 10 outliers from 13.2b tell us?

### B. Proposed Experiments

**High Priority (validate 13.4a results):**
1. **Control: Pure bf16 initialization (13.4b)**
   - Initialize directly in bfloat16 (no f32 stage)
   - Predict: Should produce 1 black hole (singularity), not 13
   - If true → proves f32→bf16 conversion is essential
2. **Scale test (13.4c)**
   - 2,221 dead tokens (match Qwen's count)
   - Longer training (10k steps)
   - Check if demographics converge toward [814, 704, 306, 228...]
3. **Robustness test**
   - Multiple random seeds
   - σ sweep (1e-6 to 1e-4)
   - Verify 13 unique vectors is stable

**Lower Priority:**
4. **Cross-model deep dive**
   - Why did Qwen 2.5 have 60 unique vectors vs Qwen 3's 13?
   - Extract Qwen's actual initialization code
   - Measure bfloat16 drift rate

### C. The Big Picture
- Float32→bfloat16 hypothesis shows **immediate promise** (13.4a)
- Need control experiments to rule out alternatives
- Monument Valley hypothesis remains **failed** for static sampling
- Key insight: Training dynamics + quantization may be inseparable

---

## X. Conclusions (Preliminary)

### What We've Learned
- Dead tokens are real, measurable, and reveal training history
- bfloat16 quantization creates discrete geometric structure
- Initialization artifacts persist through training
- Simple models (Gaussian Monument Valley) fail to explain observations

### What Remains
- The **formation mechanism** for 13 clusters
- The **demographic distribution** origin
- The **121 singletons** mystery
- Whether this is Qwen-specific or general phenomenon

### Why This Matters
- Understanding initialization → better model training
- Dead tokens reveal pretraining corpus gaps
- Geometric structure reflects training dynamics
- bfloat16 quantization has subtle effects

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
- 13.x: bfloat16 analysis & Monument Valley

---

**Document Status:** Outline Draft 1 (2025-11-09)
**Next Step:** Review with Jeffery, fill gaps, refine structure
