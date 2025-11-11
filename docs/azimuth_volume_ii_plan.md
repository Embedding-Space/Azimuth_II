# Project Azimuth Volume II: Observational Tokenology

**A list of ideas for exploring the 151,936-token galaxy**

---

## Core Philosophy

- Study the unembedding matrix as a fixed catalog of points in 2560D space
- Work primarily in γ-space (the natural training representation) 
- Build causal metric transforms as toggles to reveal hidden structure
- Do cartography: histograms, scatter plots, boring exploratory shit before imposing models
- Think astronomy, not differential geometry - this is a discrete point cloud

---

## Foundation Work

### 1. Center the coordinates
Define γ' = γ - μ where μ is the token cloud centroid. Put our telescope at the galactic center.

### 2. Build coordinate transform infrastructure (`azimuth/geometry.py`)
- `.to_causal_space()` - transform γ' → causal coordinates (logometers, stretched/isotropic)
- `.to_gamma_space()` - transform back
- Make it easy to toggle between views

---

## Visualization Techniques
*Try all of these, see what reveals structure*

### 3. Sky maps from centroid
- Use CDF flattening to spread the equatorial band
- Try different pole choices (different eigendirections)
- Look for: voids, spikes, density variations, asymmetries
- Works in both γ' and causal space

### 4. Tomographic slices (MRI-style)
- Pick 4 orthogonal directions (A, B, C, D)
- Slice along A axis into ~10 sections
- For each slice: project onto BC plane, color by D
- Reveals depth structure without crushing everything flat

### 5. 4D cross-sections (the ones that already work)
- Continue using these - they're good for exploring specific subspaces
- Try in both γ' and causal coordinates

---

## Exploratory Questions
*In rough order*

### 6. What's the radial distribution?
- Histogram of ||γ'|| in fine bins
- Is there really a shell? How tight? Any structure within it?

### 7. That spike in the sky map
- What tokens are in it?
- Why are they so tightly clustered?
- Is it linguistic (all Thai?) or structural (punctuation?)?

### 8. That void at -180°, -0.15°
- Is it real or a density dip?
- What tokens are nearest to it?
- Does it appear in other pole orientations?

### 9. Thai token overlap mystery
- Find them in space
- Measure their mutual distances
- Are there other linguistic overlaps like this?

### 10. Cluster discovery (density-based, not imposed)
- Where are the naturally occurring clumps?
- Do they correspond to linguistics? Syntax? Semantics?
- Use multiple visualization methods to confirm

### 11. Toggle experiments
- Find something clustered in γ'-space
- Transform to causal space - does it spread out?
- Find something isotropic in causal space
- Transform to γ'-space - does it cluster?
- What does this tell us about the geometry?

---

## NOT in Volume II
*Save for later or never*

- Steering experiments
- Perplexity measurements  
- Manifold boundaries
- Anything requiring model inference

---

## Key Insights from Planning Session

**"LLMs turn context into vectors and vectors into tokens."** - A damn good summary.

**The asymmetry of spherical coordinates:** Latitude and longitude aren't symmetric. Longitude is cyclic (360° rotation around pole), latitude is not (0° to 180° from pole to pole). This is why everything crushes to the equator in high-D - we're measuring *from* an axis (latitude) rather than *around* it (longitude).

**Why CDF flattening works:** Takes the concentrated equatorial band and stretches it to fill the full sphere by mapping through the cumulative distribution function. Preserves rank ordering but spreads crowded regions.

**Why tomographic slices work:** Instead of crushing all 2560 dimensions onto 2D (total orthographic projection), we take thin slices perpendicular to one axis and only project *those* tokens. Like CT scans or MRI - reveals structure without overlap confusion.

**The causal metric role:** Not for imposing structure, but as an alternative lens. Transform coordinates so clustered→isotropic or vice versa, revealing complementary geometric views of the same data.

---

*Generated: Sunday, November 2, 2025*
*From conversation with Jeffery over Johnnie Walker Black*
