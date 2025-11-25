# What We're Recording and Why

## The Four Big Tensors

For each token's embedding vector **w** ∈ ℝ⁶⁴ at each training step, we record four quantities:

---

## 1. **W** (the embeddings themselves)

**What it is:** The actual position of the token in 64D space.

**Shape:** (steps, vocab, 64)

**Math:** Just the embedding matrix at timestep t:
```
W[t] = current embedding vectors
```

**Why we need it:** This is the fundamental data. Everything else is about understanding *why* W moves the way it does.

---

## 2. **Gradients** (∂L/∂w)

**What it is:** The direction and magnitude that the loss function "wants" each token to move.

**Shape:** (steps, vocab, 64)

**Math:** The partial derivative of loss with respect to each embedding:
```
grad[t] = ∂L/∂W[t]
```

For dead tokens specifically, this comes from the unembedding path:
```
∂L/∂W = ∂L/∂logits · ∂logits/∂W
       = (predicted_probs - true_labels) · h
```

Where h is the hidden state and the gradient magnitude is proportional to the predicted probability p_i.

**Why we might need it:**
- Shows the "force" acting on each token
- Reveals whether dead tokens get tiny gradients (our hypothesis: yes, proportional to softmax probabilities)
- Can verify our freezing mechanism: if gradients stay non-zero but *updates* fall below bfloat16 quantization, that confirms the mechanism

**Why we might NOT need it:**
- We can infer a lot from W[t] alone by computing ΔW = W[t+1] - W[t]
- Gradients are pre-optimizer: what Adam *does* with them is what matters
- If we're just tracking "does the token move?", W[t] is sufficient

---

## 3. **Momentum** (Adam's exp_avg)

**What it is:** Adam's running average of past gradients. Think of it as the token's "velocity" based on recent gradient history.

**Shape:** (steps, vocab, 64)

**Math:** Exponentially weighted moving average with β₁ = 0.9:
```
m[t] = β₁ · m[t-1] + (1 - β₁) · grad[t]
```

Each component of m is roughly "the average gradient direction over the last ~10 steps."

**Why we might need it:**
- Reveals whether tokens have "momentum" that persists even when instantaneous gradients are small
- Could explain continued motion after gradients shrink
- Shows whether dead tokens ever build up directional bias

**Why we might NOT need it:**
- This is Adam's *internal state*, not directly observable in token positions
- For dead tokens (our main interest), gradients are always tiny → momentum stays near zero
- We can't easily "see" momentum in W[t] trajectories

---

## 4. **Variance** (Adam's exp_avg_sq)

**What it is:** Adam's running average of *squared* gradients. Used to normalize step sizes per dimension.

**Shape:** (steps, vocab, 64)

**Math:** Exponentially weighted moving average with β₂ = 0.999:
```
v[t] = β₂ · v[t-1] + (1 - β₂) · grad[t]²
```

Adam's actual update is:
```
W[t+1] = W[t] - lr · m[t] / (√v[t] + ε)
```

So v controls the *scaling* of updates: dimensions with historically large gradients get smaller steps.

**Why we might need it:**
- Shows whether certain dimensions are "stuck" due to high variance
- Reveals if dead tokens have different variance patterns than live tokens

**Why we might NOT need it:**
- Even more indirect than momentum
- For dead tokens with tiny gradients, variance also stays near zero
- The division by √v amplifies small momentum when variance is small, but we can compute effective step size from ΔW directly

---

## The Core Question

**Do we actually need grads, momentum, and variance to understand dead token dynamics?**

**Argument for YES:**
- They reveal the *mechanism*: we can see gradients shrink, momentum fail to accumulate, variance stay flat
- Direct evidence for our hypothesis about softmax sharpening → gradient compression
- Can distinguish "token doesn't move because no gradient" vs "gradient exists but update rounds to zero"

**Argument for NO:**
- W[t] contains the ground truth: token positions over time
- ΔW[t] = W[t+1] - W[t] directly shows movement magnitude
- We can compute:
  - Velocity: ΔW/Δt
  - Acceleration: Δ²W/Δt²
  - Whether tokens freeze: ||ΔW|| < threshold
- Grads/momentum/variance are *intermediate* quantities that produce ΔW, but we already have the result

**Analogy:** It's like studying planetary orbits:
- **With grads/momentum/variance:** We record gravitational forces, velocity, angular momentum at each timestep
- **With W[t] only:** We record positions, then compute velocities and accelerations from the trajectory

Both work. The first is more direct for understanding forces. The second is simpler and sufficient if you trust your ability to infer forces from motion.

---

## Recommendation

For **dead token dynamics specifically**, I think **W[t] alone is probably sufficient** because:

1. **Dead tokens don't train** → their gradients are always tiny and predictable
2. **We care about phase transitions** (gas → solid) which are visible in ||ΔW||
3. **The freezing mechanism** is about updates falling below bfloat16 precision, which we can detect from ΔW becoming zero
4. **Memory savings:** Recording W only = 12.8 GB instead of 51.2 GB

**What we'd lose:**
- Direct measurement of gradient magnitudes (but we can estimate from softmax probabilities)
- Smoking gun evidence if momentum somehow persists (unlikely for dead tokens)
- Variance normalization effects (minor for dead tokens)

**What we'd keep:**
- All position data: W[t] for all tokens, all timesteps, all runs
- Ability to compute velocities, accelerations, freezing signatures
- Ability to analyze epochs, phase transitions, spatial structure

**If we want to be conservative:** Record W + grads (25.6 GB) to directly verify gradient compression hypothesis. Skip momentum and variance.

What do you think?
