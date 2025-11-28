# Dead Token Gradients: Why "Dead" Tokens Aren't Frozen

> **Last updated:** 2025-11-27

## The Misconception

It's tempting to think: "Dead tokens never appear in training data, so they receive zero gradients, so they stay frozen at initialization."

**This is wrong.**

## The Two Gradient Pathways

In a transformer with tied embeddings (E = W), the embedding/unembedding matrix receives gradients from two sources:

### 1. Embedding Side (Input)
When a token appears in the input sequence, its embedding row gets a gradient:
```
∂L/∂E[token] = ∂L/∂h · (some backprop through the network)
```
**Dead tokens:** Never appear in input → zero gradient from this path.

### 2. Unembedding Side (Output)
At every sequence position, the model computes logits for ALL vocab tokens:
```
logits = h @ W.T  # [batch, seq, vocab]
probs = softmax(logits)
loss = cross_entropy(probs, targets)
```

The cross-entropy gradient w.r.t. the unembedding weight for token *i* is:
```
∂L/∂W[i] = (p_i - y_i) · h
```
where:
- `p_i` = predicted probability of token *i*
- `y_i` = 1 if token *i* is the correct answer, 0 otherwise
- `h` = hidden state at that position

**Dead tokens:** Never the correct answer (y_i = 0), but p_i > 0 whenever the model assigns them any probability. So:
```
∂L/∂W[dead] = p_dead · h  ≠ 0
```

## The Key Insight

**Every token in the vocabulary receives a gradient at every sequence position, every batch.**

For dead tokens, this gradient is always positive (pushing them down, making them less likely). The magnitude is proportional to how much probability the model accidentally assigns to them.

## Evolution Over Training

### Early Training
- Model is uncertain → wide softmax distribution
- Dead tokens get substantial probability mass (≈ 1/vocab ≈ 10⁻⁵)
- Gradient magnitude: ~6e-2 per token per position
- Dead tokens move significantly

### Late Training
- Model is confident → sharp softmax distribution
- Probability concentrates on correct tokens
- Dead token probabilities → exponentially small
- Gradient magnitude: ~10⁻⁶ or smaller
- Updates fall below bfloat16 quantization threshold
- **Dead tokens freeze**

## The Softmax Sharpening Mechanism

This explains the "Fimbulwinter" (dead token freezing):

1. Early: Wide distribution → thermal jitter (tokens move)
2. Middle: Sharpening distribution → cooling (movement slows)
3. Late: Sharp distribution → frozen (updates sub-quantization)

The phase transition from "gas" (moving) to "solid" (frozen) is governed by softmax temperature, which effectively decreases as the model learns.

## Experimental Validation

**Notebook 14.1a** (Lil Gatsby Instrumented) confirmed:
- Step 0: Dead token gradient mean = 6.12e-2
- Step 90: Dead token gradient mean = 5.90e-3
- Gradients decrease but are clearly non-zero throughout early training

## Implications for the Frozen Smoke

The frozen smoke in Qwen 3 4B represents tokens that:
1. Started at/near some initialization point
2. Received real gradients during training (not frozen from t=0)
3. Eventually froze when softmax sharpened enough
4. Their final positions reflect accumulated drift before freezing

The structure we see is NOT a pure initialization fossil—it's been shaped by training dynamics before the Fimbulwinter set in.

## Open Questions

- How does drift direction correlate with token semantics?
- Why do dead tokens move coherently (as a cloud) rather than dispersing randomly?
- What determines the final position of the frozen smoke centroid?

---

*Discovered: November 10, 2025 (Notebook 14.1a)*
*Mechanism explained: November 19, 2025 (Softmax sharpening insight)*
