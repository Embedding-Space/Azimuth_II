# From Probabilities to Weight Updates: The Complete Picture

A walkthrough of the full chain from "model spits out probabilities" to "W gets nudged."

## The Forward Pass: Probabilities

When the model predicts the next token, the final layer computes **logits** (raw scores) for each token in the vocabulary:

$$\text{logits} = W \cdot h$$

where $h$ is the hidden state (2560D for Qwen, 64D for Flannel) and $W$ is our unembedding matrix (vocab_size × hidden_dim). Each row of $W$ is one token's vector.

Then softmax converts logits to probabilities:

$$p_i = \frac{e^{\text{logit}_i}}{\sum_j e^{\text{logit}_j}}$$

So token $i$'s probability is its exponential score divided by the sum of everyone's exponential scores.

## The Backward Pass: From Loss to Gradients

**Cross-entropy loss:** For the true next token (let's call it token $t$), the loss is:

$$L = -\log(p_t)$$

Minimizing this loss means pushing $p_t$ closer to 1.

**Gradient w.r.t. logits:** The beautiful thing about cross-entropy + softmax is the gradient is stupidly simple:

$$\frac{\partial L}{\partial \text{logit}_i} = \begin{cases}
p_i - 1 & \text{if } i = t \text{ (correct token)} \\
p_i & \text{if } i \neq t \text{ (wrong token)}
\end{cases}$$

In other words: every *wrong* token gets a gradient equal to its predicted probability (pushing it down), and the *correct* token gets gradient $(p_t - 1)$, which is negative (pushing it up).

**Gradient w.r.t. W:** By chain rule:

$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial \text{logit}_i} \cdot h$$

So each token's **gradient vector** is:
- Its probability error (positive for wrong tokens, negative for correct)
- Times the hidden state vector $h$

This means:
- **Dead tokens** that never appear as the correct answer only ever get *positive* gradients (= $p_{\text{dead}} \cdot h$)
- Their gradients point in the direction of $h$
- But $p_{\text{dead}}$ is tiny (exponentially small) because they're far from the action

## Adam: The Optimizer

Okay, so now we have gradients $g_t$ at each timestep. Basic gradient descent would do:

$$W_{t+1} = W_t - \eta \cdot g_t$$

where $\eta$ is the learning rate. But Adam is fancier—it adapts the learning rate per parameter based on gradient history.

### The Two Accumulators

Adam maintains two exponential moving averages:

**1. Momentum ($m_t$):** First moment (mean) of gradients

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

This is a *velocity* term—it accumulates gradient direction over time. Standard $\beta_1 = 0.9$ means "90% of old momentum + 10% of new gradient."

**2. Variance ($v_t$):** Second moment (uncentered variance) of gradients

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

This tracks how *volatile* gradients are (squared). Standard $\beta_2 = 0.999$ means "99.9% of old variance + 0.1% of new squared gradient."

### Bias Correction

Early in training, $m$ and $v$ are biased toward zero (they start at 0). Adam corrects this:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The denominators grow toward 1 as $t$ increases, so this correction matters most early on.

### The Update Rule

Finally, the actual parameter update:

$$W_{t+1} = W_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero.

**What this means:**
- Update direction comes from $\hat{m}_t$ (momentum = smoothed gradient direction)
- Update magnitude is *scaled down* by $\sqrt{\hat{v}_t}$ (parameters with noisy/large gradients get smaller updates)
- This is **adaptive learning rate**: each parameter effectively has its own learning rate based on its gradient history

### AdamW: Weight Decay

We use **AdamW**, which adds weight decay *after* the Adam update:

$$W_{t+1} = W_{t+1} - \lambda \cdot W_t$$

where $\lambda = 0.1$ in our case. This pulls all parameters gently toward zero, independently of gradients. It's a regularization term that prevents weights from growing unbounded.

## Why This Matters for Dead Tokens

For a dead token that *never* appears in the training data:

1. **Gradient is always positive**: $g = p_{\text{dead}} \cdot h$ (always pointing along $h$)
2. **But tiny**: $p_{\text{dead}} \approx 10^{-6}$ or smaller (exponentially suppressed)
3. **Momentum accumulates**: Even tiny gradients build up in $m_t$ over time
4. **Variance stays small**: Gradients are consistent (always positive, similar magnitude), so $v_t$ stays small → Adam doesn't scale down the update much
5. **Weight decay pulls toward origin**: Independent of gradients

So dead tokens get:
- A tiny **push** in the direction of hidden states (via gradients → momentum)
- A tiny **pull** toward the origin (via weight decay)
- Potentially coherent bulk motion if gradients are correlated across timesteps

The question is: **which effect dominates?** And does the *direction* of $h$ (hidden states) have some coherent structure that causes the directional drift we observed?

---

## Summary

**The Pipeline:**
1. Model outputs probabilities via softmax
2. Gradient for each token = (probability - is_correct) × hidden_state
3. Adam uses momentum (smooth gradient direction) and variance (gradient volatility) to adapt per-parameter learning rates
4. Dead tokens get tiny but consistent gradients + weight decay pulling toward origin

**What Recording $m_t$ and $v_t$ Will Show Us:**
- Whether dead tokens have accumulated momentum (directional velocity in embedding space)
- Whether their variance is truly small (confirming consistent gradient direction)
- The balance between gradient-driven drift and weight decay
- Whether the coherent bulk motion we observed is momentum-driven or gradient-driven

---

*Written 2025-11-21*
