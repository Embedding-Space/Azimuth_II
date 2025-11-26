# Adam Step-by-Step: Understanding the Dynamics

## The Core Equations

Adam maintains two running averages for each parameter:

- **m** (momentum): exponential moving average of gradients
- **v** (variance): exponential moving average of squared gradients

At each step $t$, given gradient $g_t$:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

With typical values $\beta_1 = 0.9$ and $\beta_2 = 0.999$.

## The Bias Problem

At initialization, $m_0 = 0$ and $v_0 = 0$. This creates a problem.

After step 1:
- $m_1 = 0.9 \cdot 0 + 0.1 \cdot g_1 = 0.1 \cdot g_1$
- $v_1 = 0.999 \cdot 0 + 0.001 \cdot g_1^2 = 0.001 \cdot g_1^2$

These are *biased toward zero*—they're much smaller than the "true" running averages would be if we'd been running forever.

## Bias Correction

To fix this, Adam computes bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The correction factors $(1 - \beta^t)$ start small and approach 1 as $t \to \infty$.

| Step $t$ | $1 - \beta_1^t$ | $1 - \beta_2^t$ | Correction to $m$ | Correction to $v$ |
|----------|-----------------|-----------------|-------------------|-------------------|
| 1 | 0.1 | 0.001 | ×10 | ×1000 |
| 2 | 0.19 | 0.002 | ×5.3 | ×500 |
| 3 | 0.271 | 0.003 | ×3.7 | ×333 |
| 10 | 0.651 | 0.010 | ×1.5 | ×100 |
| 100 | 1.0 | 0.095 | ×1.0 | ×10.5 |
| 1000 | 1.0 | 0.632 | ×1.0 | ×1.6 |

Notice: the momentum correction stabilizes quickly (~10 steps), but the variance correction takes ~1000 steps to settle!

## The Update Rule

The final update is:

$$\Delta w = -\text{lr} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Let's expand this fully:

$$\Delta w = -\text{lr} \cdot \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}$$

## Step 1: The Cancellation

At $t = 1$, let's trace through with a single gradient value $g$:

**Raw values:**
- $m_1 = (1 - \beta_1) \cdot g = 0.1 \cdot g$
- $v_1 = (1 - \beta_2) \cdot g^2 = 0.001 \cdot g^2$

**Bias correction factors:**
- $1 - \beta_1^1 = 1 - 0.9 = 0.1$
- $1 - \beta_2^1 = 1 - 0.999 = 0.001$

**Bias-corrected values:**
- $\hat{m}_1 = \frac{0.1 \cdot g}{0.1} = g$
- $\hat{v}_1 = \frac{0.001 \cdot g^2}{0.001} = g^2$

**The update:**
$$\Delta w = -\text{lr} \cdot \frac{g}{\sqrt{g^2} + \epsilon} = -\text{lr} \cdot \frac{g}{|g| + \epsilon}$$

If $|g| \gg \epsilon$:
$$\Delta w \approx -\text{lr} \cdot \text{sign}(g)$$

**At step 1, every parameter moves by exactly $\pm\text{lr}$, regardless of gradient magnitude!**

## Step 2: Breaking the Symmetry

At $t = 2$, with new gradient $g_2$ (and previous gradient $g_1$):

**Raw values:**
- $m_2 = 0.9 \cdot m_1 + 0.1 \cdot g_2 = 0.9 \cdot (0.1 \cdot g_1) + 0.1 \cdot g_2 = 0.09 \cdot g_1 + 0.1 \cdot g_2$
- $v_2 = 0.999 \cdot v_1 + 0.001 \cdot g_2^2 = 0.999 \cdot (0.001 \cdot g_1^2) + 0.001 \cdot g_2^2$
- $v_2 = 0.000999 \cdot g_1^2 + 0.001 \cdot g_2^2$

**Bias correction factors:**
- $1 - \beta_1^2 = 1 - 0.81 = 0.19$
- $1 - \beta_2^2 = 1 - 0.998 = 0.002$ (approximately)

**Bias-corrected values:**
- $\hat{m}_2 = \frac{0.09 \cdot g_1 + 0.1 \cdot g_2}{0.19} \approx 0.47 \cdot g_1 + 0.53 \cdot g_2$
- $\hat{v}_2 = \frac{0.000999 \cdot g_1^2 + 0.001 \cdot g_2^2}{0.002} \approx 0.5 \cdot g_1^2 + 0.5 \cdot g_2^2$

Now the update becomes:
$$\Delta w = -\text{lr} \cdot \frac{0.47 \cdot g_1 + 0.53 \cdot g_2}{\sqrt{0.5 \cdot g_1^2 + 0.5 \cdot g_2^2} + \epsilon}$$

**The perfect cancellation is broken.** The update now depends on:
1. The *history* of gradients (both $g_1$ and $g_2$)
2. The *magnitudes* of those gradients (they don't cancel anymore)

## What This Means for Dead Tokens

**Step 1:** Every token (dead or alive) gets shoved by $\sqrt{D} \times \text{lr} \approx 0.008$ in weight space. The direction is determined by gradient signs, but the magnitude is fixed.

**Steps 2+:** The variance accumulator $v$ starts to "remember" gradient magnitudes. Tokens with consistently small gradients will have small $\hat{v}$, which means *larger* effective learning rates (dividing by a smaller number). Tokens with large gradients will have large $\hat{v}$, damping their updates.

**For dead tokens specifically:**
- They get small gradients (only "push away" signal, no "pull toward")
- So their $v$ stays small
- So their effective learning rate stays high
- So they keep moving relatively fast... until the gradients themselves shrink

The "cooling" we see isn't Adam clamping down—it's the gradient signal itself fading as the model learns and makes fewer confident mistakes.

## Summary

| Step | Behavior |
|------|----------|
| $t = 1$ | Perfect cancellation: $\Delta w = -\text{lr} \cdot \text{sign}(g)$ |
| $t = 2$ to ~10 | Transition: momentum stabilizes, variance still inflated |
| $t = 10$ to ~1000 | Variance correction slowly settling |
| $t > 1000$ | True Adam behavior: adaptive learning rates based on gradient history |

The Big Bang (step 1) is violent because Adam hasn't learned anything yet—it treats all gradients as equally important and takes a fixed-size step in each dimension. The universe "cools" as Adam accumulates gradient statistics and starts making smarter, magnitude-aware updates.
