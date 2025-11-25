---
name: bfloat16-ulp
description: Compute the Unit in Last Place (ULP) for bfloat16 tensors. Use this skill when working with bfloat16 precision analysis, quantization effects, distinguishability thresholds, or lattice coordinate systems. Essential for Azimuth research involving dead token dynamics, fimbulwinter analysis, or any work requiring precise understanding of bfloat16's discrete number line.
---

# bfloat16 ULP Calculation

## Why This Matters

bfloat16 has only 7 mantissa bits, making its ULP (the gap between adjacent representable numbers) relatively large. When analyzing whether two values are "the same" or tracking how values move through training, you need to know the actual quantum of the number format.

**Common pitfall**: `torch.nextafter(x, x+1) - x` does NOT give you ULP for bfloat16. It gives the next representable float32 value, which is vastly smaller than the bfloat16 ULP.

## The Formula

For a bfloat16 value with stored exponent bits E (the raw 8-bit unsigned integer from the representation):

```
ULP = 2^(E - 134)
```

Where 134 = 127 (exponent bias) + 7 (mantissa bits).

This works because:
- The exponent bias of 127 converts stored exponent to actual exponent
- Subtracting 7 more accounts for the mantissa precision
- Result: ULP = 2^(actual_exponent - mantissa_bits)

## Implementation

Use the script at `scripts/compute_ulp.py`. It handles:
- Extracting exponent bits via view-as-uint16 and bit shifting
- Computing ULP in float32 (to avoid precision loss)
- Proper handling of zeros and subnormals
- Batch operations on full tensors

### Quick Usage

```python
import torch
import sys
sys.path.insert(0, '/path/to/bfloat16-ulp/scripts')
from compute_ulp import compute_bfloat16_ulp

# For a tensor of bfloat16 values
W = torch.randn(1000, 2560, dtype=torch.bfloat16)
ulp = compute_bfloat16_ulp(W)  # Returns float32 tensor of ULP values
```

### What the Script Does

1. Takes absolute value (ULP depends only on magnitude, not sign)
2. Views the bfloat16 bits as uint16
3. Extracts the 8 exponent bits via right-shift by 7
4. Computes 2^(E - 134) in float32
5. Returns ULP values as float32 tensor (same shape as input)

## Sanity Checks

After computing ULP, verify your results:

1. **ULP scales with magnitude**: `ulp / abs(value)` should be roughly constant (~2^-7 ≈ 0.0078) for normal numbers
2. **Zero handling**: ULP at zero should equal the smallest positive subnormal
3. **Range check**: For typical embedding values (~0.001 to ~0.01), ULP should be ~10^-5 to ~10^-4

## Common Applications

- **Distinguishability threshold**: Two bfloat16 vectors are indistinguishable if they differ by less than 1 ULP in all components
- **Lattice coordinates**: Express displacements in units of ULP to see motion on the discrete bfloat16 grid
- **Fimbulwinter detection**: When gradient updates fall below ULP, the value is frozen
