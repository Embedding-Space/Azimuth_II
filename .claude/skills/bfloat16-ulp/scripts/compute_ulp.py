"""
Compute bfloat16 ULP (Unit in Last Place) for tensors.

The ULP is the gap between adjacent representable bfloat16 numbers.
For a value with stored exponent E: ULP = 2^(E - 134)
Where 134 = 127 (bias) + 7 (mantissa bits).
"""

import torch


def compute_bfloat16_ulp(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the ULP for each element of a bfloat16 tensor.
    
    Args:
        x: Input tensor. Will be converted to bfloat16 if not already.
        
    Returns:
        Float32 tensor of ULP values, same shape as input.
        
    Example:
        >>> W = torch.tensor([0.001, 0.01, 0.1], dtype=torch.bfloat16)
        >>> ulp = compute_bfloat16_ulp(W)
        >>> print(ulp)  # Will show ULP at each scale
    """
    # Ensure bfloat16
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    
    # ULP depends only on magnitude, not sign
    x_abs = torch.abs(x)
    
    # View the raw bits as uint16
    # bfloat16 layout: [1 sign bit][8 exponent bits][7 mantissa bits]
    bits = x_abs.view(torch.uint16)
    
    # Extract exponent bits by right-shifting away the 7 mantissa bits
    exponent = (bits >> 7).to(torch.int32)
    
    # Handle zeros and subnormals (exponent == 0)
    # For subnormals, ULP = 2^(-126 - 7) = 2^(-133)
    # Using torch.where to handle this case
    
    # For normal numbers: ULP = 2^(E - 134)
    # For subnormals (E=0): ULP = 2^(-133) (the smallest positive subnormal)
    ulp_exponent = torch.where(
        exponent == 0,
        torch.tensor(-133, dtype=torch.int32, device=x.device),
        exponent - 134
    )
    
    # Compute ULP = 2^ulp_exponent in float32
    # Using torch.pow with base 2.0
    ulp = torch.pow(2.0, ulp_exponent.to(torch.float32))
    
    return ulp


def displacement_in_ulp(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute the displacement between two bfloat16 tensors in ULP units.
    
    Uses the ULP of x1 as the reference scale.
    
    Args:
        x1: First tensor (reference for ULP scale)
        x2: Second tensor (same shape as x1)
        
    Returns:
        Float32 tensor of displacements measured in ULP units.
    """
    # Ensure bfloat16
    if x1.dtype != torch.bfloat16:
        x1 = x1.to(torch.bfloat16)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    
    # Compute displacement in float32 to avoid precision loss
    delta = x2.to(torch.float32) - x1.to(torch.float32)
    
    # Get ULP at x1's scale
    ulp = compute_bfloat16_ulp(x1)
    
    # Return displacement in ULP units
    return delta / ulp


if __name__ == "__main__":
    # Quick sanity check
    print("bfloat16 ULP Sanity Check")
    print("=" * 40)
    
    test_values = torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0], dtype=torch.bfloat16)
    ulps = compute_bfloat16_ulp(test_values)
    
    print(f"{'Value':>10} {'ULP':>15} {'ULP/Value':>15}")
    print("-" * 40)
    for v, u in zip(test_values.tolist(), ulps.tolist()):
        ratio = u / abs(v) if v != 0 else float('inf')
        print(f"{v:>10.4f} {u:>15.2e} {ratio:>15.4f}")
    
    print()
    print(f"Expected ULP/Value ratio for normal numbers: ~{2**-7:.4f}")
    print()
    
    # Test zero handling
    zero = torch.tensor([0.0], dtype=torch.bfloat16)
    zero_ulp = compute_bfloat16_ulp(zero)
    print(f"ULP at zero: {zero_ulp.item():.2e}")
    print(f"Expected (2^-133): {2**-133:.2e}")
