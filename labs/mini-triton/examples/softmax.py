"""
Softmax Example for Mini-Triton.

This example demonstrates:
- Numerically stable softmax implementation
- Reduction operations (max, sum)
- More complex kernel logic

The softmax function is fundamental to neural networks,
particularly in attention mechanisms.

Formula (numerically stable):
    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Usage:
    python examples/softmax.py
"""

import numpy as np
import mini_triton as mt
import mini_triton.language as tl


@mt.jit
def softmax_kernel(
    x_ptr,       # Input pointer
    out_ptr,     # Output pointer
    n_cols,      # Number of columns (softmax dimension)
    BLOCK_SIZE: mt.constexpr,  # Block size
):
    """
    Compute softmax over a row of data.

    Each program handles one row.
    This version assumes the row fits in one block.
    """
    # Get row index
    row_idx = tl.program_id(0)

    # Compute column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Compute memory offset for this row
    row_offset = row_idx * n_cols

    # Load the row
    x = tl.load(x_ptr + row_offset + col_offsets, mask=mask, other=float("-inf"))

    # Step 1: Find max for numerical stability
    # We subtract max to prevent overflow in exp()
    x_max = tl.max(x, axis=0)

    # Step 2: Compute exp(x - max)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)

    # Step 3: Compute sum of exponentials
    sum_exp = tl.sum(exp_x, axis=0)

    # Step 4: Normalize
    softmax = exp_x / sum_exp

    # Store result
    tl.store(out_ptr + row_offset + col_offsets, softmax, mask=mask)


@mt.jit
def softmax_kernel_2d(
    x_ptr,       # Input pointer [M, N]
    out_ptr,     # Output pointer [M, N]
    M,           # Number of rows
    N,           # Number of columns
    stride_m,    # Stride for row dimension
    stride_n,    # Stride for column dimension
    BLOCK_M: mt.constexpr,
    BLOCK_N: mt.constexpr,
):
    """
    Softmax over 2D tensor, computing softmax along the last dimension.

    This kernel handles larger tensors by iterating over columns.
    Each program processes one row.
    """
    row_idx = tl.program_id(0)

    # Initialize max and sum
    row_max = tl.full((1,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((1,), dtype=tl.float32)

    # First pass: find row maximum
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        # Load block of data
        x = tl.load(
            x_ptr + row_idx * stride_m + col_offsets * stride_n,
            mask=mask,
            other=float("-inf")
        )

        # Update max
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # Second pass: compute sum of exp
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        x = tl.load(
            x_ptr + row_idx * stride_m + col_offsets * stride_n,
            mask=mask,
            other=float("-inf")
        )

        exp_x = tl.exp(x - row_max)
        row_sum = row_sum + tl.sum(tl.where(mask, exp_x, 0.0), axis=0)

    # Third pass: write normalized values
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        x = tl.load(
            x_ptr + row_idx * stride_m + col_offsets * stride_n,
            mask=mask,
            other=float("-inf")
        )

        softmax = tl.exp(x - row_max) / row_sum
        tl.store(
            out_ptr + row_idx * stride_m + col_offsets * stride_n,
            softmax,
            mask=mask
        )


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax along given axis.

    Args:
        x: Input array
        axis: Axis to compute softmax along (default: -1)

    Returns:
        Softmax output
    """
    # For simplicity, we handle 2D case with axis=-1
    assert x.ndim == 2, "Only 2D tensors supported"
    assert axis == -1 or axis == 1, "Only last axis supported"
    assert x.dtype == np.float32, "Only float32 supported"

    m, n = x.shape
    out = np.empty_like(x)

    BLOCK_SIZE = 1024  # Assumes n <= 1024 for simple version

    if n <= BLOCK_SIZE:
        # Simple version: one row per program
        grid = (m,)
        softmax_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Multi-pass version for larger rows
        BLOCK_N = 256
        grid = (m,)
        softmax_kernel_2d[grid](
            x, out, m, n, n, 1,
            BLOCK_M=1, BLOCK_N=BLOCK_N
        )

    return out


def numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reference NumPy implementation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def main():
    print("Mini-Triton Softmax Example")
    print("=" * 50)

    # Test 1: Single row
    print("\n1. Single row softmax")
    x = np.random.randn(1, 128).astype(np.float32)

    result = softmax(x)
    expected = numpy_softmax(x)

    print(f"   Shape: {x.shape}")
    print(f"   Sum of softmax: {result.sum():.6f} (should be 1.0)")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-5)
    print("   ✓ Test passed!")

    # Test 2: Multiple rows
    print("\n2. Batch softmax")
    x = np.random.randn(32, 256).astype(np.float32)

    result = softmax(x)
    expected = numpy_softmax(x)

    print(f"   Shape: {x.shape}")
    print(f"   All rows sum to 1: {np.allclose(result.sum(axis=1), 1.0)}")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-5)
    print("   ✓ Test passed!")

    # Test 3: Large input (stress test numerical stability)
    print("\n3. Large values (numerical stability test)")
    x = np.random.randn(16, 512).astype(np.float32) * 100  # Large values

    result = softmax(x)
    expected = numpy_softmax(x)

    print(f"   Shape: {x.shape}")
    print(f"   Input range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"   No NaN/Inf in result: {not (np.isnan(result).any() or np.isinf(result).any())}")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-4)
    print("   ✓ Test passed!")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    main()
