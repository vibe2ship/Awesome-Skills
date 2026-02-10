"""
Vector Addition Example for Mini-Triton.

This is the "Hello World" of GPU programming. It demonstrates:
- Basic @jit kernel structure
- tl.program_id for block identification
- tl.arange for creating index blocks
- Masked memory operations for edge handling
- Grid-based kernel launching

Usage:
    python examples/vector_add.py
"""

import numpy as np
import mini_triton as mt
import mini_triton.language as tl


@mt.jit
def vector_add_kernel(
    x_ptr,      # Pointer to first input array
    y_ptr,      # Pointer to second input array
    out_ptr,    # Pointer to output array
    n_elements, # Number of elements to process
    BLOCK_SIZE: mt.constexpr,  # Block size (compile-time constant)
):
    """
    Element-wise vector addition: out = x + y

    Each program (block) processes BLOCK_SIZE elements.
    The grid is set up so that all elements are covered.
    """
    # Step 1: Get this program's ID
    # In a grid of programs, each has a unique ID
    pid = tl.program_id(0)

    # Step 2: Compute the offsets for this program
    # If pid=0 and BLOCK_SIZE=128, offsets = [0, 1, 2, ..., 127]
    # If pid=1 and BLOCK_SIZE=128, offsets = [128, 129, ..., 255]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Step 3: Create a mask for bounds checking
    # The last block might have offsets >= n_elements
    # We use a mask to avoid out-of-bounds access
    mask = offsets < n_elements

    # Step 4: Load data from memory
    # Masked load: only loads where mask is True
    # 'other' specifies value for masked elements (won't be used)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Step 5: Perform the computation
    result = x + y

    # Step 6: Store result to memory
    # Masked store: only stores where mask is True
    tl.store(out_ptr + offsets, result, mask=mask)


def vector_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Python wrapper for vector addition.

    Args:
        x: First input array
        y: Second input array

    Returns:
        Element-wise sum x + y
    """
    assert x.shape == y.shape, "Arrays must have same shape"
    assert x.dtype == y.dtype == np.float32, "Arrays must be float32"

    n_elements = x.size
    out = np.empty_like(x)

    # Choose block size (power of 2 for efficiency)
    BLOCK_SIZE = 128

    # Compute grid size (number of blocks)
    # We need enough blocks to cover all elements
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    # kernel[grid](...) syntax specifies grid dimensions
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


def main():
    print("Mini-Triton Vector Addition Example")
    print("=" * 50)

    # Test 1: Basic addition
    print("\n1. Basic vector addition")
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    result = vector_add(x, y)
    expected = x + y

    print(f"   Input size: {n}")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-5)
    print("   ✓ Test passed!")

    # Test 2: Non-multiple of block size
    print("\n2. Non-multiple of block size")
    n = 1000  # Not a multiple of 128
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    result = vector_add(x, y)
    expected = x + y

    print(f"   Input size: {n} (not multiple of 128)")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-5)
    print("   ✓ Test passed!")

    # Test 3: Large array
    print("\n3. Large array")
    n = 1_000_000
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    result = vector_add(x, y)
    expected = x + y

    print(f"   Input size: {n:,}")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-5)
    print("   ✓ Test passed!")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    main()
