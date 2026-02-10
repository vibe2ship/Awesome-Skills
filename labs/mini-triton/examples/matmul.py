"""
Matrix Multiplication Example for Mini-Triton.

This is the most important kernel in deep learning - GEMM (General Matrix Multiply).
This example demonstrates:
- 2D grid launching (program_id for both dimensions)
- Block-based tiling for memory efficiency
- Accumulation loop for large matrices
- tl.dot for hardware-accelerated matrix multiply

Understanding this kernel is crucial for:
- Understanding PyTorch/TensorFlow internals
- Optimizing transformer models
- Building Flash Attention

Usage:
    python examples/matmul.py
"""

import numpy as np
import mini_triton as mt
import mini_triton.language as tl


@mt.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (how many elements to skip to get to next row/col)
    stride_am, stride_ak,  # A strides
    stride_bk, stride_bn,  # B strides
    stride_cm, stride_cn,  # C strides
    # Meta-parameters (compile-time constants)
    BLOCK_M: mt.constexpr,
    BLOCK_N: mt.constexpr,
    BLOCK_K: mt.constexpr,
):
    """
    Compute C = A @ B where:
        A is (M, K)
        B is (K, N)
        C is (M, N)

    This kernel uses block-based tiling:
    - The output C is divided into BLOCK_M x BLOCK_N tiles
    - Each program computes one tile
    - For each tile, we iterate over K in BLOCK_K chunks

    Memory access pattern:
        For each (BLOCK_M x BLOCK_N) output tile:
            Load BLOCK_M x BLOCK_K from A
            Load BLOCK_K x BLOCK_N from B
            Accumulate: tile += A_block @ B_block
    """
    # Step 1: Identify which output tile this program computes
    pid_m = tl.program_id(0)  # Row tile index
    pid_n = tl.program_id(1)  # Column tile index

    # Step 2: Compute the row/col indices for this tile
    # These are the starting indices in the output matrix
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Step 3: Initialize the accumulator
    # This will hold the partial result for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Step 4: Iterate over K dimension in chunks
    for k in range(0, K, BLOCK_K):
        # Compute indices for this K-block
        rk = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A block: (BLOCK_M, BLOCK_K)
        # A[rm, rk] - need 2D indexing
        # rm[:, None] creates (BLOCK_M, 1) - broadcast for rows
        # rk[None, :] creates (1, BLOCK_K) - broadcast for cols
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0
        )

        # Load B block: (BLOCK_K, BLOCK_N)
        b = tl.load(
            b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0
        )

        # Compute partial product and accumulate
        # a: (BLOCK_M, BLOCK_K)
        # b: (BLOCK_K, BLOCK_N)
        # result: (BLOCK_M, BLOCK_N)
        acc = tl.dot(a, b, acc)

    # Step 5: Store the result
    # Create output indices
    # c[rm, rn] = acc
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute matrix multiplication C = A @ B.

    Args:
        a: Matrix of shape (M, K)
        b: Matrix of shape (K, N)

    Returns:
        Matrix of shape (M, N)
    """
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} @ {b.shape}"
    assert a.dtype == b.dtype == np.float32

    M, K = a.shape
    K, N = b.shape
    c = np.empty((M, N), dtype=np.float32)

    # Tile sizes (must be powers of 2 for efficiency)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    # Compute grid dimensions
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)

    # Get strides (elements, not bytes)
    # For row-major: stride_row = num_cols, stride_col = 1
    stride_am, stride_ak = a.shape[1], 1
    stride_bk, stride_bn = b.shape[1], 1
    stride_cm, stride_cn = c.shape[1], 1

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c


def main():
    print("Mini-Triton Matrix Multiplication Example")
    print("=" * 50)

    # Test 1: Small square matrices
    print("\n1. Small square matrices (64x64)")
    M, N, K = 64, 64, 64
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    result = matmul(a, b)
    expected = a @ b

    print(f"   Shape: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-4)
    print("   ✓ Test passed!")

    # Test 2: Rectangular matrices
    print("\n2. Rectangular matrices")
    M, N, K = 128, 256, 64
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    result = matmul(a, b)
    expected = a @ b

    print(f"   Shape: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-4)
    print("   ✓ Test passed!")

    # Test 3: Non-multiple of block size
    print("\n3. Non-multiple of block size")
    M, N, K = 100, 150, 75  # Not multiples of 32
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    result = matmul(a, b)
    expected = a @ b

    print(f"   Shape: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-4)
    print("   ✓ Test passed!")

    # Test 4: Larger matrices
    print("\n4. Larger matrices")
    M, N, K = 512, 512, 512
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    result = matmul(a, b)
    expected = a @ b

    print(f"   Shape: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"   Operations: {2 * M * N * K:,} FLOPs")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-3)
    print("   ✓ Test passed!")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("\nNote: Real Triton achieves massive speedups on GPU")
    print("through hardware tensor cores and memory optimization.")


if __name__ == "__main__":
    main()
