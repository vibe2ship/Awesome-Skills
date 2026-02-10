"""
End-to-end tests for Mini-Triton.

These tests verify the complete pipeline from @jit decorated functions
to actual execution with NumPy arrays.
"""

import pytest
import numpy as np
from mini_triton import jit, constexpr
import mini_triton.language as tl


class TestVectorAdd:
    """Tests for vector addition kernel."""

    def test_vector_add_basic(self):
        """Test basic vector addition."""

        @jit
        def vector_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        # Create test data
        n = 1024
        x = np.random.randn(n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        out = np.zeros_like(x)

        # Compute expected result
        expected = x + y

        # Launch kernel
        BLOCK_SIZE = 128
        grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        vector_add[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)

        # Verify
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_vector_add_non_multiple(self):
        """Test vector addition with non-multiple of block size."""

        @jit
        def vector_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        # 1000 is not a multiple of 128
        n = 1000
        x = np.random.randn(n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        out = np.zeros_like(x)
        expected = x + y

        BLOCK_SIZE = 128
        grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        vector_add[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)

        np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestElementwiseOps:
    """Tests for element-wise operations."""

    def test_relu(self):
        """Test ReLU kernel."""

        @jit
        def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.maximum(x, 0.0)
            tl.store(out_ptr + offsets, out, mask=mask)

        n = 1024
        x = np.random.randn(n).astype(np.float32)
        out = np.zeros_like(x)
        expected = np.maximum(x, 0)

        BLOCK_SIZE = 128
        grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)

        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_exp(self):
        """Test exp kernel."""

        @jit
        def exp_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.exp(x)
            tl.store(out_ptr + offsets, out, mask=mask)

        n = 1024
        x = np.random.randn(n).astype(np.float32)
        out = np.zeros_like(x)
        expected = np.exp(x)

        BLOCK_SIZE = 128
        grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        exp_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)

        np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestReductions:
    """Tests for reduction kernels."""

    def test_sum_1d(self):
        """Test 1D sum reduction."""

        @jit
        def sum_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: constexpr):
            pid = tl.program_id(0)
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            total = tl.sum(x, axis=0)
            tl.store(out_ptr, total)

        n = 128
        x = np.random.randn(n).astype(np.float32)
        out = np.zeros(1, dtype=np.float32)
        expected = np.sum(x)

        # Single block for this simple case
        sum_kernel[1](x, out, n, BLOCK_SIZE=128)

        np.testing.assert_allclose(out[0], expected, rtol=1e-4)


class TestSoftmax:
    """Tests for softmax kernel."""

    def test_softmax_1d(self):
        """Test 1D softmax."""

        @jit
        def softmax_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: constexpr):
            pid = tl.program_id(0)
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            # Load input
            x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))

            # Numerically stable softmax
            x_max = tl.max(x, axis=0)
            x_shifted = x - x_max
            exp_x = tl.exp(x_shifted)
            sum_exp = tl.sum(exp_x, axis=0)
            softmax = exp_x / sum_exp

            tl.store(out_ptr + offsets, softmax, mask=mask)

        n = 128
        x = np.random.randn(n).astype(np.float32)
        out = np.zeros_like(x)

        # NumPy reference
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        expected = exp_x / np.sum(exp_x)

        softmax_kernel[1](x, out, n, BLOCK_SIZE=128)

        np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestMatMul:
    """Tests for matrix multiplication."""

    def test_matmul_simple(self):
        """Test simple matrix multiplication."""

        @jit
        def matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            BLOCK_M: constexpr, BLOCK_N: constexpr, BLOCK_K: constexpr
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            # Initialize accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Iterate over K dimension
            for k in range(0, K, BLOCK_K):
                # Load A block
                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_k = k + tl.arange(0, BLOCK_K)
                a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])

                # Load B block
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])

                # Accumulate
                acc = tl.dot(a, b, acc)

            # Store result
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

        M, N, K = 64, 64, 64
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        expected = a @ b

        BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
        grid = (M // BLOCK_M, N // BLOCK_N)
        matmul_kernel[grid](
            a, b, c, M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        np.testing.assert_allclose(c, expected, rtol=1e-4)


class TestJITCaching:
    """Tests for JIT compilation caching."""

    def test_same_constexpr_reuses_cache(self):
        """Test that same constexpr values reuse compiled kernel."""

        @jit
        def simple_kernel(x_ptr, out_ptr, BLOCK: constexpr):
            offsets = tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offsets)
            tl.store(out_ptr + offsets, x + 1.0)

        x = np.zeros(128, dtype=np.float32)
        out1 = np.zeros_like(x)
        out2 = np.zeros_like(x)

        # First call - compiles
        simple_kernel[1](x, out1, BLOCK=128)

        # Second call - should reuse cached kernel
        simple_kernel[1](x, out2, BLOCK=128)

        np.testing.assert_array_equal(out1, out2)

    def test_different_constexpr_creates_new_kernel(self):
        """Test that different constexpr values create new kernels."""

        @jit
        def sized_kernel(x_ptr, out_ptr, BLOCK: constexpr):
            offsets = tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offsets)
            tl.store(out_ptr + offsets, x)

        # Different block sizes should work
        x64 = np.ones(64, dtype=np.float32)
        x128 = np.ones(128, dtype=np.float32)
        out64 = np.zeros_like(x64)
        out128 = np.zeros_like(x128)

        sized_kernel[1](x64, out64, BLOCK=64)
        sized_kernel[1](x128, out128, BLOCK=128)

        np.testing.assert_array_equal(out64, x64)
        np.testing.assert_array_equal(out128, x128)


class TestErrorHandling:
    """Tests for error handling."""

    def test_no_grid_error(self):
        """Test that calling kernel without grid raises error."""

        @jit
        def kernel(x_ptr):
            pass

        with pytest.raises((TypeError, RuntimeError)):
            kernel(np.zeros(10))  # Missing grid
