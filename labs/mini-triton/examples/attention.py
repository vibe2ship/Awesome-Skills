"""
Simplified Attention Example for Mini-Triton.

This example demonstrates the building blocks of attention mechanisms,
which are the core of transformer models (GPT, BERT, etc.).

This is a simplified version - real Flash Attention is more complex
with online softmax computation and memory-efficient tiling.

Standard Attention:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

Where:
    Q: Query matrix (seq_len, head_dim)
    K: Key matrix (seq_len, head_dim)
    V: Value matrix (seq_len, head_dim)

Usage:
    python examples/attention.py
"""

import numpy as np
import mini_triton as mt
import mini_triton.language as tl


@mt.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len, head_dim,
    stride_q_seq, stride_q_dim,
    stride_k_seq, stride_k_dim,
    stride_v_seq, stride_v_dim,
    stride_o_seq, stride_o_dim,
    scale,
    BLOCK_SEQ: mt.constexpr,
    BLOCK_DIM: mt.constexpr,
):
    """
    Compute attention for one sequence position.

    This is a simplified attention kernel that processes one query at a time.
    Each program handles one row of the output.

    Real Flash Attention uses:
    - Block-based processing for memory efficiency
    - Online softmax to avoid materializing full attention matrix
    - Tiling over the sequence dimension
    """
    # Which row of output this program computes
    row_idx = tl.program_id(0)

    # Step 1: Load the query vector for this row
    # Q[row_idx, :] - shape (head_dim,)
    q_offsets = tl.arange(0, BLOCK_DIM)
    q_mask = q_offsets < head_dim
    q = tl.load(
        q_ptr + row_idx * stride_q_seq + q_offsets * stride_q_dim,
        mask=q_mask,
        other=0.0
    )  # (BLOCK_DIM,)

    # Step 2: Initialize output accumulator and softmax normalizer
    acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)  # For V accumulation
    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)  # Running max
    l_i = tl.zeros((1,), dtype=tl.float32)  # Running sum of exp

    # Step 3: Iterate over key/value positions
    for j_start in range(0, seq_len, BLOCK_SEQ):
        # Compute scores for this block: Q[i] @ K[j:j+BLOCK]^T
        j_offsets = j_start + tl.arange(0, BLOCK_SEQ)
        j_mask = j_offsets < seq_len

        # Load keys for this block: K[j:j+BLOCK, :] - shape (BLOCK_SEQ, BLOCK_DIM)
        k = tl.load(
            k_ptr + j_offsets[:, None] * stride_k_seq + q_offsets[None, :] * stride_k_dim,
            mask=j_mask[:, None] & q_mask[None, :],
            other=0.0
        )  # (BLOCK_SEQ, BLOCK_DIM)

        # Compute attention scores: q @ k^T
        # q: (BLOCK_DIM,) -> need (1, BLOCK_DIM)
        # k: (BLOCK_SEQ, BLOCK_DIM)
        # scores: (BLOCK_SEQ,)
        scores = tl.sum(q[None, :] * k, axis=1) * scale  # (BLOCK_SEQ,)

        # Mask out invalid positions
        scores = tl.where(j_mask, scores, float("-inf"))

        # Online softmax update
        m_j = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_j)

        # Update running sum
        exp_scores = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(exp_scores, axis=0)
        m_i = m_new

        # Load values: V[j:j+BLOCK, :] - shape (BLOCK_SEQ, BLOCK_DIM)
        v = tl.load(
            v_ptr + j_offsets[:, None] * stride_v_seq + q_offsets[None, :] * stride_v_dim,
            mask=j_mask[:, None] & q_mask[None, :],
            other=0.0
        )  # (BLOCK_SEQ, BLOCK_DIM)

        # Update accumulator: weighted sum of values
        # exp_scores: (BLOCK_SEQ,) -> (BLOCK_SEQ, 1)
        # v: (BLOCK_SEQ, BLOCK_DIM)
        # contribution: (BLOCK_DIM,)
        acc = alpha * acc + tl.sum(exp_scores[:, None] * v, axis=0)

    # Step 4: Normalize by softmax sum
    out = acc / l_i

    # Step 5: Store output
    out_offsets = tl.arange(0, BLOCK_DIM)
    out_mask = out_offsets < head_dim
    tl.store(
        out_ptr + row_idx * stride_o_seq + out_offsets * stride_o_dim,
        out,
        mask=out_mask
    )


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product attention.

    Args:
        q: Query matrix (seq_len, head_dim)
        k: Key matrix (seq_len, head_dim)
        v: Value matrix (seq_len, head_dim)

    Returns:
        Attention output (seq_len, head_dim)
    """
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype == np.float32

    seq_len, head_dim = q.shape
    out = np.empty_like(q)

    # Scaling factor
    scale = 1.0 / np.sqrt(head_dim)

    # Block sizes
    BLOCK_SEQ = 32  # Process this many KV positions at a time
    BLOCK_DIM = 64  # Must be >= head_dim (or we need to tile)

    # Validate (simplified version has constraints)
    assert head_dim <= BLOCK_DIM, f"head_dim {head_dim} > BLOCK_DIM {BLOCK_DIM}"

    # Grid: one program per output row
    grid = (seq_len,)

    # Get strides
    stride_q_seq, stride_q_dim = q.strides[0] // 4, q.strides[1] // 4
    stride_k_seq, stride_k_dim = k.strides[0] // 4, k.strides[1] // 4
    stride_v_seq, stride_v_dim = v.strides[0] // 4, v.strides[1] // 4
    stride_o_seq, stride_o_dim = out.strides[0] // 4, out.strides[1] // 4

    attention_kernel[grid](
        q, k, v, out,
        seq_len, head_dim,
        stride_q_seq, stride_q_dim,
        stride_k_seq, stride_k_dim,
        stride_v_seq, stride_v_dim,
        stride_o_seq, stride_o_dim,
        scale,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
    )

    return out


def numpy_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation."""
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = q @ k.T * scale  # (seq_len, seq_len)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    softmax_scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return softmax_scores @ v


def main():
    print("Mini-Triton Attention Example")
    print("=" * 50)

    # Test 1: Small attention
    print("\n1. Small attention")
    seq_len, head_dim = 32, 32
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)

    result = attention(q, k, v)
    expected = numpy_attention(q, k, v)

    print(f"   Shape: ({seq_len}, {head_dim})")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-4)
    print("   ✓ Test passed!")

    # Test 2: Larger attention
    print("\n2. Larger attention")
    seq_len, head_dim = 128, 64
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)

    result = attention(q, k, v)
    expected = numpy_attention(q, k, v)

    print(f"   Shape: ({seq_len}, {head_dim})")
    print(f"   Max error: {np.max(np.abs(result - expected)):.2e}")
    assert np.allclose(result, expected, rtol=1e-4)
    print("   ✓ Test passed!")

    # Test 3: Verify attention properties
    print("\n3. Attention properties")
    seq_len, head_dim = 64, 32
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)

    # Identity attention: when Q=K, attention to self should be strong
    q_same = k.copy()
    result = attention(q_same, k, v)

    print(f"   Testing self-attention consistency")
    print(f"   Output shape: {result.shape}")
    print(f"   Output range: [{result.min():.3f}, {result.max():.3f}]")

    # Compare with expected
    expected = numpy_attention(q_same, k, v)
    assert np.allclose(result, expected, rtol=1e-4)
    print("   ✓ Test passed!")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("\nNote: This is a simplified attention implementation.")
    print("Real Flash Attention uses more sophisticated tiling")
    print("to avoid materializing the full attention matrix,")
    print("reducing memory from O(n²) to O(n).")


if __name__ == "__main__":
    main()
