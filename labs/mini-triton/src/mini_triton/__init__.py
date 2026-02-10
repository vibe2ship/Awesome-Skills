"""
Mini-Triton: A pedagogical GPU kernel compiler inspired by OpenAI Triton.

This is a learning project that implements the core concepts of the Triton compiler:
- Tile-based programming model
- Python DSL to IR transformation
- Code generation for NumPy (and optionally Numba CUDA)

Example usage:
    import mini_triton as mt
    import mini_triton.language as tl

    @mt.jit
    def vector_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: mt.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
"""

from mini_triton.runtime.jit import jit, JITFunction, constexpr
from mini_triton.ir.types import DType, float16, float32, float64, int32, int64, bool_

__version__ = "0.1.0"

__all__ = [
    "jit",
    "JITFunction",
    "constexpr",
    "DType",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "bool_",
]
