"""
Mini-Triton Code Generation.

This module generates executable code from the IR.

Available backends:
- numpy_gen: NumPy-based Python code (for verification and debugging)
- c_gen: C code generation (optional, more advanced)
"""

from mini_triton.codegen.base import CodeGenerator
from mini_triton.codegen.numpy_gen import NumpyCodeGenerator

__all__ = ["CodeGenerator", "NumpyCodeGenerator"]
