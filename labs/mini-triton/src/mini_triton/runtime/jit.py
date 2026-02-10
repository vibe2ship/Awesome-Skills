"""
JIT Compilation for Mini-Triton.

This module provides the @jit decorator that converts Python functions
into compiled Triton-like kernels.

Usage:
    import mini_triton as mt
    import mini_triton.language as tl

    @mt.jit
    def my_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: mt.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ...

    # Call the kernel
    my_kernel[grid](x, y, n, BLOCK_SIZE=128)

The JIT compilation process:
1. Parse the Python function's AST
2. Convert to Mini-Triton IR
3. Run optimization passes
4. Generate target code (NumPy or Numba)
5. Cache the compiled kernel

TODO: Implement the JIT decorator
"""

from __future__ import annotations
import functools
import hashlib
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from mini_triton.ir.types import Type, ScalarType, PointerType, float32, int32
from mini_triton.ir.ops import Function
from mini_triton.frontend.ast_visitor import trace_function
from mini_triton.codegen.numpy_gen import NumpyCodeGenerator


class constexpr:
    """
    Marker class for compile-time constant parameters.

    Use as a type annotation to indicate that a parameter is
    known at compile time and can be used for specialization.

    Example:
        @mt.jit
        def my_kernel(x_ptr, n, BLOCK_SIZE: mt.constexpr):
            ...  # BLOCK_SIZE is known at compile time
    """
    pass


@dataclass
class LaunchConfig:
    """
    Configuration for kernel launch.

    Attributes:
        grid: Grid dimensions (num_blocks_x, num_blocks_y, num_blocks_z)
    """
    grid: Tuple[int, ...]

    def __post_init__(self):
        # Normalize grid to 3D
        if len(self.grid) == 1:
            self.grid = (self.grid[0], 1, 1)
        elif len(self.grid) == 2:
            self.grid = (self.grid[0], self.grid[1], 1)


class JITFunction:
    """
    A JIT-compiled Triton-like kernel.

    This class wraps a Python function and provides:
    1. Lazy compilation on first call
    2. Caching of compiled kernels
    3. Grid-based launch syntax (kernel[grid](...))
    """

    def __init__(self, func: Callable):
        """
        Initialize JIT function wrapper.

        Args:
            func: The Python function to JIT compile
        """
        # TODO:
        # self._func = func
        # self._name = func.__name__
        # self._cache: Dict[str, Callable] = {}  # specialization key -> compiled kernel
        # self._ir_cache: Dict[str, Function] = {}  # for debugging
        raise NotImplementedError("TODO: Implement JITFunction.__init__")

    def __getitem__(self, grid: Union[int, Tuple[int, ...]]) -> "_KernelLauncher":
        """
        Indexing syntax for specifying the grid.

        Example:
            kernel[1024]  # 1D grid with 1024 blocks
            kernel[(32, 32)]  # 2D grid
        """
        # TODO:
        # 1. Normalize grid to tuple
        # 2. Return _KernelLauncher with self and grid
        raise NotImplementedError("TODO: Implement JITFunction.__getitem__")

    def _get_specialization_key(self, constexpr_values: Dict[str, Any]) -> str:
        """
        Generate a cache key for this specialization.

        Different constexpr values produce different compiled kernels.
        """
        # TODO:
        # Create a hashable key from constexpr_values
        # e.g., "BLOCK_SIZE=128,NUM_WARPS=4"
        raise NotImplementedError("TODO: Implement JITFunction._get_specialization_key")

    def _compile(
        self,
        constexpr_values: Dict[str, Any],
        param_types: Dict[str, Type]
    ) -> Callable:
        """
        Compile the function for given constexpr values.

        Args:
            constexpr_values: Values for constexpr parameters
            param_types: Types for each parameter

        Returns:
            Compiled kernel function
        """
        # TODO:
        # 1. Check cache first
        # 2. If not cached:
        #    a. Trace the function to IR
        #    b. Run optimization passes
        #    c. Generate code
        #    d. Compile code
        #    e. Cache the result
        # 3. Return compiled kernel
        raise NotImplementedError("TODO: Implement JITFunction._compile")

    def _infer_param_types(self, args: Tuple, kwargs: Dict[str, Any]) -> Dict[str, Type]:
        """
        Infer parameter types from runtime arguments.

        Uses numpy array dtype and shape to determine pointer types.
        """
        # TODO:
        # 1. Get function parameter names
        # 2. For each numpy array, create PointerType
        # 3. For each scalar, create ScalarType
        raise NotImplementedError("TODO: Implement JITFunction._infer_param_types")

    def __call__(self, *args, **kwargs):
        """
        Direct call (without grid) - for debugging.
        """
        # TODO: Raise error or run with default grid
        raise NotImplementedError("TODO: Implement JITFunction.__call__")


class _KernelLauncher:
    """
    Helper class for launching a kernel with a specific grid.

    Created when you write kernel[grid](...).
    """

    def __init__(self, jit_func: JITFunction, grid: Tuple[int, ...]):
        """
        Initialize the launcher.

        Args:
            jit_func: The JIT function to launch
            grid: Grid dimensions
        """
        # TODO:
        # self._jit_func = jit_func
        # self._grid = grid
        raise NotImplementedError("TODO: Implement _KernelLauncher.__init__")

    def __call__(self, *args, **kwargs):
        """
        Launch the kernel with given arguments.

        Args:
            *args: Positional arguments (arrays, scalars)
            **kwargs: Keyword arguments (including constexpr values)
        """
        # TODO:
        # 1. Separate constexpr values from runtime arguments
        # 2. Infer parameter types from arguments
        # 3. Compile the kernel (may be cached)
        # 4. Execute the kernel with grid and arguments
        raise NotImplementedError("TODO: Implement _KernelLauncher.__call__")


def jit(func: Callable) -> JITFunction:
    """
    Decorator to JIT compile a Triton-like kernel.

    Usage:
        @jit
        def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            ...

        # Launch with 1D grid
        grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        my_kernel[grid](x, y, out, n, BLOCK_SIZE=128)

    Args:
        func: The Python function to JIT compile

    Returns:
        JITFunction wrapper
    """
    return JITFunction(func)
