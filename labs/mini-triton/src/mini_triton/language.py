"""
Mini-Triton Language Module (tl.* API).

This module provides the user-facing API that mimics Triton's tl.* namespace.
These functions are used inside @jit decorated kernels.

Usage:
    import mini_triton as mt
    import mini_triton.language as tl

    @mt.jit
    def my_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: mt.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.exp(x)
        tl.store(y_ptr + offsets, y, mask=mask)

During JIT compilation, these functions don't execute directly - instead,
they are traced to build the IR. The actual implementation is in the
frontend/ast_visitor.py module.

For tracing to work, we use "proxy" objects that record operations.

TODO: Implement the proxy classes and tl.* functions below
"""

from __future__ import annotations
from typing import Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass

from mini_triton.ir.types import DType, float32, int32, bool_
from mini_triton.ir.ops import BinaryOpKind, UnaryOpKind, ReduceOpKind

if TYPE_CHECKING:
    from mini_triton.ir.builder import IRBuilder
    from mini_triton.ir.ops import Value


# Global state for tracing (set during JIT compilation)
_current_builder: Optional["IRBuilder"] = None


def _get_builder() -> "IRBuilder":
    """Get the current IR builder (set during tracing)."""
    if _current_builder is None:
        raise RuntimeError(
            "Triton operations can only be used inside @jit decorated functions"
        )
    return _current_builder


def _set_builder(builder: Optional["IRBuilder"]) -> None:
    """Set the current IR builder (called by JIT compiler)."""
    global _current_builder
    _current_builder = builder


class TensorProxy:
    """
    Proxy object representing a tensor/block during tracing.

    When a @jit function is traced, this object captures operations
    and records them in the IR.

    Supports Python operator overloading (+, -, *, /, <, >, etc.)
    so that normal Python expressions work.
    """

    def __init__(self, value: "Value"):
        """
        Create a TensorProxy wrapping an IR Value.

        Args:
            value: The IR Value this proxy represents
        """
        # TODO: Store the value
        # self._value = value
        raise NotImplementedError("TODO: Implement TensorProxy.__init__")

    @property
    def value(self) -> "Value":
        """Get the underlying IR Value."""
        # TODO: Return self._value
        raise NotImplementedError("TODO: Implement TensorProxy.value")

    # ========== Arithmetic Operators ==========

    def __add__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Addition: self + other"""
        # TODO:
        # 1. If other is a Python scalar, create a constant
        # 2. Call builder.add() with both values
        # 3. Return new TensorProxy wrapping the result
        raise NotImplementedError("TODO: Implement TensorProxy.__add__")

    def __radd__(self, other: Union[int, float]) -> "TensorProxy":
        """Reverse addition: other + self"""
        return self.__add__(other)

    def __sub__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Subtraction: self - other"""
        # TODO: Similar to __add__ but with subtraction
        raise NotImplementedError("TODO: Implement TensorProxy.__sub__")

    def __rsub__(self, other: Union[int, float]) -> "TensorProxy":
        """Reverse subtraction: other - self"""
        # TODO: Create constant for other, then other - self
        raise NotImplementedError("TODO: Implement TensorProxy.__rsub__")

    def __mul__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Multiplication: self * other"""
        # TODO: Similar to __add__ but with multiplication
        raise NotImplementedError("TODO: Implement TensorProxy.__mul__")

    def __rmul__(self, other: Union[int, float]) -> "TensorProxy":
        """Reverse multiplication: other * self"""
        return self.__mul__(other)

    def __truediv__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Division: self / other"""
        # TODO: Similar to __add__ but with division
        raise NotImplementedError("TODO: Implement TensorProxy.__truediv__")

    def __rtruediv__(self, other: Union[int, float]) -> "TensorProxy":
        """Reverse division: other / self"""
        # TODO: Create constant for other, then other / self
        raise NotImplementedError("TODO: Implement TensorProxy.__rtruediv__")

    def __mod__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Modulo: self % other"""
        # TODO: Similar to __add__ but with modulo
        raise NotImplementedError("TODO: Implement TensorProxy.__mod__")

    def __neg__(self) -> "TensorProxy":
        """Negation: -self"""
        # TODO: Call builder.neg()
        raise NotImplementedError("TODO: Implement TensorProxy.__neg__")

    # ========== Comparison Operators ==========

    def __lt__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Less than: self < other"""
        # TODO: Call builder.lt()
        raise NotImplementedError("TODO: Implement TensorProxy.__lt__")

    def __le__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Less than or equal: self <= other"""
        # TODO: Call builder.le()
        raise NotImplementedError("TODO: Implement TensorProxy.__le__")

    def __gt__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Greater than: self > other"""
        # TODO: Call builder.gt()
        raise NotImplementedError("TODO: Implement TensorProxy.__gt__")

    def __ge__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":
        """Greater than or equal: self >= other"""
        # TODO: Call builder.ge()
        raise NotImplementedError("TODO: Implement TensorProxy.__ge__")

    def __eq__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":  # type: ignore
        """Equal: self == other"""
        # TODO: Call builder.eq()
        raise NotImplementedError("TODO: Implement TensorProxy.__eq__")

    def __ne__(self, other: Union["TensorProxy", int, float]) -> "TensorProxy":  # type: ignore
        """Not equal: self != other"""
        # TODO: Call builder.ne()
        raise NotImplementedError("TODO: Implement TensorProxy.__ne__")

    # ========== Type Casting ==========

    def to(self, dtype: DType) -> "TensorProxy":
        """Cast to a different dtype."""
        # TODO: Call builder.cast()
        raise NotImplementedError("TODO: Implement TensorProxy.to")

    def __repr__(self) -> str:
        # TODO: Return a string representation for debugging
        raise NotImplementedError("TODO: Implement TensorProxy.__repr__")


# ========== Core Triton-like Functions ==========

def program_id(axis: int) -> TensorProxy:
    """
    Get the program (block) ID for the given axis.

    In a GPU grid launch, each block has a unique ID.
    This is how blocks know which data to process.

    Args:
        axis: The axis (0, 1, or 2)

    Returns:
        Scalar int32 containing the program ID

    Example:
        pid = tl.program_id(0)  # Get x-axis block ID
        # In a 1D grid of 10 blocks, pid will be 0-9
    """
    # TODO:
    # 1. Get the current builder
    # 2. Call builder.program_id(axis)
    # 3. Return TensorProxy wrapping the result
    raise NotImplementedError("TODO: Implement program_id")


def arange(start: int, end: int) -> TensorProxy:
    """
    Create a 1D range of integers [start, end).

    This is fundamental to Triton programming - it creates
    a block of indices used for memory access.

    Args:
        start: Start value (inclusive)
        end: End value (exclusive)

    Returns:
        1D block of int32 values

    Example:
        offsets = tl.arange(0, 128)  # [0, 1, 2, ..., 127]
        ptr = base_ptr + offsets     # Creates 128 pointers
    """
    # TODO:
    # 1. Get the current builder
    # 2. Call builder.arange(start, end)
    # 3. Return TensorProxy wrapping the result
    raise NotImplementedError("TODO: Implement arange")


def zeros(shape: tuple, dtype: DType) -> TensorProxy:
    """
    Create a block of zeros.

    Args:
        shape: Shape of the block
        dtype: Data type

    Returns:
        Block filled with zeros
    """
    # TODO:
    # 1. Get the current builder
    # 2. Call builder.zeros(shape, dtype)
    # 3. Return TensorProxy wrapping the result
    raise NotImplementedError("TODO: Implement zeros")


def full(shape: tuple, fill_value: Any, dtype: DType) -> TensorProxy:
    """
    Create a block filled with a value.

    Args:
        shape: Shape of the block
        fill_value: Value to fill with
        dtype: Data type

    Returns:
        Block filled with fill_value
    """
    # TODO:
    # 1. Get the current builder
    # 2. Call builder.full(shape, fill_value, dtype)
    # 3. Return TensorProxy wrapping the result
    raise NotImplementedError("TODO: Implement full")


def load(
    ptr: TensorProxy,
    mask: Optional[TensorProxy] = None,
    other: Optional[Union[TensorProxy, int, float]] = None
) -> TensorProxy:
    """
    Load values from memory.

    This is one of the two fundamental memory operations (with store).

    Args:
        ptr: Pointer(s) to load from
        mask: Optional boolean mask (only load where True)
        other: Value to use for masked elements (where mask is False)

    Returns:
        Loaded values

    Example:
        # Load 128 elements with masking for the last block
        offsets = tl.arange(0, 128)
        mask = offsets < n_elements  # Don't load past end
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    """
    # TODO:
    # 1. Get the current builder
    # 2. Handle 'other' if it's a scalar
    # 3. Call builder.load()
    # 4. Return TensorProxy wrapping the result
    raise NotImplementedError("TODO: Implement load")


def store(
    ptr: TensorProxy,
    value: TensorProxy,
    mask: Optional[TensorProxy] = None
) -> None:
    """
    Store values to memory.

    Args:
        ptr: Pointer(s) to store to
        value: Values to store
        mask: Optional boolean mask (only store where True)

    Example:
        offsets = tl.arange(0, 128)
        mask = offsets < n_elements
        tl.store(out_ptr + offsets, result, mask=mask)
    """
    # TODO:
    # 1. Get the current builder
    # 2. Call builder.store()
    raise NotImplementedError("TODO: Implement store")


def dot(
    a: TensorProxy,
    b: TensorProxy,
    acc: Optional[TensorProxy] = None
) -> TensorProxy:
    """
    Matrix multiplication.

    Computes a @ b, optionally adding to an accumulator.

    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N)
        acc: Optional accumulator (M, N)

    Returns:
        Result matrix (M, N)

    Example:
        # GEMM-style computation
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc = tl.dot(a, b, acc)
    """
    # TODO:
    # 1. Get the current builder
    # 2. Call builder.dot()
    # 3. Return TensorProxy wrapping the result
    raise NotImplementedError("TODO: Implement dot")


# ========== Reduction Operations ==========

def sum(x: TensorProxy, axis: int) -> TensorProxy:
    """
    Sum reduction along an axis.

    Args:
        x: Input block
        axis: Axis to reduce

    Returns:
        Reduced block (one fewer dimension)
    """
    # TODO: Call builder.sum()
    raise NotImplementedError("TODO: Implement sum")


def max(x: TensorProxy, axis: int) -> TensorProxy:
    """
    Max reduction along an axis.

    Args:
        x: Input block
        axis: Axis to reduce

    Returns:
        Reduced block (one fewer dimension)
    """
    # TODO: Call builder.max()
    raise NotImplementedError("TODO: Implement max")


def min(x: TensorProxy, axis: int) -> TensorProxy:
    """
    Min reduction along an axis.

    Args:
        x: Input block
        axis: Axis to reduce

    Returns:
        Reduced block (one fewer dimension)
    """
    # TODO: Call builder.min()
    raise NotImplementedError("TODO: Implement min")


# ========== Element-wise Math Functions ==========

def exp(x: TensorProxy) -> TensorProxy:
    """Element-wise exponential."""
    # TODO: Call builder.exp()
    raise NotImplementedError("TODO: Implement exp")


def log(x: TensorProxy) -> TensorProxy:
    """Element-wise natural logarithm."""
    # TODO: Call builder.log()
    raise NotImplementedError("TODO: Implement log")


def sqrt(x: TensorProxy) -> TensorProxy:
    """Element-wise square root."""
    # TODO: Call builder.sqrt()
    raise NotImplementedError("TODO: Implement sqrt")


def abs(x: TensorProxy) -> TensorProxy:
    """Element-wise absolute value."""
    # TODO: Call builder.abs()
    raise NotImplementedError("TODO: Implement abs")


# ========== Comparison/Selection Functions ==========

def maximum(x: TensorProxy, y: Union[TensorProxy, int, float]) -> TensorProxy:
    """Element-wise maximum."""
    # TODO: Call builder.maximum()
    raise NotImplementedError("TODO: Implement maximum")


def minimum(x: TensorProxy, y: Union[TensorProxy, int, float]) -> TensorProxy:
    """Element-wise minimum."""
    # TODO: Call builder.minimum()
    raise NotImplementedError("TODO: Implement minimum")


def where(
    condition: TensorProxy,
    x: Union[TensorProxy, int, float],
    y: Union[TensorProxy, int, float]
) -> TensorProxy:
    """
    Select values based on condition.

    Like numpy.where: returns x where condition is True, else y.

    Args:
        condition: Boolean block
        x: Values where True
        y: Values where False

    Returns:
        Selected values
    """
    # TODO: Call builder.where()
    raise NotImplementedError("TODO: Implement where")


# ========== Type Constants (re-exported for convenience) ==========

# These allow users to write tl.float32 instead of importing from types
float16 = DType("float16", 16, is_floating=True)
float32 = DType("float32", 32, is_floating=True)
float64 = DType("float64", 64, is_floating=True)
int32 = DType("int32", 32, is_floating=False)
int64 = DType("int64", 64, is_floating=False)
bool_ = DType("bool", 1, is_floating=False)
