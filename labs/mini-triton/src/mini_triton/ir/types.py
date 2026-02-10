"""
Type system for Mini-Triton IR.

This module defines the type hierarchy:
- DType: Data types (float32, int32, etc.)
- Type: Base class for all types
- ScalarType: Single values
- PointerType: Memory pointers
- BlockType: Tile/block types (the key Triton concept)

The type system is inspired by Triton's type system but simplified for learning.

Key concepts:
1. BlockType represents a tile of data - the core abstraction in Triton
2. PointerType is used for memory access, can point to scalars or blocks
3. Types are immutable and can be compared for equality

TODO: Implement the type classes below
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Union


class DType:
    """
    Data type enumeration for scalar values.

    Similar to numpy.dtype but simplified.

    Attributes:
        name: Human-readable type name
        bits: Number of bits for this type
        is_floating: Whether this is a floating-point type
    """

    def __init__(self, name: str, bits: int, is_floating: bool):
        # TODO: Store the name, bits, and is_floating attributes
        raise NotImplementedError("TODO: Implement DType.__init__")

    def __repr__(self) -> str:
        # TODO: Return a string representation like "float32"
        raise NotImplementedError("TODO: Implement DType.__repr__")

    def __eq__(self, other: object) -> bool:
        # TODO: Compare two DTypes for equality
        raise NotImplementedError("TODO: Implement DType.__eq__")

    def __hash__(self) -> int:
        # TODO: Return a hash based on the type properties
        raise NotImplementedError("TODO: Implement DType.__hash__")

    @property
    def numpy_dtype(self) -> str:
        """Return the numpy dtype string for this type."""
        # TODO: Return the corresponding numpy dtype string
        # e.g., "float32" -> "np.float32"
        raise NotImplementedError("TODO: Implement DType.numpy_dtype")


# Pre-defined data types (these should work after implementing DType)
float16 = DType("float16", 16, is_floating=True)
float32 = DType("float32", 32, is_floating=True)
float64 = DType("float64", 64, is_floating=True)
int32 = DType("int32", 32, is_floating=False)
int64 = DType("int64", 64, is_floating=False)
bool_ = DType("bool", 1, is_floating=False)


class Type(ABC):
    """
    Base class for all types in the Mini-Triton type system.

    All types must implement:
    - __repr__: String representation
    - __eq__: Equality comparison
    - __hash__: Hashing (for use in dicts/sets)
    """

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


@dataclass(frozen=True)
class ScalarType(Type):
    """
    Represents a scalar (single value) type.

    Examples:
        ScalarType(float32)  # A single float32 value
        ScalarType(int32)    # A single int32 value
    """
    dtype: DType

    def __repr__(self) -> str:
        # TODO: Return string like "scalar<float32>"
        raise NotImplementedError("TODO: Implement ScalarType.__repr__")

    def __eq__(self, other: object) -> bool:
        # TODO: Compare with another ScalarType
        raise NotImplementedError("TODO: Implement ScalarType.__eq__")

    def __hash__(self) -> int:
        # TODO: Return hash based on dtype
        raise NotImplementedError("TODO: Implement ScalarType.__hash__")


@dataclass(frozen=True)
class BlockType(Type):
    """
    Represents a block (tile) of values - the core Triton concept.

    A BlockType has:
    - dtype: The element type
    - shape: The shape of the block (tuple of ints)

    In Triton, operations work on entire blocks at once, enabling
    efficient parallel execution on GPUs.

    Examples:
        BlockType(float32, (128,))       # 1D block of 128 floats
        BlockType(float32, (64, 64))     # 2D block of 64x64 floats

    Shape elements can be:
    - Positive integers: Fixed size dimensions
    - In real Triton, shapes can be symbolic (we simplify to concrete shapes)
    """
    dtype: DType
    shape: Tuple[int, ...]

    def __repr__(self) -> str:
        # TODO: Return string like "block<float32, [128]>" or "block<float32, [64, 64]>"
        raise NotImplementedError("TODO: Implement BlockType.__repr__")

    def __eq__(self, other: object) -> bool:
        # TODO: Compare dtype and shape
        raise NotImplementedError("TODO: Implement BlockType.__eq__")

    def __hash__(self) -> int:
        # TODO: Hash based on dtype and shape
        raise NotImplementedError("TODO: Implement BlockType.__hash__")

    @property
    def rank(self) -> int:
        """Return the number of dimensions (rank) of the block."""
        # TODO: Return len(self.shape)
        raise NotImplementedError("TODO: Implement BlockType.rank")

    @property
    def numel(self) -> int:
        """Return the total number of elements in the block."""
        # TODO: Return product of shape dimensions
        raise NotImplementedError("TODO: Implement BlockType.numel")

    def broadcast_to(self, new_shape: Tuple[int, ...]) -> "BlockType":
        """
        Return a new BlockType with the given shape (for broadcasting).

        Raises ValueError if shapes are incompatible.
        """
        # TODO: Validate that broadcasting is legal and return new BlockType
        # Broadcasting rules (same as NumPy):
        # 1. Shapes are compared from right to left
        # 2. Dimensions are compatible if they are equal or one of them is 1
        raise NotImplementedError("TODO: Implement BlockType.broadcast_to")


@dataclass(frozen=True)
class PointerType(Type):
    """
    Represents a pointer to memory.

    Pointers in Triton are used for memory access (load/store).
    They can point to either scalars or blocks.

    In real Triton, pointer arithmetic is common:
        ptr + offsets  # Returns a block of pointers

    Examples:
        PointerType(float32)              # Pointer to single float32
        PointerType(float32, (128,))      # Pointer with block offset shape

    The offset_shape field tracks the shape when pointer arithmetic
    creates a "block of pointers" (e.g., ptr + tl.arange(0, 128)).
    """
    pointee_dtype: DType
    offset_shape: Optional[Tuple[int, ...]] = None

    def __repr__(self) -> str:
        # TODO: Return string like "ptr<float32>" or "ptr<float32, [128]>"
        raise NotImplementedError("TODO: Implement PointerType.__repr__")

    def __eq__(self, other: object) -> bool:
        # TODO: Compare pointee_dtype and offset_shape
        raise NotImplementedError("TODO: Implement PointerType.__eq__")

    def __hash__(self) -> int:
        # TODO: Hash based on pointee_dtype and offset_shape
        raise NotImplementedError("TODO: Implement PointerType.__hash__")

    def with_offset_shape(self, shape: Tuple[int, ...]) -> "PointerType":
        """Return a new PointerType with the given offset shape."""
        # TODO: Create new PointerType with updated offset_shape
        raise NotImplementedError("TODO: Implement PointerType.with_offset_shape")

    @property
    def is_block_pointer(self) -> bool:
        """Return True if this is a 'block of pointers' (has offset_shape)."""
        # TODO: Check if offset_shape is not None
        raise NotImplementedError("TODO: Implement PointerType.is_block_pointer")


# Type aliases for convenience
TensorType = Union[ScalarType, BlockType]
AnyType = Union[ScalarType, BlockType, PointerType]


def get_broadcast_shape(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute the broadcast shape of two shapes.

    Follows NumPy broadcasting rules:
    1. Pad shorter shape with 1s on the left
    2. Each dimension must be equal or one must be 1

    Args:
        shape1: First shape
        shape2: Second shape

    Returns:
        The broadcast result shape

    Raises:
        ValueError: If shapes cannot be broadcast together

    Examples:
        get_broadcast_shape((128,), (128,)) -> (128,)
        get_broadcast_shape((1,), (128,)) -> (128,)
        get_broadcast_shape((64, 1), (1, 128)) -> (64, 128)
        get_broadcast_shape((64,), (128,)) -> ValueError
    """
    # TODO: Implement NumPy-style broadcasting rules
    raise NotImplementedError("TODO: Implement get_broadcast_shape")


def infer_binary_op_type(lhs: Type, rhs: Type) -> Type:
    """
    Infer the result type of a binary operation.

    Rules:
    1. scalar op scalar -> scalar
    2. block op scalar -> block (scalar is broadcast)
    3. block op block -> block (shapes must be broadcast-compatible)

    The result dtype follows standard promotion rules:
    - float + int -> float
    - smaller + larger -> larger

    Args:
        lhs: Left operand type
        rhs: Right operand type

    Returns:
        The result type of the binary operation
    """
    # TODO: Implement type inference for binary operations
    raise NotImplementedError("TODO: Implement infer_binary_op_type")
