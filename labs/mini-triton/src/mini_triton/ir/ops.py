"""
IR Operations for Mini-Triton.

This module defines all the IR nodes that represent Triton operations.
The IR is in SSA (Static Single Assignment) form.

Key concepts:
1. Every operation produces a Value (result)
2. Values are typed (via the Type system)
3. The IR is immutable - transformations create new nodes

Operations are categorized as:
- Memory ops: LoadOp, StoreOp
- Arithmetic ops: BinaryOp, UnaryOp
- Block ops: MakeRangeOp, BroadcastOp, ReshapeOp
- Control ops: ProgramIdOp
- Reduction ops: ReduceOp, DotOp

TODO: Implement the IR node classes below
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any, Sequence

from mini_triton.ir.types import Type, DType, ScalarType, BlockType, PointerType


class BinaryOpKind(Enum):
    """Kinds of binary operations."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    AND = auto()  # Bitwise/logical AND
    OR = auto()   # Bitwise/logical OR
    XOR = auto()  # Bitwise XOR
    SHL = auto()  # Left shift
    SHR = auto()  # Right shift (arithmetic)
    EQ = auto()   # Equal
    NE = auto()   # Not equal
    LT = auto()   # Less than
    LE = auto()   # Less than or equal
    GT = auto()   # Greater than
    GE = auto()   # Greater than or equal
    MAX = auto()  # Element-wise max
    MIN = auto()  # Element-wise min


class UnaryOpKind(Enum):
    """Kinds of unary operations."""
    NEG = auto()     # Negation (-x)
    NOT = auto()     # Logical not
    EXP = auto()     # Exponential
    LOG = auto()     # Natural log
    SQRT = auto()    # Square root
    SIN = auto()     # Sine
    COS = auto()     # Cosine
    ABS = auto()     # Absolute value


class ReduceOpKind(Enum):
    """Kinds of reduction operations."""
    SUM = auto()
    MAX = auto()
    MIN = auto()
    PROD = auto()


class IRNode(ABC):
    """
    Base class for all IR nodes.

    Every IR node has:
    - A unique ID (for debugging/printing)
    - A method to get child nodes (for traversal)
    """

    _id_counter: int = 0

    def __init__(self):
        self._id = IRNode._id_counter
        IRNode._id_counter += 1

    @property
    def id(self) -> int:
        return self._id

    @abstractmethod
    def children(self) -> List["IRNode"]:
        """Return all child nodes of this node."""
        pass


class Value(IRNode):
    """
    A value in the IR - the result of some computation.

    Every Value has a Type. Values are immutable and are in SSA form
    (each Value is defined exactly once).

    Attributes:
        result_type: The type of this value
        name: Optional name for debugging (e.g., variable names from source)
        defining_op: The operation that produces this value (set during IR construction)
    """

    def __init__(self, result_type: Type, name: Optional[str] = None):
        super().__init__()
        # TODO: Initialize result_type, name, and defining_op (initially None)
        raise NotImplementedError("TODO: Implement Value.__init__")

    @property
    def type(self) -> Type:
        """Return the type of this value."""
        # TODO: Return self.result_type
        raise NotImplementedError("TODO: Implement Value.type")

    def children(self) -> List[IRNode]:
        # Values don't have children in the IR tree sense
        return []

    def __repr__(self) -> str:
        # TODO: Return a string like "%0: float32" or "%x: block<float32, [128]>"
        raise NotImplementedError("TODO: Implement Value.__repr__")


@dataclass
class Constant(IRNode):
    """
    A constant value (literal).

    Examples:
        Constant(42, int32)
        Constant(3.14, float32)
        Constant([1, 2, 3], BlockType(int32, (3,)))  # Block constant

    Attributes:
        value: The constant value (Python int, float, or list)
        dtype: The data type
    """
    value: Any
    dtype: DType

    def __post_init__(self):
        super().__init__()

    def children(self) -> List[IRNode]:
        return []

    @property
    def type(self) -> Type:
        """Return the type of this constant."""
        # TODO: Return ScalarType for scalar values, BlockType for array values
        raise NotImplementedError("TODO: Implement Constant.type")

    def __repr__(self) -> str:
        # TODO: Return string like "const(42: int32)" or "const([1,2,3]: block<int32,[3]>)"
        raise NotImplementedError("TODO: Implement Constant.__repr__")


@dataclass
class BinaryOp(IRNode):
    """
    Binary operation on two values.

    Supports element-wise operations on scalars and blocks.
    When operating on blocks, broadcasting is applied automatically.

    Examples:
        BinaryOp(BinaryOpKind.ADD, lhs, rhs)  # lhs + rhs
        BinaryOp(BinaryOpKind.MUL, lhs, rhs)  # lhs * rhs
    """
    op: BinaryOpKind
    lhs: Value
    rhs: Value
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result Value with inferred type
        # Use infer_binary_op_type from types.py
        raise NotImplementedError("TODO: Implement BinaryOp.__post_init__")

    def children(self) -> List[IRNode]:
        return [self.lhs, self.rhs]

    def __repr__(self) -> str:
        # TODO: Return string like "add(%0, %1) -> %2"
        raise NotImplementedError("TODO: Implement BinaryOp.__repr__")


@dataclass
class UnaryOp(IRNode):
    """
    Unary operation on a single value.

    Examples:
        UnaryOp(UnaryOpKind.NEG, x)   # -x
        UnaryOp(UnaryOpKind.EXP, x)   # exp(x)
    """
    op: UnaryOpKind
    operand: Value
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result with same type as operand
        raise NotImplementedError("TODO: Implement UnaryOp.__post_init__")

    def children(self) -> List[IRNode]:
        return [self.operand]

    def __repr__(self) -> str:
        # TODO: Return string like "neg(%0) -> %1"
        raise NotImplementedError("TODO: Implement UnaryOp.__repr__")


@dataclass
class LoadOp(IRNode):
    """
    Load values from memory.

    This is one of the core Triton operations. It loads a block of values
    from memory using a pointer and optional mask.

    Examples:
        LoadOp(ptr, mask=None)           # Load without mask
        LoadOp(ptr, mask=mask_value)     # Masked load
        LoadOp(ptr, mask=mask, other=0)  # Masked load with default value

    In real GPU execution:
    - Unmasked elements are loaded from memory
    - Masked elements (where mask is False) return 'other' value
    - This enables handling edge cases (e.g., last block of data)

    Attributes:
        ptr: Pointer value (PointerType)
        mask: Optional boolean mask (BlockType with bool_ dtype)
        other: Default value for masked elements
    """
    ptr: Value
    mask: Optional[Value] = None
    other: Optional[Value] = None
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result Value
        # Result type depends on ptr:
        # - If ptr has offset_shape, result is BlockType with that shape
        # - Otherwise, result is ScalarType
        raise NotImplementedError("TODO: Implement LoadOp.__post_init__")

    def children(self) -> List[IRNode]:
        children = [self.ptr]
        if self.mask is not None:
            children.append(self.mask)
        if self.other is not None:
            children.append(self.other)
        return children

    def __repr__(self) -> str:
        # TODO: Return string like "load(%ptr, mask=%mask) -> %result"
        raise NotImplementedError("TODO: Implement LoadOp.__repr__")


@dataclass
class StoreOp(IRNode):
    """
    Store values to memory.

    The counterpart to LoadOp. Stores a block of values to memory.

    Examples:
        StoreOp(ptr, value)               # Store without mask
        StoreOp(ptr, value, mask=mask)    # Masked store

    Attributes:
        ptr: Pointer to store location
        value: Value to store
        mask: Optional boolean mask (only store where mask is True)
    """
    ptr: Value
    value: Value
    mask: Optional[Value] = None

    def __post_init__(self):
        super().__init__()
        # No result value for store (it's a side effect)

    def children(self) -> List[IRNode]:
        children = [self.ptr, self.value]
        if self.mask is not None:
            children.append(self.mask)
        return children

    def __repr__(self) -> str:
        # TODO: Return string like "store(%ptr, %value, mask=%mask)"
        raise NotImplementedError("TODO: Implement StoreOp.__repr__")


@dataclass
class MakeRangeOp(IRNode):
    """
    Create a range of integers (like tl.arange).

    This is fundamental to Triton programming - it creates a 1D block
    of consecutive integers used for indexing.

    Examples:
        MakeRangeOp(0, 128)  # Creates [0, 1, 2, ..., 127]

    In Triton code:
        offsets = tl.arange(0, BLOCK_SIZE)  # Create index block
        ptr = base_ptr + offsets            # Create block of pointers

    Attributes:
        start: Start value (inclusive)
        end: End value (exclusive)
    """
    start: int
    end: int
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result as BlockType with shape (end - start,) and int32 dtype
        raise NotImplementedError("TODO: Implement MakeRangeOp.__post_init__")

    def children(self) -> List[IRNode]:
        return []

    @property
    def size(self) -> int:
        """Return the number of elements in the range."""
        return self.end - self.start

    def __repr__(self) -> str:
        # TODO: Return string like "arange(0, 128) -> %0"
        raise NotImplementedError("TODO: Implement MakeRangeOp.__repr__")


@dataclass
class BroadcastOp(IRNode):
    """
    Broadcast a value to a larger shape.

    Used when combining values of different shapes.

    Examples:
        BroadcastOp(scalar, (128,))      # Broadcast scalar to 1D block
        BroadcastOp(block_1d, (64, 128)) # Broadcast 1D to 2D
    """
    operand: Value
    shape: Tuple[int, ...]
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result as BlockType with new shape
        # Validate that broadcast is legal
        raise NotImplementedError("TODO: Implement BroadcastOp.__post_init__")

    def children(self) -> List[IRNode]:
        return [self.operand]

    def __repr__(self) -> str:
        # TODO: Return string like "broadcast(%0, [64, 128]) -> %1"
        raise NotImplementedError("TODO: Implement BroadcastOp.__repr__")


@dataclass
class RangeOp(IRNode):
    """
    (Deprecated in favor of MakeRangeOp, kept for compatibility)
    """
    start: Value
    end: Value
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        raise NotImplementedError("TODO: Implement RangeOp.__post_init__")

    def children(self) -> List[IRNode]:
        return [self.start, self.end]


@dataclass
class ProgramIdOp(IRNode):
    """
    Get the program (block) ID for a given axis.

    In Triton, kernels are launched as a grid of programs (blocks).
    Each program has a unique ID in each axis.

    Examples:
        ProgramIdOp(0)  # Get ID in first axis (like blockIdx.x in CUDA)
        ProgramIdOp(1)  # Get ID in second axis (like blockIdx.y in CUDA)

    This is how different blocks know which part of the data to process.

    Attributes:
        axis: The axis to get the ID for (0, 1, or 2)
    """
    axis: int
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result as ScalarType(int32)
        # Validate axis is 0, 1, or 2
        raise NotImplementedError("TODO: Implement ProgramIdOp.__post_init__")

    def children(self) -> List[IRNode]:
        return []

    def __repr__(self) -> str:
        # TODO: Return string like "program_id(0) -> %0"
        raise NotImplementedError("TODO: Implement ProgramIdOp.__repr__")


@dataclass
class DotOp(IRNode):
    """
    Matrix multiplication (dot product) of two 2D blocks.

    This is the core operation for implementing GEMM and attention.

    Examples:
        DotOp(a, b)  # Matrix multiply: result = a @ b

    Shapes must satisfy: (M, K) @ (K, N) -> (M, N)

    Attributes:
        lhs: Left operand, shape (M, K)
        rhs: Right operand, shape (K, N)
        acc: Optional accumulator, shape (M, N) - for fused multiply-add
    """
    lhs: Value
    rhs: Value
    acc: Optional[Value] = None
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result
        # Validate shapes are compatible for matmul
        # Result shape: (M, N) where lhs is (M, K) and rhs is (K, N)
        raise NotImplementedError("TODO: Implement DotOp.__post_init__")

    def children(self) -> List[IRNode]:
        children = [self.lhs, self.rhs]
        if self.acc is not None:
            children.append(self.acc)
        return children

    def __repr__(self) -> str:
        # TODO: Return string like "dot(%a, %b) -> %c" or "dot(%a, %b, acc=%acc) -> %c"
        raise NotImplementedError("TODO: Implement DotOp.__repr__")


@dataclass
class ReduceOp(IRNode):
    """
    Reduce a block along an axis.

    Examples:
        ReduceOp(ReduceOpKind.SUM, x, axis=0)   # Sum along first axis
        ReduceOp(ReduceOpKind.MAX, x, axis=-1)  # Max along last axis

    The result has one fewer dimension than the input.

    Attributes:
        op: The reduction operation (sum, max, min, prod)
        operand: The block to reduce
        axis: The axis to reduce along
    """
    op: ReduceOpKind
    operand: Value
    axis: int
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result
        # Result shape has the reduced axis removed
        # e.g., (64, 128) with axis=1 -> (64,)
        raise NotImplementedError("TODO: Implement ReduceOp.__post_init__")

    def children(self) -> List[IRNode]:
        return [self.operand]

    def __repr__(self) -> str:
        # TODO: Return string like "reduce_sum(%x, axis=0) -> %y"
        raise NotImplementedError("TODO: Implement ReduceOp.__repr__")


@dataclass
class WhereOp(IRNode):
    """
    Select values based on a condition (like numpy.where or tl.where).

    Examples:
        WhereOp(cond, true_val, false_val)
        # Returns true_val where cond is True, else false_val

    Attributes:
        condition: Boolean block/scalar
        true_value: Value to use where condition is True
        false_value: Value to use where condition is False
    """
    condition: Value
    true_value: Value
    false_value: Value
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result
        # Result type is broadcast of true_value and false_value types
        raise NotImplementedError("TODO: Implement WhereOp.__post_init__")

    def children(self) -> List[IRNode]:
        return [self.condition, self.true_value, self.false_value]

    def __repr__(self) -> str:
        # TODO: Return string like "where(%cond, %true, %false) -> %result"
        raise NotImplementedError("TODO: Implement WhereOp.__repr__")


@dataclass
class CastOp(IRNode):
    """
    Cast a value to a different dtype.

    Examples:
        CastOp(x, float32)  # Cast x to float32

    Attributes:
        operand: Value to cast
        target_dtype: Target data type
    """
    operand: Value
    target_dtype: DType
    result: Value = field(init=False)

    def __post_init__(self):
        super().__init__()
        # TODO: Create self.result with same shape but new dtype
        raise NotImplementedError("TODO: Implement CastOp.__post_init__")

    def children(self) -> List[IRNode]:
        return [self.operand]

    def __repr__(self) -> str:
        # TODO: Return string like "cast(%x, float32) -> %y"
        raise NotImplementedError("TODO: Implement CastOp.__repr__")


@dataclass
class Block:
    """
    A basic block in the IR - a sequence of operations.

    In our simplified IR, a Block is a linear sequence of operations
    (no control flow within a block).

    Attributes:
        ops: List of operations in this block
        args: Block arguments (like function parameters)
    """
    ops: List[IRNode] = field(default_factory=list)
    args: List[Value] = field(default_factory=list)

    def add_op(self, op: IRNode) -> None:
        """Add an operation to this block."""
        # TODO: Append op to self.ops
        raise NotImplementedError("TODO: Implement Block.add_op")

    def __repr__(self) -> str:
        # TODO: Return a multi-line string showing all ops
        raise NotImplementedError("TODO: Implement Block.__repr__")


@dataclass
class Function:
    """
    A function (kernel) in the IR.

    Represents a Triton kernel function.

    Attributes:
        name: Function name
        params: List of (name, type) tuples for parameters
        body: The function body as a Block
        constexpr_params: Set of parameter names that are constexpr
    """
    name: str
    params: List[Tuple[str, Type]]
    body: Block
    constexpr_params: set = field(default_factory=set)

    def get_param_value(self, name: str) -> Optional[Value]:
        """Get the Value for a parameter by name."""
        # TODO: Find and return the Value in body.args matching the name
        raise NotImplementedError("TODO: Implement Function.get_param_value")

    def __repr__(self) -> str:
        # TODO: Return a function representation like:
        # def kernel_name(param1: type1, param2: type2):
        #     <body ops>
        raise NotImplementedError("TODO: Implement Function.__repr__")
