"""
IR Builder for Mini-Triton.

The IRBuilder provides a convenient API for constructing IR.
It tracks the current insertion point and creates Values automatically.

Usage example:
    builder = IRBuilder()
    func = builder.create_function("my_kernel", [("x_ptr", ptr_type), ("n", int32_type)])

    pid = builder.program_id(0)
    offsets = builder.arange(0, 128)
    ptr = builder.add(x_ptr, offsets)
    x = builder.load(ptr)
    ...

TODO: Implement the IRBuilder methods below
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union, Any

from mini_triton.ir.types import (
    Type, DType, ScalarType, BlockType, PointerType,
    float32, int32, bool_
)
from mini_triton.ir.ops import (
    IRNode, Value, Constant, BinaryOp, UnaryOp, LoadOp, StoreOp,
    DotOp, ReduceOp, MakeRangeOp, BroadcastOp, ProgramIdOp,
    WhereOp, CastOp, Function, Block,
    BinaryOpKind, UnaryOpKind, ReduceOpKind,
)


class IRBuilder:
    """
    Builder for constructing Mini-Triton IR.

    The builder maintains:
    - Current function being built
    - Current block for inserting ops
    - Symbol table for variable names

    Typical usage:
        builder = IRBuilder()
        func = builder.create_function(...)
        builder.set_insertion_point(func.body)
        ... add ops ...
    """

    def __init__(self):
        # TODO: Initialize:
        # - self._current_block: Optional[Block] = None
        # - self._current_function: Optional[Function] = None
        # - self._symbol_table: Dict[str, Value] = {}
        raise NotImplementedError("TODO: Implement IRBuilder.__init__")

    def create_function(
        self,
        name: str,
        params: List[Tuple[str, Type]],
        constexpr_params: Optional[set] = None
    ) -> Function:
        """
        Create a new function and set it as the current function.

        Args:
            name: Function name
            params: List of (param_name, param_type) tuples
            constexpr_params: Set of parameter names that are compile-time constants

        Returns:
            The created Function
        """
        # TODO:
        # 1. Create a Block for the function body
        # 2. Create Value objects for each parameter and add to block.args
        # 3. Add parameters to symbol table
        # 4. Create Function object
        # 5. Set as current function and block
        raise NotImplementedError("TODO: Implement IRBuilder.create_function")

    def set_insertion_point(self, block: Block) -> None:
        """Set the block where new operations will be inserted."""
        # TODO: Set self._current_block = block
        raise NotImplementedError("TODO: Implement IRBuilder.set_insertion_point")

    def get_value(self, name: str) -> Optional[Value]:
        """Look up a value by name in the symbol table."""
        # TODO: Return self._symbol_table.get(name)
        raise NotImplementedError("TODO: Implement IRBuilder.get_value")

    def set_value(self, name: str, value: Value) -> None:
        """Add or update a value in the symbol table."""
        # TODO: Set self._symbol_table[name] = value
        raise NotImplementedError("TODO: Implement IRBuilder.set_value")

    def _add_op(self, op: IRNode) -> IRNode:
        """Add an operation to the current block."""
        # TODO: Add op to current block and return it
        raise NotImplementedError("TODO: Implement IRBuilder._add_op")

    # ========== Constant Creation ==========

    def constant(self, value: Any, dtype: DType) -> Value:
        """Create a constant value."""
        # TODO:
        # 1. Create Constant node
        # 2. Add to current block
        # 3. Return the result Value
        raise NotImplementedError("TODO: Implement IRBuilder.constant")

    def zeros(self, shape: Tuple[int, ...], dtype: DType) -> Value:
        """Create a block of zeros."""
        # TODO: Create a constant block filled with zeros
        raise NotImplementedError("TODO: Implement IRBuilder.zeros")

    def full(self, shape: Tuple[int, ...], fill_value: Any, dtype: DType) -> Value:
        """Create a block filled with a value."""
        # TODO: Create a constant block filled with fill_value
        raise NotImplementedError("TODO: Implement IRBuilder.full")

    # ========== Program ID ==========

    def program_id(self, axis: int) -> Value:
        """
        Get the program ID for the given axis.

        Args:
            axis: Axis (0, 1, or 2)

        Returns:
            Scalar int32 value containing the program ID
        """
        # TODO:
        # 1. Create ProgramIdOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder.program_id")

    # ========== Range/Indexing ==========

    def arange(self, start: int, end: int) -> Value:
        """
        Create a 1D range of integers.

        Equivalent to tl.arange(start, end).

        Args:
            start: Start value (inclusive)
            end: End value (exclusive)

        Returns:
            1D block of int32 values [start, start+1, ..., end-1]
        """
        # TODO:
        # 1. Create MakeRangeOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder.arange")

    # ========== Binary Operations ==========

    def _binary_op(self, op: BinaryOpKind, lhs: Value, rhs: Value) -> Value:
        """Helper to create a binary operation."""
        # TODO:
        # 1. Create BinaryOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder._binary_op")

    def add(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise addition."""
        return self._binary_op(BinaryOpKind.ADD, lhs, rhs)

    def sub(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise subtraction."""
        return self._binary_op(BinaryOpKind.SUB, lhs, rhs)

    def mul(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise multiplication."""
        return self._binary_op(BinaryOpKind.MUL, lhs, rhs)

    def div(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise division."""
        return self._binary_op(BinaryOpKind.DIV, lhs, rhs)

    def mod(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise modulo."""
        return self._binary_op(BinaryOpKind.MOD, lhs, rhs)

    def maximum(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise maximum."""
        return self._binary_op(BinaryOpKind.MAX, lhs, rhs)

    def minimum(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise minimum."""
        return self._binary_op(BinaryOpKind.MIN, lhs, rhs)

    # Comparison operations
    def eq(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise equality comparison."""
        return self._binary_op(BinaryOpKind.EQ, lhs, rhs)

    def ne(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise not-equal comparison."""
        return self._binary_op(BinaryOpKind.NE, lhs, rhs)

    def lt(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise less-than comparison."""
        return self._binary_op(BinaryOpKind.LT, lhs, rhs)

    def le(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise less-than-or-equal comparison."""
        return self._binary_op(BinaryOpKind.LE, lhs, rhs)

    def gt(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise greater-than comparison."""
        return self._binary_op(BinaryOpKind.GT, lhs, rhs)

    def ge(self, lhs: Value, rhs: Value) -> Value:
        """Element-wise greater-than-or-equal comparison."""
        return self._binary_op(BinaryOpKind.GE, lhs, rhs)

    # ========== Unary Operations ==========

    def _unary_op(self, op: UnaryOpKind, operand: Value) -> Value:
        """Helper to create a unary operation."""
        # TODO:
        # 1. Create UnaryOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder._unary_op")

    def neg(self, operand: Value) -> Value:
        """Negation."""
        return self._unary_op(UnaryOpKind.NEG, operand)

    def exp(self, operand: Value) -> Value:
        """Exponential."""
        return self._unary_op(UnaryOpKind.EXP, operand)

    def log(self, operand: Value) -> Value:
        """Natural logarithm."""
        return self._unary_op(UnaryOpKind.LOG, operand)

    def sqrt(self, operand: Value) -> Value:
        """Square root."""
        return self._unary_op(UnaryOpKind.SQRT, operand)

    def abs(self, operand: Value) -> Value:
        """Absolute value."""
        return self._unary_op(UnaryOpKind.ABS, operand)

    # ========== Memory Operations ==========

    def load(
        self,
        ptr: Value,
        mask: Optional[Value] = None,
        other: Optional[Value] = None
    ) -> Value:
        """
        Load values from memory.

        Args:
            ptr: Pointer to load from
            mask: Optional mask (boolean, same shape as load)
            other: Value to use for masked elements

        Returns:
            Loaded values
        """
        # TODO:
        # 1. Create LoadOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder.load")

    def store(
        self,
        ptr: Value,
        value: Value,
        mask: Optional[Value] = None
    ) -> None:
        """
        Store values to memory.

        Args:
            ptr: Pointer to store to
            value: Values to store
            mask: Optional mask (only store where True)
        """
        # TODO:
        # 1. Create StoreOp
        # 2. Add to current block
        raise NotImplementedError("TODO: Implement IRBuilder.store")

    # ========== Dot Product ==========

    def dot(
        self,
        lhs: Value,
        rhs: Value,
        acc: Optional[Value] = None
    ) -> Value:
        """
        Matrix multiplication.

        Args:
            lhs: Left matrix (M, K)
            rhs: Right matrix (K, N)
            acc: Optional accumulator (M, N) for fused multiply-add

        Returns:
            Result matrix (M, N)
        """
        # TODO:
        # 1. Create DotOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder.dot")

    # ========== Reductions ==========

    def _reduce_op(self, op: ReduceOpKind, operand: Value, axis: int) -> Value:
        """Helper to create a reduce operation."""
        # TODO:
        # 1. Create ReduceOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder._reduce_op")

    def sum(self, operand: Value, axis: int) -> Value:
        """Sum reduction along axis."""
        return self._reduce_op(ReduceOpKind.SUM, operand, axis)

    def max(self, operand: Value, axis: int) -> Value:
        """Max reduction along axis."""
        return self._reduce_op(ReduceOpKind.MAX, operand, axis)

    def min(self, operand: Value, axis: int) -> Value:
        """Min reduction along axis."""
        return self._reduce_op(ReduceOpKind.MIN, operand, axis)

    # ========== Other Operations ==========

    def broadcast(self, operand: Value, shape: Tuple[int, ...]) -> Value:
        """Broadcast a value to a larger shape."""
        # TODO:
        # 1. Create BroadcastOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder.broadcast")

    def where(self, condition: Value, true_value: Value, false_value: Value) -> Value:
        """Select values based on condition."""
        # TODO:
        # 1. Create WhereOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder.where")

    def cast(self, operand: Value, dtype: DType) -> Value:
        """Cast a value to a different dtype."""
        # TODO:
        # 1. Create CastOp
        # 2. Add to current block
        # 3. Return result value
        raise NotImplementedError("TODO: Implement IRBuilder.cast")

    # ========== Convenience Methods ==========

    def ptr_add(self, ptr: Value, offset: Value) -> Value:
        """
        Add an offset to a pointer.

        When adding a block of offsets to a pointer, the result is
        a "block of pointers" with the offset shape.
        """
        # TODO:
        # 1. This is a BinaryOp ADD but with special type handling
        # 2. If offset is a block, result should be PointerType with offset_shape
        raise NotImplementedError("TODO: Implement IRBuilder.ptr_add")
