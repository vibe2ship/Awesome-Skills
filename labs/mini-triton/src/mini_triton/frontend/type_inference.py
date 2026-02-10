"""
Type Inference for Mini-Triton IR.

This module implements type inference and checking for the IR.
It ensures all operations have consistent types and computes
result types for operations.

Type inference runs after AST conversion and before code generation.
It's responsible for:
1. Inferring types for values that don't have explicit types
2. Checking that operations have compatible operand types
3. Propagating shape information through the IR

TODO: Implement type inference
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass

from mini_triton.ir.types import (
    Type, DType, ScalarType, BlockType, PointerType,
    get_broadcast_shape, infer_binary_op_type
)
from mini_triton.ir.ops import (
    IRNode, Value, Constant, BinaryOp, UnaryOp, LoadOp, StoreOp,
    DotOp, ReduceOp, MakeRangeOp, BroadcastOp, ProgramIdOp,
    WhereOp, CastOp, Function, Block
)


@dataclass
class TypeError:
    """Represents a type error found during inference."""
    message: str
    node: IRNode


class TypeInference:
    """
    Type inference and checking for Mini-Triton IR.

    Walks the IR and ensures all types are consistent.
    Reports errors if there are type mismatches.
    """

    def __init__(self):
        # TODO:
        # self._errors: List[TypeError] = []
        raise NotImplementedError("TODO: Implement TypeInference.__init__")

    @property
    def errors(self) -> List[TypeError]:
        """Return the list of type errors."""
        # TODO: Return self._errors
        raise NotImplementedError("TODO: Implement TypeInference.errors")

    def infer_function(self, func: Function) -> bool:
        """
        Run type inference on a function.

        Args:
            func: The function to type check

        Returns:
            True if no errors, False if there are type errors
        """
        # TODO:
        # 1. Clear any previous errors
        # 2. Visit each operation in the function body
        # 3. Return len(self._errors) == 0
        raise NotImplementedError("TODO: Implement TypeInference.infer_function")

    def _infer_op(self, op: IRNode) -> None:
        """Infer types for a single operation."""
        # TODO: Dispatch to specific methods based on op type
        raise NotImplementedError("TODO: Implement TypeInference._infer_op")

    def _infer_binary_op(self, op: BinaryOp) -> None:
        """
        Infer types for a binary operation.

        Checks that operand types are compatible and infers result type.
        """
        # TODO:
        # 1. Get types of lhs and rhs
        # 2. Check compatibility using infer_binary_op_type
        # 3. Set result type
        # 4. Report errors if types are incompatible
        raise NotImplementedError("TODO: Implement TypeInference._infer_binary_op")

    def _infer_unary_op(self, op: UnaryOp) -> None:
        """Infer types for a unary operation."""
        # TODO: Result type is same as operand type (usually)
        raise NotImplementedError("TODO: Implement TypeInference._infer_unary_op")

    def _infer_load_op(self, op: LoadOp) -> None:
        """
        Infer types for a load operation.

        Result type depends on pointer type:
        - Scalar pointer -> Scalar result
        - Block pointer -> Block result (with pointer's offset shape)
        """
        # TODO:
        # 1. Check that ptr is a PointerType
        # 2. If ptr has offset_shape, result is BlockType
        # 3. Otherwise, result is ScalarType
        # 4. If mask provided, check it's boolean with matching shape
        raise NotImplementedError("TODO: Implement TypeInference._infer_load_op")

    def _infer_store_op(self, op: StoreOp) -> None:
        """Infer types for a store operation (no result, just check)."""
        # TODO:
        # 1. Check ptr is PointerType
        # 2. Check value type matches pointer's pointee type
        # 3. Check mask (if provided) is boolean with matching shape
        raise NotImplementedError("TODO: Implement TypeInference._infer_store_op")

    def _infer_dot_op(self, op: DotOp) -> None:
        """
        Infer types for a dot (matrix multiply) operation.

        Checks that shapes are compatible: (M, K) @ (K, N) -> (M, N)
        """
        # TODO:
        # 1. Check both operands are 2D BlockTypes
        # 2. Check inner dimensions match (lhs columns == rhs rows)
        # 3. Result shape is (lhs rows, rhs columns)
        raise NotImplementedError("TODO: Implement TypeInference._infer_dot_op")

    def _infer_reduce_op(self, op: ReduceOp) -> None:
        """Infer types for a reduce operation."""
        # TODO:
        # 1. Check operand is BlockType
        # 2. Check axis is valid
        # 3. Result shape has reduced axis removed
        raise NotImplementedError("TODO: Implement TypeInference._infer_reduce_op")

    def _infer_make_range_op(self, op: MakeRangeOp) -> None:
        """Infer types for a make_range (arange) operation."""
        # TODO: Result is BlockType(int32, (end - start,))
        raise NotImplementedError("TODO: Implement TypeInference._infer_make_range_op")

    def _infer_broadcast_op(self, op: BroadcastOp) -> None:
        """Infer types for a broadcast operation."""
        # TODO:
        # 1. Check that broadcast is valid (shapes are compatible)
        # 2. Set result type with new shape
        raise NotImplementedError("TODO: Implement TypeInference._infer_broadcast_op")

    def _infer_where_op(self, op: WhereOp) -> None:
        """Infer types for a where operation."""
        # TODO:
        # 1. Check condition is boolean
        # 2. Check true/false values have compatible types
        # 3. Result is broadcast of true/false types
        raise NotImplementedError("TODO: Implement TypeInference._infer_where_op")

    def _infer_cast_op(self, op: CastOp) -> None:
        """Infer types for a cast operation."""
        # TODO: Result has same shape, different dtype
        raise NotImplementedError("TODO: Implement TypeInference._infer_cast_op")

    def _add_error(self, message: str, node: IRNode) -> None:
        """Add a type error."""
        # TODO: Append TypeError to self._errors
        raise NotImplementedError("TODO: Implement TypeInference._add_error")


def type_check(func: Function) -> List[TypeError]:
    """
    Convenience function to type check a function.

    Args:
        func: Function to check

    Returns:
        List of type errors (empty if no errors)
    """
    inference = TypeInference()
    inference.infer_function(func)
    return inference.errors
