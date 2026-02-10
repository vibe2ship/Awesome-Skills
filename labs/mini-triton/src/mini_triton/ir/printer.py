"""
IR Printer for Mini-Triton.

Pretty-prints the IR for debugging and visualization.
This is essential for understanding what the compiler is doing.

Example output:
    func @vector_add(%x_ptr: ptr<float32>, %y_ptr: ptr<float32>, %out_ptr: ptr<float32>, %n: int32) {
      %0 = program_id(0) : int32
      %1 = constant(128) : int32
      %2 = mul(%0, %1) : int32
      %3 = arange(0, 128) : block<int32, [128]>
      %4 = add(%2, %3) : block<int32, [128]>
      %5 = lt(%4, %n) : block<bool, [128]>
      %6 = ptr_add(%x_ptr, %4) : ptr<float32, [128]>
      %7 = load(%6, mask=%5) : block<float32, [128]>
      ...
    }

TODO: Implement the IRPrinter methods below
"""

from __future__ import annotations
from typing import List, Dict, Optional
from io import StringIO

from mini_triton.ir.types import Type, DType, ScalarType, BlockType, PointerType
from mini_triton.ir.ops import (
    IRNode, Value, Constant, BinaryOp, UnaryOp, LoadOp, StoreOp,
    DotOp, ReduceOp, MakeRangeOp, BroadcastOp, ProgramIdOp,
    WhereOp, CastOp, Function, Block,
    BinaryOpKind, UnaryOpKind, ReduceOpKind,
)


class IRPrinter:
    """
    Pretty-printer for Mini-Triton IR.

    Can print individual operations, blocks, or entire functions.
    Useful for debugging and understanding compiler transformations.
    """

    def __init__(self, indent_size: int = 2):
        # TODO: Initialize:
        # - self._indent_size = indent_size
        # - self._value_names: Dict[int, str] = {}  # value.id -> name
        # - self._name_counter = 0
        raise NotImplementedError("TODO: Implement IRPrinter.__init__")

    def _get_value_name(self, value: Value) -> str:
        """
        Get or create a name for a value.

        Named values use their name, anonymous values get %0, %1, etc.
        """
        # TODO:
        # 1. If value already has a name mapping, return it
        # 2. If value.name is set, use that (prefixed with %)
        # 3. Otherwise, generate %N where N is incremented
        raise NotImplementedError("TODO: Implement IRPrinter._get_value_name")

    def _format_type(self, ty: Type) -> str:
        """Format a type as a string."""
        # TODO:
        # - ScalarType(float32) -> "float32"
        # - BlockType(float32, (128,)) -> "block<float32, [128]>"
        # - PointerType(float32) -> "ptr<float32>"
        # - PointerType(float32, (128,)) -> "ptr<float32, [128]>"
        raise NotImplementedError("TODO: Implement IRPrinter._format_type")

    def _format_binary_op(self, op: BinaryOpKind) -> str:
        """Format a binary operation kind as a string."""
        # TODO: Return "add", "sub", "mul", etc. based on op
        raise NotImplementedError("TODO: Implement IRPrinter._format_binary_op")

    def _format_unary_op(self, op: UnaryOpKind) -> str:
        """Format a unary operation kind as a string."""
        # TODO: Return "neg", "exp", "log", etc. based on op
        raise NotImplementedError("TODO: Implement IRPrinter._format_unary_op")

    def _format_reduce_op(self, op: ReduceOpKind) -> str:
        """Format a reduce operation kind as a string."""
        # TODO: Return "sum", "max", "min", "prod" based on op
        raise NotImplementedError("TODO: Implement IRPrinter._format_reduce_op")

    def print_value(self, value: Value) -> str:
        """Print a value reference."""
        # TODO: Return the value name like "%0" or "%x"
        raise NotImplementedError("TODO: Implement IRPrinter.print_value")

    def print_constant(self, const: Constant) -> str:
        """Print a constant."""
        # TODO: Return "constant(value) : type"
        raise NotImplementedError("TODO: Implement IRPrinter.print_constant")

    def print_op(self, op: IRNode) -> str:
        """
        Print a single operation.

        Dispatches to the appropriate print method based on op type.
        """
        # TODO: Pattern match on op type and call appropriate method
        # Example outputs:
        # - BinaryOp: "%2 = add(%0, %1) : block<float32, [128]>"
        # - LoadOp: "%5 = load(%4, mask=%3) : block<float32, [128]>"
        # - StoreOp: "store(%ptr, %val, mask=%mask)"
        raise NotImplementedError("TODO: Implement IRPrinter.print_op")

    def print_block(self, block: Block, indent: int = 0) -> str:
        """Print a basic block."""
        # TODO:
        # 1. Print block arguments if any
        # 2. Print each operation with proper indentation
        raise NotImplementedError("TODO: Implement IRPrinter.print_block")

    def print_function(self, func: Function) -> str:
        """
        Print a complete function.

        Example:
            func @kernel_name(%arg0: ptr<float32>, %arg1: int32) {
              %0 = program_id(0) : int32
              ...
            }
        """
        # TODO:
        # 1. Print function signature
        # 2. Print body block with indentation
        # 3. Close with "}"
        raise NotImplementedError("TODO: Implement IRPrinter.print_function")

    def __call__(self, ir: IRNode | Function | Block) -> str:
        """
        Print any IR object.

        Convenience method to print functions, blocks, or ops.
        """
        # TODO: Dispatch to appropriate method based on type
        raise NotImplementedError("TODO: Implement IRPrinter.__call__")


def print_ir(ir: IRNode | Function | Block) -> str:
    """Convenience function to print IR."""
    printer = IRPrinter()
    return printer(ir)


def dump_ir(ir: IRNode | Function | Block) -> None:
    """Print IR to stdout."""
    print(print_ir(ir))
