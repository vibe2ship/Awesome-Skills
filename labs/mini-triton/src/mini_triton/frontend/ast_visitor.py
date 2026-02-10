"""
Python AST to IR conversion for Mini-Triton.

This module implements the "tracing" mechanism that converts a Python
function decorated with @jit into Mini-Triton IR.

How tracing works:
1. Parse the Python function's AST
2. Walk the AST, converting Python constructs to IR operations
3. Handle control flow (if/for) specially
4. Return the constructed IR Function

Key challenges:
- Python is dynamic, IR is static
- Need to handle constexpr parameters (compile-time constants)
- Need to resolve tl.* calls to IR operations
- Need to track variable bindings

TODO: Implement the AST visitor
"""

from __future__ import annotations
import ast
import inspect
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

from mini_triton.ir.types import Type, DType, ScalarType, BlockType, PointerType, int32, float32
from mini_triton.ir.ops import Function, Block, Value
from mini_triton.ir.builder import IRBuilder


@dataclass
class FunctionSignature:
    """
    Parsed signature of a Triton kernel function.

    Attributes:
        name: Function name
        params: List of (name, annotation) tuples
        constexpr_params: Set of parameter names marked as constexpr
    """
    name: str
    params: List[Tuple[str, Optional[str]]]
    constexpr_params: set


def parse_signature(func: Callable) -> FunctionSignature:
    """
    Parse a Python function's signature.

    Extracts parameter names, type annotations, and identifies
    constexpr parameters.

    Args:
        func: The function to parse

    Returns:
        Parsed FunctionSignature
    """
    # TODO:
    # 1. Get function signature using inspect.signature
    # 2. Extract parameter names and annotations
    # 3. Identify constexpr parameters (annotated with mt.constexpr)
    # 4. Return FunctionSignature
    raise NotImplementedError("TODO: Implement parse_signature")


class ASTVisitor(ast.NodeVisitor):
    """
    AST visitor that converts Python code to Mini-Triton IR.

    This is the core of the frontend - it walks the Python AST
    and generates IR operations.

    Example:
        def my_kernel(x_ptr, n):
            pid = tl.program_id(0)  # -> ProgramIdOp
            x = tl.load(x_ptr)      # -> LoadOp

    The visitor converts each Python statement/expression into
    corresponding IR operations.
    """

    def __init__(
        self,
        builder: IRBuilder,
        constexpr_values: Dict[str, Any],
        local_namespace: Dict[str, Any]
    ):
        """
        Initialize the AST visitor.

        Args:
            builder: IRBuilder to use for constructing IR
            constexpr_values: Compile-time constant values
            local_namespace: Namespace for resolving names (tl, etc.)
        """
        # TODO:
        # self._builder = builder
        # self._constexpr = constexpr_values
        # self._namespace = local_namespace
        # self._variables: Dict[str, Value] = {}  # Python var name -> IR Value
        raise NotImplementedError("TODO: Implement ASTVisitor.__init__")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a function definition.

        This is the entry point - it sets up parameters and visits the body.
        """
        # TODO:
        # 1. Create IR function using builder.create_function
        # 2. Map parameter names to IR Values in self._variables
        # 3. Visit each statement in the function body
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_FunctionDef")

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Visit an assignment statement.

        Example: x = tl.load(ptr)
        """
        # TODO:
        # 1. Evaluate the right-hand side (visit node.value)
        # 2. For each target, store the value in self._variables
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Assign")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """
        Visit an augmented assignment (+=, *=, etc.).

        Example: acc += x
        """
        # TODO:
        # 1. Get current value of target
        # 2. Apply the operation
        # 3. Store result back
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_AugAssign")

    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Visit an expression statement.

        Example: tl.store(ptr, value)
        """
        # TODO: Just visit the expression (for side effects like store)
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Expr")

    def visit_For(self, node: ast.For) -> None:
        """
        Visit a for loop.

        For loops are unrolled at compile time if the range is constexpr.

        Example:
            for k in range(0, K, BLOCK_K):  # BLOCK_K is constexpr
                ...

        This gets unrolled into multiple copies of the body.
        """
        # TODO:
        # 1. Check if the range is constexpr
        # 2. If yes, unroll the loop
        # 3. If no, raise an error (we don't support dynamic loops)
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_For")

    def visit_If(self, node: ast.If) -> None:
        """
        Visit an if statement.

        If the condition is constexpr, we can evaluate it at compile time.
        Otherwise, we need to generate a select/where operation.
        """
        # TODO:
        # 1. Evaluate condition
        # 2. If constexpr, take appropriate branch
        # 3. If runtime, generate appropriate control flow
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_If")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """
        Visit a binary operation (+, -, *, /, etc.).

        Example: x + y, pid * BLOCK_SIZE
        """
        # TODO:
        # 1. Visit left and right operands
        # 2. If both are constexpr, compute at compile time
        # 3. Otherwise, generate IR binary operation
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_BinOp")

    def visit_Compare(self, node: ast.Compare) -> Any:
        """
        Visit a comparison operation (<, >, ==, etc.).

        Example: offsets < n
        """
        # TODO:
        # 1. Visit left operand
        # 2. For each comparator, generate comparison
        # 3. Chain multiple comparisons with AND
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Compare")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """
        Visit a unary operation (-, not, etc.).

        Example: -x, not mask
        """
        # TODO:
        # 1. Visit operand
        # 2. Generate appropriate unary operation
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_UnaryOp")

    def visit_Call(self, node: ast.Call) -> Any:
        """
        Visit a function call.

        This handles tl.* calls like tl.load, tl.store, tl.arange, etc.

        Example: tl.load(ptr, mask=mask)
        """
        # TODO:
        # 1. Resolve the function being called
        # 2. Evaluate arguments
        # 3. If it's a tl.* function, generate appropriate IR
        # 4. If it's a builtin (range, etc.), handle specially
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Call")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """
        Visit an attribute access.

        Example: tl.float32, x.to(dtype)
        """
        # TODO:
        # 1. Visit the value (e.g., 'tl')
        # 2. Get the attribute (e.g., 'float32')
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Attribute")

    def visit_Name(self, node: ast.Name) -> Any:
        """
        Visit a name (variable reference).

        Example: x, pid, BLOCK_SIZE
        """
        # TODO:
        # 1. Check if it's in self._variables (IR value)
        # 2. Check if it's in self._constexpr (compile-time constant)
        # 3. Check if it's in self._namespace (tl, etc.)
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Name")

    def visit_Constant(self, node: ast.Constant) -> Any:
        """
        Visit a constant literal.

        Example: 0, 128, 3.14
        """
        # TODO: Return the constant value (for constexpr evaluation)
        # or create an IR constant (for runtime)
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Constant")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """
        Visit a subscript operation.

        Example: x[i], ptr[mask]
        """
        # TODO: This is complex - may need special handling
        raise NotImplementedError("TODO: Implement ASTVisitor.visit_Subscript")


def trace_function(
    func: Callable,
    constexpr_values: Dict[str, Any],
    param_types: Dict[str, Type]
) -> Function:
    """
    Trace a Python function and convert it to IR.

    This is the main entry point for the frontend.

    Args:
        func: The Python function to trace
        constexpr_values: Values for constexpr parameters
        param_types: Types for each parameter

    Returns:
        The IR Function

    Example:
        @jit
        def my_kernel(x_ptr, n, BLOCK: constexpr):
            ...

        ir = trace_function(
            my_kernel,
            constexpr_values={"BLOCK": 128},
            param_types={"x_ptr": PointerType(float32), "n": ScalarType(int32)}
        )
    """
    # TODO:
    # 1. Parse the function signature
    # 2. Get the function's source code and parse AST
    # 3. Create IRBuilder
    # 4. Create ASTVisitor
    # 5. Visit the function AST
    # 6. Return the IR Function
    raise NotImplementedError("TODO: Implement trace_function")
