"""
Base Code Generator for Mini-Triton.

Defines the interface that all code generators must implement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any

from mini_triton.ir.ops import Function


class CodeGenerator(ABC):
    """
    Abstract base class for code generators.

    Code generators convert IR to executable code for a specific target.
    """

    @abstractmethod
    def generate(self, func: Function) -> str:
        """
        Generate code for a function.

        Args:
            func: The IR function to generate code for

        Returns:
            Generated code as a string
        """
        pass

    @abstractmethod
    def compile(self, func: Function) -> Callable:
        """
        Compile a function to executable code.

        Args:
            func: The IR function to compile

        Returns:
            A callable that executes the kernel
        """
        pass


class CompilationResult:
    """
    Result of compiling a kernel.

    Contains the generated code and the callable kernel.
    """

    def __init__(self, code: str, kernel: Callable, metadata: Dict[str, Any]):
        """
        Initialize compilation result.

        Args:
            code: Generated source code
            kernel: Compiled kernel function
            metadata: Additional metadata (e.g., register usage, etc.)
        """
        # TODO:
        # self.code = code
        # self.kernel = kernel
        # self.metadata = metadata
        raise NotImplementedError("TODO: Implement CompilationResult.__init__")

    def __call__(self, *args, **kwargs):
        """Execute the kernel."""
        # TODO: Call self.kernel with args
        raise NotImplementedError("TODO: Implement CompilationResult.__call__")
