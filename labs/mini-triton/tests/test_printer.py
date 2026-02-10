"""
Tests for the IR Printer.

These tests verify that the IR printer produces correct output.
"""

import pytest
from mini_triton.ir.types import (
    ScalarType, BlockType, PointerType,
    float32, int32, bool_
)
from mini_triton.ir.ops import (
    MakeRangeOp, ProgramIdOp, Constant, Function, Block
)
from mini_triton.ir.builder import IRBuilder
from mini_triton.ir.printer import IRPrinter, print_ir


class TestIRPrinterBasic:
    """Basic IR printer tests."""

    def test_printer_creation(self):
        """Test that IRPrinter can be created."""
        printer = IRPrinter()
        assert printer is not None

    def test_format_type_scalar(self):
        """Test formatting scalar type."""
        printer = IRPrinter()
        result = printer._format_type(ScalarType(float32))
        assert "float32" in result

    def test_format_type_block(self):
        """Test formatting block type."""
        printer = IRPrinter()
        result = printer._format_type(BlockType(float32, (128,)))
        assert "float32" in result
        assert "128" in result

    def test_format_type_pointer(self):
        """Test formatting pointer type."""
        printer = IRPrinter()
        result = printer._format_type(PointerType(float32))
        assert "ptr" in result.lower() or "pointer" in result.lower()
        assert "float32" in result


class TestIRPrinterOps:
    """Tests for printing operations."""

    def test_print_constant(self):
        """Test printing constant."""
        printer = IRPrinter()
        const = Constant(42, int32)
        result = printer.print_constant(const)
        assert "42" in result

    def test_print_make_range_op(self):
        """Test printing arange operation."""
        printer = IRPrinter()
        op = MakeRangeOp(0, 128)
        result = printer.print_op(op)
        assert "0" in result
        assert "128" in result
        # Should mention arange or range
        assert "arange" in result.lower() or "range" in result.lower()

    def test_print_program_id_op(self):
        """Test printing program_id operation."""
        printer = IRPrinter()
        op = ProgramIdOp(0)
        result = printer.print_op(op)
        assert "program_id" in result.lower() or "pid" in result.lower()
        assert "0" in result


class TestIRPrinterFunction:
    """Tests for printing complete functions."""

    def test_print_simple_function(self):
        """Test printing a simple function."""
        builder = IRBuilder()
        func = builder.create_function(
            "test_kernel",
            [
                ("x_ptr", PointerType(float32)),
                ("n", ScalarType(int32)),
            ]
        )

        # Add some ops
        pid = builder.program_id(0)
        offsets = builder.arange(0, 128)

        printer = IRPrinter()
        result = printer.print_function(func)

        assert "test_kernel" in result
        assert "x_ptr" in result
        # Should have program_id and arange
        assert "program_id" in result.lower() or "pid" in result.lower()

    def test_print_function_with_constexpr(self):
        """Test printing function with constexpr."""
        builder = IRBuilder()
        func = builder.create_function(
            "kernel",
            [
                ("ptr", PointerType(float32)),
                ("BLOCK", ScalarType(int32)),
            ],
            constexpr_params={"BLOCK"}
        )

        printer = IRPrinter()
        result = printer.print_function(func)

        assert "kernel" in result
        assert "BLOCK" in result


class TestPrintIRConvenience:
    """Tests for convenience functions."""

    def test_print_ir_function(self):
        """Test print_ir with function."""
        builder = IRBuilder()
        func = builder.create_function("test", [])
        builder.program_id(0)

        result = print_ir(func)
        assert "test" in result

    def test_print_ir_op(self):
        """Test print_ir with single op."""
        op = MakeRangeOp(0, 64)
        result = print_ir(op)
        assert "64" in result
