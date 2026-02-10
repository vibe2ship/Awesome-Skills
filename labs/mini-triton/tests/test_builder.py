"""
Tests for the IR Builder.

These tests verify that the IRBuilder correctly constructs IR.
"""

import pytest
from mini_triton.ir.types import (
    ScalarType, BlockType, PointerType,
    float32, int32, bool_
)
from mini_triton.ir.ops import (
    MakeRangeOp, ProgramIdOp, BinaryOp, UnaryOp, LoadOp, StoreOp,
    BinaryOpKind, UnaryOpKind,
)
from mini_triton.ir.builder import IRBuilder


class TestIRBuilderBasic:
    """Basic IRBuilder tests."""

    def test_builder_creation(self):
        """Test that IRBuilder can be created."""
        builder = IRBuilder()
        assert builder is not None

    def test_create_function(self):
        """Test creating a function."""
        builder = IRBuilder()
        func = builder.create_function(
            "test_kernel",
            [
                ("x_ptr", PointerType(float32)),
                ("n", ScalarType(int32)),
            ]
        )

        assert func.name == "test_kernel"
        assert len(func.params) == 2

    def test_create_function_with_constexpr(self):
        """Test creating a function with constexpr params."""
        builder = IRBuilder()
        func = builder.create_function(
            "test_kernel",
            [
                ("x_ptr", PointerType(float32)),
                ("BLOCK", ScalarType(int32)),
            ],
            constexpr_params={"BLOCK"}
        )

        assert "BLOCK" in func.constexpr_params


class TestIRBuilderConstants:
    """Tests for constant creation."""

    def test_constant_int(self):
        """Test creating integer constant."""
        builder = IRBuilder()
        builder.create_function("test", [])

        c = builder.constant(42, int32)
        assert c.type.dtype == int32

    def test_constant_float(self):
        """Test creating float constant."""
        builder = IRBuilder()
        builder.create_function("test", [])

        c = builder.constant(3.14, float32)
        assert c.type.dtype == float32

    def test_zeros(self):
        """Test creating zeros block."""
        builder = IRBuilder()
        builder.create_function("test", [])

        z = builder.zeros((128,), float32)
        assert isinstance(z.type, BlockType)
        assert z.type.shape == (128,)

    def test_full(self):
        """Test creating block filled with value."""
        builder = IRBuilder()
        builder.create_function("test", [])

        f = builder.full((64, 64), 1.0, float32)
        assert isinstance(f.type, BlockType)
        assert f.type.shape == (64, 64)


class TestIRBuilderOps:
    """Tests for operation creation."""

    def test_program_id(self):
        """Test creating program_id operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        pid = builder.program_id(0)
        assert isinstance(pid.type, ScalarType)
        assert pid.type.dtype == int32

    def test_arange(self):
        """Test creating arange operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        offsets = builder.arange(0, 128)
        assert isinstance(offsets.type, BlockType)
        assert offsets.type.shape == (128,)
        assert offsets.type.dtype == int32

    def test_add(self):
        """Test creating add operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        a = builder.constant(1, int32)
        b = builder.constant(2, int32)
        c = builder.add(a, b)

        assert isinstance(c.type, ScalarType)

    def test_add_block(self):
        """Test adding blocks."""
        builder = IRBuilder()
        builder.create_function("test", [])

        a = builder.arange(0, 128)
        b = builder.constant(10, int32)
        c = builder.add(a, b)

        assert isinstance(c.type, BlockType)
        assert c.type.shape == (128,)

    def test_mul(self):
        """Test creating mul operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        a = builder.constant(3, int32)
        b = builder.constant(4, int32)
        c = builder.mul(a, b)

        assert isinstance(c.type, ScalarType)

    def test_comparison(self):
        """Test creating comparison operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        a = builder.arange(0, 128)
        b = builder.constant(100, int32)
        mask = builder.lt(a, b)

        assert isinstance(mask.type, BlockType)
        assert mask.type.dtype == bool_


class TestIRBuilderMemory:
    """Tests for memory operations."""

    def test_load(self):
        """Test creating load operation."""
        builder = IRBuilder()
        func = builder.create_function(
            "test",
            [("ptr", PointerType(float32, (128,)))]
        )

        ptr = builder.get_value("ptr")
        x = builder.load(ptr)

        assert isinstance(x.type, BlockType)
        assert x.type.shape == (128,)

    def test_load_with_mask(self):
        """Test creating masked load."""
        builder = IRBuilder()
        func = builder.create_function(
            "test",
            [("ptr", PointerType(float32, (128,)))]
        )

        ptr = builder.get_value("ptr")
        mask = builder.lt(builder.arange(0, 128), builder.constant(100, int32))
        other = builder.constant(0.0, float32)
        x = builder.load(ptr, mask=mask, other=other)

        assert isinstance(x.type, BlockType)

    def test_store(self):
        """Test creating store operation."""
        builder = IRBuilder()
        func = builder.create_function(
            "test",
            [("ptr", PointerType(float32, (128,)))]
        )

        ptr = builder.get_value("ptr")
        value = builder.zeros((128,), float32)
        builder.store(ptr, value)

        # Store doesn't return a value
        # Just check it doesn't raise

    def test_store_with_mask(self):
        """Test creating masked store."""
        builder = IRBuilder()
        func = builder.create_function(
            "test",
            [("ptr", PointerType(float32, (128,)))]
        )

        ptr = builder.get_value("ptr")
        value = builder.zeros((128,), float32)
        mask = builder.lt(builder.arange(0, 128), builder.constant(100, int32))
        builder.store(ptr, value, mask=mask)


class TestIRBuilderUnary:
    """Tests for unary operations."""

    def test_neg(self):
        """Test negation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        x = builder.constant(5.0, float32)
        y = builder.neg(x)

        assert y.type.dtype == float32

    def test_exp(self):
        """Test exponential."""
        builder = IRBuilder()
        builder.create_function("test", [])

        x = builder.arange(0, 128)
        x = builder.cast(x, float32)
        y = builder.exp(x)

        assert isinstance(y.type, BlockType)
        assert y.type.dtype == float32

    def test_log(self):
        """Test logarithm."""
        builder = IRBuilder()
        builder.create_function("test", [])

        x = builder.constant(2.718, float32)
        y = builder.log(x)

        assert y.type.dtype == float32


class TestIRBuilderReduction:
    """Tests for reduction operations."""

    def test_sum(self):
        """Test sum reduction."""
        builder = IRBuilder()
        builder.create_function("test", [])

        x = builder.zeros((128,), float32)
        y = builder.sum(x, axis=0)

        assert isinstance(y.type, ScalarType)

    def test_max(self):
        """Test max reduction."""
        builder = IRBuilder()
        builder.create_function("test", [])

        x = builder.zeros((64, 128), float32)
        y = builder.max(x, axis=1)

        assert isinstance(y.type, BlockType)
        assert y.type.shape == (64,)


class TestIRBuilderDot:
    """Tests for dot product operation."""

    def test_dot(self):
        """Test matrix multiplication."""
        builder = IRBuilder()
        builder.create_function("test", [])

        a = builder.zeros((64, 32), float32)
        b = builder.zeros((32, 128), float32)
        c = builder.dot(a, b)

        assert isinstance(c.type, BlockType)
        assert c.type.shape == (64, 128)

    def test_dot_with_acc(self):
        """Test dot with accumulator."""
        builder = IRBuilder()
        builder.create_function("test", [])

        a = builder.zeros((64, 32), float32)
        b = builder.zeros((32, 128), float32)
        acc = builder.zeros((64, 128), float32)
        c = builder.dot(a, b, acc=acc)

        assert c.type.shape == (64, 128)


class TestIRBuilderOther:
    """Tests for other operations."""

    def test_broadcast(self):
        """Test broadcast operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        x = builder.constant(1.0, float32)
        y = builder.broadcast(x, (128,))

        assert isinstance(y.type, BlockType)
        assert y.type.shape == (128,)

    def test_where(self):
        """Test where operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        cond = builder.lt(builder.arange(0, 128), builder.constant(64, int32))
        true_val = builder.zeros((128,), float32)
        false_val = builder.full((128,), 1.0, float32)
        result = builder.where(cond, true_val, false_val)

        assert isinstance(result.type, BlockType)
        assert result.type.shape == (128,)

    def test_cast(self):
        """Test cast operation."""
        builder = IRBuilder()
        builder.create_function("test", [])

        x = builder.arange(0, 128)  # int32
        y = builder.cast(x, float32)

        assert y.type.dtype == float32
        assert y.type.shape == (128,)

    def test_ptr_add(self):
        """Test pointer arithmetic."""
        builder = IRBuilder()
        func = builder.create_function(
            "test",
            [("ptr", PointerType(float32))]
        )

        ptr = builder.get_value("ptr")
        offsets = builder.arange(0, 128)
        ptr_block = builder.ptr_add(ptr, offsets)

        assert isinstance(ptr_block.type, PointerType)
        assert ptr_block.type.offset_shape == (128,)
