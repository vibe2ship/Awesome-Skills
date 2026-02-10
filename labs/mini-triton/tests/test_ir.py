"""
Tests for the Mini-Triton IR operations.

These tests verify that IR nodes work correctly:
- Value creation and typing
- All operation types (BinaryOp, UnaryOp, LoadOp, etc.)
- Function and Block structures
"""

import pytest
from mini_triton.ir.types import (
    ScalarType, BlockType, PointerType,
    float32, int32, bool_
)
from mini_triton.ir.ops import (
    Value, Constant, BinaryOp, UnaryOp, LoadOp, StoreOp,
    MakeRangeOp, ProgramIdOp, DotOp, ReduceOp, BroadcastOp,
    WhereOp, CastOp, Block, Function,
    BinaryOpKind, UnaryOpKind, ReduceOpKind,
)


class TestValue:
    """Tests for Value class."""

    def test_value_creation(self):
        """Test that Values can be created with a type."""
        v = Value(ScalarType(float32))
        assert v.type == ScalarType(float32)

    def test_value_with_name(self):
        """Test Value with a name."""
        v = Value(ScalarType(float32), name="x")
        assert v.type == ScalarType(float32)

    def test_value_unique_ids(self):
        """Test that Values get unique IDs."""
        v1 = Value(ScalarType(float32))
        v2 = Value(ScalarType(float32))
        assert v1.id != v2.id

    def test_value_repr(self):
        """Test Value string representation."""
        v = Value(ScalarType(float32), name="x")
        repr_str = repr(v)
        # Should contain the value identifier
        assert "%" in repr_str or "x" in repr_str.lower()


class TestConstant:
    """Tests for Constant class."""

    def test_scalar_constant_int(self):
        """Test integer scalar constant."""
        c = Constant(42, int32)
        assert c.value == 42
        assert c.dtype == int32
        assert isinstance(c.type, ScalarType)

    def test_scalar_constant_float(self):
        """Test float scalar constant."""
        c = Constant(3.14, float32)
        assert c.value == 3.14
        assert c.dtype == float32

    def test_block_constant(self):
        """Test block (array) constant."""
        c = Constant([1, 2, 3, 4], int32)
        assert c.value == [1, 2, 3, 4]
        # Block constant should have BlockType
        assert isinstance(c.type, BlockType)
        assert c.type.shape == (4,)

    def test_constant_repr(self):
        """Test Constant string representation."""
        c = Constant(42, int32)
        repr_str = repr(c)
        assert "42" in repr_str


class TestBinaryOp:
    """Tests for BinaryOp class."""

    def test_add_scalar_scalar(self):
        """Test scalar + scalar addition."""
        lhs = Value(ScalarType(float32))
        rhs = Value(ScalarType(float32))
        op = BinaryOp(BinaryOpKind.ADD, lhs, rhs)

        assert op.op == BinaryOpKind.ADD
        assert op.lhs == lhs
        assert op.rhs == rhs
        assert isinstance(op.result.type, ScalarType)

    def test_add_block_block(self):
        """Test block + block addition."""
        lhs = Value(BlockType(float32, (128,)))
        rhs = Value(BlockType(float32, (128,)))
        op = BinaryOp(BinaryOpKind.ADD, lhs, rhs)

        assert isinstance(op.result.type, BlockType)
        assert op.result.type.shape == (128,)

    def test_add_block_scalar(self):
        """Test block + scalar addition (broadcast)."""
        lhs = Value(BlockType(float32, (128,)))
        rhs = Value(ScalarType(float32))
        op = BinaryOp(BinaryOpKind.ADD, lhs, rhs)

        assert isinstance(op.result.type, BlockType)
        assert op.result.type.shape == (128,)

    def test_mul_operation(self):
        """Test multiplication operation."""
        lhs = Value(ScalarType(int32))
        rhs = Value(ScalarType(int32))
        op = BinaryOp(BinaryOpKind.MUL, lhs, rhs)

        assert op.op == BinaryOpKind.MUL
        assert isinstance(op.result.type, ScalarType)

    def test_comparison_operation(self):
        """Test comparison operations return bool type."""
        lhs = Value(BlockType(int32, (128,)))
        rhs = Value(ScalarType(int32))
        op = BinaryOp(BinaryOpKind.LT, lhs, rhs)

        assert op.result.type.dtype == bool_

    def test_children(self):
        """Test that children returns operands."""
        lhs = Value(ScalarType(float32))
        rhs = Value(ScalarType(float32))
        op = BinaryOp(BinaryOpKind.ADD, lhs, rhs)

        assert lhs in op.children()
        assert rhs in op.children()


class TestUnaryOp:
    """Tests for UnaryOp class."""

    def test_neg_scalar(self):
        """Test scalar negation."""
        operand = Value(ScalarType(float32))
        op = UnaryOp(UnaryOpKind.NEG, operand)

        assert op.op == UnaryOpKind.NEG
        assert op.operand == operand
        assert op.result.type == operand.type

    def test_exp_block(self):
        """Test block exponential."""
        operand = Value(BlockType(float32, (128,)))
        op = UnaryOp(UnaryOpKind.EXP, operand)

        assert op.op == UnaryOpKind.EXP
        assert op.result.type == operand.type


class TestMakeRangeOp:
    """Tests for MakeRangeOp class."""

    def test_arange_basic(self):
        """Test basic arange operation."""
        op = MakeRangeOp(0, 128)

        assert op.start == 0
        assert op.end == 128
        assert op.size == 128

    def test_arange_result_type(self):
        """Test that arange produces correct type."""
        op = MakeRangeOp(0, 128)

        assert isinstance(op.result.type, BlockType)
        assert op.result.type.shape == (128,)
        assert op.result.type.dtype == int32

    def test_arange_nonzero_start(self):
        """Test arange with non-zero start."""
        op = MakeRangeOp(10, 20)

        assert op.start == 10
        assert op.end == 20
        assert op.size == 10
        assert op.result.type.shape == (10,)


class TestProgramIdOp:
    """Tests for ProgramIdOp class."""

    def test_program_id_axis_0(self):
        """Test program_id for axis 0."""
        op = ProgramIdOp(0)

        assert op.axis == 0
        assert isinstance(op.result.type, ScalarType)
        assert op.result.type.dtype == int32

    def test_program_id_axis_1(self):
        """Test program_id for axis 1."""
        op = ProgramIdOp(1)
        assert op.axis == 1

    def test_program_id_invalid_axis(self):
        """Test that invalid axis raises error."""
        with pytest.raises((ValueError, AssertionError)):
            ProgramIdOp(3)  # Only 0, 1, 2 are valid


class TestLoadOp:
    """Tests for LoadOp class."""

    def test_load_scalar_pointer(self):
        """Test load from scalar pointer."""
        ptr = Value(PointerType(float32))
        op = LoadOp(ptr)

        assert op.ptr == ptr
        assert isinstance(op.result.type, ScalarType)
        assert op.result.type.dtype == float32

    def test_load_block_pointer(self):
        """Test load from block pointer (pointer with offset)."""
        ptr = Value(PointerType(float32, (128,)))
        op = LoadOp(ptr)

        assert isinstance(op.result.type, BlockType)
        assert op.result.type.shape == (128,)
        assert op.result.type.dtype == float32

    def test_load_with_mask(self):
        """Test load with mask."""
        ptr = Value(PointerType(float32, (128,)))
        mask = Value(BlockType(bool_, (128,)))
        op = LoadOp(ptr, mask=mask)

        assert op.mask == mask

    def test_load_with_mask_and_other(self):
        """Test load with mask and default value."""
        ptr = Value(PointerType(float32, (128,)))
        mask = Value(BlockType(bool_, (128,)))
        other = Value(ScalarType(float32))
        op = LoadOp(ptr, mask=mask, other=other)

        assert op.mask == mask
        assert op.other == other


class TestStoreOp:
    """Tests for StoreOp class."""

    def test_store_basic(self):
        """Test basic store operation."""
        ptr = Value(PointerType(float32, (128,)))
        value = Value(BlockType(float32, (128,)))
        op = StoreOp(ptr, value)

        assert op.ptr == ptr
        assert op.value == value

    def test_store_with_mask(self):
        """Test store with mask."""
        ptr = Value(PointerType(float32, (128,)))
        value = Value(BlockType(float32, (128,)))
        mask = Value(BlockType(bool_, (128,)))
        op = StoreOp(ptr, value, mask=mask)

        assert op.mask == mask


class TestDotOp:
    """Tests for DotOp class."""

    def test_dot_basic(self):
        """Test basic matrix multiplication."""
        lhs = Value(BlockType(float32, (64, 32)))  # (M, K)
        rhs = Value(BlockType(float32, (32, 128)))  # (K, N)
        op = DotOp(lhs, rhs)

        assert isinstance(op.result.type, BlockType)
        assert op.result.type.shape == (64, 128)  # (M, N)

    def test_dot_with_accumulator(self):
        """Test dot with accumulator."""
        lhs = Value(BlockType(float32, (64, 32)))
        rhs = Value(BlockType(float32, (32, 128)))
        acc = Value(BlockType(float32, (64, 128)))
        op = DotOp(lhs, rhs, acc=acc)

        assert op.acc == acc
        assert op.result.type.shape == (64, 128)


class TestReduceOp:
    """Tests for ReduceOp class."""

    def test_reduce_sum_axis_0(self):
        """Test sum reduction along axis 0."""
        operand = Value(BlockType(float32, (64, 128)))
        op = ReduceOp(ReduceOpKind.SUM, operand, axis=0)

        assert op.op == ReduceOpKind.SUM
        assert op.axis == 0
        assert isinstance(op.result.type, BlockType)
        assert op.result.type.shape == (128,)

    def test_reduce_sum_axis_1(self):
        """Test sum reduction along axis 1."""
        operand = Value(BlockType(float32, (64, 128)))
        op = ReduceOp(ReduceOpKind.SUM, operand, axis=1)

        assert op.result.type.shape == (64,)

    def test_reduce_max(self):
        """Test max reduction."""
        operand = Value(BlockType(float32, (128,)))
        op = ReduceOp(ReduceOpKind.MAX, operand, axis=0)

        # Reducing 1D along axis 0 gives scalar
        assert isinstance(op.result.type, ScalarType)


class TestBroadcastOp:
    """Tests for BroadcastOp class."""

    def test_broadcast_scalar_to_block(self):
        """Test broadcasting scalar to block."""
        operand = Value(ScalarType(float32))
        op = BroadcastOp(operand, (128,))

        assert isinstance(op.result.type, BlockType)
        assert op.result.type.shape == (128,)

    def test_broadcast_1d_to_2d(self):
        """Test broadcasting 1D to 2D."""
        operand = Value(BlockType(float32, (128,)))
        op = BroadcastOp(operand, (64, 128))

        assert op.result.type.shape == (64, 128)


class TestWhereOp:
    """Tests for WhereOp class."""

    def test_where_basic(self):
        """Test basic where operation."""
        condition = Value(BlockType(bool_, (128,)))
        true_val = Value(BlockType(float32, (128,)))
        false_val = Value(BlockType(float32, (128,)))
        op = WhereOp(condition, true_val, false_val)

        assert op.condition == condition
        assert op.true_value == true_val
        assert op.false_value == false_val
        assert op.result.type.shape == (128,)


class TestCastOp:
    """Tests for CastOp class."""

    def test_cast_dtype(self):
        """Test casting dtype."""
        operand = Value(BlockType(int32, (128,)))
        op = CastOp(operand, float32)

        assert op.result.type.dtype == float32
        assert op.result.type.shape == (128,)


class TestBlock:
    """Tests for Block class."""

    def test_block_creation(self):
        """Test Block creation."""
        block = Block()
        assert len(block.ops) == 0
        assert len(block.args) == 0

    def test_block_add_op(self):
        """Test adding operations to Block."""
        block = Block()
        op = MakeRangeOp(0, 128)
        block.add_op(op)

        assert len(block.ops) == 1
        assert block.ops[0] == op


class TestFunction:
    """Tests for Function class."""

    def test_function_creation(self):
        """Test Function creation."""
        params = [
            ("x_ptr", PointerType(float32)),
            ("n", ScalarType(int32)),
        ]
        body = Block()
        func = Function("test_kernel", params, body)

        assert func.name == "test_kernel"
        assert len(func.params) == 2

    def test_function_with_constexpr(self):
        """Test Function with constexpr params."""
        params = [
            ("x_ptr", PointerType(float32)),
            ("BLOCK_SIZE", ScalarType(int32)),
        ]
        body = Block()
        func = Function("test_kernel", params, body, constexpr_params={"BLOCK_SIZE"})

        assert "BLOCK_SIZE" in func.constexpr_params
