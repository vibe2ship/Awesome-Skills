"""
Tests for the Mini-Triton type system.

These tests verify that the type system works correctly:
- DType creation and comparison
- ScalarType, BlockType, PointerType
- Broadcasting rules
- Type inference for operations
"""

import pytest
from mini_triton.ir.types import (
    DType,
    ScalarType,
    BlockType,
    PointerType,
    float16,
    float32,
    float64,
    int32,
    int64,
    bool_,
    get_broadcast_shape,
    infer_binary_op_type,
)


class TestDType:
    """Tests for DType class."""

    def test_dtype_creation(self):
        """Test that DTypes can be created with name, bits, and is_floating."""
        dt = DType("float32", 32, is_floating=True)
        assert dt.name == "float32"
        assert dt.bits == 32
        assert dt.is_floating is True

    def test_dtype_repr(self):
        """Test DType string representation."""
        assert repr(float32) == "float32"
        assert repr(int32) == "int32"

    def test_dtype_equality(self):
        """Test DType equality comparison."""
        dt1 = DType("float32", 32, is_floating=True)
        dt2 = DType("float32", 32, is_floating=True)
        dt3 = DType("float64", 64, is_floating=True)

        assert dt1 == dt2
        assert dt1 != dt3
        assert float32 == dt1

    def test_dtype_hash(self):
        """Test that DTypes are hashable (can be used in dicts/sets)."""
        dt1 = DType("float32", 32, is_floating=True)
        dt2 = DType("float32", 32, is_floating=True)

        # Equal objects must have equal hashes
        assert hash(dt1) == hash(dt2)

        # Can be used in sets
        dtype_set = {dt1, dt2}
        assert len(dtype_set) == 1

    def test_predefined_dtypes(self):
        """Test that predefined DTypes are correct."""
        assert float16.bits == 16
        assert float16.is_floating is True

        assert float32.bits == 32
        assert float32.is_floating is True

        assert float64.bits == 64
        assert float64.is_floating is True

        assert int32.bits == 32
        assert int32.is_floating is False

        assert int64.bits == 64
        assert int64.is_floating is False

        assert bool_.bits == 1
        assert bool_.is_floating is False

    def test_numpy_dtype(self):
        """Test numpy dtype string conversion."""
        assert float32.numpy_dtype == "np.float32"
        assert int32.numpy_dtype == "np.int32"


class TestScalarType:
    """Tests for ScalarType class."""

    def test_scalar_type_creation(self):
        """Test ScalarType creation."""
        st = ScalarType(float32)
        assert st.dtype == float32

    def test_scalar_type_repr(self):
        """Test ScalarType string representation."""
        st = ScalarType(float32)
        assert "float32" in repr(st)

    def test_scalar_type_equality(self):
        """Test ScalarType equality."""
        st1 = ScalarType(float32)
        st2 = ScalarType(float32)
        st3 = ScalarType(int32)

        assert st1 == st2
        assert st1 != st3

    def test_scalar_type_hash(self):
        """Test ScalarType is hashable."""
        st1 = ScalarType(float32)
        st2 = ScalarType(float32)

        assert hash(st1) == hash(st2)
        assert len({st1, st2}) == 1


class TestBlockType:
    """Tests for BlockType class."""

    def test_block_type_1d(self):
        """Test 1D BlockType creation."""
        bt = BlockType(float32, (128,))
        assert bt.dtype == float32
        assert bt.shape == (128,)

    def test_block_type_2d(self):
        """Test 2D BlockType creation."""
        bt = BlockType(float32, (64, 64))
        assert bt.dtype == float32
        assert bt.shape == (64, 64)

    def test_block_type_repr(self):
        """Test BlockType string representation."""
        bt = BlockType(float32, (128,))
        repr_str = repr(bt)
        assert "float32" in repr_str
        assert "128" in repr_str

    def test_block_type_equality(self):
        """Test BlockType equality."""
        bt1 = BlockType(float32, (128,))
        bt2 = BlockType(float32, (128,))
        bt3 = BlockType(float32, (256,))
        bt4 = BlockType(int32, (128,))

        assert bt1 == bt2
        assert bt1 != bt3  # Different shape
        assert bt1 != bt4  # Different dtype

    def test_block_type_rank(self):
        """Test BlockType rank property."""
        bt1d = BlockType(float32, (128,))
        bt2d = BlockType(float32, (64, 64))
        bt3d = BlockType(float32, (32, 32, 32))

        assert bt1d.rank == 1
        assert bt2d.rank == 2
        assert bt3d.rank == 3

    def test_block_type_numel(self):
        """Test BlockType numel property."""
        bt1d = BlockType(float32, (128,))
        bt2d = BlockType(float32, (64, 64))
        bt3d = BlockType(float32, (2, 3, 4))

        assert bt1d.numel == 128
        assert bt2d.numel == 4096
        assert bt3d.numel == 24

    def test_block_type_broadcast_to_valid(self):
        """Test valid broadcasting."""
        bt = BlockType(float32, (1,))
        bt_broadcast = bt.broadcast_to((128,))
        assert bt_broadcast.shape == (128,)
        assert bt_broadcast.dtype == float32

        bt2d = BlockType(float32, (64, 1))
        bt2d_broadcast = bt2d.broadcast_to((64, 128))
        assert bt2d_broadcast.shape == (64, 128)

    def test_block_type_broadcast_to_invalid(self):
        """Test invalid broadcasting raises error."""
        bt = BlockType(float32, (64,))
        with pytest.raises(ValueError):
            bt.broadcast_to((128,))  # 64 cannot broadcast to 128


class TestPointerType:
    """Tests for PointerType class."""

    def test_pointer_type_scalar(self):
        """Test scalar PointerType creation."""
        pt = PointerType(float32)
        assert pt.pointee_dtype == float32
        assert pt.offset_shape is None

    def test_pointer_type_with_offset(self):
        """Test PointerType with offset shape."""
        pt = PointerType(float32, (128,))
        assert pt.pointee_dtype == float32
        assert pt.offset_shape == (128,)

    def test_pointer_type_repr(self):
        """Test PointerType string representation."""
        pt = PointerType(float32)
        repr_str = repr(pt)
        assert "ptr" in repr_str.lower() or "pointer" in repr_str.lower()
        assert "float32" in repr_str

    def test_pointer_type_equality(self):
        """Test PointerType equality."""
        pt1 = PointerType(float32)
        pt2 = PointerType(float32)
        pt3 = PointerType(int32)
        pt4 = PointerType(float32, (128,))

        assert pt1 == pt2
        assert pt1 != pt3
        assert pt1 != pt4

    def test_pointer_with_offset_shape(self):
        """Test with_offset_shape method."""
        pt = PointerType(float32)
        pt_with_offset = pt.with_offset_shape((128,))

        assert pt_with_offset.pointee_dtype == float32
        assert pt_with_offset.offset_shape == (128,)

    def test_is_block_pointer(self):
        """Test is_block_pointer property."""
        pt_scalar = PointerType(float32)
        pt_block = PointerType(float32, (128,))

        assert pt_scalar.is_block_pointer is False
        assert pt_block.is_block_pointer is True


class TestBroadcastShape:
    """Tests for get_broadcast_shape function."""

    def test_same_shape(self):
        """Test broadcasting same shapes."""
        result = get_broadcast_shape((128,), (128,))
        assert result == (128,)

        result = get_broadcast_shape((64, 128), (64, 128))
        assert result == (64, 128)

    def test_broadcast_with_1(self):
        """Test broadcasting with dimension size 1."""
        result = get_broadcast_shape((1,), (128,))
        assert result == (128,)

        result = get_broadcast_shape((128,), (1,))
        assert result == (128,)

    def test_broadcast_2d(self):
        """Test 2D broadcasting."""
        result = get_broadcast_shape((64, 1), (1, 128))
        assert result == (64, 128)

        result = get_broadcast_shape((64, 128), (128,))
        assert result == (64, 128)

    def test_broadcast_different_ranks(self):
        """Test broadcasting with different ranks."""
        result = get_broadcast_shape((128,), (64, 128))
        assert result == (64, 128)

        result = get_broadcast_shape((1,), (64, 128))
        assert result == (64, 128)

    def test_broadcast_incompatible(self):
        """Test that incompatible shapes raise ValueError."""
        with pytest.raises(ValueError):
            get_broadcast_shape((64,), (128,))

        with pytest.raises(ValueError):
            get_broadcast_shape((64, 64), (128, 128))


class TestInferBinaryOpType:
    """Tests for infer_binary_op_type function."""

    def test_scalar_scalar(self):
        """Test scalar + scalar type inference."""
        lhs = ScalarType(float32)
        rhs = ScalarType(float32)
        result = infer_binary_op_type(lhs, rhs)

        assert isinstance(result, ScalarType)
        assert result.dtype == float32

    def test_block_scalar(self):
        """Test block + scalar type inference."""
        lhs = BlockType(float32, (128,))
        rhs = ScalarType(float32)
        result = infer_binary_op_type(lhs, rhs)

        assert isinstance(result, BlockType)
        assert result.shape == (128,)

    def test_scalar_block(self):
        """Test scalar + block type inference."""
        lhs = ScalarType(float32)
        rhs = BlockType(float32, (128,))
        result = infer_binary_op_type(lhs, rhs)

        assert isinstance(result, BlockType)
        assert result.shape == (128,)

    def test_block_block_same_shape(self):
        """Test block + block with same shape."""
        lhs = BlockType(float32, (128,))
        rhs = BlockType(float32, (128,))
        result = infer_binary_op_type(lhs, rhs)

        assert isinstance(result, BlockType)
        assert result.shape == (128,)

    def test_block_block_broadcast(self):
        """Test block + block with broadcasting."""
        lhs = BlockType(float32, (64, 1))
        rhs = BlockType(float32, (1, 128))
        result = infer_binary_op_type(lhs, rhs)

        assert isinstance(result, BlockType)
        assert result.shape == (64, 128)

    def test_dtype_promotion(self):
        """Test that dtypes are promoted correctly."""
        # float32 + int32 -> float32
        lhs = ScalarType(float32)
        rhs = ScalarType(int32)
        result = infer_binary_op_type(lhs, rhs)

        assert result.dtype == float32
