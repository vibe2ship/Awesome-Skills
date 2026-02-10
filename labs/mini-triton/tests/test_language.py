"""
Tests for the Mini-Triton language API (tl.*).

These tests verify that the language API works correctly for tracing.
"""

import pytest
from mini_triton.ir.types import float32, int32, bool_
from mini_triton.ir.builder import IRBuilder
from mini_triton.language import (
    TensorProxy,
    _set_builder, _get_builder,
    program_id, arange, zeros, full,
    load, store, dot,
    sum, max, min,
    exp, log, sqrt, abs,
    maximum, minimum, where,
)


@pytest.fixture
def setup_builder():
    """Set up an IR builder for testing."""
    builder = IRBuilder()
    builder.create_function("test", [])
    _set_builder(builder)
    yield builder
    _set_builder(None)


class TestTensorProxy:
    """Tests for TensorProxy class."""

    def test_proxy_creation(self, setup_builder):
        """Test creating TensorProxy."""
        builder = setup_builder
        value = builder.constant(1.0, float32)
        proxy = TensorProxy(value)
        assert proxy.value == value

    def test_proxy_add_proxy(self, setup_builder):
        """Test adding two proxies."""
        builder = setup_builder
        a = TensorProxy(builder.constant(1.0, float32))
        b = TensorProxy(builder.constant(2.0, float32))
        c = a + b

        assert isinstance(c, TensorProxy)

    def test_proxy_add_scalar(self, setup_builder):
        """Test adding proxy and Python scalar."""
        builder = setup_builder
        a = TensorProxy(builder.constant(1.0, float32))
        c = a + 2.0

        assert isinstance(c, TensorProxy)

    def test_proxy_radd(self, setup_builder):
        """Test reverse addition."""
        builder = setup_builder
        a = TensorProxy(builder.constant(1.0, float32))
        c = 2.0 + a

        assert isinstance(c, TensorProxy)

    def test_proxy_sub(self, setup_builder):
        """Test subtraction."""
        builder = setup_builder
        a = TensorProxy(builder.constant(5.0, float32))
        b = TensorProxy(builder.constant(2.0, float32))
        c = a - b

        assert isinstance(c, TensorProxy)

    def test_proxy_mul(self, setup_builder):
        """Test multiplication."""
        builder = setup_builder
        a = TensorProxy(builder.constant(3.0, float32))
        b = TensorProxy(builder.constant(4.0, float32))
        c = a * b

        assert isinstance(c, TensorProxy)

    def test_proxy_div(self, setup_builder):
        """Test division."""
        builder = setup_builder
        a = TensorProxy(builder.constant(10.0, float32))
        b = TensorProxy(builder.constant(2.0, float32))
        c = a / b

        assert isinstance(c, TensorProxy)

    def test_proxy_neg(self, setup_builder):
        """Test negation."""
        builder = setup_builder
        a = TensorProxy(builder.constant(5.0, float32))
        b = -a

        assert isinstance(b, TensorProxy)

    def test_proxy_comparison_lt(self, setup_builder):
        """Test less-than comparison."""
        builder = setup_builder
        a = TensorProxy(builder.arange(0, 128))
        b = 64
        c = a < b

        assert isinstance(c, TensorProxy)

    def test_proxy_comparison_eq(self, setup_builder):
        """Test equality comparison."""
        builder = setup_builder
        a = TensorProxy(builder.constant(1, int32))
        b = TensorProxy(builder.constant(1, int32))
        c = a == b

        assert isinstance(c, TensorProxy)


class TestLanguageFunctions:
    """Tests for tl.* language functions."""

    def test_program_id(self, setup_builder):
        """Test tl.program_id."""
        pid = program_id(0)
        assert isinstance(pid, TensorProxy)

    def test_arange(self, setup_builder):
        """Test tl.arange."""
        offsets = arange(0, 128)
        assert isinstance(offsets, TensorProxy)

    def test_zeros(self, setup_builder):
        """Test tl.zeros."""
        z = zeros((128,), float32)
        assert isinstance(z, TensorProxy)

    def test_full(self, setup_builder):
        """Test tl.full."""
        f = full((64, 64), 1.0, float32)
        assert isinstance(f, TensorProxy)


class TestLanguageMath:
    """Tests for math functions."""

    def test_exp(self, setup_builder):
        """Test tl.exp."""
        builder = setup_builder
        x = TensorProxy(builder.zeros((128,), float32))
        y = exp(x)
        assert isinstance(y, TensorProxy)

    def test_log(self, setup_builder):
        """Test tl.log."""
        builder = setup_builder
        x = TensorProxy(builder.full((128,), 1.0, float32))
        y = log(x)
        assert isinstance(y, TensorProxy)

    def test_sqrt(self, setup_builder):
        """Test tl.sqrt."""
        builder = setup_builder
        x = TensorProxy(builder.full((128,), 4.0, float32))
        y = sqrt(x)
        assert isinstance(y, TensorProxy)

    def test_maximum(self, setup_builder):
        """Test tl.maximum."""
        builder = setup_builder
        x = TensorProxy(builder.zeros((128,), float32))
        y = maximum(x, 0.0)
        assert isinstance(y, TensorProxy)

    def test_minimum(self, setup_builder):
        """Test tl.minimum."""
        builder = setup_builder
        x = TensorProxy(builder.zeros((128,), float32))
        y = minimum(x, 1.0)
        assert isinstance(y, TensorProxy)


class TestLanguageReduction:
    """Tests for reduction functions."""

    def test_sum(self, setup_builder):
        """Test tl.sum."""
        builder = setup_builder
        x = TensorProxy(builder.zeros((128,), float32))
        y = sum(x, axis=0)
        assert isinstance(y, TensorProxy)

    def test_max(self, setup_builder):
        """Test tl.max."""
        builder = setup_builder
        x = TensorProxy(builder.zeros((64, 128), float32))
        y = max(x, axis=1)
        assert isinstance(y, TensorProxy)


class TestLanguageWhere:
    """Tests for where function."""

    def test_where(self, setup_builder):
        """Test tl.where."""
        builder = setup_builder
        cond = TensorProxy(builder.lt(builder.arange(0, 128), builder.constant(64, int32)))
        true_val = TensorProxy(builder.zeros((128,), float32))
        false_val = TensorProxy(builder.full((128,), 1.0, float32))
        result = where(cond, true_val, false_val)
        assert isinstance(result, TensorProxy)


class TestNoBuilder:
    """Tests for error handling when no builder is set."""

    def test_program_id_no_builder(self):
        """Test that using tl.* without builder raises error."""
        _set_builder(None)
        with pytest.raises(RuntimeError):
            program_id(0)

    def test_arange_no_builder(self):
        """Test arange without builder raises error."""
        _set_builder(None)
        with pytest.raises(RuntimeError):
            arange(0, 128)
