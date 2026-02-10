"""
Tests for the NumPy code generator.

These tests verify that the code generator produces correct NumPy code.
"""

import pytest
from mini_triton.ir.types import (
    ScalarType, BlockType, PointerType,
    float32, int32, bool_
)
from mini_triton.ir.builder import IRBuilder
from mini_triton.codegen.numpy_gen import NumpyCodeGenerator, generate_numpy_kernel


class TestNumpyCodeGenBasic:
    """Basic code generator tests."""

    def test_codegen_creation(self):
        """Test that NumpyCodeGenerator can be created."""
        gen = NumpyCodeGenerator()
        assert gen is not None


class TestGenerateSimpleKernel:
    """Tests for generating simple kernels."""

    def test_generate_empty_function(self):
        """Test generating an empty function."""
        builder = IRBuilder()
        func = builder.create_function("empty_kernel", [])

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "def" in code
        assert "empty_kernel" in code

    def test_generate_with_params(self):
        """Test generating function with parameters."""
        builder = IRBuilder()
        func = builder.create_function(
            "kernel_with_params",
            [
                ("x_ptr", PointerType(float32)),
                ("n", ScalarType(int32)),
            ]
        )

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "x_ptr" in code
        assert "n" in code

    def test_generate_program_id(self):
        """Test generating program_id."""
        builder = IRBuilder()
        func = builder.create_function("pid_kernel", [])
        pid = builder.program_id(0)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        # Should reference the program id
        assert "program_id" in code.lower() or "_pid" in code.lower()

    def test_generate_arange(self):
        """Test generating arange."""
        builder = IRBuilder()
        func = builder.create_function("arange_kernel", [])
        offsets = builder.arange(0, 128)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "np.arange" in code or "arange" in code
        assert "128" in code

    def test_generate_binary_op(self):
        """Test generating binary operations."""
        builder = IRBuilder()
        func = builder.create_function("add_kernel", [])

        a = builder.constant(1, int32)
        b = builder.constant(2, int32)
        c = builder.add(a, b)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        # Should have addition
        assert "+" in code

    def test_generate_comparison(self):
        """Test generating comparison operations."""
        builder = IRBuilder()
        func = builder.create_function("cmp_kernel", [])

        offsets = builder.arange(0, 128)
        n = builder.constant(100, int32)
        mask = builder.lt(offsets, n)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "<" in code


class TestGenerateMemoryOps:
    """Tests for generating memory operations."""

    def test_generate_load(self):
        """Test generating load operation."""
        builder = IRBuilder()
        func = builder.create_function(
            "load_kernel",
            [("x_ptr", PointerType(float32, (128,)))]
        )

        ptr = builder.get_value("x_ptr")
        x = builder.load(ptr)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        # Should have array indexing
        assert "[" in code and "]" in code

    def test_generate_store(self):
        """Test generating store operation."""
        builder = IRBuilder()
        func = builder.create_function(
            "store_kernel",
            [("out_ptr", PointerType(float32, (128,)))]
        )

        ptr = builder.get_value("out_ptr")
        value = builder.zeros((128,), float32)
        builder.store(ptr, value)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        # Should have assignment to array
        assert "=" in code

    def test_generate_masked_load(self):
        """Test generating masked load."""
        builder = IRBuilder()
        func = builder.create_function(
            "masked_load_kernel",
            [("x_ptr", PointerType(float32, (128,)))]
        )

        ptr = builder.get_value("x_ptr")
        mask = builder.lt(builder.arange(0, 128), builder.constant(100, int32))
        other = builder.constant(0.0, float32)
        x = builder.load(ptr, mask=mask, other=other)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        # Should use np.where for masking
        assert "where" in code.lower()


class TestGenerateMathOps:
    """Tests for generating math operations."""

    def test_generate_exp(self):
        """Test generating exp."""
        builder = IRBuilder()
        func = builder.create_function("exp_kernel", [])

        x = builder.zeros((128,), float32)
        y = builder.exp(x)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "np.exp" in code or "exp" in code

    def test_generate_log(self):
        """Test generating log."""
        builder = IRBuilder()
        func = builder.create_function("log_kernel", [])

        x = builder.full((128,), 1.0, float32)
        y = builder.log(x)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "np.log" in code or "log" in code


class TestGenerateReductions:
    """Tests for generating reduction operations."""

    def test_generate_sum(self):
        """Test generating sum reduction."""
        builder = IRBuilder()
        func = builder.create_function("sum_kernel", [])

        x = builder.zeros((128,), float32)
        y = builder.sum(x, axis=0)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "np.sum" in code or "sum" in code

    def test_generate_max(self):
        """Test generating max reduction."""
        builder = IRBuilder()
        func = builder.create_function("max_kernel", [])

        x = builder.zeros((64, 128), float32)
        y = builder.max(x, axis=1)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "np.max" in code or "max" in code


class TestGenerateDot:
    """Tests for generating dot product."""

    def test_generate_dot(self):
        """Test generating matrix multiplication."""
        builder = IRBuilder()
        func = builder.create_function("dot_kernel", [])

        a = builder.zeros((64, 32), float32)
        b = builder.zeros((32, 128), float32)
        c = builder.dot(a, b)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "np.dot" in code or "@" in code or "matmul" in code.lower()


class TestGenerateWhere:
    """Tests for generating where operation."""

    def test_generate_where(self):
        """Test generating where."""
        builder = IRBuilder()
        func = builder.create_function("where_kernel", [])

        cond = builder.lt(builder.arange(0, 128), builder.constant(64, int32))
        true_val = builder.zeros((128,), float32)
        false_val = builder.full((128,), 1.0, float32)
        result = builder.where(cond, true_val, false_val)

        gen = NumpyCodeGenerator()
        code = gen.generate(func)

        assert "np.where" in code or "where" in code


class TestCodegenConvenience:
    """Tests for convenience functions."""

    def test_generate_numpy_kernel_function(self):
        """Test generate_numpy_kernel function."""
        builder = IRBuilder()
        func = builder.create_function("test_kernel", [])
        builder.program_id(0)

        code = generate_numpy_kernel(func)
        assert "test_kernel" in code
