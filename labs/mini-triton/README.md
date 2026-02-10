# Mini-Triton: A Pedagogical GPU Kernel Compiler

Mini-Triton is an educational implementation of a GPU kernel compiler inspired by [OpenAI's Triton](https://github.com/openai/triton). It's designed to help you understand the core concepts of AI compilers through hands-on implementation.

## 🎯 Learning Objectives

By implementing this project, you will learn:

- **Compiler Fundamentals**: IR design, type systems, optimization passes
- **Triton/GPU Concepts**: Tile-based programming, program IDs, memory coalescing
- **Deep Learning Compilation**: How PyTorch 2.0's `torch.compile` works under the hood
- **Systems Programming**: Building compilers, code generation, JIT compilation

## 📊 Architecture

```
┌─────────────────────────────────────┐
│         @mt.jit decorator           │
│      (Python DSL Entry Point)       │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│           Frontend                  │
│    (Python AST → Triton IR)         │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│           IR Layer                  │
│  (Types, Operations, Builder)       │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Optimization Passes          │
│  (Canonicalize, Type Inference)     │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│          Code Generator             │
│        (IR → NumPy Code)            │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Clone and setup
cd mini_triton
pip install -e ".[dev]"

# Run tests (they will fail until you implement!)
pytest tests/

# Try an example (after implementation)
python examples/vector_add.py
```

## 📁 Project Structure

```
mini_triton/
├── src/mini_triton/
│   ├── ir/                    # Intermediate Representation
│   │   ├── types.py          # Type system (DType, BlockType, PointerType)
│   │   ├── ops.py            # IR operations (Load, Store, Dot, etc.)
│   │   ├── builder.py        # IR construction utilities
│   │   └── printer.py        # IR pretty-printing
│   ├── language.py           # tl.* API (load, store, arange, etc.)
│   ├── frontend/             # Python AST → IR
│   │   ├── ast_visitor.py    # AST traversal and conversion
│   │   └── type_inference.py # Type checking and inference
│   ├── transforms/           # Optimization passes
│   │   └── canonicalize.py   # Constant folding, simplification
│   ├── codegen/              # Code generation
│   │   └── numpy_gen.py      # Generate NumPy code
│   └── runtime/              # Execution runtime
│       ├── jit.py            # @jit decorator
│       └── launcher.py       # Kernel launching
├── tests/                    # Comprehensive test suite
├── examples/                 # Working examples
│   ├── vector_add.py        # Hello World
│   ├── softmax.py           # Reductions
│   ├── matmul.py            # GEMM
│   └── attention.py         # Transformer building block
└── pyproject.toml
```

## 📋 Implementation TODO List

### Milestone 1: Type System & IR (Foundation)

- [ ] **Task 1.1**: Implement `ir/types.py`
  - DType, ScalarType, BlockType, PointerType
  - Broadcasting rules
  - Run: `pytest tests/test_types.py`

- [ ] **Task 1.2**: Implement `ir/ops.py`
  - Value, Constant, BinaryOp, UnaryOp
  - LoadOp, StoreOp, MakeRangeOp, ProgramIdOp
  - DotOp, ReduceOp, WhereOp
  - Run: `pytest tests/test_ir.py`

- [ ] **Task 1.3**: Implement `ir/builder.py`
  - IRBuilder class with all operation creation methods
  - Run: `pytest tests/test_builder.py`

### Milestone 2: Language API (Frontend)

- [ ] **Task 2.1**: Implement `language.py`
  - TensorProxy class with operator overloading
  - tl.* functions (program_id, arange, load, store, etc.)
  - Run: `pytest tests/test_language.py`

- [ ] **Task 2.2**: Implement `frontend/ast_visitor.py`
  - Python AST → IR conversion
  - Handle assignments, expressions, control flow

- [ ] **Task 2.3**: Implement `frontend/type_inference.py`
  - Type checking and inference
  - Run: `pytest tests/test_frontend.py`

### Milestone 3: Code Generation

- [ ] **Task 3.1**: Implement `codegen/numpy_gen.py`
  - Generate NumPy code from IR
  - Run: `pytest tests/test_codegen.py`

- [ ] **Task 3.2**: Implement `runtime/jit.py`
  - @jit decorator
  - Kernel caching
  - Grid-based launching

### Milestone 4: Integration

- [ ] **Task 4.1**: Complete end-to-end pipeline
  - Run: `pytest tests/test_e2e.py`

- [ ] **Task 4.2**: Run examples
  - `python examples/vector_add.py`
  - `python examples/softmax.py`
  - `python examples/matmul.py`

## 🔑 Key Concepts

### 1. Tile-Based Programming

Unlike CUDA where you think in threads, Triton thinks in **tiles** (blocks of data):

```python
@mt.jit
def kernel(ptr, BLOCK: mt.constexpr):
    offsets = tl.arange(0, BLOCK)  # Create a tile of indices
    data = tl.load(ptr + offsets)   # Load a tile of data
    # Operations work on entire tiles at once
```

### 2. Program IDs

Each "program" (block) has a unique ID used to identify which data to process:

```python
pid = tl.program_id(0)  # Get block ID in x-dimension
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

### 3. Masking

Handle edge cases (last block might be partial):

```python
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 4. Constexpr

Compile-time constants enable specialization:

```python
def kernel(ptr, n, BLOCK: mt.constexpr):  # BLOCK is known at compile time
    offsets = tl.arange(0, BLOCK)  # Compiler knows BLOCK=128
```

## 📚 Resources

### Triton & AI Compilers
- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

### Compiler Fundamentals
- [MLIR Documentation](https://mlir.llvm.org/)
- [TVM: Deep Learning Compiler](https://tvm.apache.org/)

### GPU Programming
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GPU Memory Hierarchy](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

## 🏆 Challenges (After Completion)

1. **Add Numba CUDA Backend**: Generate actual GPU code
2. **Implement Auto-Tuning**: Try different tile sizes
3. **Add Loop Fusion**: Fuse elementwise operations
4. **Build Flash Attention**: Implement the full algorithm

## 📝 License

MIT

---

Happy learning! 🎓
