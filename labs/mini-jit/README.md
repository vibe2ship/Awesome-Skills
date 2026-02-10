# Mini-JIT

A minimal JIT (Just-In-Time) compiler implementation in C++, demonstrating x86-64 code generation.

## Overview

Mini-JIT teaches the fundamentals of JIT compilation:
- **Memory management**: Allocating executable memory
- **Code generation**: Emitting x86-64 machine code
- **Calling conventions**: System V AMD64 ABI
- **Runtime compilation**: Compiling and executing code at runtime

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mini-JIT                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Source Code (e.g., Brainfuck, Expression)                     │
│       │                                                          │
│       ▼                                                          │
│   ┌───────────────┐                                              │
│   │    Parser     │  Source → IR/AST                             │
│   └───────┬───────┘                                              │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────┐                                              │
│   │   Compiler    │  IR → x86-64 instructions                    │
│   └───────┬───────┘                                              │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────┐                                              │
│   │   CodeGen     │  Emit machine code bytes                     │
│   └───────┬───────┘                                              │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────┐                                              │
│   │ Executable    │  mmap with PROT_EXEC                         │
│   │   Memory      │                                              │
│   └───────┬───────┘                                              │
│           │                                                      │
│           ▼                                                      │
│       Execute!    reinterpret_cast<fn_ptr>(code)()              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## x86-64 Basics

### Registers
```
┌─────────────────────────────────────────────────────────────────┐
│                      x86-64 Registers                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  General Purpose (64-bit):                                       │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐             │
│  │ RAX │ RBX │ RCX │ RDX │ RSI │ RDI │ RBP │ RSP │             │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘             │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐             │
│  │ R8  │ R9  │ R10 │ R11 │ R12 │ R13 │ R14 │ R15 │             │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘             │
│                                                                  │
│  System V AMD64 Calling Convention:                              │
│  - Arguments: RDI, RSI, RDX, RCX, R8, R9                        │
│  - Return: RAX                                                   │
│  - Callee-saved: RBX, RBP, R12-R15                              │
│  - Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Instruction Encoding
```
┌─────────────────────────────────────────────────────────────────┐
│                  x86-64 Instruction Format                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Prefixes] [REX] [Opcode] [ModR/M] [SIB] [Disp] [Imm]         │
│                                                                  │
│  REX Prefix (for 64-bit operands):                              │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                             │
│  │ 0 │ 1 │ 0 │ 0 │ W │ R │ X │ B │                             │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                             │
│  W: 64-bit operand size                                          │
│  R: ModR/M reg extension                                         │
│  X: SIB index extension                                          │
│  B: ModR/M r/m or SIB base extension                            │
│                                                                  │
│  ModR/M Byte:                                                    │
│  ┌─────────┬─────────┬───────────┐                              │
│  │ Mod(2)  │ Reg(3)  │  R/M(3)   │                              │
│  └─────────┴─────────┴───────────┘                              │
│                                                                  │
│  Example: mov rax, rbx                                           │
│  48 89 d8 = REX.W + MOV r/m64,r64 + ModR/M(11,011,000)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mini-jit/
├── include/
│   └── mini_jit/
│       ├── memory.hpp      # Executable memory management
│       ├── codegen.hpp     # x86-64 code generator
│       ├── compiler.hpp    # High-level compiler
│       └── jit.hpp         # JIT engine
├── src/
│   ├── memory.cpp
│   ├── codegen.cpp
│   ├── compiler.cpp
│   ├── jit.cpp
│   └── main.cpp
├── examples/
│   ├── brainfuck.cpp       # Brainfuck JIT compiler
│   └── calculator.cpp      # Expression calculator
├── tests/
│   └── test_codegen.cpp
├── CMakeLists.txt
└── README.md
```

## Learning Objectives

### Milestone 1: Executable Memory
- [ ] Implement mmap-based memory allocation
- [ ] Handle page alignment
- [ ] Manage memory protection (RW → RX)
- [ ] Implement memory deallocation

### Milestone 2: Basic Code Generation
- [ ] Emit function prologue/epilogue
- [ ] Implement mov instructions
- [ ] Implement arithmetic (add, sub, mul, div)
- [ ] Implement comparison and jumps

### Milestone 3: Calling Convention
- [ ] Handle function arguments (System V ABI)
- [ ] Handle return values
- [ ] Save/restore callee-saved registers
- [ ] Implement function calls

### Milestone 4: Brainfuck JIT
- [ ] Parse Brainfuck source
- [ ] Compile to x86-64
- [ ] Implement all 8 operations
- [ ] Add optimizations (run-length encoding)

### Milestone 5: Expression JIT
- [ ] Parse arithmetic expressions
- [ ] Build expression tree
- [ ] Compile to x86-64
- [ ] Support variables

## Example: Simple Function

```cpp
// Goal: JIT compile: int64_t add(int64_t a, int64_t b) { return a + b; }

CodeGen gen;
gen.push(RBP);           // push rbp
gen.mov(RBP, RSP);       // mov rbp, rsp
gen.mov(RAX, RDI);       // mov rax, rdi (first arg)
gen.add(RAX, RSI);       // add rax, rsi (second arg)
gen.pop(RBP);            // pop rbp
gen.ret();               // ret

auto fn = gen.finalize<int64_t(*)(int64_t, int64_t)>();
int64_t result = fn(3, 4);  // result = 7
```

## References

- [Intel x86-64 Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [System V AMD64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Brainfuck](https://en.wikipedia.org/wiki/Brainfuck)
- [Writing a JIT Compiler](https://eli.thegreenplace.net/2017/adventures-in-jit-compilation-part-1-an-interpreter/)
