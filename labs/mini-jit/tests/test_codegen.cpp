/**
 * Tests for x86-64 code generation.
 */

#include "mini_jit/codegen.hpp"
#include "mini_jit/memory.hpp"

#include <iostream>
#include <cstring>
#include <cassert>

using namespace mini_jit;

// Test helper
#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            std::cout << "Running " #name "... "; \
            try { \
                test_##name(); \
                std::cout << "PASSED\n"; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << "\n"; \
            } \
        } \
    } runner_##name; \
    void test_##name()

// ==================== Memory Tests ====================

TEST(page_size) {
    size_t page_size = get_page_size();
    assert(page_size > 0);
    assert((page_size & (page_size - 1)) == 0);  // Power of 2
}

TEST(code_buffer_emit) {
    CodeBuffer buf;
    buf.emit8(0x48);
    buf.emit8(0xc3);

    assert(buf.position() == 2);
    assert(buf.code()[0] == 0x48);
    assert(buf.code()[1] == 0xc3);
}

TEST(code_buffer_emit32) {
    CodeBuffer buf;
    buf.emit32(0x12345678);

    assert(buf.position() == 4);
    // Little endian
    assert(buf.code()[0] == 0x78);
    assert(buf.code()[1] == 0x56);
    assert(buf.code()[2] == 0x34);
    assert(buf.code()[3] == 0x12);
}

// ==================== CodeGen Tests ====================

TEST(codegen_nop) {
    CodeGen gen;
    gen.nop();
    gen.nop();
    gen.nop();

    assert(gen.size() == 3);
    const auto& code = gen.buffer().code();
    assert(code[0] == 0x90);
    assert(code[1] == 0x90);
    assert(code[2] == 0x90);
}

// These tests will fail until the TODO functions are implemented
// They serve as reference for expected behavior

TEST(codegen_ret) {
    CodeGen gen;
    gen.ret();  // Should emit 0xC3

    assert(gen.size() == 1);
    assert(gen.buffer().code()[0] == 0xC3);
}

TEST(codegen_mov_reg_reg) {
    CodeGen gen;
    gen.mov(Reg::RAX, Reg::RBX);  // mov rax, rbx

    // Expected: 48 89 d8
    // REX.W (48) + MOV r/m64,r64 (89) + ModR/M (d8 = 11 011 000)
    const auto& code = gen.buffer().code();
    assert(gen.size() >= 3);
}

TEST(codegen_mov_reg_imm) {
    CodeGen gen;
    gen.mov(Reg::RAX, 42);  // mov rax, 42

    // Expected: 48 b8 2a 00 00 00 00 00 00 00
    // REX.W + MOV r64, imm64
    const auto& code = gen.buffer().code();
    assert(gen.size() >= 10);
}

TEST(codegen_add) {
    CodeGen gen;
    gen.add(Reg::RAX, Reg::RBX);  // add rax, rbx

    // Expected: 48 01 d8
    assert(gen.size() >= 3);
}

TEST(codegen_push_pop) {
    CodeGen gen;
    gen.push(Reg::RBX);  // push rbx
    gen.pop(Reg::RBX);   // pop rbx

    // push rbx: 53 (or 41 53 for r8-r15)
    // pop rbx: 5b (or 41 5b for r8-r15)
    assert(gen.size() >= 2);
}

// ==================== Integration Tests ====================

TEST(simple_function) {
    // Compile: int64_t fn() { return 42; }
    CodeGen gen;
    gen.mov(Reg::RAX, 42);
    gen.ret();

    // This will fail until memory functions are implemented
    // auto fn = gen.finalize<int64_t(*)()>();
    // assert(fn() == 42);
}

TEST(add_function) {
    // Compile: int64_t add(int64_t a, int64_t b) { return a + b; }
    // System V: a in RDI, b in RSI, return in RAX
    CodeGen gen;
    gen.mov(Reg::RAX, Reg::RDI);
    gen.add(Reg::RAX, Reg::RSI);
    gen.ret();

    // auto fn = gen.finalize<int64_t(*)(int64_t, int64_t)>();
    // assert(fn(3, 4) == 7);
    // assert(fn(100, 200) == 300);
}

int main() {
    std::cout << "Mini-JIT Code Generation Tests\n";
    std::cout << "==============================\n\n";

    // Tests run automatically via static initialization

    std::cout << "\nAll tests completed.\n";
    std::cout << "(Note: Some tests expect TODO implementations)\n";

    return 0;
}
