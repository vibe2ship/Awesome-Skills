#pragma once

#include "memory.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace mini_jit {

/**
 * x86-64 registers.
 */
enum class Reg : uint8_t {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3,
    RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8 = 8, R9 = 9, R10 = 10, R11 = 11,
    R12 = 12, R13 = 13, R14 = 14, R15 = 15,
};

/**
 * Returns register name for debugging.
 */
const char* reg_name(Reg reg);

/**
 * Condition codes for jumps.
 */
enum class Cond : uint8_t {
    O = 0,    // Overflow
    NO = 1,   // No overflow
    B = 2,    // Below (unsigned <)
    AE = 3,   // Above or equal (unsigned >=)
    E = 4,    // Equal
    NE = 5,   // Not equal
    BE = 6,   // Below or equal (unsigned <=)
    A = 7,    // Above (unsigned >)
    S = 8,    // Sign (negative)
    NS = 9,   // No sign (positive)
    P = 10,   // Parity even
    NP = 11,  // Parity odd
    L = 12,   // Less (signed <)
    GE = 13,  // Greater or equal (signed >=)
    LE = 14,  // Less or equal (signed <=)
    G = 15,   // Greater (signed >)
};

/**
 * Label for jump targets.
 */
struct Label {
    size_t offset = 0;
    bool bound = false;
    std::vector<size_t> patches; // Offsets needing patching
};

/**
 * x86-64 code generator.
 */
class CodeGen {
public:
    CodeGen();

    // ==================== Emit Instructions ====================

    /**
     * Emits function prologue.
     *
     * TODO: Implement this function
     * push rbp; mov rbp, rsp; [sub rsp, stack_size]
     */
    void prologue(size_t stack_size = 0);

    /**
     * Emits function epilogue.
     *
     * TODO: Implement this function
     * [add rsp, stack_size]; pop rbp; ret
     */
    void epilogue(size_t stack_size = 0);

    /**
     * Emits: mov dst, src (register to register).
     *
     * TODO: Implement this function
     */
    void mov(Reg dst, Reg src);

    /**
     * Emits: mov dst, imm64 (immediate to register).
     *
     * TODO: Implement this function
     */
    void mov(Reg dst, int64_t imm);

    /**
     * Emits: mov dst, [src + offset] (memory load).
     *
     * TODO: Implement this function
     */
    void mov_load(Reg dst, Reg src, int32_t offset = 0);

    /**
     * Emits: mov [dst + offset], src (memory store).
     *
     * TODO: Implement this function
     */
    void mov_store(Reg dst, int32_t offset, Reg src);

    /**
     * Emits: mov byte [dst + offset], imm8.
     *
     * TODO: Implement this function
     */
    void mov_store8(Reg dst, int32_t offset, uint8_t imm);

    /**
     * Emits: add dst, src.
     *
     * TODO: Implement this function
     */
    void add(Reg dst, Reg src);

    /**
     * Emits: add dst, imm32.
     *
     * TODO: Implement this function
     */
    void add(Reg dst, int32_t imm);

    /**
     * Emits: sub dst, src.
     *
     * TODO: Implement this function
     */
    void sub(Reg dst, Reg src);

    /**
     * Emits: sub dst, imm32.
     *
     * TODO: Implement this function
     */
    void sub(Reg dst, int32_t imm);

    /**
     * Emits: imul dst, src.
     *
     * TODO: Implement this function
     */
    void imul(Reg dst, Reg src);

    /**
     * Emits: idiv src (divides RDX:RAX by src).
     *
     * TODO: Implement this function
     */
    void idiv(Reg src);

    /**
     * Emits: cqo (sign-extend RAX into RDX:RAX).
     *
     * TODO: Implement this function
     */
    void cqo();

    /**
     * Emits: cmp left, right.
     *
     * TODO: Implement this function
     */
    void cmp(Reg left, Reg right);

    /**
     * Emits: cmp reg, imm32.
     *
     * TODO: Implement this function
     */
    void cmp(Reg reg, int32_t imm);

    /**
     * Emits: test reg, reg.
     *
     * TODO: Implement this function
     */
    void test(Reg reg, Reg reg2);

    /**
     * Emits: push reg.
     *
     * TODO: Implement this function
     */
    void push(Reg reg);

    /**
     * Emits: pop reg.
     *
     * TODO: Implement this function
     */
    void pop(Reg reg);

    /**
     * Emits: call target (absolute address).
     *
     * TODO: Implement this function
     */
    void call(void* target);

    /**
     * Emits: call reg.
     *
     * TODO: Implement this function
     */
    void call(Reg reg);

    /**
     * Emits: ret.
     *
     * TODO: Implement this function
     */
    void ret();

    /**
     * Emits: jmp label.
     *
     * TODO: Implement this function
     */
    void jmp(Label& label);

    /**
     * Emits: jcc label (conditional jump).
     *
     * TODO: Implement this function
     */
    void jcc(Cond cond, Label& label);

    /**
     * Emits: setcc reg (set byte based on condition).
     *
     * TODO: Implement this function
     */
    void setcc(Cond cond, Reg reg);

    /**
     * Emits: xor dst, src.
     *
     * TODO: Implement this function
     */
    void xor_(Reg dst, Reg src);

    /**
     * Emits: and dst, src.
     *
     * TODO: Implement this function
     */
    void and_(Reg dst, Reg src);

    /**
     * Emits: or dst, src.
     *
     * TODO: Implement this function
     */
    void or_(Reg dst, Reg src);

    /**
     * Emits: not reg.
     *
     * TODO: Implement this function
     */
    void not_(Reg reg);

    /**
     * Emits: neg reg.
     *
     * TODO: Implement this function
     */
    void neg(Reg reg);

    /**
     * Emits: inc reg.
     *
     * TODO: Implement this function
     */
    void inc(Reg reg);

    /**
     * Emits: dec reg.
     *
     * TODO: Implement this function
     */
    void dec(Reg reg);

    /**
     * Emits: shl reg, imm.
     *
     * TODO: Implement this function
     */
    void shl(Reg reg, uint8_t imm);

    /**
     * Emits: shr reg, imm.
     *
     * TODO: Implement this function
     */
    void shr(Reg reg, uint8_t imm);

    /**
     * Emits: sar reg, imm (arithmetic shift right).
     *
     * TODO: Implement this function
     */
    void sar(Reg reg, uint8_t imm);

    /**
     * Emits: lea dst, [src + offset].
     *
     * TODO: Implement this function
     */
    void lea(Reg dst, Reg src, int32_t offset);

    /**
     * Emits: nop.
     */
    void nop();

    // ==================== Label Management ====================

    /**
     * Creates a new label.
     */
    Label create_label();

    /**
     * Binds a label to the current position.
     *
     * TODO: Implement this function
     */
    void bind(Label& label);

    // ==================== Finalization ====================

    /**
     * Finalizes and returns executable code.
     *
     * TODO: Implement this function
     */
    template<typename Fn>
    Fn finalize() {
        auto mem = buffer_.finalize();
        auto fn = mem->as_function<Fn>();
        // Transfer ownership to prevent deallocation
        // In real code, you'd manage this differently
        mem.release();
        return fn;
    }

    /**
     * Returns the code buffer for inspection.
     */
    const CodeBuffer& buffer() const { return buffer_; }

    /**
     * Returns current code size.
     */
    size_t size() const { return buffer_.position(); }

    /**
     * Dumps generated code as hex.
     */
    std::string dump_hex() const;

private:
    CodeBuffer buffer_;

    // Helper methods for encoding

    /**
     * Returns true if register needs REX.B prefix.
     */
    static bool needs_rex_b(Reg reg) { return static_cast<uint8_t>(reg) >= 8; }

    /**
     * Returns true if register needs REX.R prefix.
     */
    static bool needs_rex_r(Reg reg) { return static_cast<uint8_t>(reg) >= 8; }

    /**
     * Emits REX prefix if needed.
     */
    void emit_rex(bool w, Reg reg, Reg rm);
    void emit_rex(bool w, Reg rm);

    /**
     * Emits ModR/M byte.
     */
    void emit_modrm(uint8_t mod, Reg reg, Reg rm);
    void emit_modrm(uint8_t mod, uint8_t reg, Reg rm);

    /**
     * Emits SIB byte.
     */
    void emit_sib(uint8_t scale, Reg index, Reg base);

    /**
     * Gets the low 3 bits of a register.
     */
    static uint8_t reg_low(Reg reg) { return static_cast<uint8_t>(reg) & 0x7; }
};

} // namespace mini_jit
