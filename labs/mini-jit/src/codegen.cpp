#include "mini_jit/codegen.hpp"

#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace mini_jit {

const char* reg_name(Reg reg) {
    static const char* names[] = {
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
    };
    return names[static_cast<size_t>(reg)];
}

CodeGen::CodeGen() : buffer_(4096) {}

// REX prefix encoding
void CodeGen::emit_rex(bool w, Reg reg, Reg rm) {
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;                              // REX.W
    if (needs_rex_r(reg)) rex |= 0x04;               // REX.R
    if (needs_rex_b(rm)) rex |= 0x01;                // REX.B
    if (rex != 0x40 || w) {
        buffer_.emit8(rex);
    }
}

void CodeGen::emit_rex(bool w, Reg rm) {
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;
    if (needs_rex_b(rm)) rex |= 0x01;
    if (rex != 0x40 || w) {
        buffer_.emit8(rex);
    }
}

// ModR/M encoding
void CodeGen::emit_modrm(uint8_t mod, Reg reg, Reg rm) {
    buffer_.emit8((mod << 6) | (reg_low(reg) << 3) | reg_low(rm));
}

void CodeGen::emit_modrm(uint8_t mod, uint8_t reg, Reg rm) {
    buffer_.emit8((mod << 6) | (reg << 3) | reg_low(rm));
}

// SIB encoding
void CodeGen::emit_sib(uint8_t scale, Reg index, Reg base) {
    buffer_.emit8((scale << 6) | (reg_low(index) << 3) | reg_low(base));
}

/**
 * Emits function prologue.
 *
 * TODO: Implement this function
 * push rbp; mov rbp, rsp; [sub rsp, stack_size]
 */
void CodeGen::prologue(size_t stack_size) {
    (void)stack_size;
    // TODO: Implement
    throw std::runtime_error("TODO: implement prologue");
}

/**
 * Emits function epilogue.
 *
 * TODO: Implement this function
 * [add rsp, stack_size]; pop rbp; ret
 */
void CodeGen::epilogue(size_t stack_size) {
    (void)stack_size;
    // TODO: Implement
    throw std::runtime_error("TODO: implement epilogue");
}

/**
 * Emits: mov dst, src (register to register).
 *
 * TODO: Implement this function
 * Encoding: REX.W + 89 /r (mov r/m64, r64)
 *       or: REX.W + 8B /r (mov r64, r/m64)
 */
void CodeGen::mov(Reg dst, Reg src) {
    (void)dst;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement mov reg, reg");
}

/**
 * Emits: mov dst, imm64 (immediate to register).
 *
 * TODO: Implement this function
 * Encoding: REX.W + B8+rd io (mov r64, imm64)
 */
void CodeGen::mov(Reg dst, int64_t imm) {
    (void)dst;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement mov reg, imm");
}

/**
 * Emits: mov dst, [src + offset] (memory load).
 *
 * TODO: Implement this function
 */
void CodeGen::mov_load(Reg dst, Reg src, int32_t offset) {
    (void)dst;
    (void)src;
    (void)offset;
    // TODO: Implement
    throw std::runtime_error("TODO: implement mov_load");
}

/**
 * Emits: mov [dst + offset], src (memory store).
 *
 * TODO: Implement this function
 */
void CodeGen::mov_store(Reg dst, int32_t offset, Reg src) {
    (void)dst;
    (void)offset;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement mov_store");
}

/**
 * Emits: mov byte [dst + offset], imm8.
 *
 * TODO: Implement this function
 */
void CodeGen::mov_store8(Reg dst, int32_t offset, uint8_t imm) {
    (void)dst;
    (void)offset;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement mov_store8");
}

/**
 * Emits: add dst, src.
 *
 * TODO: Implement this function
 * Encoding: REX.W + 01 /r (add r/m64, r64)
 */
void CodeGen::add(Reg dst, Reg src) {
    (void)dst;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement add reg, reg");
}

/**
 * Emits: add dst, imm32.
 *
 * TODO: Implement this function
 */
void CodeGen::add(Reg dst, int32_t imm) {
    (void)dst;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement add reg, imm");
}

/**
 * Emits: sub dst, src.
 *
 * TODO: Implement this function
 */
void CodeGen::sub(Reg dst, Reg src) {
    (void)dst;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement sub reg, reg");
}

/**
 * Emits: sub dst, imm32.
 *
 * TODO: Implement this function
 */
void CodeGen::sub(Reg dst, int32_t imm) {
    (void)dst;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement sub reg, imm");
}

/**
 * Emits: imul dst, src.
 *
 * TODO: Implement this function
 */
void CodeGen::imul(Reg dst, Reg src) {
    (void)dst;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement imul");
}

/**
 * Emits: idiv src.
 *
 * TODO: Implement this function
 */
void CodeGen::idiv(Reg src) {
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement idiv");
}

/**
 * Emits: cqo.
 *
 * TODO: Implement this function
 */
void CodeGen::cqo() {
    // TODO: Implement
    throw std::runtime_error("TODO: implement cqo");
}

/**
 * Emits: cmp left, right.
 *
 * TODO: Implement this function
 */
void CodeGen::cmp(Reg left, Reg right) {
    (void)left;
    (void)right;
    // TODO: Implement
    throw std::runtime_error("TODO: implement cmp reg, reg");
}

/**
 * Emits: cmp reg, imm32.
 *
 * TODO: Implement this function
 */
void CodeGen::cmp(Reg reg, int32_t imm) {
    (void)reg;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement cmp reg, imm");
}

/**
 * Emits: test reg, reg.
 *
 * TODO: Implement this function
 */
void CodeGen::test(Reg reg, Reg reg2) {
    (void)reg;
    (void)reg2;
    // TODO: Implement
    throw std::runtime_error("TODO: implement test");
}

/**
 * Emits: push reg.
 *
 * TODO: Implement this function
 * Encoding: 50+rd (push r64)
 */
void CodeGen::push(Reg reg) {
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement push");
}

/**
 * Emits: pop reg.
 *
 * TODO: Implement this function
 * Encoding: 58+rd (pop r64)
 */
void CodeGen::pop(Reg reg) {
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement pop");
}

/**
 * Emits: call target.
 *
 * TODO: Implement this function
 */
void CodeGen::call(void* target) {
    (void)target;
    // TODO: Implement
    // Hint: mov rax, target; call rax
    throw std::runtime_error("TODO: implement call ptr");
}

/**
 * Emits: call reg.
 *
 * TODO: Implement this function
 */
void CodeGen::call(Reg reg) {
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement call reg");
}

/**
 * Emits: ret.
 *
 * TODO: Implement this function
 * Encoding: C3 (ret)
 */
void CodeGen::ret() {
    // TODO: Implement
    throw std::runtime_error("TODO: implement ret");
}

/**
 * Emits: jmp label.
 *
 * TODO: Implement this function
 */
void CodeGen::jmp(Label& label) {
    (void)label;
    // TODO: Implement
    throw std::runtime_error("TODO: implement jmp");
}

/**
 * Emits: jcc label.
 *
 * TODO: Implement this function
 */
void CodeGen::jcc(Cond cond, Label& label) {
    (void)cond;
    (void)label;
    // TODO: Implement
    throw std::runtime_error("TODO: implement jcc");
}

/**
 * Emits: setcc reg.
 *
 * TODO: Implement this function
 */
void CodeGen::setcc(Cond cond, Reg reg) {
    (void)cond;
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement setcc");
}

/**
 * Emits: xor dst, src.
 *
 * TODO: Implement this function
 */
void CodeGen::xor_(Reg dst, Reg src) {
    (void)dst;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement xor");
}

/**
 * Emits: and dst, src.
 *
 * TODO: Implement this function
 */
void CodeGen::and_(Reg dst, Reg src) {
    (void)dst;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement and");
}

/**
 * Emits: or dst, src.
 *
 * TODO: Implement this function
 */
void CodeGen::or_(Reg dst, Reg src) {
    (void)dst;
    (void)src;
    // TODO: Implement
    throw std::runtime_error("TODO: implement or");
}

/**
 * Emits: not reg.
 *
 * TODO: Implement this function
 */
void CodeGen::not_(Reg reg) {
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement not");
}

/**
 * Emits: neg reg.
 *
 * TODO: Implement this function
 */
void CodeGen::neg(Reg reg) {
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement neg");
}

/**
 * Emits: inc reg.
 *
 * TODO: Implement this function
 */
void CodeGen::inc(Reg reg) {
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement inc");
}

/**
 * Emits: dec reg.
 *
 * TODO: Implement this function
 */
void CodeGen::dec(Reg reg) {
    (void)reg;
    // TODO: Implement
    throw std::runtime_error("TODO: implement dec");
}

/**
 * Emits: shl reg, imm.
 *
 * TODO: Implement this function
 */
void CodeGen::shl(Reg reg, uint8_t imm) {
    (void)reg;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement shl");
}

/**
 * Emits: shr reg, imm.
 *
 * TODO: Implement this function
 */
void CodeGen::shr(Reg reg, uint8_t imm) {
    (void)reg;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement shr");
}

/**
 * Emits: sar reg, imm.
 *
 * TODO: Implement this function
 */
void CodeGen::sar(Reg reg, uint8_t imm) {
    (void)reg;
    (void)imm;
    // TODO: Implement
    throw std::runtime_error("TODO: implement sar");
}

/**
 * Emits: lea dst, [src + offset].
 *
 * TODO: Implement this function
 */
void CodeGen::lea(Reg dst, Reg src, int32_t offset) {
    (void)dst;
    (void)src;
    (void)offset;
    // TODO: Implement
    throw std::runtime_error("TODO: implement lea");
}

void CodeGen::nop() {
    buffer_.emit8(0x90);
}

Label CodeGen::create_label() {
    return Label{};
}

/**
 * Binds a label to the current position.
 *
 * TODO: Implement this function
 */
void CodeGen::bind(Label& label) {
    (void)label;
    // TODO: Implement
    // Hint:
    // 1. Set label.offset to current position
    // 2. Patch all pending references
    throw std::runtime_error("TODO: implement bind");
}

std::string CodeGen::dump_hex() const {
    std::ostringstream oss;
    const auto& code = buffer_.code();
    for (size_t i = 0; i < code.size(); ++i) {
        if (i > 0 && i % 16 == 0) oss << "\n";
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(code[i]) << " ";
    }
    return oss.str();
}

} // namespace mini_jit
