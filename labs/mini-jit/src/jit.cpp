#include "mini_jit/jit.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>

namespace mini_jit {

JIT::JIT() : bf_tape_(30000, 0) {}

JIT::~JIT() = default;

/**
 * Compiles and runs an expression.
 *
 * TODO: Implement this function
 */
int64_t JIT::eval(const std::string& expr) {
    (void)expr;
    // TODO: Implement
    // 1. Parse expression
    // 2. Compile to function
    // 3. Execute and return result
    throw std::runtime_error("TODO: implement JIT::eval");
}

/**
 * Compiles and caches Brainfuck.
 *
 * TODO: Implement this function
 */
BrainfuckCompiler::BFFunction JIT::compile_brainfuck(const std::string& source) {
    (void)source;
    // TODO: Implement
    // 1. Check cache
    // 2. If not cached, compile and cache
    // 3. Return function
    throw std::runtime_error("TODO: implement JIT::compile_brainfuck");
}

/**
 * Runs a compiled Brainfuck program.
 *
 * TODO: Implement this function
 */
void JIT::run_brainfuck(const std::string& source) {
    (void)source;
    // TODO: Implement
    // 1. Compile (with caching)
    // 2. Clear tape
    // 3. Execute
    throw std::runtime_error("TODO: implement JIT::run_brainfuck");
}

void JIT::clear_cache() {
    bf_cache_.clear();
}

/**
 * Disassembles machine code.
 *
 * TODO: Implement this function (optional)
 */
std::string disassemble(const uint8_t* code, size_t size) {
    (void)code;
    (void)size;
    // TODO: Implement (optional)
    // Could integrate with a disassembler library
    return "Disassembly not implemented";
}

void hex_dump(const uint8_t* code, size_t size) {
    for (size_t i = 0; i < size; i += 16) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << i << ": ";

        // Hex bytes
        for (size_t j = 0; j < 16 && i + j < size; ++j) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(code[i + j]) << " ";
        }

        // Padding for incomplete line
        for (size_t j = size - i; j < 16; ++j) {
            std::cout << "   ";
        }

        std::cout << " ";

        // ASCII representation
        for (size_t j = 0; j < 16 && i + j < size; ++j) {
            char c = static_cast<char>(code[i + j]);
            std::cout << (std::isprint(c) ? c : '.');
        }

        std::cout << std::endl;
    }
}

} // namespace mini_jit
