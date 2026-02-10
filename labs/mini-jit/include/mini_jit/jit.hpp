#pragma once

#include "memory.hpp"
#include "codegen.hpp"
#include "compiler.hpp"
#include <functional>
#include <string>
#include <unordered_map>

namespace mini_jit {

/**
 * JIT compilation context.
 *
 * Manages compiled code and provides a high-level interface.
 */
class JIT {
public:
    JIT();
    ~JIT();

    /**
     * Compiles and runs an expression.
     *
     * TODO: Implement this function
     *
     * @param expr The expression string (e.g., "2 + 3 * 4")
     * @return The result of evaluating the expression
     */
    int64_t eval(const std::string& expr);

    /**
     * Compiles an expression with variables.
     *
     * TODO: Implement this function
     *
     * @param expr The expression string (e.g., "x + y * 2")
     * @param vars Variable names in order
     * @return A compiled function
     */
    template<typename Fn>
    Fn compile_expr(const std::string& expr, const std::vector<std::string>& vars) {
        // TODO: Implement
        return nullptr;
    }

    /**
     * Compiles and caches Brainfuck.
     *
     * TODO: Implement this function
     */
    BrainfuckCompiler::BFFunction compile_brainfuck(const std::string& source);

    /**
     * Runs a compiled Brainfuck program.
     *
     * TODO: Implement this function
     */
    void run_brainfuck(const std::string& source);

    /**
     * Clears all cached compilations.
     */
    void clear_cache();

    /**
     * Returns compilation statistics.
     */
    struct Stats {
        size_t functions_compiled = 0;
        size_t total_code_size = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
    };

    Stats stats() const { return stats_; }

private:
    // Cache of compiled Brainfuck programs
    std::unordered_map<std::string, BrainfuckCompiler::BFFunction> bf_cache_;

    // Brainfuck memory tape
    std::vector<uint8_t> bf_tape_;

    // Statistics
    Stats stats_;
};

/**
 * Helper: Creates a JIT-compiled function from a lambda.
 *
 * This is a simplified interface that compiles expressions.
 *
 * Example:
 *   auto add = jit_function<int64_t(int64_t, int64_t)>("a + b", {"a", "b"});
 *   int64_t result = add(3, 4);  // result = 7
 *
 * TODO: Implement this function
 */
template<typename Fn>
Fn jit_function(const std::string& expr, const std::vector<std::string>& params) {
    // TODO: Implement
    return nullptr;
}

/**
 * Disassembles machine code (for debugging).
 *
 * TODO: Implement this function (optional)
 * Can integrate with a disassembler library or output raw hex.
 */
std::string disassemble(const uint8_t* code, size_t size);

/**
 * Prints code bytes as hex dump.
 */
void hex_dump(const uint8_t* code, size_t size);

} // namespace mini_jit
