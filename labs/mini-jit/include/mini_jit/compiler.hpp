#pragma once

#include "codegen.hpp"
#include <string>
#include <vector>
#include <memory>
#include <variant>

namespace mini_jit {

// ==================== Expression Compiler ====================

/**
 * Expression AST node.
 */
struct Expr;

using ExprPtr = std::unique_ptr<Expr>;

struct Expr {
    enum class Kind {
        Number,
        Variable,
        BinaryOp,
        UnaryOp,
        Call,
    };

    Kind kind;

    // Number
    int64_t number_value = 0;

    // Variable
    std::string var_name;

    // BinaryOp
    char binary_op = 0;
    ExprPtr left;
    ExprPtr right;

    // UnaryOp
    char unary_op = 0;
    ExprPtr operand;

    // Call
    std::string func_name;
    std::vector<ExprPtr> args;

    static ExprPtr make_number(int64_t value);
    static ExprPtr make_variable(const std::string& name);
    static ExprPtr make_binary(char op, ExprPtr left, ExprPtr right);
    static ExprPtr make_unary(char op, ExprPtr operand);
    static ExprPtr make_call(const std::string& name, std::vector<ExprPtr> args);
};

/**
 * Parses an expression string.
 *
 * Grammar:
 * expr    = term (('+' | '-') term)*
 * term    = factor (('*' | '/') factor)*
 * factor  = '-' factor | atom
 * atom    = NUMBER | VARIABLE | '(' expr ')' | CALL
 *
 * TODO: Implement this function
 */
ExprPtr parse_expr(const std::string& input);

/**
 * Expression compiler.
 *
 * Compiles an expression to a function that takes variables as arguments.
 */
class ExprCompiler {
public:
    ExprCompiler();

    /**
     * Compiles an expression.
     *
     * TODO: Implement this function
     * Generate code that evaluates the expression and returns result in RAX.
     */
    void compile(const Expr& expr);

    /**
     * Adds a variable binding.
     * Variables are passed as function arguments.
     */
    void add_variable(const std::string& name, size_t arg_index);

    /**
     * Finalizes and returns a function.
     *
     * @tparam Fn Function signature, e.g., int64_t(*)(int64_t, int64_t)
     */
    template<typename Fn>
    Fn finalize() {
        codegen_.epilogue();
        return codegen_.finalize<Fn>();
    }

private:
    CodeGen codegen_;
    std::unordered_map<std::string, size_t> variables_;
    size_t stack_offset_ = 0;

    /**
     * Compiles an expression recursively.
     *
     * TODO: Implement this function
     * Result is left in RAX.
     */
    void compile_expr(const Expr& expr);

    /**
     * Gets the register for an argument index.
     */
    Reg arg_reg(size_t index);
};

// ==================== Brainfuck Compiler ====================

/**
 * Brainfuck instruction.
 */
struct BFInstr {
    enum class Op {
        Add,      // + (add to current cell)
        Sub,      // - (subtract from current cell)
        Left,     // < (move pointer left)
        Right,    // > (move pointer right)
        Output,   // . (output current cell)
        Input,    // , (input to current cell)
        LoopStart,// [ (start loop)
        LoopEnd,  // ] (end loop)
    };

    Op op;
    int count = 1; // For run-length optimization
};

/**
 * Parses Brainfuck source.
 *
 * TODO: Implement this function
 * Optionally apply run-length encoding for consecutive +, -, <, >.
 */
std::vector<BFInstr> parse_brainfuck(const std::string& source);

/**
 * Brainfuck JIT compiler.
 *
 * Compiles Brainfuck to x86-64 machine code.
 *
 * Memory layout:
 * - RDI: pointer to memory tape (passed as argument)
 * - RBX: current position on tape (callee-saved)
 */
class BrainfuckCompiler {
public:
    BrainfuckCompiler();

    /**
     * Compiles Brainfuck source.
     *
     * TODO: Implement this function
     */
    void compile(const std::string& source);

    /**
     * Compiles parsed instructions.
     *
     * TODO: Implement this function
     */
    void compile(const std::vector<BFInstr>& instructions);

    /**
     * Finalizes and returns a function.
     *
     * Function signature: void bf_program(uint8_t* tape)
     */
    using BFFunction = void(*)(uint8_t*);

    BFFunction finalize();

    /**
     * Sets the putchar function pointer.
     */
    void set_putchar(void* fn) { putchar_fn_ = fn; }

    /**
     * Sets the getchar function pointer.
     */
    void set_getchar(void* fn) { getchar_fn_ = fn; }

private:
    CodeGen codegen_;
    std::vector<Label> loop_stack_;
    void* putchar_fn_ = nullptr;
    void* getchar_fn_ = nullptr;

    /**
     * Compiles a single instruction.
     *
     * TODO: Implement this function
     */
    void compile_instr(const BFInstr& instr);

    /**
     * Compiles loop start.
     *
     * TODO: Implement this function
     */
    void compile_loop_start();

    /**
     * Compiles loop end.
     *
     * TODO: Implement this function
     */
    void compile_loop_end();
};

// ==================== Simple Stack Machine ====================

/**
 * Stack machine instruction.
 */
struct StackInstr {
    enum class Op {
        Push,     // Push immediate
        Pop,      // Pop and discard
        Add,      // Pop two, push sum
        Sub,      // Pop two, push difference
        Mul,      // Pop two, push product
        Div,      // Pop two, push quotient
        Neg,      // Negate top
        Dup,      // Duplicate top
        Swap,     // Swap top two
        Load,     // Load from memory
        Store,    // Store to memory
        Call,     // Call function
        Ret,      // Return
    };

    Op op;
    int64_t operand = 0;
    void* fn_ptr = nullptr;
};

/**
 * Stack machine compiler.
 */
class StackMachineCompiler {
public:
    StackMachineCompiler();

    /**
     * Compiles stack machine instructions.
     *
     * TODO: Implement this function
     */
    void compile(const std::vector<StackInstr>& instructions);

    /**
     * Finalizes and returns a function.
     */
    template<typename Fn>
    Fn finalize() {
        return codegen_.finalize<Fn>();
    }

private:
    CodeGen codegen_;

    /**
     * Compiles a single instruction.
     *
     * TODO: Implement this function
     */
    void compile_instr(const StackInstr& instr);
};

} // namespace mini_jit
