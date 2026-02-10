#include "mini_jit/compiler.hpp"

#include <stdexcept>
#include <cctype>

namespace mini_jit {

// ==================== Expression AST ====================

ExprPtr Expr::make_number(int64_t value) {
    auto expr = std::make_unique<Expr>();
    expr->kind = Kind::Number;
    expr->number_value = value;
    return expr;
}

ExprPtr Expr::make_variable(const std::string& name) {
    auto expr = std::make_unique<Expr>();
    expr->kind = Kind::Variable;
    expr->var_name = name;
    return expr;
}

ExprPtr Expr::make_binary(char op, ExprPtr left, ExprPtr right) {
    auto expr = std::make_unique<Expr>();
    expr->kind = Kind::BinaryOp;
    expr->binary_op = op;
    expr->left = std::move(left);
    expr->right = std::move(right);
    return expr;
}

ExprPtr Expr::make_unary(char op, ExprPtr operand) {
    auto expr = std::make_unique<Expr>();
    expr->kind = Kind::UnaryOp;
    expr->unary_op = op;
    expr->operand = std::move(operand);
    return expr;
}

ExprPtr Expr::make_call(const std::string& name, std::vector<ExprPtr> args) {
    auto expr = std::make_unique<Expr>();
    expr->kind = Kind::Call;
    expr->func_name = name;
    expr->args = std::move(args);
    return expr;
}

/**
 * Parses an expression string.
 *
 * TODO: Implement this function
 * Grammar:
 * expr    = term (('+' | '-') term)*
 * term    = factor (('*' | '/' | '%') factor)*
 * factor  = '-' factor | atom
 * atom    = NUMBER | VARIABLE | '(' expr ')'
 */
ExprPtr parse_expr(const std::string& input) {
    (void)input;
    // TODO: Implement recursive descent parser
    throw std::runtime_error("TODO: implement parse_expr");
}

// ==================== Expression Compiler ====================

ExprCompiler::ExprCompiler() {
    codegen_.prologue();
}

void ExprCompiler::add_variable(const std::string& name, size_t arg_index) {
    variables_[name] = arg_index;
}

Reg ExprCompiler::arg_reg(size_t index) {
    // System V AMD64 calling convention
    static const Reg arg_regs[] = {
        Reg::RDI, Reg::RSI, Reg::RDX, Reg::RCX, Reg::R8, Reg::R9
    };
    if (index >= 6) {
        throw std::runtime_error("Too many arguments (stack args not implemented)");
    }
    return arg_regs[index];
}

/**
 * Compiles an expression.
 *
 * TODO: Implement this function
 */
void ExprCompiler::compile(const Expr& expr) {
    (void)expr;
    // TODO: Implement
    throw std::runtime_error("TODO: implement ExprCompiler::compile");
}

/**
 * Compiles an expression recursively.
 *
 * TODO: Implement this function
 * Result is left in RAX.
 */
void ExprCompiler::compile_expr(const Expr& expr) {
    (void)expr;
    // TODO: Implement
    // Handle each expression type:
    // - Number: mov rax, imm
    // - Variable: mov rax, arg_reg
    // - BinaryOp: compile left, push, compile right, pop, operate
    // - UnaryOp: compile operand, negate
    throw std::runtime_error("TODO: implement compile_expr");
}

// ==================== Brainfuck Compiler ====================

/**
 * Parses Brainfuck source.
 *
 * TODO: Implement this function
 */
std::vector<BFInstr> parse_brainfuck(const std::string& source) {
    (void)source;
    // TODO: Implement
    // Optionally: run-length encode consecutive +, -, <, >
    throw std::runtime_error("TODO: implement parse_brainfuck");
}

BrainfuckCompiler::BrainfuckCompiler() {
    // Default to standard C library functions
    putchar_fn_ = reinterpret_cast<void*>(&putchar);
    getchar_fn_ = reinterpret_cast<void*>(&getchar);
}

/**
 * Compiles Brainfuck source.
 *
 * TODO: Implement this function
 */
void BrainfuckCompiler::compile(const std::string& source) {
    (void)source;
    // TODO: Implement
    // 1. Parse source
    // 2. Generate prologue (save RBX, load tape pointer)
    // 3. Compile instructions
    // 4. Generate epilogue
    throw std::runtime_error("TODO: implement BrainfuckCompiler::compile");
}

/**
 * Compiles parsed instructions.
 *
 * TODO: Implement this function
 */
void BrainfuckCompiler::compile(const std::vector<BFInstr>& instructions) {
    (void)instructions;
    // TODO: Implement
    throw std::runtime_error("TODO: implement BrainfuckCompiler::compile(instrs)");
}

BrainfuckCompiler::BFFunction BrainfuckCompiler::finalize() {
    return codegen_.finalize<BFFunction>();
}

/**
 * Compiles a single instruction.
 *
 * TODO: Implement this function
 *
 * Memory layout:
 * - RDI: tape pointer (passed as argument, saved to stack)
 * - RBX: current position on tape (callee-saved)
 *
 * + : add byte [rdi + rbx], count
 * - : sub byte [rdi + rbx], count
 * > : add rbx, count
 * < : sub rbx, count
 * . : mov dil, [rdi + rbx]; call putchar
 * , : call getchar; mov [rdi + rbx], al
 * [ : cmp byte [rdi + rbx], 0; je end_of_loop
 * ] : jmp start_of_loop
 */
void BrainfuckCompiler::compile_instr(const BFInstr& instr) {
    (void)instr;
    // TODO: Implement
    throw std::runtime_error("TODO: implement compile_instr");
}

/**
 * Compiles loop start.
 *
 * TODO: Implement this function
 */
void BrainfuckCompiler::compile_loop_start() {
    // TODO: Implement
    // 1. Create labels for loop start and end
    // 2. Bind loop start label
    // 3. Check if current cell is zero
    // 4. Jump to end if zero
    // 5. Push labels onto stack
    throw std::runtime_error("TODO: implement compile_loop_start");
}

/**
 * Compiles loop end.
 *
 * TODO: Implement this function
 */
void BrainfuckCompiler::compile_loop_end() {
    // TODO: Implement
    // 1. Pop labels from stack
    // 2. Jump to loop start
    // 3. Bind loop end label
    throw std::runtime_error("TODO: implement compile_loop_end");
}

// ==================== Stack Machine Compiler ====================

StackMachineCompiler::StackMachineCompiler() {}

/**
 * Compiles stack machine instructions.
 *
 * TODO: Implement this function
 */
void StackMachineCompiler::compile(const std::vector<StackInstr>& instructions) {
    (void)instructions;
    // TODO: Implement
    throw std::runtime_error("TODO: implement StackMachineCompiler::compile");
}

/**
 * Compiles a single instruction.
 *
 * TODO: Implement this function
 * Use RSP as stack pointer, RAX and RCX as temporaries.
 */
void StackMachineCompiler::compile_instr(const StackInstr& instr) {
    (void)instr;
    // TODO: Implement
    throw std::runtime_error("TODO: implement StackMachineCompiler::compile_instr");
}

} // namespace mini_jit
