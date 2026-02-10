/**
 * Expression calculator example.
 *
 * Demonstrates JIT compilation of arithmetic expressions.
 */

#include "mini_jit/jit.hpp"

#include <iostream>
#include <string>

using namespace mini_jit;

void basic_eval() {
    std::cout << "=== Basic Expression Evaluation ===\n";

    JIT jit;

    // Simple expressions
    std::cout << "2 + 3 = " << jit.eval("2 + 3") << std::endl;
    std::cout << "10 - 4 = " << jit.eval("10 - 4") << std::endl;
    std::cout << "6 * 7 = " << jit.eval("6 * 7") << std::endl;
    std::cout << "20 / 4 = " << jit.eval("20 / 4") << std::endl;

    // Complex expressions
    std::cout << "2 + 3 * 4 = " << jit.eval("2 + 3 * 4") << std::endl;
    std::cout << "(2 + 3) * 4 = " << jit.eval("(2 + 3) * 4") << std::endl;
    std::cout << "10 - 3 - 2 = " << jit.eval("10 - 3 - 2") << std::endl;
}

void compile_with_variables() {
    std::cout << "\n=== Expressions with Variables ===\n";

    JIT jit;

    // Compile expression with variables
    auto add = jit.compile_expr<int64_t(*)(int64_t, int64_t)>(
        "x + y", {"x", "y"});

    std::cout << "add(3, 4) = " << add(3, 4) << std::endl;
    std::cout << "add(10, 20) = " << add(10, 20) << std::endl;

    // More complex expression
    auto quad = jit.compile_expr<int64_t(*)(int64_t, int64_t, int64_t)>(
        "a * x * x + b * x + c", {"a", "b", "c"});

    // Note: This treats the polynomial as f(a, b, c) with x=2
    // In a real implementation, you'd handle this differently
    std::cout << "Quadratic with a=1, b=2, c=1: "
              << quad(1, 2, 1) << std::endl;
}

void interactive_mode() {
    std::cout << "\n=== Interactive Mode ===\n";
    std::cout << "Enter expressions (or 'quit' to exit):\n";

    JIT jit;
    std::string line;

    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line) || line == "quit") {
            break;
        }

        if (line.empty()) continue;

        try {
            int64_t result = jit.eval(line);
            std::cout << "= " << result << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Expression Calculator Example\n";
    std::cout << "=============================\n\n";

    try {
        basic_eval();
        compile_with_variables();

        if (argc > 1 && std::string(argv[1]) == "-i") {
            interactive_mode();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "(This is expected until TODO functions are implemented)\n";
        return 1;
    }

    return 0;
}
