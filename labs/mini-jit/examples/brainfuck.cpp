/**
 * Brainfuck JIT compiler example.
 *
 * Demonstrates JIT compilation of Brainfuck programs.
 */

#include "mini_jit/jit.hpp"

#include <iostream>
#include <string>

using namespace mini_jit;

void run_hello_world() {
    std::cout << "=== Brainfuck Hello World ===\n";

    // Hello World in Brainfuck
    const std::string hello_world =
        "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]"
        ">>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";

    JIT jit;
    std::cout << "Running: ";
    jit.run_brainfuck(hello_world);
    std::cout << std::endl;
}

void run_fibonacci() {
    std::cout << "\n=== Brainfuck Fibonacci ===\n";

    // Prints first few Fibonacci numbers (as ASCII)
    const std::string fib =
        "+++++++++++>+>>>>++++++++++++++++++++++++++++++++++++++++++++>"
        "++++++++++++++++++++++++++++++++<<<<<<[>[>>>>>>+>+<<<<<<<-]>>>>>>>-]";

    JIT jit;
    std::cout << "Running Fibonacci: ";
    jit.run_brainfuck(fib);
    std::cout << std::endl;
}

int main() {
    std::cout << "Brainfuck JIT Compiler Example\n";
    std::cout << "==============================\n\n";

    try {
        run_hello_world();
        // run_fibonacci();  // Uncomment to test
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "(This is expected until TODO functions are implemented)\n";
        return 1;
    }

    return 0;
}
