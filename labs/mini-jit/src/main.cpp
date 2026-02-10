#include "mini_jit/jit.hpp"

#include <iostream>
#include <string>

using namespace mini_jit;

void print_usage() {
    std::cout << "Mini-JIT - A minimal JIT compiler\n\n";
    std::cout << "Usage:\n";
    std::cout << "  mini_jit eval <expression>     Evaluate an expression\n";
    std::cout << "  mini_jit bf <source>           Run Brainfuck code\n";
    std::cout << "  mini_jit demo                  Run demo\n";
    std::cout << "\nExamples:\n";
    std::cout << "  mini_jit eval \"2 + 3 * 4\"\n";
    std::cout << "  mini_jit bf \"++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.\"\n";
}

void run_demo() {
    std::cout << "=== Mini-JIT Demo ===\n\n";

    // Demo 1: Simple addition
    std::cout << "1. JIT-compiling a simple addition function...\n";
    std::cout << "   (Implementation pending)\n\n";

    // Demo 2: Brainfuck Hello World
    std::cout << "2. JIT-compiling Brainfuck 'Hello World'...\n";
    std::cout << "   (Implementation pending)\n\n";

    // Demo 3: Expression evaluation
    std::cout << "3. Evaluating expression '2 + 3 * 4'...\n";
    std::cout << "   (Implementation pending)\n\n";

    std::cout << "Demo complete! Implement the TODO functions to see results.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    try {
        JIT jit;

        if (command == "eval" && argc >= 3) {
            std::string expr = argv[2];
            int64_t result = jit.eval(expr);
            std::cout << result << std::endl;
        } else if (command == "bf" && argc >= 3) {
            std::string source = argv[2];
            jit.run_brainfuck(source);
        } else if (command == "demo") {
            run_demo();
        } else {
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
