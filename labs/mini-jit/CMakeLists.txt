cmake_minimum_required(VERSION 3.16)
project(mini-jit VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(MINI_JIT_BUILD_TESTS "Build tests" ON)
option(MINI_JIT_BUILD_EXAMPLES "Build examples" ON)

# Source files
set(MINI_JIT_SOURCES
    src/memory.cpp
    src/codegen.cpp
    src/compiler.cpp
    src/jit.cpp
)

# Header files
set(MINI_JIT_HEADERS
    include/mini_jit/memory.hpp
    include/mini_jit/codegen.hpp
    include/mini_jit/compiler.hpp
    include/mini_jit/jit.hpp
)

# Library
add_library(mini_jit ${MINI_JIT_SOURCES})
target_include_directories(mini_jit PUBLIC include)

# Main executable
add_executable(mini_jit_main src/main.cpp)
target_link_libraries(mini_jit_main mini_jit)

# Examples
if(MINI_JIT_BUILD_EXAMPLES)
    add_executable(example_bf examples/brainfuck.cpp)
    target_link_libraries(example_bf mini_jit)

    add_executable(example_calc examples/calculator.cpp)
    target_link_libraries(example_calc mini_jit)
endif()

# Tests
if(MINI_JIT_BUILD_TESTS)
    enable_testing()
    add_executable(test_codegen tests/test_codegen.cpp)
    target_link_libraries(test_codegen mini_jit)
    add_test(NAME test_codegen COMMAND test_codegen)
endif()
