#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

namespace mini_jit {

/**
 * Executable memory region.
 *
 * Manages memory that can be written to and then executed.
 * Uses mmap on Unix systems.
 */
class ExecutableMemory {
public:
    /**
     * Creates an executable memory region.
     *
     * @param size The size in bytes (will be rounded up to page size)
     */
    explicit ExecutableMemory(size_t size);

    ~ExecutableMemory();

    // Non-copyable
    ExecutableMemory(const ExecutableMemory&) = delete;
    ExecutableMemory& operator=(const ExecutableMemory&) = delete;

    // Movable
    ExecutableMemory(ExecutableMemory&& other) noexcept;
    ExecutableMemory& operator=(ExecutableMemory&& other) noexcept;

    /**
     * Returns pointer to the memory region.
     */
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }

    /**
     * Returns the size of the memory region.
     */
    size_t size() const { return size_; }

    /**
     * Makes the memory executable (and read-only).
     *
     * TODO: Implement this function
     * Call mprotect to change from RW to RX.
     */
    void make_executable();

    /**
     * Makes the memory writable (and non-executable).
     *
     * TODO: Implement this function
     */
    void make_writable();

    /**
     * Returns true if memory is currently executable.
     */
    bool is_executable() const { return executable_; }

    /**
     * Casts the memory to a function pointer.
     */
    template<typename Fn>
    Fn as_function() const {
        return reinterpret_cast<Fn>(data_);
    }

private:
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    bool executable_ = false;
};

/**
 * Code buffer that automatically grows.
 */
class CodeBuffer {
public:
    CodeBuffer() = default;
    explicit CodeBuffer(size_t initial_capacity);

    /**
     * Emits a single byte.
     */
    void emit8(uint8_t value);

    /**
     * Emits a 16-bit value (little endian).
     */
    void emit16(uint16_t value);

    /**
     * Emits a 32-bit value (little endian).
     */
    void emit32(uint32_t value);

    /**
     * Emits a 64-bit value (little endian).
     */
    void emit64(uint64_t value);

    /**
     * Emits raw bytes.
     */
    void emit(const uint8_t* data, size_t len);

    /**
     * Returns current position (offset).
     */
    size_t position() const { return code_.size(); }

    /**
     * Returns the code buffer.
     */
    const std::vector<uint8_t>& code() const { return code_; }
    std::vector<uint8_t>& code() { return code_; }

    /**
     * Patches a 32-bit value at an offset.
     *
     * TODO: Implement this function
     */
    void patch32(size_t offset, uint32_t value);

    /**
     * Copies code to executable memory.
     *
     * TODO: Implement this function
     */
    std::unique_ptr<ExecutableMemory> finalize();

private:
    std::vector<uint8_t> code_;
};

/**
 * Gets the system page size.
 */
size_t get_page_size();

/**
 * Allocates executable memory using mmap.
 *
 * TODO: Implement this function
 * Use mmap with PROT_READ | PROT_WRITE initially.
 */
void* allocate_executable(size_t size);

/**
 * Deallocates executable memory.
 *
 * TODO: Implement this function
 */
void deallocate_executable(void* ptr, size_t size);

} // namespace mini_jit
