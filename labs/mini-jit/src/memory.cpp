#include "mini_jit/memory.hpp"

#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace mini_jit {

size_t get_page_size() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

/**
 * Allocates executable memory using mmap.
 *
 * TODO: Implement this function
 * Use mmap with PROT_READ | PROT_WRITE initially.
 * Round size up to page boundary.
 */
void* allocate_executable(size_t size) {
    (void)size;
    // TODO: Implement
    // Hint:
    // - Round size up to page boundary
    // - Use mmap with MAP_PRIVATE | MAP_ANONYMOUS
    // - Start with PROT_READ | PROT_WRITE
    throw std::runtime_error("TODO: implement allocate_executable");
}

/**
 * Deallocates executable memory.
 *
 * TODO: Implement this function
 */
void deallocate_executable(void* ptr, size_t size) {
    (void)ptr;
    (void)size;
    // TODO: Implement
    // Hint: Use munmap
    throw std::runtime_error("TODO: implement deallocate_executable");
}

// ExecutableMemory implementation

ExecutableMemory::ExecutableMemory(size_t size) {
    size_t page_size = get_page_size();
    size_ = (size + page_size - 1) & ~(page_size - 1);
    data_ = static_cast<uint8_t*>(allocate_executable(size_));
}

ExecutableMemory::~ExecutableMemory() {
    if (data_) {
        deallocate_executable(data_, size_);
    }
}

ExecutableMemory::ExecutableMemory(ExecutableMemory&& other) noexcept
    : data_(other.data_), size_(other.size_), executable_(other.executable_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.executable_ = false;
}

ExecutableMemory& ExecutableMemory::operator=(ExecutableMemory&& other) noexcept {
    if (this != &other) {
        if (data_) {
            deallocate_executable(data_, size_);
        }
        data_ = other.data_;
        size_ = other.size_;
        executable_ = other.executable_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.executable_ = false;
    }
    return *this;
}

/**
 * Makes the memory executable (and read-only).
 *
 * TODO: Implement this function
 * Call mprotect to change from RW to RX.
 */
void ExecutableMemory::make_executable() {
    // TODO: Implement
    // Hint: Use mprotect with PROT_READ | PROT_EXEC
    throw std::runtime_error("TODO: implement make_executable");
}

/**
 * Makes the memory writable (and non-executable).
 *
 * TODO: Implement this function
 */
void ExecutableMemory::make_writable() {
    // TODO: Implement
    // Hint: Use mprotect with PROT_READ | PROT_WRITE
    throw std::runtime_error("TODO: implement make_writable");
}

// CodeBuffer implementation

CodeBuffer::CodeBuffer(size_t initial_capacity) {
    code_.reserve(initial_capacity);
}

void CodeBuffer::emit8(uint8_t value) {
    code_.push_back(value);
}

void CodeBuffer::emit16(uint16_t value) {
    code_.push_back(value & 0xFF);
    code_.push_back((value >> 8) & 0xFF);
}

void CodeBuffer::emit32(uint32_t value) {
    code_.push_back(value & 0xFF);
    code_.push_back((value >> 8) & 0xFF);
    code_.push_back((value >> 16) & 0xFF);
    code_.push_back((value >> 24) & 0xFF);
}

void CodeBuffer::emit64(uint64_t value) {
    emit32(static_cast<uint32_t>(value));
    emit32(static_cast<uint32_t>(value >> 32));
}

void CodeBuffer::emit(const uint8_t* data, size_t len) {
    code_.insert(code_.end(), data, data + len);
}

/**
 * Patches a 32-bit value at an offset.
 *
 * TODO: Implement this function
 */
void CodeBuffer::patch32(size_t offset, uint32_t value) {
    (void)offset;
    (void)value;
    // TODO: Implement
    // Hint: Write 4 bytes at the given offset
    throw std::runtime_error("TODO: implement patch32");
}

/**
 * Copies code to executable memory.
 *
 * TODO: Implement this function
 */
std::unique_ptr<ExecutableMemory> CodeBuffer::finalize() {
    // TODO: Implement
    // Hint:
    // 1. Create ExecutableMemory of appropriate size
    // 2. Copy code to memory
    // 3. Make memory executable
    // 4. Return
    throw std::runtime_error("TODO: implement finalize");
}

} // namespace mini_jit
