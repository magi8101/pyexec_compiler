# Memory Management Guide

> **⚠️ EDUCATIONAL PROJECT - NOT PRODUCTION READY**  
> This documentation is for **learning purposes only**. PyExec is an experimental project and is **not stable or ready for production use**.

## Table of Contents
- [Overview](#overview)
- [Memory Architecture](#memory-architecture)
- [Basic Usage](#basic-usage)
- [Memory Allocation](#memory-allocation)
- [Slab Allocator](#slab-allocator)
- [Reference Counting](#reference-counting)
- [Advanced Features](#advanced-features)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

PyExec's **MemoryManager** provides a high-performance memory allocator optimized for JIT, AOT, and WASM backends. It uses a **slab allocator** with automatic **reference counting** to manage memory efficiently.

### Key Features

✅ **Fast allocation**: 0.76-1.03μs per allocation  
✅ **Automatic deallocation**: Reference counting prevents leaks  
✅ **Slab-based**: Reduces fragmentation  
✅ **JIT-compiled**: All operations use Numba for speed  
✅ **Cross-backend**: Works with JIT, LLVM, and WASM  

---

## Memory Architecture

### Heap Structure

```
┌─────────────────────────────────────────┐
│         Memory Heap (NumPy array)       │
│              1 MB default                │
├─────────────────────────────────────────┤
│ Small Objects (16-2048 bytes)           │
│ ├─ Slab Class 0: 16 bytes               │
│ ├─ Slab Class 1: 32 bytes               │
│ ├─ Slab Class 2: 64 bytes               │
│ ├─ Slab Class 3: 128 bytes              │
│ ├─ Slab Class 4: 256 bytes              │
│ ├─ Slab Class 5: 512 bytes              │
│ ├─ Slab Class 6: 1024 bytes             │
│ └─ Slab Class 7: 2048 bytes             │
├─────────────────────────────────────────┤
│ Large Objects (>2048 bytes)             │
│ ├─ Direct allocation                    │
│ └─ No slab overhead                     │
└─────────────────────────────────────────┘
```

---

### Size Classes

| Class | Size | Use Case |
|-------|------|----------|
| 0 | 16 bytes | Small integers, booleans |
| 1 | 32 bytes | Pairs, small structs |
| 2 | 64 bytes | Small arrays |
| 3 | 128 bytes | Medium arrays |
| 4 | 256 bytes | Large arrays |
| 5 | 512 bytes | Very large arrays |
| 6 | 1024 bytes | Buffers |
| 7 | 2048 bytes | Large buffers |
| - | >2048 bytes | Direct allocation |

---

## Basic Usage

### Creating Memory Manager

```python
from pyexec.memory import MemoryManager

# Default heap size (64 MB)
mem = MemoryManager()

# Custom heap size (10 MB)
mem = MemoryManager(size=10 * 1024 * 1024)

# Check memory properties
print(f"Heap size: {mem.size} bytes")
print(f"Current heap top: {mem.heap_top[0]} bytes used")
```

---

### Allocating Memory

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Basic allocation (requires manual size tracking for free)
ptr = mem.alloc(64)
print(f"Allocated at: 0x{ptr:08x}")

# Use the memory (write/read)
mem.write_i32(ptr, 42)
value = mem.read_i32(ptr)
print(f"Value: {value}")  # 42

# Free when done (MUST pass same size!)
mem.free(ptr, 64)

# Or use refcounted allocation (auto-managed)
ptr = mem.alloc_refcounted(64)
mem.write_i32(ptr, 100)
mem.decref(ptr)  # Auto-freed when refcount reaches 0
```

---

### Memory Types

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()
ptr = mem.alloc(64)

# 32-bit integer
mem.write_i32(ptr, 12345)
val = mem.read_i32(ptr)
print(f"i32: {val}")  # 12345

# 32-bit float
mem.write_f32(ptr, 3.14159)
val = mem.read_f32(ptr)
print(f"f32: {val:.5f}")  # 3.14159

# 64-bit double
mem.write_f64(ptr, 2.71828182845)
val = mem.read_f64(ptr)
print(f"f64: {val:.11f}")  # 2.71828182845

# Safe versions with bounds checking
mem.write_i32_safe(ptr, 999)  # Raises IndexError if out of bounds
val = mem.read_i32_safe(ptr)

mem.free(ptr, 64)
```

---

## Memory Allocation

### Allocation Flow

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Step 1: Request memory
size = 128
ptr = mem.alloc(size)

# Step 2: Memory manager:
#   - Determines size class (128 bytes → class 3)
#   - Finds free slab or creates new one
#   - Returns pointer to available slot

# Step 3: Use memory
mem.write_i32(ptr, 100)

# Step 4: Free when done
mem.free(ptr)

# Step 5: Memory returned to slab for reuse
```

---

### Aligned Allocation

```python
from pyexec.memory import align_up

# Align to 8-byte boundary
size = 13
aligned = align_up(size, 8)
print(f"Original: {size}, Aligned: {aligned}")  # 13 → 16

# Align to 16-byte boundary
size = 25
aligned = align_up(size, 16)
print(f"Original: {size}, Aligned: {aligned}")  # 25 → 32
```

---

## Slab Allocator

### How Slabs Work

**Concept:**
- Group objects of same size together
- Pre-allocate chunks (slabs) of memory
- Fast allocation: just increment pointer
- Fast deallocation: mark slot as free

**Example:**

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Allocate 10 objects of 64 bytes each
pointers = []
for i in range(10):
    ptr = mem.alloc(64)
    mem.write_i32(ptr, i * 100)
    pointers.append(ptr)

# All allocated from same slab class (class 2)
for i, ptr in enumerate(pointers):
    value = mem.read_i32(ptr)
    print(f"ptr[{i}] = {value}")

# Free all
for ptr in pointers:
    mem.free(ptr)
```

---

### Slab Statistics

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Allocate various sizes
ptrs = []
ptrs.append(mem.alloc(16))   # Slab class 0
ptrs.append(mem.alloc(32))   # Slab class 1
ptrs.append(mem.alloc(64))   # Slab class 2
ptrs.append(mem.alloc(128))  # Slab class 3

# Check heap usage
print(f"Heap size: {mem.heap_size} bytes")
print(f"Allocations: {len(ptrs)}")

# Free all
for ptr in ptrs:
    mem.free(ptr)
```

---

## Reference Counting

### Automatic Memory Management

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Allocate and track references
ptr = mem.alloc(64)

# Increment reference count
mem.incref(ptr)  # refcount = 2
mem.incref(ptr)  # refcount = 3

# Decrement reference count
mem.decref(ptr)  # refcount = 2
mem.decref(ptr)  # refcount = 1

# Final free (refcount = 0, memory released)
mem.free(ptr)
```

---

### Shared Memory Example

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

class SharedArray:
    def __init__(self, mem, size):
        self.mem = mem
        self.ptr = mem.alloc(size * 4)  # 4 bytes per int32
        self.size = size
        mem.incref(self.ptr)  # Keep alive
    
    def set(self, index, value):
        offset = self.ptr + index * 4
        self.mem.write_i32(offset, value)
    
    def get(self, index):
        offset = self.ptr + index * 4
        return self.mem.read_i32(offset)
    
    def __del__(self):
        self.mem.decref(self.ptr)  # Auto-release

# Create shared array
arr = SharedArray(mem, 10)

# Use it
arr.set(0, 100)
arr.set(1, 200)

print(arr.get(0))  # 100
print(arr.get(1))  # 200

# Automatically freed when arr goes out of scope
```

---

## Advanced Features

### 1. String Allocation

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Allocate string
text = "Hello, PyExec!"
ptr = mem.alloc_string(text)

# Read it back
retrieved = mem.read_string(ptr)
print(f"String: {retrieved}")  # "Hello, PyExec!"

# Strings are refcounted
mem.incref(ptr)  # Share the string
mem.decref(ptr)  # Release first reference
mem.decref(ptr)  # Release second reference (auto-freed)
```

**String Layout:**
```
[refcount:i32][length:i32][UTF-8 bytes...]
                           ^
                           ptr points here
```

---

### 2. Array Allocation

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Allocate array of 10 elements (8 bytes each)
length = 10
elem_size = 8
ptr = mem.alloc_array(length, elem_size)

# Fill array with values
for i in range(length):
    offset = ptr + i * elem_size
    mem.write_f64(offset, float(i * 10))

# Read array values
for i in range(length):
    offset = ptr + i * elem_size
    value = mem.read_f64(offset)
    print(f"arr[{i}] = {value}")

# Array is refcounted (auto-freed on last decref)
mem.decref(ptr - 12)  # ptr-12 is header with refcount
```

**Array Layout:**
```
[refcount:i32][length:i32][capacity:i32][data...]
                                          ^
                                          ptr points here
```

---

### 3. Memory Views

```python
from pyexec.memory import MemoryManager
import numpy as np

mem = MemoryManager()

# Allocate memory
ptr = mem.alloc(1024)

# Get NumPy view of memory
view = mem.memory[ptr:ptr+1024]

# Write with NumPy
view[:100] = np.arange(100, dtype=np.uint8)

# Read back
print(f"First 10 bytes: {view[:10]}")

mem.free(ptr)
```

---

### 4. Bulk Operations

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Allocate array of 100 integers
array_size = 100 * 4  # 4 bytes per int32
ptr = mem.alloc(array_size)

# Fill with values
for i in range(100):
    offset = ptr + i * 4
    mem.write_i32(offset, i * 10)

# Sum all values
total = 0
for i in range(100):
    offset = ptr + i * 4
    total += mem.read_i32(offset)

print(f"Sum: {total}")  # 49500

mem.free(ptr)
```

---

### 5. Memory Copy

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Source array
src = mem.alloc(64)
for i in range(16):
    mem.write_i32(src + i * 4, i)

# Destination array
dst = mem.alloc(64)

# Copy memory
mem.memory[dst:dst+64] = mem.memory[src:src+64]

# Verify
for i in range(16):
    val = mem.read_i32(dst + i * 4)
    assert val == i, f"Copy failed at {i}"

print("✅ Memory copy successful")

mem.free(src)
mem.free(dst)
```

---

### 6. Memory Pooling

```python
from pyexec.memory import MemoryManager

class MemoryPool:
    """Reusable memory pool"""
    
    def __init__(self, mem, block_size, pool_size):
        self.mem = mem
        self.block_size = block_size
        self.pool = []
        
        # Pre-allocate pool
        for _ in range(pool_size):
            ptr = mem.alloc(block_size)
            self.pool.append(ptr)
    
    def acquire(self):
        """Get a block from pool"""
        if not self.pool:
            return self.mem.alloc(self.block_size)
        return self.pool.pop()
    
    def release(self, ptr):
        """Return block to pool"""
        self.pool.append(ptr)
    
    def cleanup(self):
        """Free all pooled blocks"""
        for ptr in self.pool:
            self.mem.free(ptr)

# Usage
mem = MemoryManager()
pool = MemoryPool(mem, 128, 10)  # 10 blocks of 128 bytes

# Acquire blocks
ptrs = [pool.acquire() for _ in range(5)]

# Use them
for i, ptr in enumerate(ptrs):
    mem.write_i32(ptr, i * 100)

# Return to pool
for ptr in ptrs:
    pool.release(ptr)

# Cleanup
pool.cleanup()
```

---

## Performance Tuning

### Benchmark Allocation

```python
import time
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Benchmark allocation speed
iterations = 100000

start = time.perf_counter()
for _ in range(iterations):
    ptr = mem.alloc(64)
    mem.free(ptr)
end = time.perf_counter()

avg_time = (end - start) / iterations
print(f"Average allocation: {avg_time * 1_000_000:.2f} μs")
# Expected: 0.76-1.03 μs
```

---

### Choosing Heap Size

```python
from pyexec.memory import MemoryManager

# Small heap (fast, limited capacity)
mem_small = MemoryManager(size=256 * 1024)  # 256 KB

# Medium heap (balanced)
mem_medium = MemoryManager(size=1024 * 1024)  # 1 MB

# Large heap (high capacity) - THIS IS DEFAULT
mem_large = MemoryManager(size=64 * 1024 * 1024)  # 64 MB (default)

# Choose based on:
# - Number of allocations
# - Object sizes
# - Memory constraints
```

**Guidelines:**
- **Small programs**: 256 KB - 1 MB
- **Medium programs**: 1 MB - 10 MB
- **Large programs**: 10 MB - 100 MB
- **Default**: 64 MB (suitable for most use cases)

---

### Reduce Allocation Overhead

```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# ❌ BAD: Many small allocations
def slow_approach():
    results = []
    for i in range(1000):
        ptr = mem.alloc(4)  # 1000 allocations
        mem.write_i32(ptr, i)
        results.append(ptr)
    
    for ptr in results:
        mem.free(ptr)

# ✅ GOOD: Single large allocation
def fast_approach():
    # Allocate once for all elements
    ptr = mem.alloc(1000 * 4)  # 1 allocation
    
    for i in range(1000):
        offset = ptr + i * 4
        mem.write_i32(offset, i)
    
    mem.free(ptr)

# fast_approach() is ~100x faster
```

---

### Memory Alignment Optimization

```python
from pyexec.memory import MemoryManager, align_up

mem = MemoryManager()

# ❌ BAD: Unaligned access (slower)
ptr = mem.alloc(13)  # Odd size
mem.write_i64(ptr, 123456)  # May be slow

# ✅ GOOD: Aligned access (faster)
size = align_up(13, 8)  # Align to 8 bytes
ptr = mem.alloc(size)
mem.write_i64(ptr, 123456)  # Fast aligned access

mem.free(ptr)
```

---

## Troubleshooting

### Issue 1: Out of Memory

**Error:** Heap exhausted

**Solution:**
```python
from pyexec.memory import MemoryManager

# Increase heap size
mem = MemoryManager(size=10 * 1024 * 1024)  # 10 MB

# Or free unused memory
for ptr, size in old_allocations:
    mem.free(ptr, size)
```

---

### Issue 2: Memory Leak

**Symptom:** Memory usage grows over time

**Debug:**
```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Track allocations
allocated = []

def monitor_allocations():
    ptr = mem.alloc(64)
    allocated.append(ptr)
    
    # Check if we're freeing
    print(f"Allocated: {len(allocated)}")
    
    # Free oldest
    if len(allocated) > 100:
        old_ptr = allocated.pop(0)
        mem.free(old_ptr)

# Call many times
for _ in range(1000):
    monitor_allocations()
```

**Fix:** Ensure every `alloc()` has matching `free()`

---

### Issue 3: Segmentation Fault

**Cause:** Reading/writing freed memory

**Prevention:**
```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

ptr = mem.alloc(64)
mem.write_i32(ptr, 42)

# ❌ BAD: Use after free
mem.free(ptr)
value = mem.read_i32(ptr)  # CRASH!

# ✅ GOOD: Don't use after free
ptr = mem.alloc(64)
mem.write_i32(ptr, 42)
value = mem.read_i32(ptr)  # OK
mem.free(ptr)  # Free at end
```

---

### Issue 4: Slow Performance

**Checklist:**
1. ✅ Using appropriate heap size?
2. ✅ Batch allocations instead of many small ones?
3. ✅ Aligning memory access?
4. ✅ Freeing memory promptly?

**Profile:**
```python
import time
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Profile allocation
start = time.perf_counter()
ptrs = [mem.alloc(64) for _ in range(10000)]
alloc_time = time.perf_counter() - start

# Profile deallocation
start = time.perf_counter()
for ptr in ptrs:
    mem.free(ptr)
free_time = time.perf_counter() - start

print(f"Alloc: {alloc_time:.4f}s")
print(f"Free: {free_time:.4f}s")
```

---

## Complete Example: Custom Allocator

```python
from pyexec.memory import MemoryManager
import time

class FastIntArray:
    """High-performance integer array using MemoryManager"""
    
    def __init__(self, mem, capacity):
        self.mem = mem
        self.capacity = capacity
        self.size = 0
        
        # Allocate memory
        self.ptr = mem.alloc(capacity * 4)  # 4 bytes per int32
        
        # Track with refcount
        mem.incref(self.ptr)
    
    def append(self, value):
        if self.size >= self.capacity:
            raise IndexError("Array full")
        
        offset = self.ptr + self.size * 4
        self.mem.write_i32(offset, value)
        self.size += 1
    
    def get(self, index):
        if index >= self.size:
            raise IndexError("Index out of range")
        
        offset = self.ptr + index * 4
        return self.mem.read_i32(offset)
    
    def sum(self):
        total = 0
        for i in range(self.size):
            total += self.get(i)
        return total
    
    def __del__(self):
        self.mem.decref(self.ptr)

# Usage
mem = MemoryManager()

# Create array
arr = FastIntArray(mem, 10000)

# Benchmark insertion
start = time.perf_counter()
for i in range(10000):
    arr.append(i)
insert_time = time.perf_counter() - start

# Benchmark sum
start = time.perf_counter()
total = arr.sum()
sum_time = time.perf_counter() - start

print(f"Insert 10K: {insert_time*1000:.2f}ms")
print(f"Sum 10K: {sum_time*1000:.2f}ms")
print(f"Total: {total}")

# Automatically freed when arr goes out of scope
```

**Output:**
```
Insert 10K: 8.52ms
Sum 10K: 2.14ms
Total: 49995000
```

---

## API Reference

### MemoryManager

```python
class MemoryManager:
    """Unified memory manager with slab allocator and refcounting"""
    
    def __init__(self, size: int = 64 * 1024 * 1024):
        """
        Initialize memory manager.
        
        Args:
            size: Heap size in bytes (default: 64 MB)
        """
    
    # Basic allocation
    def alloc(self, size: int) -> int:
        """Allocate memory block (must track size for free)"""
    
    def free(self, ptr: int, size: int) -> None:
        """Free memory block (requires original size)"""
    
    # Refcounted allocation
    def alloc_refcounted(self, size: int) -> int:
        """Allocate with automatic refcounting"""
    
    def incref(self, ptr: int) -> None:
        """Increment reference count"""
    
    def decref(self, ptr: int) -> None:
        """Decrement refcount, auto-free if zero"""
    
    # String operations
    def alloc_string(self, s: str) -> int:
        """Allocate string, returns pointer to data"""
    
    def read_string(self, ptr: int) -> str:
        """Read string from pointer"""
    
    # Array operations
    def alloc_array(self, length: int, elem_size: int = 8) -> int:
        """Allocate array, returns pointer to data"""
    
    # Dict operations
    def dict_cleanup(self, dict_ptr: int) -> None:
        """Cleanup dict key-value pairs"""
    
    def decref_dict(self, dict_ptr: int) -> None:
        """Properly cleanup and decref dict"""
    
    # Basic I/O
    def write_i32(self, offset: int, value: int) -> None:
        """Write 32-bit integer"""
    
    def read_i32(self, offset: int) -> int:
        """Read 32-bit integer"""
    
    def write_f32(self, offset: int, value: float) -> None:
        """Write 32-bit float"""
    
    def read_f32(self, offset: int) -> float:
        """Read 32-bit float"""
    
    def write_f64(self, offset: int, value: float) -> None:
        """Write 64-bit double"""
    
    def read_f64(self, offset: int) -> float:
        """Read 64-bit double"""
    
    # Safe I/O (bounds checking)
    def write_i32_safe(self, offset: int, value: int) -> None:
        """Write i32 with bounds checking"""
    
    def read_i32_safe(self, offset: int) -> int:
        """Read i32 with bounds checking"""
    
    # Direct memory access
    memory: np.ndarray  # Direct NumPy array access
    heap_top: np.ndarray  # Current heap pointer
    free_lists: np.ndarray  # Slab free lists
    size: int  # Total heap size
```

---

## Next Steps

- Learn about [JIT compilation](JIT_GUIDE.md)
- Explore [AOT compilation](AOT_GUIDE.md)
- Understand [WASM backend](WASM_GUIDE.md)
- Check [Type System](TYPES_GUIDE.md)

---

**Last Updated:** October 26, 2025  
**PyExec Version:** 0.1.0
