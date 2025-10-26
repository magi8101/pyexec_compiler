# PyExec Documentation Hub

> **⚠️ EDUCATIONAL PROJECT - NOT PRODUCTION READY**  
> This documentation is for **learning purposes only**. PyExec is an experimental project to explore compilation techniques and is **not stable or ready for production use**. Use at your own risk.

Complete in-depth guides for every PyExec component.

---

## 📚 Documentation Index

### Core Guides

1. **[JIT Compilation Guide](JIT_GUIDE.md)** 
   Learn how to use Just-In-Time compilation for rapid development.
   - Quick start examples
   - Control flow (if/while/for/recursion)
   - Parallel execution
   - Performance optimization
   - Complete troubleshooting guide

2. **[AOT Compilation Guide](AOT_GUIDE.md)**  
   Master Ahead-of-Time compilation for production deployments.
   - LLVM backend (shared libraries)
   - Nuitka backend (standalone executables)
   - Choosing the right backend
   - Deployment strategies
   - Cross-language integration

3. **[WebAssembly Guide](WASM_GUIDE.md)**   
   Compile Python to WebAssembly for browser and edge computing.
   - Browser integration
   - JavaScript bindings
   - Node.js usage
   - Complete web application examples
   - Performance optimization

4. **[Memory Management Guide](MEMORY_GUIDE.md)**   
   Understand PyExec's high-performance memory allocator.
   - Slab allocator internals
   - Reference counting
   - Memory allocation strategies
   - Performance tuning
   - Advanced patterns

5. **[Type System Guide](TYPES_GUIDE.md)**   
   Master PyExec's zero-overhead type system.
   - All primitive types (int8-int64, float32/64, bool)
   - Type conversion rules
   - Type safety best practices
   - LLVM and NumPy integration
   - Troubleshooting type errors

6. **[Runtime Guide](RUNTIME_GUIDE.md)** ⚙️  
   Control compilation modes and runtime behavior.
   - Auto mode detection
   - JIT vs AOT selection
   - Parallel compilation
   - Integration patterns
   - Advanced runtime techniques

7. **[Limitations & Workarounds Guide](LIMITATIONS_AND_WORKAROUNDS.md)** 🔧  
   Solutions for Python features not directly supported.
   - Collections workarounds (lists, dicts, sets)
   - String handling strategies
   - OOP alternatives (structs via memory)
   - Error handling without exceptions
   - WASM-specific solutions (DOM, file I/O, int64)
   - Hybrid Python + compiled approaches

---

## 🎯 Quick Navigation

### By Use Case

**I want to...**

- **Prototype quickly** → [JIT Guide](JIT_GUIDE.md)
- **Deploy to production** → [AOT Guide](AOT_GUIDE.md)
- **Run in browsers** → [WASM Guide](WASM_GUIDE.md)
- **Optimize memory** → [Memory Guide](MEMORY_GUIDE.md)
- **Fix type errors** → [Types Guide](TYPES_GUIDE.md)
- **Control compilation** → [Runtime Guide](RUNTIME_GUIDE.md)
- **Use lists/strings/classes** → [Limitations & Workarounds](LIMITATIONS_AND_WORKAROUNDS.md)

---

### By Experience Level

**Beginner** (New to PyExec)
1. Start with [JIT Guide - Quick Start](JIT_GUIDE.md#quick-start)
2. Read [Types Guide - Overview](TYPES_GUIDE.md#overview)
3. Try [JIT Guide - Basic Usage](JIT_GUIDE.md#basic-usage)

**Intermediate** (Familiar with basics)
1. Explore [AOT Guide - LLVM Backend](AOT_GUIDE.md#llvm-backend)
2. Learn [Memory Guide - Allocation](MEMORY_GUIDE.md#memory-allocation)
3. Study [Runtime Guide - Modes](RUNTIME_GUIDE.md#runtime-modes)

**Advanced** (Building production systems)
1. Master [AOT Guide - Deployment](AOT_GUIDE.md#deployment)
2. Deep dive [Memory Guide - Advanced](MEMORY_GUIDE.md#advanced-features)
3. Implement [Runtime Guide - Patterns](RUNTIME_GUIDE.md#integration-patterns)

---

### By Feature

| Feature | Guide | Section |
|---------|-------|---------|
| **Loops (while/for)** | [JIT Guide](JIT_GUIDE.md#loops-while-for) | Basic Usage |
| **If/Else** | [JIT Guide](JIT_GUIDE.md#conditional-logic-ifelse) | Control Flow |
| **Recursion** | [JIT Guide](JIT_GUIDE.md#recursive-functions) | Advanced |
| **Parallel execution** | [JIT Guide](JIT_GUIDE.md#parallel-execution) | Advanced |
| **Shared libraries** | [AOT Guide](AOT_GUIDE.md#llvm-backend) | LLVM Backend |
| **Executables** | [AOT Guide](AOT_GUIDE.md#nuitka-backend) | Nuitka Backend |
| **Browser apps** | [WASM Guide](WASM_GUIDE.md#browser-deployment) | Deployment |
| **Memory allocation** | [Memory Guide](MEMORY_GUIDE.md#memory-allocation) | Basic Usage |
| **Type conversion** | [Types Guide](TYPES_GUIDE.md#type-conversion) | Core Concepts |
| **Mode selection** | [Runtime Guide](RUNTIME_GUIDE.md#runtime-modes) | Configuration |
| **Collections** | [Limitations Guide](LIMITATIONS_AND_WORKAROUNDS.md#1-collections-list-dict-set-tuple) | Workarounds |
| **Strings** | [Limitations Guide](LIMITATIONS_AND_WORKAROUNDS.md#2-strings) | Workarounds |
| **Classes/OOP** | [Limitations Guide](LIMITATIONS_AND_WORKAROUNDS.md#3-classesoop) | Workarounds |
| **DOM access (WASM)** | [Limitations Guide](LIMITATIONS_AND_WORKAROUNDS.md#1-no-direct-dom-access) | WASM Solutions |

---

## 📖 Complete Examples

### Example 1: Simple Calculator (JIT)
```python
from pyexec import jit
from pyexec.types import int32

@jit
def add(a: int32, b: int32) -> int32:
    return a + b

print(add(10, 20))  # 30
```
**Learn more:** [JIT Guide - Quick Start](JIT_GUIDE.md#quick-start)

---

### Example 2: Native Library (AOT)
```python
from pyexec import compile_llvm
from pyexec.types import int32

def factorial(n: int32) -> int32:
    result = 1
    i = 1
    while i <= n:
        result = result * i
        i = i + 1
    return result

compile_llvm(factorial, output_dir="./lib", base_name="math")
# Creates: lib/math.dll (Windows) or lib/libmath.so (Linux)
```
**Learn more:** [AOT Guide - LLVM Backend](AOT_GUIDE.md#llvm-backend)

---

### Example 3: Web Application (WASM)
```python
from pyexec.wasm_backend import WasmCompiler
from pyexec.types import int32

compiler = WasmCompiler()

def is_prime(n: int32) -> int32:
    if n < 2: return 0
    if n == 2: return 1
    if n % 2 == 0: return 0
    i = 3
    while i * i <= n:
        if n % i == 0:
            return 0
        i = i + 2
    return 1

compiler.compile_to_wasm(
    is_prime,
    output_path="prime.wasm",
    generate_js_bindings=True
)
```
**Learn more:** [WASM Guide - Browser Deployment](WASM_GUIDE.md#browser-deployment)

---

## 🔧 Troubleshooting Hub

### Common Issues

**Compilation Errors**
- [Type mismatch errors](TYPES_GUIDE.md#issue-1-type-mismatch-error)
- [Nuitka issues](AOT_GUIDE.md#nuitka-issues)
- [LLVM problems](AOT_GUIDE.md#llvm-issues)

**Runtime Problems**
- [Slow performance](JIT_GUIDE.md#issue-3-slow-performance)
- [Memory leaks](MEMORY_GUIDE.md#issue-2-memory-leak)
- [Mode not applied](RUNTIME_GUIDE.md#issue-1-mode-not-applied)

**WASM Specific**
- [CORS errors](WASM_GUIDE.md#issue-2-cors-errors-in-browser)
- [Type conversion](WASM_GUIDE.md#issue-4-type-conversion-errors)
- [Loading failures](WASM_GUIDE.md#issue-3-wasm-file-not-loading)

---

## 📊 Performance Reference

### Compilation Times

| Backend | Compile Time | Use Case |
|---------|--------------|----------|
| **JIT (Numba)** | 4-10ms | Development, prototyping |
| **LLVM** | 1-3s | Production libraries |
| **Nuitka** | 30-60s | Standalone executables |

### Runtime Speedups

| Algorithm | vs CPython | vs JavaScript |
|-----------|------------|---------------|
| Integer math | 50-100x | 2-3x |
| Floating point | 20-50x | 1.5-2x |
| Loops | 30-80x | 2-4x |

**Details:** See individual guides for comprehensive benchmarks.

---

## 🎓 Learning Paths

### Path 1: Web Developer
1. [JIT Guide - Quick Start](JIT_GUIDE.md#quick-start) (30 min)
2. [WASM Guide - Browser Deployment](WASM_GUIDE.md#browser-deployment) (1 hour)
3. [WASM Guide - JavaScript Integration](WASM_GUIDE.md#javascript-integration) (45 min)
4. Build: Prime checker web app

---

### Path 2: Backend Developer
1. [JIT Guide - Basic Usage](JIT_GUIDE.md#basic-usage) (30 min)
2. [AOT Guide - LLVM Backend](AOT_GUIDE.md#llvm-backend) (1 hour)
3. [Runtime Guide - Integration Patterns](RUNTIME_GUIDE.md#integration-patterns) (45 min)
4. Build: REST API with compiled functions

---

### Path 3: Systems Programmer
1. [Types Guide - Complete](TYPES_GUIDE.md) (1 hour)
2. [Memory Guide - Complete](MEMORY_GUIDE.md) (1.5 hours)
3. [AOT Guide - LLVM Backend](AOT_GUIDE.md#llvm-backend) (1 hour)
4. Build: High-performance library

---

### Path 4: Data Scientist
1. [JIT Guide - Parallel Execution](JIT_GUIDE.md#parallel-execution) (30 min)
2. [Memory Guide - Bulk Operations](MEMORY_GUIDE.md#bulk-operations) (45 min)
3. [Types Guide - NumPy Integration](TYPES_GUIDE.md#numpy-dtype-conversion) (30 min)
4. Build: Numerical computation pipeline

---

## 🔗 Related Documentation

### Main Documentation
- [README.md](../README.md) - Quick start and overview
- [API_REFERENCE.md](../API_REFERENCE.md) - Complete API documentation
- [TECHNICAL.md](../TECHNICAL.md) - Honest capabilities and limitations
- [VERIFICATION.md](../VERIFICATION.md) - Code quality and portability report

### Getting Help
- **Bug reports**: Check [VERIFICATION.md](../VERIFICATION.md) first
- **Type errors**: See [Types Guide - Troubleshooting](TYPES_GUIDE.md#troubleshooting)
- **Performance issues**: See guide-specific troubleshooting sections

---

## 📝 Documentation Standards

All guides follow consistent structure:
1. **Overview** - What it is and when to use it
2. **Quick Start** - Get running in 5 minutes
3. **Basic Usage** - Common patterns and examples
4. **Advanced Features** - Deep dives and edge cases
5. **Troubleshooting** - Common issues and solutions
6. **Complete Examples** - Real-world applications

---

## 🚀 Quick Reference Cards

### JIT Cheat Sheet
```python
from pyexec import jit
from pyexec.types import int32

@jit
def func(x: int32) -> int32:
    return x * 2
```

### AOT Cheat Sheet
```python
from pyexec import compile_llvm

compile_llvm(func, output_dir="./lib", base_name="mylib")
```

### WASM Cheat Sheet
```python
from pyexec.wasm_backend import WasmCompiler

compiler = WasmCompiler()
compiler.compile_to_wasm(func, output_path="func.wasm")
```

### Memory Cheat Sheet
```python
from pyexec.memory import MemoryManager

mem = MemoryManager()
ptr = mem.alloc(64)
mem.write_i32(ptr, 42)
mem.free(ptr)
```

---

## 📅 Last Updated

**Date:** October 26, 2025  
**Version:** PyExec 0.1.0  
**Authors:** PyExec Development Team

---

## 💡 Tips for Reading

1. **Start simple**: Begin with JIT Guide Quick Start
2. **Follow examples**: Every guide has runnable code
3. **Try immediately**: Copy-paste and experiment
4. **Read troubleshooting**: Common issues are documented
5. **Cross-reference**: Guides link to related topics

---

**Ready to start?** Choose a guide above and dive in! 🎉
