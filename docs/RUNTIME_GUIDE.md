# Runtime System Guide

> **⚠️ EDUCATIONAL PROJECT - NOT PRODUCTION READY**  
> This documentation is for **learning purposes only**. PyExec is an experimental project and is **not stable or ready for production use**.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Runtime Modes](#runtime-modes)
- [Compilation API](#compilation-api)
- [Mode Detection](#mode-detection)
- [Advanced Usage](#advanced-usage)
- [Integration Patterns](#integration-patterns)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **Runtime** class is PyExec's central orchestrator, managing compilation mode selection and dispatching functions to the appropriate backend (JIT or AOT).

### Key Features

✅ **Automatic mode detection**: Chooses best backend for environment  
✅ **Unified API**: Single interface for all compilation modes  
✅ **Lazy initialization**: Backends loaded only when needed  
✅ **Mode override**: Force JIT or AOT when desired  
✅ **Parallel execution**: Automatic parallelization support  

---

## Quick Start

### Basic Usage

```python
from pyexec import Runtime
from pyexec.types import int32

# Create runtime (auto mode)
runtime = Runtime()

# Compile function
@runtime.compile
def add(a: int32, b: int32) -> int32:
    return a + b

# Use it
result = add(10, 20)
print(result)  # 30
```

**What happens:**
1. Runtime detects environment (interactive → JIT, script → AOT)
2. Function compiled with chosen backend
3. Compiled function returned

---

### Decorator Syntax

```python
from pyexec import Runtime
from pyexec.types import int32

runtime = Runtime()

@runtime.compile
def factorial(n: int32) -> int32:
    result = 1
    i = 1
    while i <= n:
        result = result * i
        i = i + 1
    return result

print(factorial(10))  # 3628800
```

---

## Runtime Modes

### Mode Options

```python
from pyexec import Runtime

# Auto mode (default) - detects best backend
runtime_auto = Runtime(mode="auto")

# Force JIT compilation
runtime_jit = Runtime(mode="jit")

# Force AOT compilation
runtime_aot = Runtime(mode="aot")
```

---

### Auto Mode Behavior

**Detection logic:**

| Environment | Mode | Reason |
|-------------|------|--------|
| Jupyter/IPython | JIT | Interactive, needs fast compile |
| REPL | JIT | Interactive session |
| Script | AOT | Production, optimize startup |
| Unknown | JIT | Safe default |

**Example:**
```python
from pyexec import Runtime

runtime = Runtime(mode="auto")

# In Jupyter → Uses JIT (fast compile)
# In script.py → Uses AOT (optimized binary)
```

---

### JIT Mode

```python
from pyexec import Runtime
from pyexec.types import int32

runtime = Runtime(mode="jit")

@runtime.compile
def compute(x: int32) -> int32:
    return x * x

# First call: compiles + runs (~5ms)
result = compute(10)

# Subsequent calls: instant
result = compute(20)
```

**Characteristics:**
- Compile time: 4-10ms
- Runtime: Optimized machine code
- Cached in memory
- No file output

---

### AOT Mode

```python
from pyexec import Runtime
from pyexec.types import int32

runtime = Runtime(mode="aot")

@runtime.compile
def compute(x: int32) -> int32:
    return x * x

# Compiles to native code
result = compute(10)
```

**Characteristics:**
- Compile time: 1-3s (LLVM)
- Runtime: Native machine code
- Creates files (.dll/.so)
- Reusable across runs

---

## Compilation API

### compile() Method

```python
from pyexec import Runtime
from pyexec.types import int32

runtime = Runtime()

def my_function(x: int32) -> int32:
    return x + 1

# Method 1: Decorator
@runtime.compile
def decorated(x: int32) -> int32:
    return x + 1

# Method 2: Direct call
compiled = runtime.compile(my_function)

# Both work identically
print(decorated(10))  # 11
print(compiled(10))   # 11
```

---

### jit_compile() Method

```python
from pyexec import Runtime
from pyexec.types import int32

runtime = Runtime()

def my_function(x: int32) -> int32:
    return x * 2

# Force JIT compilation (ignores mode setting)
jit_func = runtime.jit_compile(my_function)

print(jit_func(5))  # 10
```

**Use cases:**
- Development/debugging
- Interactive environments
- Hot code paths needing fast recompilation

---

### aot_compile() Method

```python
from pyexec import Runtime
from pyexec.types import int32

runtime = Runtime()

def my_function(x: int32) -> int32:
    return x * 2

# Force AOT compilation (ignores mode setting)
aot_func = runtime.aot_compile(my_function)

print(aot_func(5))  # 10
```

**Use cases:**
- Production deployment
- Standalone libraries
- Long-running services

---

### Parallel Compilation

```python
from pyexec import Runtime
from pyexec.types import int64

runtime = Runtime(mode="jit")

@runtime.compile(parallel=True)
def parallel_sum(n: int64) -> int64:
    """
    Automatically parallelized by Numba.
    Uses all CPU cores.
    """
    total = 0
    for i in range(n):
        total = total + i
    return total

# Runs on all available cores
result = parallel_sum(1000000)
print(result)  # 499999500000
```

**Requirements:**
- Only works in JIT mode
- Function must be parallelizable (no dependencies between iterations)
- Best for large iteration counts (>10,000)

---

## Mode Detection

### Detection Logic

```python
from pyexec import Runtime

runtime = Runtime(mode="auto")

# Internal detection process:
# 1. Check if running interactively
#    - Jupyter: sys.ps1 exists or '__IPYTHON__' in globals
#    - REPL: hasattr(sys, 'ps1')
# 2. If interactive → JIT mode
# 3. If script → AOT mode
```

---

### Manual Detection Check

```python
from pyexec import Runtime

runtime = Runtime(mode="auto")

# Check what mode was selected
mode = runtime._detect_mode()
print(f"Selected mode: {mode}")  # "jit" or "aot"

# Check if interactive
is_interactive = runtime._is_interactive()
print(f"Interactive: {is_interactive}")
```

---

### Override Detection

```python
from pyexec import Runtime

# Force specific mode regardless of environment
runtime = Runtime(mode="jit")  # Always JIT

# Or
runtime = Runtime(mode="aot")  # Always AOT
```

---

## Advanced Usage

### Multiple Runtimes

```python
from pyexec import Runtime
from pyexec.types import int32

# Different runtimes for different purposes
jit_runtime = Runtime(mode="jit")
aot_runtime = Runtime(mode="aot")

@jit_runtime.compile
def fast_prototype(x: int32) -> int32:
    """Quick iteration during development"""
    return x * 2

@aot_runtime.compile
def production_code(x: int32) -> int32:
    """Optimized for deployment"""
    return x * 2

# JIT: Fast compile, good for development
print(fast_prototype(10))

# AOT: Slower compile, optimized binary
print(production_code(10))
```

---

### Lazy Backend Loading

```python
from pyexec import Runtime

# Runtime initialized quickly
runtime = Runtime(mode="auto")

# JIT backend loaded only on first compile
@runtime.compile
def first_function(x: int32) -> int32:
    return x + 1

# Backend already loaded, subsequent compiles faster
@runtime.compile
def second_function(x: int32) -> int32:
    return x + 2
```

**Performance:**
- Runtime init: <1ms
- First compile: 4-10ms (JIT) or 1-3s (AOT)
- Subsequent compiles: Same as first

---

### Compilation Context Manager

```python
from pyexec import Runtime
from pyexec.types import int32
import time

runtime = Runtime(mode="jit")

# Measure compilation time
with time.perf_counter() as start:
    @runtime.compile
    def benchmark_func(x: int32) -> int32:
        return x * x

compile_time = time.perf_counter() - start
print(f"Compile time: {compile_time*1000:.2f}ms")
```

---

## Integration Patterns

### Pattern 1: Development + Production

```python
# config.py
import os
from pyexec import Runtime

# Use environment variable to control mode
MODE = os.getenv("PYEXEC_MODE", "auto")
runtime = Runtime(mode=MODE)

# During development:
# export PYEXEC_MODE=jit
# python app.py

# In production:
# export PYEXEC_MODE=aot
# python app.py
```

---

### Pattern 2: Conditional Compilation

```python
from pyexec import Runtime
from pyexec.types import int32

runtime = Runtime(mode="auto")

def maybe_compile(func, should_compile=True):
    """Conditionally compile function"""
    if should_compile:
        return runtime.compile(func)
    return func

# Development: skip compilation for faster iteration
@maybe_compile(should_compile=False)
def debug_function(x: int32) -> int32:
    return x + 1

# Production: compile for performance
@maybe_compile(should_compile=True)
def prod_function(x: int32) -> int32:
    return x + 1
```

---

### Pattern 3: Hybrid Approach

```python
from pyexec import Runtime
from pyexec.types import int32

jit = Runtime(mode="jit")
aot = Runtime(mode="aot")

class MathLibrary:
    """Mixed JIT/AOT for different use cases"""
    
    @jit.compile
    def fast_prototype(self, x: int32) -> int32:
        """Quick experiments"""
        return x * 2
    
    @aot.compile
    def optimized_core(self, x: int32) -> int32:
        """Performance-critical path"""
        return x * 2

lib = MathLibrary()
print(lib.fast_prototype(10))    # JIT-compiled
print(lib.optimized_core(10))    # AOT-compiled
```

---

### Pattern 4: Plugin System

```python
from pyexec import Runtime
from pyexec.types import int32

class PluginManager:
    def __init__(self, mode="auto"):
        self.runtime = Runtime(mode=mode)
        self.plugins = {}
    
    def register(self, name):
        """Decorator to register and compile plugin"""
        def decorator(func):
            compiled = self.runtime.compile(func)
            self.plugins[name] = compiled
            return compiled
        return decorator
    
    def call(self, name, *args):
        """Call registered plugin"""
        return self.plugins[name](*args)

# Usage
manager = PluginManager(mode="jit")

@manager.register("double")
def plugin_double(x: int32) -> int32:
    return x * 2

@manager.register("square")
def plugin_square(x: int32) -> int32:
    return x * x

print(manager.call("double", 5))  # 10
print(manager.call("square", 5))  # 25
```

---

## Troubleshooting

### Issue 1: Mode Not Applied

**Problem:** Runtime ignores mode setting

**Debug:**
```python
from pyexec import Runtime

runtime = Runtime(mode="jit")

# Check internal state
print(f"Mode: {runtime.mode}")  # Should be "jit"

# Check what's actually used
@runtime.compile
def test(x: int32) -> int32:
    return x

# Verify backend
print(f"Using JIT backend: {runtime._jit_backend is not None}")
print(f"Using AOT backend: {runtime._aot_compiler is not None}")
```

---

### Issue 2: Auto Mode Picks Wrong Backend

**Problem:** Auto mode chooses JIT when you want AOT (or vice versa)

**Solution:** Force specific mode:
```python
from pyexec import Runtime

# Instead of auto
runtime = Runtime(mode="auto")

# Force desired mode
runtime = Runtime(mode="aot")  # Always AOT
```

---

### Issue 3: Slow First Compilation

**Expected behavior:**
- JIT: 4-10ms (normal)
- AOT: 1-3s (normal for LLVM)

**If slower:**
```python
from pyexec import Runtime
import time

runtime = Runtime(mode="jit")

# Profile compilation
start = time.perf_counter()

@runtime.compile
def test(x: int32) -> int32:
    return x

compile_time = time.perf_counter() - start

if compile_time > 0.1:  # >100ms for JIT is slow
    print(f"⚠️ Slow compilation: {compile_time*1000:.2f}ms")
    print("Check: Numba installed correctly?")
else:
    print(f"✅ Normal: {compile_time*1000:.2f}ms")
```

---

### Issue 4: Parallel Not Working

**Problem:** `parallel=True` doesn't speed up code

**Check:**
```python
from pyexec import Runtime
from pyexec.types import int64
import time

runtime = Runtime(mode="jit")

# Baseline (serial)
@runtime.compile(parallel=False)
def serial_sum(n: int64) -> int64:
    total = 0
    for i in range(n):
        total = total + i
    return total

# Parallel
@runtime.compile(parallel=True)
def parallel_sum(n: int64) -> int64:
    total = 0
    for i in range(n):
        total = total + i
    return total

# Benchmark
n = 10_000_000

start = time.perf_counter()
serial_sum(n)
serial_time = time.perf_counter() - start

start = time.perf_counter()
parallel_sum(n)
parallel_time = time.perf_counter() - start

speedup = serial_time / parallel_time
print(f"Serial: {serial_time:.4f}s")
print(f"Parallel: {parallel_time:.4f}s")
print(f"Speedup: {speedup:.2f}x")

if speedup < 1.5:
    print("⚠️ Parallel not effective")
    print("Reasons:")
    print("  - Loop too small (need >10K iterations)")
    print("  - Dependencies between iterations")
    print("  - Overhead dominates")
```

---

## Complete Example: Adaptive Runtime

```python
from pyexec import Runtime
from pyexec.types import int32
import os
import time

class AdaptiveRuntime:
    """
    Runtime that adapts based on:
    - Environment (dev/prod)
    - Function complexity
    - Performance requirements
    """
    
    def __init__(self):
        # Detect environment
        self.is_production = os.getenv("ENV") == "production"
        
        # Create runtimes
        self.jit = Runtime(mode="jit")
        self.aot = Runtime(mode="aot")
        
        # Statistics
        self.compile_times = {}
    
    def compile(self, func, prefer_speed=False):
        """
        Compile function with adaptive backend selection.
        
        Args:
            func: Function to compile
            prefer_speed: If True, prefer fast compilation (JIT)
        """
        func_name = func.__name__
        
        # Decision logic
        if prefer_speed or not self.is_production:
            # Development or speed priority → JIT
            start = time.perf_counter()
            compiled = self.jit.compile(func)
            compile_time = time.perf_counter() - start
            backend = "JIT"
        else:
            # Production → AOT
            start = time.perf_counter()
            compiled = self.aot.compile(func)
            compile_time = time.perf_counter() - start
            backend = "AOT"
        
        # Record statistics
        self.compile_times[func_name] = {
            "backend": backend,
            "time": compile_time
        }
        
        print(f"Compiled {func_name} with {backend} in {compile_time*1000:.2f}ms")
        
        return compiled
    
    def stats(self):
        """Print compilation statistics"""
        print("\nCompilation Statistics:")
        print("-" * 40)
        for name, info in self.compile_times.items():
            print(f"{name:20} {info['backend']:5} {info['time']*1000:6.2f}ms")

# Usage
runtime = AdaptiveRuntime()

@runtime.compile(prefer_speed=True)
def quick_test(x: int32) -> int32:
    """Fast iteration during development"""
    return x * 2

@runtime.compile(prefer_speed=False)
def optimized_core(x: int32) -> int32:
    """Production-critical code"""
    return x * 2

# Use functions
print(quick_test(10))
print(optimized_core(10))

# View statistics
runtime.stats()
```

**Output:**
```
Compiled quick_test with JIT in 5.23ms
Compiled optimized_core with AOT in 1284.56ms
20
20

Compilation Statistics:
----------------------------------------
quick_test           JIT     5.23ms
optimized_core       AOT  1284.56ms
```

---

## API Reference

### Runtime Class

```python
class Runtime:
    def __init__(self, mode: Literal["auto", "jit", "aot"] = "auto")
    
    def compile(self, func: Callable, parallel: bool = False) -> Callable:
        """Compile with automatic mode detection"""
    
    def jit_compile(self, func: Callable, parallel: bool = False) -> Callable:
        """Force JIT compilation"""
    
    def aot_compile(self, func: Callable) -> Callable:
        """Force AOT compilation"""
    
    def _detect_mode(self) -> Literal["jit", "aot"]:
        """Detect appropriate mode (internal)"""
    
    def _is_interactive(self) -> bool:
        """Check if running interactively (internal)"""
```

---

## Next Steps

- Learn about [JIT compilation](JIT_GUIDE.md)
- Explore [AOT compilation](AOT_GUIDE.md)
- Understand [WASM backend](WASM_GUIDE.md)
- Check [Type System](TYPES_GUIDE.md)
- Read [Memory Management](MEMORY_GUIDE.md)

---

**Last Updated:** October 26, 2025  
**PyExec Version:** 0.1.0
