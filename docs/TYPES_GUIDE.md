# Type System Guide

> **⚠️ EDUCATIONAL PROJECT - NOT PRODUCTION READY**  
> This documentation is for **learning purposes only**. PyExec is an experimental project and is **not stable or ready for production use**.

## Table of Contents
- [Overview](#overview)
- [Primitive Types](#primitive-types)
- [Type Usage](#type-usage)
- [Type Conversion](#type-conversion)
- [Advanced Types](#advanced-types)
- [Type Safety](#type-safety)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

PyExec's type system provides **explicit, zero-overhead type descriptors** for compiled code. All types are immutable and designed for high-performance numerical computing.

### Design Goals

✅ **Zero overhead**: Types compiled away at runtime  
✅ **Type safety**: Catch errors at compile time  
✅ **LLVM compatibility**: Direct mapping to LLVM IR  
✅ **NumPy integration**: Seamless NumPy dtype conversion  
✅ **WASM support**: Full WebAssembly compatibility  

---

## Primitive Types

### Integer Types

```python
from pyexec.types import (
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64
)

# Signed integers
x: int8    # -128 to 127
y: int16   # -32,768 to 32,767
z: int32   # -2,147,483,648 to 2,147,483,647
w: int64   # -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807

# Unsigned integers
a: uint8   # 0 to 255
b: uint16  # 0 to 65,535
c: uint32  # 0 to 4,294,967,295
d: uint64  # 0 to 18,446,744,073,709,551,615
```

---

### Floating-Point Types

```python
from pyexec.types import float32, float64

x: float32  # Single precision (7 digits, ±3.4E38)
y: float64  # Double precision (15 digits, ±1.7E308)
```

---

### Boolean Type

```python
from pyexec.types import bool_

flag: bool_  # True or False (1 bit in IR, 8 bits in memory)
```

---

### Type Specifications

| Type | Size | Range | Use Case |
|------|------|-------|----------|
| `int8` | 1 byte | -128 to 127 | Small integers, flags |
| `int16` | 2 bytes | -32K to 32K | Medium integers |
| `int32` | 4 bytes | -2B to 2B | **Default integer** |
| `int64` | 8 bytes | -9E18 to 9E18 | Large integers |
| `uint8` | 1 byte | 0 to 255 | Bytes, pixels |
| `uint16` | 2 bytes | 0 to 65K | Unicode chars |
| `uint32` | 4 bytes | 0 to 4B | IDs, hashes |
| `uint64` | 8 bytes | 0 to 18E18 | Timestamps |
| `float32` | 4 bytes | ±3.4E38 | Graphics, ML |
| `float64` | 8 bytes | ±1.7E308 | **Default float** |
| `bool_` | 1 byte | True/False | Conditions |

---

## Type Usage

### Function Signatures

```python
from pyexec import jit
from pyexec.types import int32, float64, bool_

@jit
def add_integers(a: int32, b: int32) -> int32:
    """All parameters and return must be typed"""
    return a + b

@jit
def divide(a: float64, b: float64) -> float64:
    """Floating-point division"""
    return a / b

@jit
def is_positive(x: int32) -> bool_:
    """Return boolean"""
    return x > 0

# Usage
print(add_integers(10, 20))  # 30
print(divide(10.0, 3.0))     # 3.333...
print(is_positive(-5))       # False
```

---

### Type Annotations Required

```python
from pyexec import jit
from pyexec.types import int32

# ❌ BAD: No type hints
@jit
def bad(x, y):
    return x + y

# ✅ GOOD: Explicit types
@jit
def good(x: int32, y: int32) -> int32:
    return x + y
```

**Why required?**
- Compiler needs to know types ahead of time
- Enables optimization (no type checks at runtime)
- Prevents runtime type errors

---

### Variable Type Inference

```python
from pyexec import jit
from pyexec.types import int32

@jit
def compute(x: int32) -> int32:
    # Compiler infers types from operations
    y = x + 10      # y is int32 (same as x)
    z = y * 2       # z is int32 (same as y)
    result = z - 5  # result is int32
    
    return result

print(compute(5))  # 25
```

**Inference rules:**
- Operation on `int32` produces `int32`
- Operation on `float64` produces `float64`
- Mixed operations promote to wider type

---

## Type Conversion

### Implicit Conversion

```python
from pyexec import jit
from pyexec.types import int32, int64

@jit
def implicit_conversion(x: int32) -> int64:
    # int32 automatically converted to int64
    return x

print(implicit_conversion(42))  # 42 (as int64)
```

**Conversion hierarchy:**
```
int8 → int16 → int32 → int64
uint8 → uint16 → uint32 → uint64
float32 → float64
int → float (widening)
```

---

### Explicit Conversion

```python
from pyexec import jit
from pyexec.types import int32, float64

@jit
def float_to_int(x: float64) -> int32:
    # Truncates decimal part
    return int(x)

@jit
def int_to_float(x: int32) -> float64:
    # Exact conversion
    return float(x)

print(float_to_int(3.14))   # 3
print(int_to_float(42))     # 42.0
```

**Conversion behavior:**
- `int()`: Truncates (3.9 → 3)
- `float()`: Exact conversion when possible
- Overflow behavior: Undefined (wrap-around)

---

### Mixed-Type Operations

```python
from pyexec import jit
from pyexec.types import int32, float64

@jit
def mixed_math(x: int32, y: float64) -> float64:
    # int32 promoted to float64 for operation
    result = x + y  # result is float64
    return result

print(mixed_math(10, 3.5))  # 13.5
```

**Promotion rules:**
1. Integer + Float → Float
2. Smaller + Larger → Larger
3. Signed + Unsigned → Signed (when safe)

---

## Advanced Types

### Type Properties

```python
from pyexec.types import int32, float64, TypeKind

# Inspect type properties
print(int32.kind)       # TypeKind.INT32
print(int32.size)       # 4 (bytes)
print(int32.alignment)  # 4 (bytes)
print(int32.signed)     # True

print(float64.kind)     # TypeKind.FLOAT64
print(float64.size)     # 8 (bytes)
print(float64.alignment)# 8 (bytes)
print(float64.signed)   # None (not applicable)
```

---

### LLVM IR Conversion

```python
from pyexec.types import int32, float64
from llvmlite import ir

# Convert to LLVM types
llvm_int = int32.to_llvm()
llvm_float = float64.to_llvm()

print(llvm_int)    # i32
print(llvm_float)  # double
```

**LLVM mappings:**
- `int8` → `i8`
- `int32` → `i32`
- `int64` → `i64`
- `float32` → `float`
- `float64` → `double`
- `bool_` → `i1`

---

### NumPy Dtype Conversion

```python
from pyexec.types import int32, float64
import numpy as np

# Convert to NumPy dtype
np_int = int32.to_numpy_dtype()
np_float = float64.to_numpy_dtype()

print(np_int)    # int32
print(np_float)  # float64

# Use in NumPy arrays
arr = np.array([1, 2, 3], dtype=np_int)
print(arr.dtype)  # int32
```

---

### Type Registry

```python
from pyexec.types import TypeRegistry, TypeKind

# Create registry
registry = TypeRegistry()

# Get type by kind
int_type = registry.get(TypeKind.INT32)
float_type = registry.get(TypeKind.FLOAT64)

print(int_type.size)    # 4
print(float_type.size)  # 8
```

**Purpose:** Centralized type management for compiler internals

---

## Type Safety

### Compile-Time Type Checking

```python
from pyexec import jit
from pyexec.types import int32

@jit
def type_safe(x: int32) -> int32:
    # ✅ VALID: Same type
    y: int32 = x + 10
    return y

# ❌ INVALID: Type mismatch (caught at compile time)
@jit
def type_error(x: int32) -> int32:
    y: float64 = x + 10  # Error: Can't assign float64 to int32
    return y
```

---

### Overflow Behavior

```python
from pyexec import jit
from pyexec.types import int8, int32

@jit
def overflow_8bit(x: int8) -> int8:
    # int8 range: -128 to 127
    return x + 1

@jit
def overflow_32bit(x: int32) -> int32:
    # int32 range: -2,147,483,648 to 2,147,483,647
    return x + 1

# Overflow wraps around (undefined behavior)
print(overflow_8bit(127))  # -128 (wraps)
print(overflow_32bit(2147483647))  # -2147483648 (wraps)
```

**Prevention:**
```python
from pyexec import jit
from pyexec.types import int32

@jit
def safe_add(a: int32, b: int32) -> int32:
    # Check for overflow before operation
    max_int = 2147483647
    if a > max_int - b:
        return max_int  # Clamp
    return a + b
```

---

### Division by Zero

```python
from pyexec import jit
from pyexec.types import int32, float64

@jit
def safe_divide_int(a: int32, b: int32) -> int32:
    if b == 0:
        return 0  # Or handle error
    return a // b

@jit
def safe_divide_float(a: float64, b: float64) -> float64:
    if b == 0.0:
        return 0.0  # Or return infinity
    return a / b

print(safe_divide_int(10, 0))    # 0
print(safe_divide_float(10.0, 0.0))  # 0.0
```

---

## Best Practices

### 1. Use Smallest Appropriate Type

```python
from pyexec import jit
from pyexec.types import int8, int32

# ❌ BAD: Wastes memory for small values
@jit
def count_pixels(r: int32, g: int32, b: int32) -> int32:
    return r + g + b

# ✅ GOOD: Pixel values 0-255 fit in uint8
from pyexec.types import uint8, int32

@jit
def count_pixels(r: uint8, g: uint8, b: uint8) -> int32:
    # Promote to int32 for result
    return int(r) + int(g) + int(b)
```

---

### 2. Prefer int32/float64 for General Use

```python
from pyexec import jit
from pyexec.types import int32, float64

# ✅ GOOD: Standard types for most use cases
@jit
def standard_types(x: int32, y: float64) -> float64:
    return float(x) + y

# int32: -2B to 2B (sufficient for most integers)
# float64: 15 digits precision (standard for science)
```

---

### 3. Document Type Choices

```python
from pyexec import jit
from pyexec.types import uint32, int64

@jit
def hash_combine(hash1: uint32, hash2: uint32) -> uint32:
    """
    Combine two 32-bit hash values.
    
    Uses uint32 because:
    - Hash values are always positive
    - 32 bits matches common hash algorithms
    - Overflow wrapping is desired behavior
    """
    return hash1 * 31 + hash2

@jit
def timestamp_diff(ts1: int64, ts2: int64) -> int64:
    """
    Calculate difference between Unix timestamps.
    
    Uses int64 because:
    - Timestamps overflow int32 in 2038
    - Need to represent times beyond 1970-2038
    """
    return ts2 - ts1
```

---

### 4. Avoid Unnecessary Conversions

```python
from pyexec import jit
from pyexec.types import int32, float64

# ❌ BAD: Multiple conversions
@jit
def bad_conversions(x: int32) -> int32:
    y = float(x)      # int32 → float64
    z = int(y) + 1    # float64 → int32
    return z

# ✅ GOOD: Stay in same type
@jit
def good_no_conversions(x: int32) -> int32:
    return x + 1
```

---

## Troubleshooting

### Issue 1: Type Mismatch Error

**Error:** `Cannot convert float64 to int32`

**Solution:**
```python
from pyexec import jit
from pyexec.types import int32, float64

# ❌ BAD
@jit
def bad(x: float64) -> int32:
    return x  # Error: Type mismatch

# ✅ GOOD
@jit
def good(x: float64) -> int32:
    return int(x)  # Explicit conversion
```

---

### Issue 2: Import Error

**Error:** `ImportError: cannot import name 'int32'`

**Solution:**
```python
# ❌ BAD
from pyexec import int32  # Wrong import path

# ✅ GOOD
from pyexec.types import int32
```

---

### Issue 3: Overflow Not Detected

**Problem:** Large numbers wrap around

**Explanation:**
```python
from pyexec import jit
from pyexec.types import int32

@jit
def add(a: int32, b: int32) -> int32:
    return a + b

# This wraps around (undefined behavior in C)
print(add(2147483647, 1))  # -2147483648
```

**Solution:** Use larger type or check manually:
```python
from pyexec import jit
from pyexec.types import int64

@jit
def safe_add(a: int64, b: int64) -> int64:
    return a + b  # int64 has much larger range
```

---

### Issue 4: WASM int64 Issues

**Problem:** JavaScript doesn't support 64-bit integers natively

**Solution:** PyExec auto-converts int64 to int32 for WASM:
```python
from pyexec.wasm_backend import WasmCompiler
from pyexec.types import int32  # Use int32 for WASM

compiler = WasmCompiler()

@compiler.compile_to_wasm
def wasm_safe(x: int32, y: int32) -> int32:
    return x + y

# ✅ Works in browser (JavaScript number is 53-bit max)
```

---

## Complete Example: Type-Safe Vector Math

```python
from pyexec import jit
from pyexec.types import float64

@jit
def vector_length(x: float64, y: float64, z: float64) -> float64:
    """
    Calculate 3D vector length: sqrt(x^2 + y^2 + z^2)
    
    Uses float64 because:
    - Need precision for square root
    - Coordinates may be fractional
    - Result may be very small/large
    """
    return (x*x + y*y + z*z) ** 0.5

@jit
def vector_normalize(
    x: float64, y: float64, z: float64
) -> tuple[float64, float64, float64]:
    """
    Normalize vector to unit length.
    
    Returns: (nx, ny, nz) where length = 1.0
    """
    length = vector_length(x, y, z)
    
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    
    return (x / length, y / length, z / length)

@jit
def vector_dot(
    x1: float64, y1: float64, z1: float64,
    x2: float64, y2: float64, z2: float64
) -> float64:
    """Dot product of two vectors"""
    return x1*x2 + y1*y2 + z1*z2

# Usage
print(vector_length(3.0, 4.0, 0.0))  # 5.0
print(vector_normalize(10.0, 0.0, 0.0))  # (1.0, 0.0, 0.0)
print(vector_dot(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))  # 0.0 (perpendicular)
```

---

## Type System Reference

### All Available Types

```python
from pyexec.types import (
    # Signed integers
    int8, int16, int32, int64,
    
    # Unsigned integers
    uint8, uint16, uint32, uint64,
    
    # Floating point
    float32, float64,
    
    # Boolean
    bool_,
    
    # Type utilities
    Type, TypeKind, TypeRegistry
)
```

---

### Type Hierarchy

```
Type (base class)
├── Integer Types
│   ├── Signed: int8, int16, int32, int64
│   └── Unsigned: uint8, uint16, uint32, uint64
├── Floating Types
│   ├── float32
│   └── float64
└── Boolean Type
    └── bool_
```

---

## Next Steps

- Learn about [JIT compilation](JIT_GUIDE.md)
- Explore [AOT compilation](AOT_GUIDE.md)
- Understand [WASM backend](WASM_GUIDE.md)
- Check [Memory Management](MEMORY_GUIDE.md)
- Read [Runtime Guide](RUNTIME_GUIDE.md)

---

**Last Updated:** October 26, 2025  
**PyExec Version:** 0.1.0
