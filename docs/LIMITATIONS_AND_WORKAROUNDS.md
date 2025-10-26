# PyExec Limitations & Workarounds Guide

> **⚠️ EDUCATIONAL PROJECT - NOT PRODUCTION READY**  
> This documentation is for **learning purposes only**. PyExec is an experimental project and is **not stable or ready for production use**.

## Table of Contents
- [Current Limitations](#current-limitations)
- [Workarounds & Solutions](#workarounds--solutions)
- [Hybrid Approaches](#hybrid-approaches)
- [Future Roadmap](#future-roadmap)

---

## Current Limitations

### ❌ Not Currently Supported

#### 1. Collections (list, dict, set, tuple)
**Why:** Dynamic data structures require runtime memory management that conflicts with static compilation.

**Workaround:**
```python
# ❌ DOESN'T WORK: Python lists
@jit
def bad_example(n: int32) -> int32:
    items = [1, 2, 3]  # Error: Use of unsupported opcode BUILD_LIST
    return sum(items)

# ✅ WORKS: Use Memory Manager for arrays
from pyexec.memory import MemoryManager
from pyexec.types import int32

mem = MemoryManager()

@jit
def array_sum(ptr: int32, length: int32) -> int32:
    """Sum array stored in memory"""
    total = 0
    for i in range(length):
        offset = ptr + i * 4
        total = total + mem.read_i32(offset)
    return total

# Allocate array
arr_ptr = mem.alloc(10 * 4)  # 10 integers
for i in range(10):
    mem.write_i32(arr_ptr + i * 4, i + 1)

result = array_sum(arr_ptr, 10)  # 55
mem.free(arr_ptr)
```

**Alternative: Pre-compute in Python**
```python
# Prepare data in Python
data = [1, 2, 3, 4, 5]
data_sum = sum(data)
data_max = max(data)

# Pass to compiled function
@jit
def process(total: int32, maximum: int32) -> int32:
    return total * maximum

result = process(data_sum, data_max)
```

---

#### 2. Strings
**Why:** Strings are variable-length objects requiring heap allocation and UTF-8 encoding.

**Workaround Option 1: Memory Manager (for WASM/AOT)**
```python
from pyexec.memory import MemoryManager

mem = MemoryManager()

# Allocate string in memory
text = "Hello, World!"
str_ptr = mem.alloc_string(text)

# Read back
result = mem.read_string(str_ptr)
print(result)  # "Hello, World!"

# Free
mem.free(str_ptr)
```

**Workaround Option 2: Process strings in Python**
```python
# ❌ DOESN'T WORK: String operations in compiled code
@jit
def bad_string(name: str) -> str:
    return "Hello, " + name  # Not supported

# ✅ WORKS: Pass string metadata as integers
@jit
def process_string_length(length: int32) -> int32:
    """Process string length"""
    return length * 2

# Python handles strings
text = "Hello"
result_length = process_string_length(len(text))
result = text * 2  # Python does string ops
```

**Workaround Option 3: Encode as bytes**
```python
@jit
def process_bytes(byte_val: int32) -> int32:
    """Process individual bytes of string"""
    return byte_val + 1

# Python encodes/decodes
text = "Hello"
bytes_data = text.encode('utf-8')
processed = bytes(process_bytes(b) for b in bytes_data)
result = processed.decode('utf-8')
```

---

#### 3. Classes/OOP
**Why:** Classes require vtables, dynamic dispatch, and object lifetimes that complicate static compilation.

**Workaround: Use structs via Memory Manager**
```python
from pyexec.memory import MemoryManager
from pyexec.types import int32, float64

mem = MemoryManager()

# ❌ DOESN'T WORK: Classes
class Point:  # Can't compile this
    def __init__(self, x, y):
        self.x = x
        self.y = y

# ✅ WORKS: Struct in memory
# Layout: [x:float64][y:float64] = 16 bytes

def point_create(x: float64, y: float64) -> int32:
    """Create point struct, return pointer"""
    ptr = mem.alloc(16)
    mem.write_f64(ptr, x)
    mem.write_f64(ptr + 8, y)
    return ptr

def point_distance(ptr: int32) -> float64:
    """Calculate distance from origin"""
    x = mem.read_f64(ptr)
    y = mem.read_f64(ptr + 8)
    return (x*x + y*y) ** 0.5

# Usage
p = point_create(3.0, 4.0)
dist = point_distance(p)  # 5.0
mem.free(p)
```

**Alternative: Separate functions instead of methods**
```python
# Instead of class with methods, use plain functions

@jit
def vector_length(x: float64, y: float64, z: float64) -> float64:
    return (x*x + y*y + z*z) ** 0.5

@jit
def vector_dot(x1: float64, y1: float64, z1: float64,
                x2: float64, y2: float64, z2: float64) -> float64:
    return x1*x2 + y1*y2 + z1*z2

# Use like methods
v1 = (1.0, 0.0, 0.0)
v2 = (0.0, 1.0, 0.0)
length = vector_length(*v1)
dot = vector_dot(*v1, *v2)
```

---

#### 4. Exceptions (try/except)
**Why:** Exception handling requires stack unwinding and runtime type checking.

**Workaround: Return error codes**
```python
from pyexec.types import int32

# ❌ DOESN'T WORK: Exceptions
@jit
def bad_divide(a: int32, b: int32) -> int32:
    try:
        return a // b
    except ZeroDivisionError:
        return 0

# ✅ WORKS: Error codes
@jit
def safe_divide(a: int32, b: int32) -> int32:
    """Returns 0 on division by zero"""
    if b == 0:
        return 0
    return a // b

# ✅ WORKS: Tuple return for error + result
@jit
def divide_with_status(a: int32, b: int32) -> tuple:
    """Returns (success, result)"""
    if b == 0:
        return (0, 0)  # Failed
    return (1, a // b)  # Success

# Python checks status
success, result = divide_with_status(10, 0)
if success:
    print(f"Result: {result}")
else:
    print("Division by zero!")
```

---

#### 5. Imports
**Why:** Dynamic imports require Python's import machinery at runtime.

**Workaround: Inline constants or pre-compute**
```python
# ❌ DOESN'T WORK: Imports inside function
@jit
def bad_example(x: int32) -> float64:
    import math  # Not supported
    return math.sqrt(x)

# ✅ WORKS: Implement math functions
@jit
def sqrt(x: float64) -> float64:
    """Newton's method square root"""
    if x < 0.0:
        return 0.0
    
    guess = x / 2.0
    epsilon = 0.000001
    
    for _ in range(20):  # Max iterations
        next_guess = (guess + x / guess) / 2.0
        diff = guess - next_guess
        if diff < 0.0:
            diff = -diff
        if diff < epsilon:
            break
        guess = next_guess
    
    return guess

# ✅ WORKS: Pre-compute constants in Python
import math

PI = 3.14159265359
E = 2.71828182846

@jit
def circle_area(radius: float64) -> float64:
    return PI * radius * radius
```

---

#### 6. Nested Functions/Closures
**Why:** Closures require capturing outer scope variables, which needs heap allocation.

**Workaround: Pass as parameters**
```python
# ❌ DOESN'T WORK: Closures
def create_adder(n: int32):
    @jit
    def add(x: int32) -> int32:
        return x + n  # Captures 'n' from outer scope
    return add

# ✅ WORKS: Pass as parameter
@jit
def add(x: int32, n: int32) -> int32:
    return x + n

# Python handles the "closure"
def create_adder(n: int32):
    return lambda x: add(x, n)

adder = create_adder(10)
print(adder(5))  # 15
```

---

#### 7. Comprehensions
**Why:** List/dict/set comprehensions create dynamic collections.

**Workaround: Use explicit loops**
```python
# ❌ DOESN'T WORK: Comprehensions
@jit
def bad_example(n: int32) -> int32:
    squares = [x*x for x in range(n)]  # Not supported
    return sum(squares)

# ✅ WORKS: Explicit loop
@jit
def sum_squares(n: int32) -> int32:
    total = 0
    for i in range(n):
        total = total + i * i
    return total
```

---

#### 8. Generators/yield
**Why:** Generators require state machines and heap-allocated frame objects.

**Workaround: Return all values at once or use iterator pattern**
```python
# ❌ DOESN'T WORK: Generators
@jit
def fibonacci_gen(n: int32):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# ✅ WORKS: Compute all values in Python, process in compiled code
def fibonacci_list(n):
    """Python generator"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

@jit
def sum_fibonacci(count: int32) -> int32:
    """Sum first 'count' fibonacci numbers"""
    total = 0
    a = 0
    b = 1
    for _ in range(count):
        total = total + a
        temp = a
        a = b
        b = temp + b
    return total
```

---

#### 9. Decorators (except @jit/@aot)
**Why:** Decorators are metaprogramming features that execute at definition time.

**Workaround: Apply logic manually**
```python
# ❌ DOESN'T WORK: Custom decorators
def timer_decorator(func):
    def wrapper(*args):
        import time
        start = time.time()
        result = func(*args)
        print(f"Time: {time.time() - start}")
        return result
    return wrapper

@jit
@timer_decorator  # Won't work
def my_func(x: int32) -> int32:
    return x * 2

# ✅ WORKS: Timing in Python layer
@jit
def my_func(x: int32) -> int32:
    return x * 2

import time
def timed_call(func, *args):
    start = time.time()
    result = func(*args)
    print(f"Time: {time.time() - start}")
    return result

result = timed_call(my_func, 100)
```

---

#### 10. Async/await
**Why:** Async requires event loops and cooperative multitasking.

**Workaround: Use threading in Python layer**
```python
# ❌ DOESN'T WORK: Async
@jit
async def bad_async(n: int32) -> int32:
    await asyncio.sleep(1)
    return n * 2

# ✅ WORKS: Compile synchronous, wrap with async in Python
@jit
def compute(n: int32) -> int32:
    """CPU-intensive work"""
    total = 0
    for i in range(n):
        total = total + i
    return total

# Python handles async
import asyncio

async def async_compute(n: int32) -> int32:
    """Async wrapper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, compute, n)

# Usage
result = asyncio.run(async_compute(1000000))
```

---

#### 11. Context Managers (with)
**Why:** Context managers use `__enter__`/`__exit__` protocol.

**Workaround: Manual setup/teardown**
```python
# ❌ DOESN'T WORK: Context managers
@jit
def bad_example():
    with open('file.txt') as f:  # Not supported
        data = f.read()

# ✅ WORKS: Explicit resource management
from pyexec.memory import MemoryManager

mem = MemoryManager()

def process_data():
    # Setup
    ptr = mem.alloc(1024)
    
    try:
        # Use resource
        result = compiled_function(ptr)
    finally:
        # Cleanup
        mem.free(ptr)
    
    return result
```

---

## WASM-Specific Limitations

### 1. No Direct DOM Access

**Problem:** WASM runs in sandbox, can't access browser DOM directly.

**Solution: JavaScript Bindings**
```python
# Python (compile to WASM)
from pyexec.wasm_backend import WasmCompiler
from pyexec.types import int32

compiler = WasmCompiler()

@compiler.compile_to_wasm
def calculate_score(kills: int32, deaths: int32) -> int32:
    if deaths == 0:
        return kills * 100
    return (kills * 100) // deaths

compiler.compile_to_wasm(
    calculate_score,
    output_path="game.wasm",
    generate_js_bindings=True
)
```

**JavaScript (DOM manipulation)**
```javascript
// Load WASM module
loadWasmModule('game.wasm').then(game => {
    // Get DOM elements
    const killsInput = document.getElementById('kills');
    const deathsInput = document.getElementById('deaths');
    const scoreDisplay = document.getElementById('score');
    
    // Button click handler
    document.getElementById('calculate').addEventListener('click', () => {
        const kills = parseInt(killsInput.value);
        const deaths = parseInt(deathsInput.value);
        
        // Call WASM function
        const score = game.calculate_score(kills, deaths);
        
        // Update DOM
        scoreDisplay.textContent = `Score: ${score}`;
    });
});
```

**HTML**
```html
<input id="kills" type="number" placeholder="Kills">
<input id="deaths" type="number" placeholder="Deaths">
<button id="calculate">Calculate Score</button>
<div id="score"></div>
<script src="game.js"></script>
```

---

### 2. No File I/O

**Problem:** WASM sandbox blocks file system access.

**Solution 1: WASI (WebAssembly System Interface)**
```bash
# Use WASI runtime (not browser)
wasmtime your_module.wasm
```

**Solution 2: Pass data via JavaScript**
```javascript
// JavaScript reads file
fetch('data.txt')
    .then(response => response.text())
    .then(text => {
        // Convert to byte array
        const bytes = new TextEncoder().encode(text);
        
        // Pass to WASM for processing
        // (WASM processes byte array, not file directly)
        const result = wasmModule.process_bytes(bytes);
    });
```

**Solution 3: IndexedDB/LocalStorage**
```javascript
// Store in browser storage
localStorage.setItem('gameState', JSON.stringify(data));

// Load and pass to WASM
const data = JSON.parse(localStorage.getItem('gameState'));
const result = wasmModule.process(data.score, data.level);

// Save result
localStorage.setItem('gameState', JSON.stringify({
    score: result,
    level: data.level + 1
}));
```

---

### 3. 32-bit Integers Only

**Problem:** JavaScript numbers are 53-bit max, WASM i64 requires BigInt.

**Current Solution: PyExec auto-converts int64 → int32**
```python
from pyexec.wasm_backend import WasmCompiler
from pyexec.types import int64, int32

compiler = WasmCompiler()

# Define with int64
def process(n: int64) -> int64:
    return n * 2

# PyExec automatically converts to int32 for WASM!
compiler.compile_to_wasm(process, output_path="calc.wasm")
```

**Manual Workaround: Split 64-bit into two 32-bit values**
```python
@jit
def add_64bit(low1: int32, high1: int32, 
               low2: int32, high2: int32) -> tuple:
    """
    Add two 64-bit numbers represented as (low, high) pairs.
    Returns (result_low, result_high)
    """
    # Add low parts
    low_sum = low1 + low2
    
    # Check for overflow (carry)
    carry = 0
    if low_sum < 0 and low1 > 0 and low2 > 0:
        carry = 1
    
    # Add high parts + carry
    high_sum = high1 + high2 + carry
    
    return (low_sum, high_sum)

# JavaScript side
function to64bit(low, high) {
    return BigInt(high) << 32n | BigInt(low >>> 0);
}

function from64bit(value) {
    return {
        low: Number(value & 0xFFFFFFFFn),
        high: Number(value >> 32n)
    };
}

// Usage
const a = 9876543210n;
const b = 1234567890n;

const a_parts = from64bit(a);
const b_parts = from64bit(b);

const [low, high] = wasmModule.add_64bit(
    a_parts.low, a_parts.high,
    b_parts.low, b_parts.high
);

const result = to64bit(low, high);
console.log(result);  // 11111111100n
```

---

## Hybrid Approaches

### Pattern: Python Orchestration + Compiled Core

```python
# High-level logic in Python
class GameEngine:
    def __init__(self):
        self.mem = MemoryManager()
        self.players = []
    
    def add_player(self, name, x, y):
        """Python handles objects"""
        player_id = len(self.players)
        self.players.append({
            'name': name,
            'id': player_id
        })
        
        # Compiled code handles positions
        self._set_position(player_id, x, y)
    
    @jit
    def _set_position(self, player_id: int32, x: float64, y: float64):
        """Compiled: performance-critical"""
        # Fast compiled code
        pass
    
    @jit
    def _calculate_physics(self, dt: float64) -> int32:
        """Compiled: CPU-intensive physics"""
        # Runs at native speed
        pass
    
    def update(self, dt):
        """Python orchestrates"""
        # Compiled does heavy lifting
        self._calculate_physics(dt)
        
        # Python handles I/O, networking, etc.
        self._render()
        self._handle_network()
```

---

## Future Roadmap

### Planned Features

#### 1. Static Arrays (In Progress)
```python
# Future syntax (not yet implemented)
from pyexec.types import Array, int32

@jit
def sum_array(arr: Array[int32, 10]) -> int32:
    """Fixed-size array support"""
    total = 0
    for i in range(10):
        total = total + arr[i]
    return total
```

#### 2. Struct Types (Planned)
```python
# Future syntax
from pyexec.types import Struct, int32, float64

@struct
class Point:
    x: float64
    y: float64

@jit
def distance(p1: Point, p2: Point) -> float64:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return (dx*dx + dy*dy) ** 0.5
```

#### 3. Basic String Support (Research)
```python
# Potential future syntax
from pyexec.types import String

@jit
def string_length(s: String) -> int32:
    return len(s)  # Compiled string operations
```

---

## Summary: What Works Now

### ✅ Fully Supported
- Primitive types (int8-int64, float32/64, bool)
- Arithmetic operations (+, -, *, /, //, %, **)
- Comparisons (<, <=, >, >=, ==, !=)
- Control flow (if/elif/else, while, for, break, continue)
- Recursion
- Function calls
- Type conversions
- Memory management (via MemoryManager)

### ⚠️ Workarounds Available
- Arrays (use MemoryManager)
- Strings (process in Python or use MemoryManager)
- Structs (use MemoryManager layouts)
- Error handling (use error codes)
- File I/O (handle in Python layer)
- DOM access (JavaScript bindings)

### ❌ Not Supported (No Workaround)
- Dynamic collections at runtime
- Full OOP with inheritance
- Exception handling with try/except
- Generators/yield
- Async/await within compiled code

---

## Best Practices

1. **Compute in compiled code, orchestrate in Python**
2. **Use Memory Manager for complex data**
3. **Pass metadata (lengths, counts) as integers**
4. **Pre-process in Python, compute in compiled, post-process in Python**
5. **For WASM: Use JavaScript for I/O, WASM for computation**

---

**Last Updated:** October 26, 2025  
**PyExec Version:** 0.1.0
