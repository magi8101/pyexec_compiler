"""
WASM memory management using unified MemoryManager.
Production-grade memory allocation for WASM backend.
"""

from typing import Optional
from pathlib import Path
import numpy as np
from .memory import MemoryManager

# WASM linear memory starts at 64KB (first 64KB reserved for stack/globals)
HEAP_START = 65536


class WasmMemoryManager:
    """
    WASM-compatible memory manager using unified MemoryManager.
    Provides WASM-specific interface while using production-grade allocator.
    """
    
    def __init__(self, initial_pages: int = 256):
        """Initialize WASM memory manager with given page count (1 page = 64KB)."""
        total_size = initial_pages * 65536
        
        # Use unified memory manager
        self.mm = MemoryManager(size=total_size)
        
        # WASM starts heap at 64KB, but MemoryManager starts at 0
        # We'll translate addresses when needed
        self.heap_offset = HEAP_START
        
        self.pages = initial_pages
    
    def alloc(self, size: int) -> int:
        """Allocate memory block of given size."""
        return self.mm.alloc(size)
    
    def free(self, ptr: int, size: int) -> None:
        """Free memory block."""
        self.mm.free(ptr, size)
    
    def alloc_string(self, s: str) -> int:
        """Allocate string in WASM memory."""
        return self.mm.alloc_string(s)
    
    def read_string(self, ptr: int) -> str:
        """Read string from WASM memory."""
        return self.mm.read_string(ptr)
    
    def alloc_array(self, length: int, elem_size: int = 8) -> int:
        """Allocate array in WASM memory."""
        return self.mm.alloc_array(length, elem_size)
    
    def alloc_dict(self, capacity: int = 16) -> int:
        """Allocate dict in WASM memory."""
        return self.mm.alloc_dict(capacity)
    
    def alloc_list(self, capacity: int = 8, elem_size: int = 8) -> int:
        """Allocate list in WASM memory."""
        return self.mm.alloc_list(capacity, elem_size)
    
    def incref(self, ptr: int) -> None:
        """Increment reference count."""
        self.mm.incref(ptr)
    
    def decref(self, ptr: int) -> None:
        """Decrement reference count."""
        self.mm.decref(ptr)
    
    def decref_dict(self, ptr: int) -> None:
        """Decrement dict reference count with proper cleanup."""
        self.mm.decref_dict(ptr)
    
    def decref_list_auto(self, ptr: int) -> None:
        """Decrement list reference count with auto cleanup."""
        self.mm.decref_list_auto(ptr)
    
    def get_memory_view(self) -> np.ndarray:
        """Get direct view of WASM linear memory."""
        return self.mm.memory
    
    def get_heap_size(self) -> int:
        """Get current heap usage."""
        used, total = self.mm.get_usage()
        return used - HEAP_START
    
    def get_total_size(self) -> int:
        """Get total memory size."""
        return self.mm.size
    
    def grow_memory(self, pages: int) -> bool:
        """
        Grow WASM memory by given number of pages.
        Returns True if successful, False if failed.
        """
        new_size = self.mm.size + (pages * 65536)
        
        if new_size > 4 * 1024 * 1024 * 1024:  # 4GB max for WASM
            return False
        
        # Create new larger memory manager
        new_mm = MemoryManager(size=new_size)
        new_mm.memory[:self.mm.size] = self.mm.memory
        new_mm.heap_top[0] = self.mm.heap_top[0]
        new_mm.free_lists[:] = self.mm.free_lists
        
        self.mm = new_mm
        self.pages += pages
        return True
    
    def reset(self) -> None:
        """Reset memory manager (for testing)."""
        self.mm.reset()
        self.mm.heap_top[0] = HEAP_START


def generate_allocator_wasm() -> bytes:
    """
    Generate WASM module with malloc/free functions.
    Uses unified memory manager approach.
    Returns raw WASM bytecode for memory allocator.
    """
    from llvmlite import ir
    import llvmlite.binding as llvm
    
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    
    module = ir.Module(name="allocator")
    module.triple = "wasm32-unknown-unknown"
    module.data_layout = "e-m:e-p:32:32-i64:64-n32:64-S128"
    
    i32 = ir.IntType(32)
    i8 = ir.IntType(8)
    
    # Global heap pointer starting at 64KB
    heap_ptr = ir.GlobalVariable(module, i32, "heap_ptr")
    heap_ptr.initializer = ir.Constant(i32, HEAP_START)
    heap_ptr.linkage = "internal"
    
    # malloc(size: i32) -> i32
    malloc_ty = ir.FunctionType(i32, [i32])
    malloc_fn = ir.Function(module, malloc_ty, "malloc")
    malloc_fn.attributes.add("nounwind")
    
    entry = malloc_fn.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    
    size = malloc_fn.args[0]
    size.name = "size"
    
    # Load current heap pointer
    current_ptr = builder.load(heap_ptr, "current_ptr")
    
    # Align to 8 bytes: (ptr + 7) & ~7
    seven = ir.Constant(i32, 7)
    neg_eight = ir.Constant(i32, -8)
    aligned = builder.and_(builder.add(current_ptr, seven), neg_eight, "aligned")
    
    # Update heap pointer
    new_ptr = builder.add(aligned, size, "new_ptr")
    builder.store(new_ptr, heap_ptr)
    
    # Return aligned pointer
    builder.ret(aligned)
    
    # free(ptr: i32) -> void
    free_ty = ir.FunctionType(ir.VoidType(), [i32])
    free_fn = ir.Function(module, free_ty, "free")
    free_fn.attributes.add("nounwind")
    
    entry = free_fn.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    builder.ret_void()  # No-op for now (managed by Python layer)
    
    # Compile to WASM object
    llvm_ir = str(module)
    binding_module = llvm.parse_assembly(llvm_ir)
    binding_module.verify()
    
    target = llvm.Target.from_triple("wasm32-unknown-unknown")
    target_machine = target.create_target_machine(
        cpu='',
        features='',
        opt=3,
        reloc='pic',
        codemodel='default',
    )
    
    return target_machine.emit_object(binding_module)


def generate_runtime_helpers() -> str:
    """Generate JavaScript runtime helpers for WASM data structures."""
    return """
// WASM Runtime Helpers - Unified Memory Manager Interface

class WasmRuntime {
    constructor(memory, malloc, free) {
        this.memory = memory;
        this.malloc = malloc;
        this.free = free;
    }
    
    // String operations: [refcount:i32][length:i32][data...]
    decodeString(ptr) {
        const view = new DataView(this.memory.buffer);
        const length = view.getInt32(ptr - 4, true);  // -4 to get length from header
        const bytes = new Uint8Array(this.memory.buffer, ptr, length);
        return new TextDecoder().decode(bytes);
    }
    
    encodeString(str) {
        const utf8 = new TextEncoder().encode(str);
        const ptr = this.malloc(8 + utf8.length);  // 8-byte header
        const view = new DataView(this.memory.buffer);
        view.setInt32(ptr, 1, true);  // refcount = 1
        view.setInt32(ptr + 4, utf8.length, true);  // length
        new Uint8Array(this.memory.buffer).set(utf8, ptr + 8);
        return ptr + 8;  // Return pointer to data
    }
    
    // Array operations: [refcount:i32][length:i32][capacity:i32][data...]
    decodeArray(ptr, itemSize = 8) {
        const view = new DataView(this.memory.buffer);
        const length = view.getInt32(ptr - 8, true);  // -8 to get length from header
        const result = [];
        
        for (let i = 0; i < length; i++) {
            if (itemSize === 4) {
                result.push(view.getInt32(ptr + i * 4, true));
            } else {
                result.push(view.getFloat64(ptr + i * 8, true));
            }
        }
        return result;
    }
    
    encodeArray(arr, itemSize = 8) {
        const ptr = this.malloc(12 + arr.length * itemSize);
        const view = new DataView(this.memory.buffer);
        view.setInt32(ptr, 1, true);  // refcount
        view.setInt32(ptr + 4, arr.length, true);  // length
        view.setInt32(ptr + 8, arr.length, true);  // capacity
        
        for (let i = 0; i < arr.length; i++) {
            if (itemSize === 4) {
                view.setInt32(ptr + 12 + i * 4, arr[i], true);
            } else {
                view.setFloat64(ptr + 12 + i * 8, arr[i], true);
            }
        }
        return ptr + 12;  // Return pointer to data
    }
    
    // Dict operations: [refcount:i32][size:i32][capacity:i32][slots...]
    createDict(capacity = 16) {
        const ptr = this.malloc(12 + capacity * 12);
        const view = new DataView(this.memory.buffer);
        view.setInt32(ptr, 1, true);  // refcount
        view.setInt32(ptr + 4, 0, true);  // size
        view.setInt32(ptr + 8, capacity, true);  // capacity
        
        // Initialize slots to 0
        for (let i = 0; i < capacity; i++) {
            const slotPtr = ptr + 12 + i * 12;
            view.setInt32(slotPtr, 0, true);  // hash
            view.setInt32(slotPtr + 4, 0, true);  // key_ptr
            view.setInt32(slotPtr + 8, 0, true);  // value_ptr
        }
        return ptr + 12;  // Return pointer to data
    }
    
    // List operations: [refcount:i32][length:i32][capacity:i32][elem_size:i32][data...]
    createList(capacity = 8, elemSize = 8) {
        const ptr = this.malloc(16 + capacity * elemSize);
        const view = new DataView(this.memory.buffer);
        view.setInt32(ptr, 1, true);  // refcount
        view.setInt32(ptr + 4, 0, true);  // length
        view.setInt32(ptr + 8, capacity, true);  // capacity
        view.setInt32(ptr + 12, elemSize, true);  // elem_size
        return ptr + 16;  // Return pointer to data
    }
    
    // Memory info
    getHeapSize(heapTopPtr) {
        const view = new DataView(this.memory.buffer);
        return view.getInt32(heapTopPtr, true) - 65536;
    }
}

// Export for use in browser/Node
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WasmRuntime;
}
"""


if __name__ == '__main__':
    # Test WASM memory manager
    print("=== WASM Memory Manager Test ===\n")
    
    mm = WasmMemoryManager(initial_pages=1)
    
    # Test string allocation
    str_ptr = mm.alloc_string("Hello WASM")
    print(f"Allocated string at: {str_ptr}")
    print(f"String content: {mm.read_string(str_ptr)}")
    
    # Test array allocation
    arr_ptr = mm.alloc_array(10, elem_size=8)
    print(f"Allocated array at: {arr_ptr}")
    
    # Test dict allocation
    dict_ptr = mm.alloc_dict(capacity=16)
    print(f"Allocated dict at: {dict_ptr}")
    
    # Test memory usage
    heap_size = mm.get_heap_size()
    total_size = mm.get_total_size()
    print(f"\nHeap usage: {heap_size} / {total_size} bytes")
    print(f"Utilization: {100 * heap_size / total_size:.2f}%")
    
    # Cleanup
    mm.decref(str_ptr)
    mm.decref(arr_ptr)
    mm.decref_dict(dict_ptr)
    
    print("\n✅ WASM Memory Manager working with unified MemoryManager")
