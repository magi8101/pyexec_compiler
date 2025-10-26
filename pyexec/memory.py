"""
Unified memory management for pyexec: JIT, AOT, and WASM.
Slab allocator + refcounting using Numba JIT.
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict as NumbaDict
from typing import Optional, Tuple
import ctypes


SIZE_CLASSES = np.array([16, 32, 64, 128, 256, 512, 1024, 2048], dtype=np.int32)
SLAB_CHUNK_SIZE = 4096
MIN_LARGE_SIZE = 2048


@njit
def size_to_class_idx(size: int) -> int:
    """Map allocation size to slab class index."""
    for i in range(len(SIZE_CLASSES)):
        if size <= SIZE_CLASSES[i]:
            return i
    return -1


@njit
def align_up(value: int, alignment: int) -> int:
    """Align value up to alignment boundary."""
    return (value + alignment - 1) & ~(alignment - 1)


@njit
def read_i32(memory: np.ndarray, offset: int) -> int:
    """Read i32 from memory."""
    val = (int(memory[offset]) | 
           (int(memory[offset+1]) << 8) | 
           (int(memory[offset+2]) << 16) | 
           (int(memory[offset+3]) << 24))
    if val >= 0x80000000:
        val -= 0x100000000
    return int(val)


@njit
def write_i32(memory: np.ndarray, offset: int, value: int) -> None:
    """Write i32 to memory."""
    v = int(value)
    if v < 0:
        v = v + 0x100000000
    memory[offset] = v & 0xFF
    memory[offset+1] = (v >> 8) & 0xFF
    memory[offset+2] = (v >> 16) & 0xFF
    memory[offset+3] = (v >> 24) & 0xFF


@njit
def read_i64(memory: np.ndarray, offset: int) -> int:
    """Read i64 from memory."""
    int64_view = memory.view(np.int64)
    idx = offset // 8
    return int64_view[idx]


@njit
def write_i64(memory: np.ndarray, offset: int, value: int) -> None:
    """Write i64 to memory."""
    int64_view = memory.view(np.int64)
    idx = offset // 8
    int64_view[idx] = value


@njit
def read_f32(memory: np.ndarray, offset: int) -> float:
    """Read f32 from memory."""
    float32_view = memory.view(np.float32)
    idx = offset // 4
    return float(float32_view[idx])


@njit
def write_f32(memory: np.ndarray, offset: int, value: float) -> None:
    """Write f32 to memory."""
    float32_view = memory.view(np.float32)
    idx = offset // 4
    float32_view[idx] = value


@njit
def read_f64(memory: np.ndarray, offset: int) -> float:
    """Read f64 from memory."""
    float64_view = memory.view(np.float64)
    idx = offset // 8
    return float(float64_view[idx])


@njit
def write_f64(memory: np.ndarray, offset: int, value: float) -> None:
    """Write f64 to memory."""
    float64_view = memory.view(np.float64)
    idx = offset // 8
    float64_view[idx] = value


@njit
def slab_create_chunk(memory: np.ndarray, heap_top: int, size_class: int) -> int:
    """Create new slab chunk and return head of free list."""
    chunk_start = align_up(heap_top, 8)
    num_blocks = SLAB_CHUNK_SIZE // size_class
    
    for i in range(num_blocks - 1):
        block_addr = chunk_start + i * size_class
        next_addr = block_addr + size_class
        write_i32(memory, block_addr, next_addr)
    
    last_block = chunk_start + (num_blocks - 1) * size_class
    write_i32(memory, last_block, 0)
    
    return chunk_start


@njit
def slab_alloc_core(memory: np.ndarray, free_list_heads: np.ndarray, 
                    heap_top_ref: np.ndarray, class_idx: int) -> int:
    """Allocate from slab. heap_top_ref[0] is mutable heap pointer."""
    # Check if free list is empty (using -1 as empty marker)
    if free_list_heads[class_idx] == -1:
        # Create new chunk
        new_chunk = slab_create_chunk(memory, heap_top_ref[0], SIZE_CLASSES[class_idx])
        free_list_heads[class_idx] = new_chunk
        heap_top_ref[0] = new_chunk + SLAB_CHUNK_SIZE
    
    # Allocate from free list
    ptr = free_list_heads[class_idx]
    next_ptr = read_i32(memory, ptr)
    free_list_heads[class_idx] = next_ptr
    
    return ptr


@njit
def slab_free_core(memory: np.ndarray, free_list_heads: np.ndarray, 
                   ptr: int, class_idx: int) -> None:
    """Free slab allocation."""
    write_i32(memory, ptr, free_list_heads[class_idx])
    free_list_heads[class_idx] = ptr


@njit
def refcount_alloc_core(memory: np.ndarray, heap_top_ref: np.ndarray, 
                        data_size: int) -> int:
    """
    Allocate with refcount.
    Layout: [refcount:i32][size:i32][data...]
    Returns pointer to data (not header), or 0 if out of memory.
    """
    total_size = 8 + data_size
    ptr = align_up(heap_top_ref[0], 8)
    
    # Check for heap overflow
    new_heap_top = ptr + total_size
    if new_heap_top > len(memory):
        return 0  # Out of memory
    
    write_i32(memory, ptr, 1)
    write_i32(memory, ptr + 4, data_size)
    
    heap_top_ref[0] = new_heap_top
    return ptr + 8


@njit
def incref_core(memory: np.ndarray, data_ptr: int) -> None:
    """Increment refcount."""
    if data_ptr == 0:
        return
    header_ptr = data_ptr - 8
    refcount = read_i32(memory, header_ptr)
    write_i32(memory, header_ptr, refcount + 1)


@njit
def decref_core(memory: np.ndarray, free_list_heads: np.ndarray, 
                data_ptr: int) -> int:
    """
    Decrement refcount. Returns 1 if freed, 0 otherwise.
    """
    if data_ptr == 0:
        return 0
    
    header_ptr = data_ptr - 8
    refcount = read_i32(memory, header_ptr)
    
    if refcount <= 1:
        size = read_i32(memory, header_ptr + 4)
        total_size = 8 + size
        
        class_idx = size_to_class_idx(total_size)
        if class_idx >= 0:
            slab_free_core(memory, free_list_heads, header_ptr, class_idx)
        
        return 1
    else:
        write_i32(memory, header_ptr, refcount - 1)
        return 0


@njit
def dict_cleanup_core(memory: np.ndarray, free_list_heads: np.ndarray, dict_ptr: int) -> None:
    """Cleanup all key-value pairs in dict before freeing."""
    header_ptr = dict_ptr - 12
    capacity = read_i32(memory, header_ptr + 8)
    
    for i in range(capacity):
        slot_offset = dict_ptr + i * 12
        stored_hash = read_i32(memory, slot_offset)
        
        if stored_hash > 0:
            key_ptr = read_i32(memory, slot_offset + 4)
            val_ptr = read_i32(memory, slot_offset + 8)
            
            if key_ptr != 0:
                decref_core(memory, free_list_heads, key_ptr)
            if val_ptr != 0:
                decref_core(memory, free_list_heads, val_ptr)


@njit
def dict_needs_resize(memory: np.ndarray, dict_ptr: int) -> bool:
    """Check if dict needs resizing (load factor > 0.7)."""
    header_ptr = dict_ptr - 12
    size = read_i32(memory, header_ptr + 4)
    capacity = read_i32(memory, header_ptr + 8)
    return size * 10 > capacity * 7


@njit
def dict_resize_core(memory: np.ndarray, free_list_heads: np.ndarray,
                     heap_top_ref: np.ndarray, dict_ptr: int) -> int:
    """Resize dict to 2x capacity. Returns new dict pointer."""
    header_ptr = dict_ptr - 12
    old_capacity = read_i32(memory, header_ptr + 8)
    new_capacity = old_capacity * 2
    
    # Allocate new dict
    new_dict_ptr = dict_alloc_core(memory, free_list_heads, heap_top_ref, new_capacity)
    if new_dict_ptr == 0:
        return 0  # Out of memory
    
    # Rehash all active entries
    for i in range(old_capacity):
        slot_offset = dict_ptr + i * 12
        stored_hash = read_i32(memory, slot_offset)
        
        if stored_hash > 0:  # Active entry
            key_ptr = read_i32(memory, slot_offset + 4)
            val_ptr = read_i32(memory, slot_offset + 8)
            
            # Insert into new dict (dict_set_core will handle refcounting)
            success = dict_set_core(memory, free_list_heads, new_dict_ptr, key_ptr, val_ptr)
            if not success:
                # Handle insertion failure
                dict_cleanup_core(memory, free_list_heads, new_dict_ptr)
                # Free the new dict structure
                new_header_ptr = new_dict_ptr - 12
                total_size = 12 + new_capacity * 12
                class_idx = size_to_class_idx(total_size)
                if class_idx >= 0:
                    slab_free_core(memory, free_list_heads, new_header_ptr, class_idx)
                return 0
    
    return new_dict_ptr


@njit
def string_alloc_core(memory: np.ndarray, free_list_heads: np.ndarray,
                      heap_top_ref: np.ndarray, string_bytes: np.ndarray) -> int:
    """
    Allocate string with refcount.
    Layout: [refcount:i32][length:i32][data...]
    Returns pointer to data, or 0 if out of memory.
    """
    length = len(string_bytes)
    total_size = 8 + length
    
    class_idx = size_to_class_idx(total_size)
    
    if class_idx >= 0:
        ptr = slab_alloc_core(memory, free_list_heads, heap_top_ref, class_idx)
    else:
        ptr = align_up(heap_top_ref[0], 8)
        new_heap_top = ptr + total_size
        if new_heap_top > len(memory):
            return 0  # Out of memory
        heap_top_ref[0] = new_heap_top
    
    write_i32(memory, ptr, 1)
    write_i32(memory, ptr + 4, length)
    
    data_ptr = ptr + 8
    memory[data_ptr:data_ptr+length] = string_bytes
    
    return data_ptr


@njit
def array_alloc_core(memory: np.ndarray, free_list_heads: np.ndarray,
                     heap_top_ref: np.ndarray, length: int, elem_size: int) -> int:
    """
    Allocate array with refcount.
    Layout: [refcount:i32][length:i32][capacity:i32][data...]
    """
    capacity = max(length, 8)
    data_size = capacity * elem_size
    total_size = 12 + data_size
    
    class_idx = size_to_class_idx(total_size)
    
    if class_idx >= 0:
        ptr = slab_alloc_core(memory, free_list_heads, heap_top_ref, class_idx)
    else:
        ptr = align_up(heap_top_ref[0], 8)
        heap_top_ref[0] = ptr + total_size
    
    write_i32(memory, ptr, 1)
    write_i32(memory, ptr + 4, length)
    write_i32(memory, ptr + 8, capacity)
    
    return ptr + 12


@njit
def hash_string(data: np.ndarray, length: int) -> int:
    """
    FNV-1a hash for strings. Returns positive i32 for cross-platform compatibility.
    Range: [1, 2147483646] (avoids 0=empty, -1=tombstone, negative=portability issues).
    """
    hash_val = 2166136261
    for i in range(length):
        hash_val = ((hash_val ^ int(data[i])) * 16777619) & 0xFFFFFFFF
    
    # Keep in safe positive range for cross-platform modulo operations
    result = (hash_val % 2147483645) + 1
    return result


@njit
def dict_alloc_core(memory: np.ndarray, free_list_heads: np.ndarray,
                    heap_top_ref: np.ndarray, capacity: int) -> int:
    """
    Allocate dict with open addressing.
    Layout: [refcount:i32][size:i32][capacity:i32][hashes...][keys...][values...]
    Each slot: hash:i32, key_ptr:i32, value_ptr:i32
    Returns pointer to data, or 0 if out of memory.
    """
    cap = max(capacity, 8)
    slot_size = 12
    data_size = cap * slot_size
    total_size = 12 + data_size
    
    ptr = align_up(heap_top_ref[0], 8)
    new_heap_top = ptr + total_size
    if new_heap_top > len(memory):
        return 0  # Out of memory
    heap_top_ref[0] = new_heap_top
    
    write_i32(memory, ptr, 1)
    write_i32(memory, ptr + 4, 0)
    write_i32(memory, ptr + 8, cap)
    
    data_ptr = ptr + 12
    for i in range(cap):
        slot_offset = data_ptr + i * slot_size
        write_i32(memory, slot_offset, 0)
        write_i32(memory, slot_offset + 4, 0)
        write_i32(memory, slot_offset + 8, 0)
    
    return data_ptr


@njit
def dict_find_slot(memory: np.ndarray, dict_ptr: int, key_hash: int, 
                   key_ptr: int, key_len: int) -> int:
    """
    Find slot for key using improved quadratic probing.
    Uses formula: (base + probe + probe²) % capacity
    This provides better slot coverage than simple probe².
    Returns slot index, or -1 if table full.
    """
    header_ptr = dict_ptr - 12
    capacity = read_i32(memory, header_ptr + 8)
    
    base_idx = key_hash % capacity
    first_tombstone = -1
    
    for probe in range(capacity):
        # Improved quadratic: (base + probe + probe²) % capacity
        idx = (base_idx + probe + probe * probe) % capacity
        
        slot_offset = dict_ptr + idx * 12
        stored_hash = read_i32(memory, slot_offset)
        
        if stored_hash == 0:
            # Return first tombstone if we found one, else this empty slot
            return first_tombstone if first_tombstone >= 0 else idx
        
        if stored_hash == -1:
            # Remember first tombstone for potential reuse
            if first_tombstone < 0:
                first_tombstone = idx
            continue
        
        if stored_hash == key_hash:
            stored_key_ptr = read_i32(memory, slot_offset + 4)
            if stored_key_ptr != 0:
                stored_key_header = stored_key_ptr - 8
                stored_key_len = read_i32(memory, stored_key_header + 4)
                
                if stored_key_len == key_len:
                    match = True
                    for i in range(key_len):
                        if memory[stored_key_ptr + i] != memory[key_ptr + i]:
                            match = False
                            break
                    if match:
                        return idx
    
    # Table full - return tombstone if found, else -1
    return first_tombstone if first_tombstone >= 0 else -1


@njit
def dict_set_core(memory: np.ndarray, free_list_heads: np.ndarray,
                  dict_ptr: int, key_ptr: int, value_ptr: int) -> int:
    """Set key-value pair in dict. Returns 1 on success, 0 if table full."""
    header_ptr = dict_ptr - 12
    key_header = key_ptr - 8
    key_len = read_i32(memory, key_header + 4)
    
    key_hash = hash_string(memory[key_ptr:key_ptr+key_len], key_len)
    
    slot_idx = dict_find_slot(memory, dict_ptr, key_hash, key_ptr, key_len)
    
    if slot_idx < 0:
        return 0  # Table full, caller should resize
    
    slot_offset = dict_ptr + slot_idx * 12
    stored_hash = read_i32(memory, slot_offset)
    
    if stored_hash == 0:
        size = read_i32(memory, header_ptr + 4)
        write_i32(memory, header_ptr + 4, size + 1)
    else:
        old_key = read_i32(memory, slot_offset + 4)
        old_val = read_i32(memory, slot_offset + 8)
        if old_key != 0:
            decref_core(memory, free_list_heads, old_key)
        if old_val != 0:
            decref_core(memory, free_list_heads, old_val)
    
    write_i32(memory, slot_offset, key_hash)
    write_i32(memory, slot_offset + 4, key_ptr)
    write_i32(memory, slot_offset + 8, value_ptr)
    
    incref_core(memory, key_ptr)
    incref_core(memory, value_ptr)
    
    return 1


@njit
def dict_get_core(memory: np.ndarray, dict_ptr: int, key_ptr: int) -> int:
    """Get value for key. Returns 0 if not found."""
    key_header = key_ptr - 8
    key_len = read_i32(memory, key_header + 4)
    key_hash = hash_string(memory[key_ptr:key_ptr+key_len], key_len)
    
    slot_idx = dict_find_slot(memory, dict_ptr, key_hash, key_ptr, key_len)
    
    if slot_idx >= 0:
        slot_offset = dict_ptr + slot_idx * 12
        stored_hash = read_i32(memory, slot_offset)
        
        if stored_hash != 0 and stored_hash != -1:
            return read_i32(memory, slot_offset + 8)
    
    return 0


@njit
def dict_delete_core(memory: np.ndarray, free_list_heads: np.ndarray,
                     dict_ptr: int, key_ptr: int) -> int:
    """Delete key from dict. Returns 1 if found and deleted, 0 otherwise."""
    key_header = key_ptr - 8
    key_len = read_i32(memory, key_header + 4)
    key_hash = hash_string(memory[key_ptr:key_ptr+key_len], key_len)
    
    slot_idx = dict_find_slot(memory, dict_ptr, key_hash, key_ptr, key_len)
    
    if slot_idx >= 0:
        slot_offset = dict_ptr + slot_idx * 12
        stored_hash = read_i32(memory, slot_offset)
        
        if stored_hash != 0 and stored_hash != -1:
            header_ptr = dict_ptr - 12
            
            old_key = read_i32(memory, slot_offset + 4)
            old_val = read_i32(memory, slot_offset + 8)
            
            decref_core(memory, free_list_heads, old_key)
            decref_core(memory, free_list_heads, old_val)
            
            write_i32(memory, slot_offset, -1)
            write_i32(memory, slot_offset + 4, 0)
            write_i32(memory, slot_offset + 8, 0)
            
            size = read_i32(memory, header_ptr + 4)
            write_i32(memory, header_ptr + 4, size - 1)
            
            return 1
    
    return 0


@njit
def array_get_i32_core(memory: np.ndarray, array_ptr: int, index: int) -> int:
    """Get i32 from array at index."""
    header_ptr = array_ptr - 12
    length = read_i32(memory, header_ptr + 4)
    
    if index < 0 or index >= length:
        return 0
    
    return read_i32(memory, array_ptr + index * 4)


@njit
def array_set_i32_core(memory: np.ndarray, array_ptr: int, index: int, value: int) -> None:
    """Set i32 in array at index."""
    header_ptr = array_ptr - 12
    length = read_i32(memory, header_ptr + 4)
    
    if index < 0 or index >= length:
        return
    
    write_i32(memory, array_ptr + index * 4, value)


@njit
def array_get_f64_core(memory: np.ndarray, array_ptr: int, index: int) -> float:
    """Get f64 from array at index."""
    header_ptr = array_ptr - 12
    length = read_i32(memory, header_ptr + 4)
    
    if index < 0 or index >= length:
        return 0.0
    
    return read_f64(memory, array_ptr + index * 8)


@njit
def array_set_f64_core(memory: np.ndarray, array_ptr: int, index: int, value: float) -> None:
    """Set f64 in array at index."""
    header_ptr = array_ptr - 12
    length = read_i32(memory, header_ptr + 4)
    
    if index < 0 or index >= length:
        return
    
    write_f64(memory, array_ptr + index * 8, value)


@njit
def list_alloc_core(memory: np.ndarray, free_list_heads: np.ndarray,
                    heap_top_ref: np.ndarray, capacity: int, elem_size: int) -> int:
    """
    Allocate list with dynamic capacity.
    Layout: [refcount:i32][length:i32][capacity:i32][elem_size:i32][data...]
    Returns pointer to data, or 0 if out of memory.
    """
    cap = max(capacity, 8)
    data_size = cap * elem_size
    total_size = 16 + data_size
    
    ptr = align_up(heap_top_ref[0], 8)
    new_heap_top = ptr + total_size
    if new_heap_top > len(memory):
        return 0  # Out of memory
    heap_top_ref[0] = new_heap_top
    
    write_i32(memory, ptr, 1)
    write_i32(memory, ptr + 4, 0)
    write_i32(memory, ptr + 8, cap)
    write_i32(memory, ptr + 12, elem_size)
    
    return ptr + 16


@njit
def list_resize_core(memory: np.ndarray, free_list_heads: np.ndarray,
                     heap_top_ref: np.ndarray, list_ptr: int) -> int:
    """Resize list to 2x capacity. Returns new list pointer."""
    header_ptr = list_ptr - 16
    old_refcount = read_i32(memory, header_ptr)
    length = read_i32(memory, header_ptr + 4)
    capacity = read_i32(memory, header_ptr + 8)
    elem_size = read_i32(memory, header_ptr + 12)
    
    new_capacity = capacity * 2
    new_list_ptr = list_alloc_core(memory, free_list_heads, heap_top_ref, 
                                   new_capacity, elem_size)
    if new_list_ptr == 0:
        return 0  # Out of memory
    
    new_header = new_list_ptr - 16
    write_i32(memory, new_header + 4, length)  # Set length
    
    # Copy data
    for i in range(length * elem_size):
        memory[new_list_ptr + i] = memory[list_ptr + i]
    
    # Free old list if we have the only reference
    if old_refcount == 1:
        size_needed = 16 + capacity * elem_size
        class_idx = size_to_class_idx(size_needed)
        if class_idx >= 0:
            slab_free_core(memory, free_list_heads, header_ptr, class_idx)
    
    return new_list_ptr


@njit
def list_append_core(memory: np.ndarray, free_list_heads: np.ndarray,
                     heap_top_ref: np.ndarray, list_ptr_ref: np.ndarray,
                     value: float) -> int:
    """
    Append value to list. list_ptr_ref[0] may be updated if resized.
    Returns 0 on success, -1 if list has multiple references (cannot resize).
    """
    list_ptr = list_ptr_ref[0]
    header_ptr = list_ptr - 16
    length = read_i32(memory, header_ptr + 4)
    capacity = read_i32(memory, header_ptr + 8)
    elem_size = read_i32(memory, header_ptr + 12)
    
    if length >= capacity:
        # Check refcount before resizing
        refcount = read_i32(memory, header_ptr)
        if refcount > 1:
            return -1  # ERROR: Cannot resize shared list
        
        new_list = list_resize_core(memory, free_list_heads, heap_top_ref, list_ptr)
        if new_list == 0:
            return -2  # ERROR: Out of memory during resize
        list_ptr_ref[0] = new_list
        list_ptr = new_list
        header_ptr = list_ptr - 16
        capacity = read_i32(memory, header_ptr + 8)  # Update capacity
    
    # Append value
    offset = list_ptr + length * elem_size
    
    if elem_size == 4:
        write_f32(memory, offset, value)
    elif elem_size == 8:
        write_f64(memory, offset, value)
    
    write_i32(memory, header_ptr + 4, length + 1)
    return 0


@njit
def list_append_ptr_core(memory: np.ndarray, free_list_heads: np.ndarray,
                         heap_top_ref: np.ndarray, list_ptr_ref: np.ndarray,
                         ptr_value: int) -> int:
    """
    Append pointer to list (for lists of objects). Increments refcount.
    Returns 0 on success, -1 if shared list, -2 if wrong elem_size.
    """
    list_ptr = list_ptr_ref[0]
    header_ptr = list_ptr - 16
    length = read_i32(memory, header_ptr + 4)
    capacity = read_i32(memory, header_ptr + 8)
    elem_size = read_i32(memory, header_ptr + 12)
    
    if elem_size != 4:
        return -2  # Wrong element size for pointers
    
    if length >= capacity:
        refcount = read_i32(memory, header_ptr)
        if refcount > 1:
            return -1  # Cannot resize shared list
        
        new_list = list_resize_core(memory, free_list_heads, heap_top_ref, list_ptr)
        if new_list == 0:
            return -3  # ERROR: Out of memory during resize
        list_ptr_ref[0] = new_list
        list_ptr = new_list
        header_ptr = list_ptr - 16
    
    offset = list_ptr + length * 4
    write_i32(memory, offset, ptr_value)
    incref_core(memory, ptr_value)
    
    write_i32(memory, header_ptr + 4, length + 1)
    return 0


@njit
def list_get_ptr_core(memory: np.ndarray, list_ptr: int, index: int) -> int:
    """Get pointer at index. Does NOT incref. Returns 0 if out of bounds."""
    header_ptr = list_ptr - 16
    length = read_i32(memory, header_ptr + 4)
    elem_size = read_i32(memory, header_ptr + 12)
    
    if index < 0 or index >= length or elem_size != 4:
        return 0
    
    return read_i32(memory, list_ptr + index * 4)


@njit
def list_cleanup_ptrs_core(memory: np.ndarray, free_list_heads: np.ndarray,
                           list_ptr: int) -> None:
    """Cleanup pointer list - decref all elements."""
    header_ptr = list_ptr - 16
    length = read_i32(memory, header_ptr + 4)
    elem_size = read_i32(memory, header_ptr + 12)
    
    if elem_size == 4:  # Pointer list
        for i in range(length):
            ptr_val = read_i32(memory, list_ptr + i * 4)
            if ptr_val != 0:
                decref_core(memory, free_list_heads, ptr_val)


@njit
def list_get_core(memory: np.ndarray, list_ptr: int, index: int) -> float:
    """Get value at index. Returns 0.0 if out of bounds."""
    header_ptr = list_ptr - 16
    length = read_i32(memory, header_ptr + 4)
    elem_size = read_i32(memory, header_ptr + 12)
    
    if index < 0 or index >= length:
        return 0.0
    
    offset = list_ptr + index * elem_size
    
    if elem_size == 4:
        return read_f32(memory, offset)
    elif elem_size == 8:
        return read_f64(memory, offset)
    
    return 0.0


@njit
def list_set_core(memory: np.ndarray, list_ptr: int, index: int, value: float) -> None:
    """Set value at index."""
    header_ptr = list_ptr - 16
    length = read_i32(memory, header_ptr + 4)
    elem_size = read_i32(memory, header_ptr + 12)
    
    if index < 0 or index >= length:
        return
    
    offset = list_ptr + index * elem_size
    
    if elem_size == 4:
        write_f32(memory, offset, value)
    elif elem_size == 8:
        write_f64(memory, offset, value)


class MemoryManager:
    """Unified memory manager for all backends."""
    
    def __init__(self, size: int = 64 * 1024 * 1024):
        self.memory = np.zeros(size, dtype=np.uint8)
        self.heap_top = np.array([0], dtype=np.int32)
        self.free_lists = np.full(len(SIZE_CLASSES), -1, dtype=np.int32)  # -1 = empty
        self.size = size
    
    def alloc(self, size: int) -> int:
        """Allocate memory block."""
        class_idx = size_to_class_idx(size)
        
        if class_idx >= 0:
            return slab_alloc_core(self.memory, self.free_lists, self.heap_top, class_idx)
        else:
            return refcount_alloc_core(self.memory, self.heap_top, size)
    
    def free(self, ptr: int, size: int) -> None:
        """Free memory block."""
        class_idx = size_to_class_idx(size)
        
        if class_idx >= 0:
            slab_free_core(self.memory, self.free_lists, ptr, class_idx)
    
    def alloc_refcounted(self, size: int) -> int:
        """Allocate with automatic refcounting."""
        ptr = refcount_alloc_core(self.memory, self.heap_top, size)
        if ptr == 0:
            raise MemoryError(f"Out of memory: cannot allocate {size} bytes")
        return ptr
    
    def incref(self, ptr: int) -> None:
        """
        Increment reference count.
        WARNING: Only works for 8-byte header allocations (strings, arrays, refcounted values).
        """
        incref_core(self.memory, ptr)
    
    def decref(self, ptr: int) -> None:
        """
        Decrement reference count and free if zero.
        WARNING: Only works for 8-byte header allocations (strings, arrays, refcounted values).
        """
        decref_core(self.memory, self.free_lists, ptr)
    
    def dict_cleanup(self, dict_ptr: int) -> None:
        """Cleanup dict contents before freeing."""
        dict_cleanup_core(self.memory, self.free_lists, dict_ptr)
    
    def decref_dict(self, dict_ptr: int) -> None:
        """Properly cleanup and decref a dict."""
        if dict_ptr == 0:
            return
        
        # Cleanup all key-value pairs first
        self.dict_cleanup(dict_ptr)
        
        # Decrement dict refcount and free if needed
        header_ptr = dict_ptr - 12
        refcount = read_i32(self.memory, header_ptr)
        
        if refcount <= 1:
            # Free the dict structure
            capacity = read_i32(self.memory, header_ptr + 8)
            total_size = 12 + capacity * 12
            class_idx = size_to_class_idx(total_size)
            if class_idx >= 0:
                slab_free_core(self.memory, self.free_lists, header_ptr, class_idx)
        else:
            write_i32(self.memory, header_ptr, refcount - 1)
    
    def alloc_string(self, s: str) -> int:
        """Allocate string and return pointer to data."""
        string_bytes = np.frombuffer(s.encode('utf-8'), dtype=np.uint8)
        ptr = string_alloc_core(self.memory, self.free_lists, self.heap_top, string_bytes)
        if ptr == 0:
            raise MemoryError(f"Out of memory: cannot allocate string of length {len(s)}")
        return ptr
    
    def read_string(self, ptr: int) -> str:
        """Read string from pointer."""
        header_ptr = ptr - 8
        length = int(read_i32(self.memory, header_ptr + 4))
        string_bytes = bytes(self.memory[ptr:ptr+length])
        return string_bytes.decode('utf-8')
    
    def alloc_array(self, length: int, elem_size: int = 8) -> int:
        """Allocate array and return pointer to data."""
        return array_alloc_core(self.memory, self.free_lists, self.heap_top, length, elem_size)
    
    def write_i32(self, offset: int, value: int) -> None:
        """Write i32 at offset."""
        write_i32(self.memory, offset, value)
    
    def read_i32(self, offset: int) -> int:
        """Read i32 from offset."""
        return read_i32(self.memory, offset)
    
    def write_i32_safe(self, offset: int, value: int) -> None:
        """Write i32 at offset with bounds checking."""
        if offset < 0 or offset + 4 > self.size:
            raise IndexError(f"Memory access out of bounds: offset={offset}, size={self.size}")
        write_i32(self.memory, offset, value)
    
    def read_i32_safe(self, offset: int) -> int:
        """Read i32 from offset with bounds checking."""
        if offset < 0 or offset + 4 > self.size:
            raise IndexError(f"Memory access out of bounds: offset={offset}, size={self.size}")
        return read_i32(self.memory, offset)
    
    def write_f64(self, offset: int, value: float) -> None:
        """Write f64 at offset."""
        write_f64(self.memory, offset, value)
    
    def read_f64(self, offset: int) -> float:
        """Read f64 from offset."""
        return read_f64(self.memory, offset)
    
    def write_f32(self, offset: int, value: float) -> None:
        """Write f32 at offset."""
        write_f32(self.memory, offset, value)
    
    def read_f32(self, offset: int) -> float:
        """Read f32 from offset."""
        return read_f32(self.memory, offset)
    
    def write_i64(self, offset: int, value: int) -> None:
        """Write i64 at offset."""
        write_i64(self.memory, offset, value)
    
    def read_i64(self, offset: int) -> int:
        """Read i64 from offset."""
        return read_i64(self.memory, offset)
    
    def alloc_dict(self, capacity: int = 16) -> int:
        """Allocate dict and return pointer to data."""
        ptr = dict_alloc_core(self.memory, self.free_lists, self.heap_top, capacity)
        if ptr == 0:
            raise MemoryError(f"Out of memory: cannot allocate dict with capacity {capacity}")
        return ptr
    
    def dict_set(self, dict_ptr: int, key: str, value_ptr: int) -> None:
        """Set key-value in dict. Value must be a pointer."""
        key_ptr = self.alloc_string(key)
        result = dict_set_core(self.memory, self.free_lists, dict_ptr, key_ptr, value_ptr)
        self.decref(key_ptr)
        if result == 0:
            raise RuntimeError(f"Dict is full, cannot insert key '{key}'. Use dict_set_auto_resize.")
    
    def dict_set_auto_resize(self, dict_ptr_holder: np.ndarray, key: str, value_ptr: int) -> None:
        """Set key-value with automatic resize. Pass dict_ptr in array holder."""
        dict_ptr = dict_ptr_holder[0]
        
        # Check if resize needed before insertion
        if dict_needs_resize(self.memory, dict_ptr):
            new_dict_ptr = dict_resize_core(self.memory, self.free_lists, self.heap_top, dict_ptr)
            if new_dict_ptr == 0:
                raise MemoryError("Out of memory during dict resize")
            dict_ptr_holder[0] = new_dict_ptr
            dict_ptr = new_dict_ptr
        
        key_ptr = self.alloc_string(key)
        result = dict_set_core(self.memory, self.free_lists, dict_ptr, key_ptr, value_ptr)
        self.decref(key_ptr)
        
        if result == 0:
            # Still full after resize attempt
            raise RuntimeError(f"Dict insertion failed even after resize")
    
    def dict_get(self, dict_ptr: int, key: str) -> int:
        """Get value pointer for key. Returns 0 if not found."""
        key_ptr = self.alloc_string(key)
        result = dict_get_core(self.memory, dict_ptr, key_ptr)
        self.decref(key_ptr)
        return result
    
    def dict_set_int(self, dict_ptr: int, key: str, value: int) -> None:
        """Set integer value in dict."""
        val_ptr = self.alloc_refcounted(4)
        self.write_i32(val_ptr, value)
        self.dict_set(dict_ptr, key, val_ptr)
        self.decref(val_ptr)
    
    def dict_set_int_auto_resize(self, dict_ptr_holder: np.ndarray, key: str, value: int) -> None:
        """Set integer value with auto-resize."""
        val_ptr = self.alloc_refcounted(4)
        self.write_i32(val_ptr, value)
        self.dict_set_auto_resize(dict_ptr_holder, key, val_ptr)
        self.decref(val_ptr)
    
    def dict_set_float(self, dict_ptr: int, key: str, value: float) -> None:
        """Set float value in dict."""
        val_ptr = self.alloc_refcounted(8)
        self.write_f64(val_ptr, value)
        self.dict_set(dict_ptr, key, val_ptr)
        self.decref(val_ptr)
    
    def dict_set_float_auto_resize(self, dict_ptr_holder: np.ndarray, key: str, value: float) -> None:
        """Set float value with auto-resize."""
        val_ptr = self.alloc_refcounted(8)
        self.write_f64(val_ptr, value)
        self.dict_set_auto_resize(dict_ptr_holder, key, val_ptr)
        self.decref(val_ptr)
    
    def dict_get_int(self, dict_ptr: int, key: str) -> int:
        """Get integer value from dict. Returns 0 if not found."""
        val_ptr = self.dict_get(dict_ptr, key)
        if val_ptr == 0:
            return 0
        return self.read_i32(val_ptr)
    
    def dict_get_float(self, dict_ptr: int, key: str) -> float:
        """Get float value from dict. Returns 0.0 if not found."""
        val_ptr = self.dict_get(dict_ptr, key)
        if val_ptr == 0:
            return 0.0
        return self.read_f64(val_ptr)
    
    def dict_delete(self, dict_ptr: int, key: str) -> bool:
        """Delete key from dict. Returns True if found and deleted."""
        key_ptr = self.alloc_string(key)
        result = dict_delete_core(self.memory, self.free_lists, dict_ptr, key_ptr)
        self.decref(key_ptr)
        return result == 1
    
    def array_get_i32(self, array_ptr: int, index: int) -> int:
        """Get i32 from array."""
        return array_get_i32_core(self.memory, array_ptr, index)
    
    def array_set_i32(self, array_ptr: int, index: int, value: int) -> None:
        """Set i32 in array."""
        array_set_i32_core(self.memory, array_ptr, index, value)
    
    def array_get_f64(self, array_ptr: int, index: int) -> float:
        """Get f64 from array."""
        return array_get_f64_core(self.memory, array_ptr, index)
    
    def array_set_f64(self, array_ptr: int, index: int, value: float) -> None:
        """Set f64 in array."""
        array_set_f64_core(self.memory, array_ptr, index, value)
    
    def array_length(self, array_ptr: int) -> int:
        """Get array length."""
        header_ptr = array_ptr - 12
        return read_i32(self.memory, header_ptr + 4)
    
    def alloc_list(self, capacity: int = 8, elem_size: int = 8) -> int:
        """Allocate list and return pointer to data."""
        ptr = list_alloc_core(self.memory, self.free_lists, self.heap_top, capacity, elem_size)
        if ptr == 0:
            raise MemoryError(f"Out of memory: cannot allocate list with capacity {capacity}")
        return ptr
    
    def list_append(self, list_ptr_holder: np.ndarray, value: float) -> None:
        """Append value to list. Pass list_ptr in array for potential resize."""
        result = list_append_core(self.memory, self.free_lists, self.heap_top, list_ptr_holder, value)
        if result == -1:
            raise RuntimeError("Cannot append to shared list (refcount > 1). Lists cannot be resized when shared.")
        elif result == -2:
            raise MemoryError("Out of memory during list resize")
    
    def list_append_ptr(self, list_ptr_holder: np.ndarray, ptr: int) -> None:
        """Append pointer to list (for lists of objects)."""
        result = list_append_ptr_core(self.memory, self.free_lists, self.heap_top, 
                                      list_ptr_holder, ptr)
        if result == -1:
            raise RuntimeError("Cannot append to shared list (refcount > 1)")
        elif result == -2:
            raise RuntimeError("List element size must be 4 for pointers")
        elif result == -3:
            raise MemoryError("Out of memory during list resize")
    
    def list_get_ptr(self, list_ptr: int, index: int) -> int:
        """Get pointer at index (for lists of objects). Does not incref."""
        return list_get_ptr_core(self.memory, list_ptr, index)
    
    def decref_list_of_ptrs(self, list_ptr: int) -> None:
        """Properly cleanup and decref a list of pointers."""
        list_cleanup_ptrs_core(self.memory, self.free_lists, list_ptr)
        self.decref(list_ptr)
    
    def decref_list_auto(self, list_ptr: int) -> None:
        """
        Automatically cleanup list based on element size.
        elem_size=4 is treated as pointers and decreffed.
        elem_size=8 is treated as numeric values.
        """
        header_ptr = list_ptr - 16
        elem_size = read_i32(self.memory, header_ptr + 12)
        
        if elem_size == 4:
            # Assume pointer list
            list_cleanup_ptrs_core(self.memory, self.free_lists, list_ptr)
        
        self.decref(list_ptr)
    
    def list_get(self, list_ptr: int, index: int) -> float:
        """Get value at index."""
        return list_get_core(self.memory, list_ptr, index)
    
    def list_set(self, list_ptr: int, index: int, value: float) -> None:
        """Set value at index."""
        list_set_core(self.memory, list_ptr, index, value)
    
    def list_length(self, list_ptr: int) -> int:
        """Get list length."""
        header_ptr = list_ptr - 16
        return read_i32(self.memory, header_ptr + 4)
    
    def get_usage(self) -> Tuple[int, int]:
        """Get (used_bytes, total_bytes)."""
        return (self.heap_top[0], self.size)
    
    def get_ctypes_pointer(self) -> ctypes.c_void_p:
        """Get ctypes pointer to memory (for FFI)."""
        return self.memory.ctypes.data_as(ctypes.c_void_p)
    
    def reset(self) -> None:
        """Reset allocator (for testing)."""
        self.heap_top[0] = 0
        self.free_lists.fill(-1)  # Reset all free lists to empty
