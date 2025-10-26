"""
Type system for pyexec compiler.

Defines primitive types, array types, and type operations without external dependencies.
Uses dataclasses for zero-overhead type descriptors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any
from llvmlite import ir
import numpy as np


class TypeKind(Enum):
    """Enumeration of supported type kinds."""
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    COMPLEX64 = auto()
    COMPLEX128 = auto()
    BOOL = auto()
    ARRAY = auto()
    STRING = auto()  # UTF-8 string in WASM memory
    BYTES = auto()   # Raw byte array
    STRUCT = auto()  # User-defined struct
    POINTER = auto() # Memory pointer (i32 in WASM)
    VOID = auto()


@dataclass(frozen=True, eq=True)
class Type:
    """
    Immutable type descriptor with hash-consing for identity comparison.
    
    Attributes:
        kind: Type kind enum value
        size: Size in bytes
        alignment: Alignment requirement in bytes
        signed: Whether integer type is signed (None for non-integer)
    """
    kind: TypeKind
    size: int
    alignment: int
    signed: Optional[bool] = None
    
    def __post_init__(self) -> None:
        """Validate type parameters."""
        # VOID type has size 0, which is valid
        if self.kind != TypeKind.VOID and self.size <= 0:
            raise ValueError(f"Type size must be positive, got {self.size}")
        if self.alignment <= 0:
            raise ValueError(f"Type alignment must be positive, got {self.alignment}")
        if self.kind in (TypeKind.INT8, TypeKind.INT16, TypeKind.INT32, TypeKind.INT64,
                        TypeKind.UINT8, TypeKind.UINT16, TypeKind.UINT32, TypeKind.UINT64):
            if self.signed is None:
                raise ValueError(f"Integer type {self.kind} requires signed parameter")
    
    def to_llvm(self) -> ir.Type:
        """Convert to LLVM IR type."""
        mapping: Dict[TypeKind, ir.Type] = {
            TypeKind.INT8: ir.IntType(8),
            TypeKind.INT16: ir.IntType(16),
            TypeKind.INT32: ir.IntType(32),
            TypeKind.INT64: ir.IntType(64),
            TypeKind.UINT8: ir.IntType(8),
            TypeKind.UINT16: ir.IntType(16),
            TypeKind.UINT32: ir.IntType(32),
            TypeKind.UINT64: ir.IntType(64),
            TypeKind.FLOAT32: ir.FloatType(),
            TypeKind.FLOAT64: ir.DoubleType(),
            TypeKind.BOOL: ir.IntType(1),
            TypeKind.STRING: ir.IntType(32),  # i32 pointer in WASM
            TypeKind.BYTES: ir.IntType(32),   # i32 pointer
            TypeKind.POINTER: ir.IntType(32), # i32 pointer
            TypeKind.VOID: ir.VoidType(),
        }
        
        llvm_type = mapping.get(self.kind)
        if llvm_type is None:
            raise NotImplementedError(f"LLVM conversion not implemented for {self.kind}")
        return llvm_type
    
    def to_numpy_dtype(self) -> np.dtype:
        """Convert to NumPy dtype."""
        mapping: Dict[TypeKind, Any] = {
            TypeKind.INT8: np.int8,
            TypeKind.INT16: np.int16,
            TypeKind.INT32: np.int32,
            TypeKind.INT64: np.int64,
            TypeKind.UINT8: np.uint8,
            TypeKind.UINT16: np.uint16,
            TypeKind.UINT32: np.uint32,
            TypeKind.UINT64: np.uint64,
            TypeKind.FLOAT32: np.float32,
            TypeKind.FLOAT64: np.float64,
            TypeKind.COMPLEX64: np.complex64,
            TypeKind.COMPLEX128: np.complex128,
            TypeKind.BOOL: np.bool_,
        }
        
        dtype_cls = mapping.get(self.kind)
        if dtype_cls is None:
            raise NotImplementedError(f"NumPy dtype conversion not implemented for {self.kind}")
        return np.dtype(dtype_cls)
    
    def to_cffi_type(self) -> str:
        """Convert to CFFI type string."""
        mapping: Dict[TypeKind, str] = {
            TypeKind.INT8: "int8_t",
            TypeKind.INT16: "int16_t",
            TypeKind.INT32: "int32_t",
            TypeKind.INT64: "int64_t",
            TypeKind.UINT8: "uint8_t",
            TypeKind.UINT16: "uint16_t",
            TypeKind.UINT32: "uint32_t",
            TypeKind.UINT64: "uint64_t",
            TypeKind.FLOAT32: "float",
            TypeKind.FLOAT64: "double",
            TypeKind.BOOL: "bool",
            TypeKind.VOID: "void",
        }
        
        cffi_type = mapping.get(self.kind)
        if cffi_type is None:
            raise NotImplementedError(f"CFFI type conversion not implemented for {self.kind}")
        return cffi_type


@dataclass(frozen=True, eq=True)
class ArrayType:
    """
    Array type descriptor.
    
    Attributes:
        element_type: Type of array elements
        ndim: Number of dimensions
        shape: Tuple of dimension sizes (None for dynamic)
        c_contiguous: Whether array is C-contiguous (row-major)
    """
    element_type: Type
    ndim: int
    shape: Optional[Tuple[int, ...]] = None
    c_contiguous: bool = True
    
    def __post_init__(self) -> None:
        """Validate array type parameters."""
        if self.ndim <= 0:
            raise ValueError(f"Array ndim must be positive, got {self.ndim}")
        if self.shape is not None:
            if len(self.shape) != self.ndim:
                raise ValueError(f"Shape length {len(self.shape)} != ndim {self.ndim}")
            if any(dim <= 0 for dim in self.shape):
                raise ValueError(f"All shape dimensions must be positive, got {self.shape}")
    
    def to_llvm(self) -> ir.Type:
        """
        Convert to LLVM array type.
        
        Returns pointer to element type for dynamic arrays,
        or array type for static shapes.
        """
        elem_type = self.element_type.to_llvm()
        
        if self.shape is None:
            return elem_type.as_pointer()
        
        result_type = elem_type
        for dim in reversed(self.shape):
            result_type = ir.ArrayType(result_type, dim)
        
        return result_type
    
    def total_size(self) -> Optional[int]:
        """Calculate total number of elements, or None if dynamic."""
        if self.shape is None:
            return None
        
        size = 1
        for dim in self.shape:
            size *= dim
        return size


class TypeRegistry:
    """
    Singleton registry for type descriptors with hash-consing.
    
    Ensures type identity comparison works correctly by reusing instances.
    """
    
    _instance: Optional[TypeRegistry] = None
    _types: Dict[TypeKind, Type] = {}
    
    def __new__(cls) -> TypeRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_types()
        return cls._instance
    
    def _initialize_types(self) -> None:
        """Initialize primitive type descriptors."""
        self._types = {
            TypeKind.INT8: Type(TypeKind.INT8, 1, 1, signed=True),
            TypeKind.INT16: Type(TypeKind.INT16, 2, 2, signed=True),
            TypeKind.INT32: Type(TypeKind.INT32, 4, 4, signed=True),
            TypeKind.INT64: Type(TypeKind.INT64, 8, 8, signed=True),
            TypeKind.UINT8: Type(TypeKind.UINT8, 1, 1, signed=False),
            TypeKind.UINT16: Type(TypeKind.UINT16, 2, 2, signed=False),
            TypeKind.UINT32: Type(TypeKind.UINT32, 4, 4, signed=False),
            TypeKind.UINT64: Type(TypeKind.UINT64, 8, 8, signed=False),
            TypeKind.FLOAT32: Type(TypeKind.FLOAT32, 4, 4),
            TypeKind.FLOAT64: Type(TypeKind.FLOAT64, 8, 8),
            TypeKind.COMPLEX64: Type(TypeKind.COMPLEX64, 8, 4),
            TypeKind.COMPLEX128: Type(TypeKind.COMPLEX128, 16, 8),
            TypeKind.BOOL: Type(TypeKind.BOOL, 1, 1),
            TypeKind.VOID: Type(TypeKind.VOID, 0, 1),
        }
    
    def get(self, kind: TypeKind) -> Type:
        """Get type descriptor by kind."""
        type_desc = self._types.get(kind)
        if type_desc is None:
            raise ValueError(f"Unknown type kind: {kind}")
        return type_desc
    
    def from_numpy_dtype(self, dtype: np.dtype) -> Type:
        """Convert NumPy dtype to Type."""
        mapping: Dict[Any, TypeKind] = {
            np.dtype(np.int8): TypeKind.INT8,
            np.dtype(np.int16): TypeKind.INT16,
            np.dtype(np.int32): TypeKind.INT32,
            np.dtype(np.int64): TypeKind.INT64,
            np.dtype(np.uint8): TypeKind.UINT8,
            np.dtype(np.uint16): TypeKind.UINT16,
            np.dtype(np.uint32): TypeKind.UINT32,
            np.dtype(np.uint64): TypeKind.UINT64,
            np.dtype(np.float32): TypeKind.FLOAT32,
            np.dtype(np.float64): TypeKind.FLOAT64,
            np.dtype(np.complex64): TypeKind.COMPLEX64,
            np.dtype(np.complex128): TypeKind.COMPLEX128,
            np.dtype(np.bool_): TypeKind.BOOL,
        }
        
        kind = mapping.get(dtype)
        if kind is None:
            raise ValueError(f"Unsupported NumPy dtype: {dtype}")
        return self.get(kind)
    
    def from_python_type(self, py_type: type) -> Type:
        """Convert Python builtin type to Type."""
        mapping: Dict[type, TypeKind] = {
            int: TypeKind.INT64,
            float: TypeKind.FLOAT64,
            bool: TypeKind.BOOL,
            complex: TypeKind.COMPLEX128,
        }
        
        kind = mapping.get(py_type)
        if kind is None:
            raise ValueError(f"Unsupported Python type: {py_type}")
        return self.get(kind)
    
    def promote(self, type1: Type, type2: Type) -> Type:
        """
        Compute promoted type for binary operations.
        
        Uses promotion hierarchy: bool < int8 < ... < int64 < float32 < float64 < complex128
        """
        hierarchy = [
            TypeKind.BOOL,
            TypeKind.INT8, TypeKind.INT16, TypeKind.INT32, TypeKind.INT64,
            TypeKind.UINT8, TypeKind.UINT16, TypeKind.UINT32, TypeKind.UINT64,
            TypeKind.FLOAT32, TypeKind.FLOAT64,
            TypeKind.COMPLEX64, TypeKind.COMPLEX128,
        ]
        
        try:
            idx1 = hierarchy.index(type1.kind)
            idx2 = hierarchy.index(type2.kind)
        except ValueError as e:
            raise TypeError(f"Cannot promote types {type1.kind} and {type2.kind}") from e
        
        promoted_kind = hierarchy[max(idx1, idx2)]
        return self.get(promoted_kind)


# Singleton instance
_registry = TypeRegistry()

# Public API: pre-instantiated type objects
int8 = _registry.get(TypeKind.INT8)
int16 = _registry.get(TypeKind.INT16)
int32 = _registry.get(TypeKind.INT32)
int64 = _registry.get(TypeKind.INT64)
uint8 = _registry.get(TypeKind.UINT8)
uint16 = _registry.get(TypeKind.UINT16)
uint32 = _registry.get(TypeKind.UINT32)
uint64 = _registry.get(TypeKind.UINT64)
float32 = _registry.get(TypeKind.FLOAT32)
float64 = _registry.get(TypeKind.FLOAT64)
complex64 = _registry.get(TypeKind.COMPLEX64)
complex128 = _registry.get(TypeKind.COMPLEX128)
bool_ = _registry.get(TypeKind.BOOL)
void = _registry.get(TypeKind.VOID)
