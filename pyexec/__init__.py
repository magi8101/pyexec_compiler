"""
pyexec: Python JIT/AOT compiler with Numba and LLVM backends.

Provides dual compilation modes:
- JIT: Just-in-time compilation via Numba (development)
- AOT: Ahead-of-time compilation via LLVM (deployment)
- WASM: WebAssembly compilation with JavaScript bindings
"""

from typing import Callable, Literal, TypeVar, Any, List, Dict
from pathlib import Path

from .runtime import Runtime
from .types import int8, int16, int32, int64, uint8, uint16, uint32, uint64
from .types import float32, float64, complex64, complex128, bool_
from .aot_backend import AotCompiler

__version__ = "0.1.0"
__all__ = [
    "Runtime",
    "compile",
    "jit",
    "aot",
    "aot_compile",
    "compile_llvm",
    "wasm",
    "wasm_multi",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
    "complex64", "complex128",
    "bool_",
]

F = TypeVar("F", bound=Callable[..., Any])

_default_runtime = Runtime(mode="auto")


def compile(func: F) -> F:
    """
    Compile function with automatic mode detection.
    
    Uses JIT in development, AOT when PYEXEC_MODE=production.
    
    Args:
        func: Function to compile
        
    Returns:
        Compiled function
    """
    return _default_runtime.compile(func)


def jit(func: F) -> F:
    """
    Force JIT compilation via Numba.
    
    Args:
        func: Function to compile
        
    Returns:
        JIT-compiled function
    """
    return _default_runtime.jit_compile(func)


def aot(func: F) -> F:
    """
    Force AOT compilation via LLVM.
    
    Args:
        func: Function to compile
        
    Returns:
        AOT-compiled function
    """
    return _default_runtime.aot_compile(func)


def wasm(
    func: Callable[..., Any],
    output: str,
    optimize: int = 2,
    export_name: str = None,
    generate_html: bool = True,
) -> Dict[str, Path]:
    """
    Compile function to WebAssembly module.
    
    Generates .wasm binary, .js JavaScript bindings, and .html demo page.
    
    Args:
        func: Function to compile (must have type annotations)
        output: Output path (extension will be changed to .wasm)
        optimize: Optimization level (0-3, default 2)
        export_name: Custom export name (default: function name)
        generate_html: Generate HTML demo page (default: True)
        
    Returns:
        Dictionary with 'wasm', 'js', and optionally 'html' Path objects
        
    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> files = wasm(add, "add.wasm")
        >>> # Creates add.wasm and add.js
    """
    from .wasm_backend import WasmCompiler
    
    compiler = WasmCompiler()
    result = compiler.compile_to_wasm(
        func,
        Path(output),
        optimize=optimize,
        export_name=export_name,
    )
    
    return result


def wasm_multi(
    funcs: List[Callable[..., Any]],
    output: str,
    optimize: int = 2,
) -> Dict[str, Path]:
    """
    Compile multiple functions to single WebAssembly module.
    
    Args:
        funcs: List of functions to compile
        output: Output path for .wasm file
        optimize: Optimization level (0-3)
        
    Returns:
        Dictionary with 'wasm' and 'js' paths
        
    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> def mul(a: int, b: int) -> int:
        ...     return a * b
        >>> files = wasm_multi([add, mul], "math.wasm")
    """
    from .wasm_backend import WasmCompiler
    
    compiler = WasmCompiler()
    return compiler.compile_multiple_to_wasm(
        funcs,
        Path(output),
        optimize=optimize,
    )


# =============================================================================
# AOT Compilation API
# =============================================================================

def aot_compile(
    func: Callable,
    output: str,
    show_progress: bool = False
) -> Path:
    """
    Compile Python function to native binary using Nuitka.
    
    Produces standalone executable for current OS:
    - Windows: .exe
    - Linux: binary
    - macOS: binary
    
    Args:
        func: Python function to compile
        output: Output file path
        show_progress: Show compilation progress (default: False)
        
    Returns:
        Path to generated native binary
        
    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> binary = aot_compile(add, "add.exe")  # Windows
        >>> binary = aot_compile(add, "add")      # Linux/Mac
        >>> # Run: ./add 5 3
    """
    compiler = AotCompiler()
    return compiler.compile_to_executable(
        func, output,
        standalone=True,
        onefile=True,
        show_progress=show_progress
    )


def compile_llvm(func: Callable, base_name: str, output_dir: str = '.') -> Dict[str, Path]:
    """
    Compile Python function to native binaries using LLVM.
    
    Generates all files at once:
    - .o (object file)
    - .dll/.so (shared library)
    - .h (C header file)
    
    Args:
        func: Python function to compile (simple arithmetic only)
        base_name: Base name for output files
        output_dir: Output directory (default: current dir)
        
    Returns:
        Dictionary with 'object', 'library', 'header' paths
        
    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> files = compile_llvm(add, "add")
        >>> # Creates: add.o, add.dll, add.h
        >>> print(files['object'])   # add.o
        >>> print(files['library'])  # add.dll
        >>> print(files['header'])   # add.h
    """
    compiler = AotCompiler()
    return compiler.compile_llvm_full_package(func, base_name, output_dir)

