"""
Runtime orchestration for JIT/AOT mode selection.

Handles automatic mode detection and unified compilation API.
"""

from typing import Callable, TypeVar, Literal, Any, Optional
import os

from .jit_backend import NumbaJITBackend


F = TypeVar("F", bound=Callable[..., Any])


class Runtime:
    """
    Main runtime coordinator for pyexec.
    
    Manages JIT/AOT mode selection and compilation dispatch.
    """
    
    def __init__(self, mode: Literal["auto", "jit", "aot"] = "auto") -> None:
        """
        Initialize runtime.
        
        Args:
            mode: Compilation mode
                - "auto": Detect based on environment
                - "jit": Force JIT compilation (Numba)
                - "aot": Force AOT compilation (LLVM)
        """
        if mode not in ("auto", "jit", "aot"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'auto', 'jit', or 'aot'")
        
        self.mode = mode
        self._jit_backend = NumbaJITBackend()
        self._aot_compiler: Optional[Any] = None  # Lazy load AOT compiler
    
    def compile(self, func: F, parallel: bool = False) -> F:
        """
        Compile function with automatic mode detection.
        
        Args:
            func: Function to compile
            parallel: Enable automatic parallelization (JIT only)
            
        Returns:
            Compiled function
        """
        mode = self._detect_mode()
        
        if mode == "jit":
            return self.jit_compile(func, parallel=parallel)
        elif mode == "aot":
            return self.aot_compile(func)
        else:
            raise RuntimeError(f"Unknown compilation mode: {mode}")
    
    def jit_compile(self, func: F, parallel: bool = False) -> F:
        """
        Force JIT compilation via Numba.
        
        Args:
            func: Function to compile
            parallel: Enable automatic parallelization
            
        Returns:
            JIT-compiled function
        """
        if parallel:
            return self._jit_backend.compile_parallel(func)
        return self._jit_backend.compile(func)
    
    def aot_compile(self, func: F) -> F:
        """
        Force AOT compilation via LLVM.
        
        Args:
            func: Function to compile
            
        Returns:
            AOT-compiled function
        """
        if self._aot_compiler is None:
            from .aot_compiler import AOTCompiler
            self._aot_compiler = AOTCompiler()
        
        return self._aot_compiler.compile(func)
    
    def _detect_mode(self) -> Literal["jit", "aot"]:
        """
        Detect compilation mode from environment.
        
        Returns:
            Detected mode ("jit" or "aot")
        """
        if self.mode != "auto":
            return self.mode  # type: ignore
        
        env_mode = os.getenv("PYEXEC_MODE", "").lower()
        
        if env_mode == "production":
            return "aot"
        elif env_mode == "development":
            return "jit"
        
        if self._is_interactive():
            return "jit"
        
        return "jit"
    
    def _is_interactive(self) -> bool:
        """
        Check if running in interactive environment.
        
        Returns:
            True if interactive (REPL, Jupyter), False otherwise
        """
        try:
            __IPYTHON__  # type: ignore
            return True
        except NameError:
            pass
        
        import sys
        return hasattr(sys, "ps1")
