"""
JIT backend using Numba.

Thin wrapper around Numba's JIT compiler for consistent API.
"""

from typing import Callable, TypeVar, Optional, Any
import numba
from numba import types as numba_types


F = TypeVar("F", bound=Callable[..., Any])


class NumbaJITBackend:
    """
    JIT compilation backend using Numba.
    
    Delegates all compilation to Numba for maximum compatibility and performance.
    """
    
    def __init__(self) -> None:
        """Initialize Numba JIT backend."""
        self._cache_enabled = True
    
    def compile(
        self,
        func: F,
        signature: Optional[str] = None,
        nopython: bool = True,
        nogil: bool = True,
        cache: bool = True,
    ) -> F:
        """
        JIT compile function using Numba.
        
        Args:
            func: Python function to compile
            signature: Optional Numba signature string
            nopython: Force nopython mode (recommended)
            nogil: Release GIL during execution (recommended)
            cache: Enable disk caching of compiled functions
            
        Returns:
            Numba-compiled function
        """
        return numba.jit(
            signature,
            nopython=nopython,
            nogil=nogil,
            cache=cache and self._cache_enabled,
        )(func)
    
    def compile_parallel(
        self,
        func: F,
        signature: Optional[str] = None,
    ) -> F:
        """
        JIT compile with automatic parallelization.
        
        Args:
            func: Python function to compile
            signature: Optional Numba signature string
            
        Returns:
            Numba-compiled function with parallelization
        """
        return numba.jit(
            signature,
            nopython=True,
            nogil=True,
            parallel=True,
            cache=self._cache_enabled,
        )(func)
    
    def vectorize(
        self,
        func: F,
        signature: Optional[list[str]] = None,
    ) -> F:
        """
        Create vectorized ufunc.
        
        Args:
            func: Scalar function to vectorize
            signature: List of type signatures
            
        Returns:
            Vectorized function (NumPy ufunc)
        """
        if signature is None:
            return numba.vectorize(nopython=True, cache=self._cache_enabled)(func)
        return numba.vectorize(signature, nopython=True, cache=self._cache_enabled)(func)
    
    def disable_cache(self) -> None:
        """Disable function caching."""
        self._cache_enabled = False
    
    def enable_cache(self) -> None:
        """Enable function caching."""
        self._cache_enabled = True


# Simple decorator interface
def jit(func: F) -> F:
    """
    Simple JIT decorator using Numba.
    
    Usage:
        @jit
        def add(x: int, y: int) -> int:
            return x + y
    """
    return numba.jit(nopython=True, nogil=True, cache=True)(func)
