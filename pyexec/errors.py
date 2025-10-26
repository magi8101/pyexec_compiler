"""
Error types for pyexec compiler.

Production-grade error handling with source location tracking.
"""

from typing import Optional
import ast


class PyexecError(Exception):
    """Base exception for all pyexec errors."""
    pass


class CompilationError(PyexecError):
    """
    Error during compilation phase.
    
    Attributes:
        message: Error message
        source_node: Optional AST node where error occurred
    """
    
    def __init__(
        self,
        message: str,
        source_node: Optional[ast.AST] = None,
    ) -> None:
        """
        Initialize compilation error.
        
        Args:
            message: Error message
            source_node: AST node for source location
        """
        self.message = message
        self.source_node = source_node
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with source location."""
        if self.source_node is None:
            return self.message
        
        lineno = getattr(self.source_node, 'lineno', None)
        col_offset = getattr(self.source_node, 'col_offset', None)
        
        if lineno is None:
            return self.message
        
        if col_offset is None:
            return f"Line {lineno}: {self.message}"
        
        return f"Line {lineno}, column {col_offset}: {self.message}"


class TypeInferenceError(CompilationError):
    """Error during type inference."""
    pass


class CodeGenerationError(CompilationError):
    """Error during code generation."""
    pass


class LinkingError(PyexecError):
    """Error during linking phase."""
    pass
