"""
Intermediate representation for pyexec compiler.

AST-independent typed IR for analysis and code generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum, auto
import ast

from .types import Type, ArrayType


class BinOp(Enum):
    """Binary operators."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOORDIV = auto()
    MOD = auto()
    POW = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    BITOR = auto()
    BITXOR = auto()
    BITAND = auto()
    AND = auto()
    OR = auto()


class UnaryOp(Enum):
    """Unary operators."""
    INVERT = auto()
    NOT = auto()
    UADD = auto()
    USUB = auto()


class CmpOp(Enum):
    """Comparison operators."""
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LTE = auto()
    GT = auto()
    GTE = auto()


@dataclass
class IRNode:
    """
    Base class for IR nodes.
    
    Attributes:
        type_: Result type of expression (None for statements)
        source: Optional source AST node for error reporting
    """
    source: Optional[ast.AST] = field(default=None, kw_only=True)
    
    def get_lineno(self) -> Optional[int]:
        """Get source line number if available."""
        return getattr(self.source, 'lineno', None)
    
    def get_col_offset(self) -> Optional[int]:
        """Get source column offset if available."""
        return getattr(self.source, 'col_offset', None)


@dataclass
class IRConstant(IRNode):
    """
    Constant literal value.
    
    Attributes:
        value: Python literal value (int, float, bool, complex)
        type_: Type of the constant
    """
    value: Union[int, float, bool, complex]
    type_: Union[Type, ArrayType]
    
    def __post_init__(self) -> None:
        """Validate constant value matches type."""
        if isinstance(self.type_, ArrayType):
            raise ValueError("IRConstant cannot have array type")


@dataclass
class IRVariable(IRNode):
    """
    Variable reference.
    
    Attributes:
        name: Variable identifier
        type_: Type of the variable
    """
    name: str
    type_: Union[Type, ArrayType]
    
    def __post_init__(self) -> None:
        """Validate variable name."""
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Invalid variable name: {self.name}")


@dataclass
class IRBinOp(IRNode):
    """
    Binary operation.
    
    Attributes:
        op: Operator enum
        left: Left operand IR node
        right: Right operand IR node
        type_: Result type of operation
    """
    op: BinOp
    left: IRNode
    right: IRNode
    type_: Union[Type, ArrayType]
    
    def __post_init__(self) -> None:
        """Validate operand types."""
        if self.left.type_ is None or self.right.type_ is None:
            raise ValueError("IRBinOp operands must have types")


@dataclass
class IRUnaryOp(IRNode):
    """
    Unary operation.
    
    Attributes:
        op: Operator enum
        operand: Operand IR node
        type_: Result type of operation
    """
    op: UnaryOp
    operand: IRNode
    type_: Union[Type, ArrayType]
    
    def __post_init__(self) -> None:
        """Validate operand type."""
        if self.operand.type_ is None:
            raise ValueError("IRUnaryOp operand must have type")


@dataclass
class IRCompare(IRNode):
    """
    Comparison operation.
    
    Attributes:
        op: Comparison operator
        left: Left operand
        right: Right operand
        type_: Result type (always bool)
    """
    op: CmpOp
    left: IRNode
    right: IRNode
    type_: Type
    
    def __post_init__(self) -> None:
        """Validate comparison operands."""
        if self.left.type_ is None or self.right.type_ is None:
            raise ValueError("IRCompare operands must have types")


@dataclass
class IRIfExp(IRNode):
    """
    Ternary conditional expression (a if cond else b).
    
    Attributes:
        condition: Boolean condition
        true_value: Value if condition is true
        false_value: Value if condition is false
        type_: Result type
    """
    condition: IRNode
    true_value: IRNode
    false_value: IRNode
    type_: Type
    
    def __post_init__(self) -> None:
        """Validate conditional expression."""
        if self.condition.type_ is None:
            raise ValueError("IRIfExp condition must have type")
        if self.true_value.type_ is None or self.false_value.type_ is None:
            raise ValueError("IRIfExp branches must have types")


@dataclass
class IRCall(IRNode):
    """
    Function call.
    
    Attributes:
        func: Function to call (IRNode for first-class functions)
        args: Positional argument IR nodes
        kwargs: Keyword arguments (currently unsupported)
        type_: Return type of the function
    """
    func: IRNode
    args: List[IRNode]
    type_: Union[Type, ArrayType]
    kwargs: Dict[str, IRNode] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate call arguments."""
        if self.kwargs:
            raise NotImplementedError("Keyword arguments not yet supported")
        
        for arg in self.args:
            if arg.type_ is None:
                raise ValueError("Call arguments must have types")


@dataclass
class IRArrayLoad(IRNode):
    """
    Array element load.
    
    Attributes:
        array: Array IR node
        index: Index IR node (must be integer type)
        type_: Element type from array
    """
    array: IRNode
    index: IRNode
    type_: Union[Type, ArrayType]
    
    def __post_init__(self) -> None:
        """Validate array load."""
        if not isinstance(self.array.type_, ArrayType):
            raise ValueError(f"Array load requires array type, got {self.array.type_}")
        
        if self.index.type_ is None:
            raise ValueError("Array index must have type")


@dataclass
class IRArrayStore(IRNode):
    """
    Array element store.
    
    Attributes:
        array: Array IR node
        index: Index IR node
        value: Value to store
    """
    array: IRNode
    index: IRNode
    value: IRNode
    
    def __post_init__(self) -> None:
        """Validate array store."""
        if not isinstance(self.array.type_, ArrayType):
            raise ValueError(f"Array store requires array type, got {self.array.type_}")
        
        if self.index.type_ is None or self.value.type_ is None:
            raise ValueError("Array index and value must have types")


@dataclass
class IRAssign(IRNode):
    """
    Assignment statement.
    
    Attributes:
        target: Variable name
        value: Expression to assign
    """
    target: str
    value: IRNode
    
    def __post_init__(self) -> None:
        """Validate assignment."""
        if not self.target or not self.target.isidentifier():
            raise ValueError(f"Invalid assignment target: {self.target}")
        
        if self.value.type_ is None:
            raise ValueError("Assignment value must have type")


@dataclass
class IRReturn(IRNode):
    """
    Return statement.
    
    Attributes:
        value: Optional return value
    """
    value: Optional[IRNode] = None
    
    def __post_init__(self) -> None:
        """Validate return statement."""
        if self.value is not None and self.value.type_ is None:
            raise ValueError("Return value must have type")


@dataclass
class IRIf(IRNode):
    """
    Conditional statement.
    
    Attributes:
        condition: Boolean condition
        then_body: Statements to execute if true
        else_body: Statements to execute if false
    """
    condition: IRNode
    then_body: List[IRNode]
    else_body: List[IRNode]
    
    def __post_init__(self) -> None:
        """Validate conditional."""
        if self.condition.type_ is None:
            raise ValueError("Condition must have type")


@dataclass
class IRWhile(IRNode):
    """
    While loop.
    
    Attributes:
        condition: Loop condition
        body: Loop body statements
    """
    condition: IRNode
    body: List[IRNode]
    
    def __post_init__(self) -> None:
        """Validate while loop."""
        if self.condition.type_ is None:
            raise ValueError("Loop condition must have type")


@dataclass
class IRFor(IRNode):
    """
    For loop (range-based).
    
    Attributes:
        target: Loop variable name
        start: Start value
        stop: Stop value (exclusive)
        step: Step value
        body: Loop body statements
    """
    target: str
    start: IRNode
    stop: IRNode
    step: IRNode
    body: List[IRNode]
    
    def __post_init__(self) -> None:
        """Validate for loop."""
        if not self.target or not self.target.isidentifier():
            raise ValueError(f"Invalid loop variable: {self.target}")
        
        if any(node.type_ is None for node in [self.start, self.stop, self.step]):
            raise ValueError("Loop range values must have types")


@dataclass
class IRBlock(IRNode):
    """
    Statement block.
    
    Attributes:
        statements: List of statement IR nodes
    """
    statements: List[IRNode]


@dataclass
class IRFunction(IRNode):
    """
    Function definition.
    
    Attributes:
        name: Function name
        params: List of (name, type) tuples
        return_type: Function return type
        body: Function body statements
    """
    name: str
    params: List[tuple[str, Union[Type, ArrayType]]]
    return_type: Optional[Union[Type, ArrayType]]
    body: List[IRNode]
    
    def __post_init__(self) -> None:
        """Validate function definition."""
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Invalid function name: {self.name}")
        
        seen_params = set()
        for param_name, param_type in self.params:
            if not param_name or not param_name.isidentifier():
                raise ValueError(f"Invalid parameter name: {param_name}")
            
            if param_name in seen_params:
                raise ValueError(f"Duplicate parameter name: {param_name}")
            seen_params.add(param_name)
            
            if param_type is None:
                raise ValueError(f"Parameter {param_name} must have type")


def ast_binop_to_ir(op: ast.operator) -> BinOp:
    """Convert AST binary operator to IR BinOp."""
    mapping = {
        ast.Add: BinOp.ADD,
        ast.Sub: BinOp.SUB,
        ast.Mult: BinOp.MUL,
        ast.Div: BinOp.DIV,
        ast.FloorDiv: BinOp.FLOORDIV,
        ast.Mod: BinOp.MOD,
        ast.Pow: BinOp.POW,
        ast.LShift: BinOp.LSHIFT,
        ast.RShift: BinOp.RSHIFT,
        ast.BitOr: BinOp.BITOR,
        ast.BitXor: BinOp.BITXOR,
        ast.BitAnd: BinOp.BITAND,
    }
    
    op_ir = mapping.get(type(op))
    if op_ir is None:
        raise ValueError(f"Unsupported binary operator: {op}")
    return op_ir


def ast_unaryop_to_ir(op: ast.unaryop) -> UnaryOp:
    """Convert AST unary operator to IR UnaryOp."""
    mapping = {
        ast.Invert: UnaryOp.INVERT,
        ast.Not: UnaryOp.NOT,
        ast.UAdd: UnaryOp.UADD,
        ast.USub: UnaryOp.USUB,
    }
    
    op_ir = mapping.get(type(op))
    if op_ir is None:
        raise ValueError(f"Unsupported unary operator: {op}")
    return op_ir


def ast_cmpop_to_ir(op: ast.cmpop) -> CmpOp:
    """Convert AST comparison operator to IR CmpOp."""
    mapping = {
        ast.Eq: CmpOp.EQ,
        ast.NotEq: CmpOp.NEQ,
        ast.Lt: CmpOp.LT,
        ast.LtE: CmpOp.LTE,
        ast.Gt: CmpOp.GT,
        ast.GtE: CmpOp.GTE,
    }
    
    op_ir = mapping.get(type(op))
    if op_ir is None:
        raise ValueError(f"Unsupported comparison operator: {op}")
    return op_ir
