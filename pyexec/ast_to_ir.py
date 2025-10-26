"""
Convert Python AST to typed IR.

Implements type inference and IR construction from annotated Python AST.
"""

from typing import Optional, List, Dict, Union, Any
import ast

from .types import Type, ArrayType, TypeRegistry, TypeKind
from .ir import (
    IRNode, IRConstant, IRVariable, IRBinOp, IRUnaryOp, IRCompare, IRIfExp,
    IRCall, IRAssign, IRReturn, IRIf, IRWhile, IRFor, IRBlock, IRFunction,
    BinOp, UnaryOp, CmpOp,
    ast_binop_to_ir, ast_unaryop_to_ir, ast_cmpop_to_ir,
)
from .errors import TypeInferenceError, CompilationError


class Scope:
    """
    Lexical scope for variable tracking.
    
    Attributes:
        parent: Parent scope (None for global)
        variables: Mapping of variable names to types
    """
    
    def __init__(self, parent: Optional['Scope'] = None) -> None:
        """
        Initialize scope.
        
        Args:
            parent: Parent scope for lexical scoping
        """
        self.parent = parent
        self.variables: Dict[str, Union[Type, ArrayType]] = {}
    
    def define(self, name: str, type_: Union[Type, ArrayType]) -> None:
        """
        Define variable in current scope.
        
        Args:
            name: Variable name
            type_: Variable type
        """
        self.variables[name] = type_
    
    def lookup(self, name: str) -> Optional[Union[Type, ArrayType]]:
        """
        Lookup variable type in scope chain.
        
        Args:
            name: Variable name
            
        Returns:
            Variable type or None if not found
        """
        if name in self.variables:
            return self.variables[name]
        
        if self.parent is not None:
            return self.parent.lookup(name)
        
        return None


class ASTToIRConverter:
    """
    Convert Python AST to typed IR.
    
    Performs type checking and inference from annotations.
    """
    
    def __init__(self, type_registry: TypeRegistry) -> None:
        """
        Initialize converter.
        
        Args:
            type_registry: Type registry for type lookups
        """
        self._type_registry = type_registry
        self._scope: Optional[Scope] = None
    
    def convert_function(self, func_def: ast.FunctionDef) -> IRFunction:
        """
        Convert function definition to IR.
        
        Args:
            func_def: Python AST function definition
            
        Returns:
            Typed IR function
            
        Raises:
            CompilationError: If function has no type annotations
        """
        self._scope = Scope()
        
        params: List[tuple[str, Union[Type, ArrayType]]] = []
        for arg in func_def.args.args:
            if arg.annotation is None:
                raise CompilationError(
                    f"Parameter '{arg.arg}' missing type annotation",
                    arg,
                )
            
            param_type = self._resolve_type_annotation(arg.annotation)
            params.append((arg.arg, param_type))
            self._scope.define(arg.arg, param_type)
        
        return_type: Optional[Union[Type, ArrayType]] = None
        if func_def.returns is not None:
            return_type = self._resolve_type_annotation(func_def.returns)
        
        body: List[IRNode] = []
        for stmt in func_def.body:
            ir_stmt = self._convert_statement(stmt)
            if ir_stmt is not None:
                body.append(ir_stmt)
        
        return IRFunction(
            name=func_def.name,
            params=params,
            return_type=return_type,
            body=body,
            source=func_def,
        )
    
    def _convert_statement(self, stmt: ast.stmt) -> Optional[IRNode]:
        """
        Convert statement AST node to IR.
        
        Args:
            stmt: AST statement
            
        Returns:
            IR node or None for pass statements
        """
        if isinstance(stmt, ast.Assign):
            return self._convert_assign(stmt)
        elif isinstance(stmt, ast.AnnAssign):
            return self._convert_ann_assign(stmt)
        elif isinstance(stmt, ast.Return):
            return self._convert_return(stmt)
        elif isinstance(stmt, ast.If):
            return self._convert_if(stmt)
        elif isinstance(stmt, ast.While):
            return self._convert_while(stmt)
        elif isinstance(stmt, ast.For):
            return self._convert_for(stmt)
        elif isinstance(stmt, ast.Pass):
            return None
        elif isinstance(stmt, ast.Expr):
            return self._convert_expression(stmt.value)
        else:
            raise CompilationError(
                f"Unsupported statement type: {type(stmt).__name__}",
                stmt,
            )
    
    def _convert_assign(self, assign: ast.Assign) -> IRAssign:
        """Convert simple assignment."""
        if len(assign.targets) != 1:
            raise CompilationError(
                "Multiple assignment targets not supported",
                assign,
            )
        
        target = assign.targets[0]
        if not isinstance(target, ast.Name):
            raise CompilationError(
                "Only simple variable assignment supported",
                target,
            )
        
        value = self._convert_expression(assign.value)
        
        if value.type_ is None:
            raise TypeInferenceError(
                "Cannot infer type for assignment value",
                assign.value,
            )
        
        assert self._scope is not None
        self._scope.define(target.id, value.type_)
        
        return IRAssign(
            target=target.id,
            value=value,
            source=assign,
        )
    
    def _convert_ann_assign(self, assign: ast.AnnAssign) -> IRAssign:
        """Convert annotated assignment."""
        if not isinstance(assign.target, ast.Name):
            raise CompilationError(
                "Only simple variable assignment supported",
                assign.target,
            )
        
        target_type = self._resolve_type_annotation(assign.annotation)
        
        assert self._scope is not None
        self._scope.define(assign.target.id, target_type)
        
        if assign.value is None:
            value = IRConstant(
                type_=target_type,
                value=0,
                source=assign,
            )
        else:
            value = self._convert_expression(assign.value)
            if value.type_ != target_type:
                raise TypeInferenceError(
                    f"Type mismatch: expected {target_type}, got {value.type_}",
                    assign.value,
                )
        
        return IRAssign(
            target=assign.target.id,
            value=value,
            source=assign,
        )
    
    def _convert_return(self, ret: ast.Return) -> IRReturn:
        """Convert return statement."""
        if ret.value is None:
            return IRReturn(value=None, source=ret)
        
        value = self._convert_expression(ret.value)
        return IRReturn(value=value, source=ret)
    
    def _convert_if(self, if_stmt: ast.If) -> IRIf:
        """Convert if statement."""
        condition = self._convert_expression(if_stmt.test)
        
        then_body: List[IRNode] = []
        for stmt in if_stmt.body:
            ir_stmt = self._convert_statement(stmt)
            if ir_stmt is not None:
                then_body.append(ir_stmt)
        
        else_body: List[IRNode] = []
        for stmt in if_stmt.orelse:
            ir_stmt = self._convert_statement(stmt)
            if ir_stmt is not None:
                else_body.append(ir_stmt)
        
        return IRIf(
            condition=condition,
            then_body=then_body,
            else_body=else_body,
            source=if_stmt,
        )
    
    def _convert_while(self, while_stmt: ast.While) -> IRWhile:
        """Convert while loop."""
        condition = self._convert_expression(while_stmt.test)
        
        body: List[IRNode] = []
        for stmt in while_stmt.body:
            ir_stmt = self._convert_statement(stmt)
            if ir_stmt is not None:
                body.append(ir_stmt)
        
        return IRWhile(
            condition=condition,
            body=body,
            source=while_stmt,
        )
    
    def _convert_for(self, for_stmt: ast.For) -> IRFor:
        """Convert for loop (range only)."""
        if not isinstance(for_stmt.target, ast.Name):
            raise CompilationError(
                "Only simple loop variables supported",
                for_stmt.target,
            )
        
        if not isinstance(for_stmt.iter, ast.Call):
            raise CompilationError(
                "Only range() loops supported",
                for_stmt.iter,
            )
        
        if not isinstance(for_stmt.iter.func, ast.Name) or for_stmt.iter.func.id != 'range':
            raise CompilationError(
                "Only range() loops supported",
                for_stmt.iter,
            )
        
        args = for_stmt.iter.args
        if len(args) == 1:
            start = IRConstant(type_=self._type_registry.get(TypeKind.INT64), value=0)
            stop = self._convert_expression(args[0])
            step = IRConstant(type_=self._type_registry.get(TypeKind.INT64), value=1)
        elif len(args) == 2:
            start = self._convert_expression(args[0])
            stop = self._convert_expression(args[1])
            step = IRConstant(type_=self._type_registry.get(TypeKind.INT64), value=1)
        elif len(args) == 3:
            start = self._convert_expression(args[0])
            stop = self._convert_expression(args[1])
            step = self._convert_expression(args[2])
        else:
            raise CompilationError(
                "range() requires 1-3 arguments",
                for_stmt.iter,
            )
        
        assert self._scope is not None
        assert start.type_ is not None
        self._scope.define(for_stmt.target.id, start.type_)
        
        body: List[IRNode] = []
        for stmt in for_stmt.body:
            ir_stmt = self._convert_statement(stmt)
            if ir_stmt is not None:
                body.append(ir_stmt)
        
        return IRFor(
            target=for_stmt.target.id,
            start=start,
            stop=stop,
            step=step,
            body=body,
            source=for_stmt,
        )
    
    def _convert_expression(self, expr: ast.expr) -> IRNode:
        """
        Convert expression to IR.
        
        Args:
            expr: AST expression
            
        Returns:
            Typed IR expression
        """
        if isinstance(expr, ast.Constant):
            return self._convert_constant(expr)
        elif isinstance(expr, ast.Name):
            return self._convert_name(expr)
        elif isinstance(expr, ast.BinOp):
            return self._convert_binop(expr)
        elif isinstance(expr, ast.UnaryOp):
            return self._convert_unaryop(expr)
        elif isinstance(expr, ast.Compare):
            return self._convert_compare(expr)
        elif isinstance(expr, ast.BoolOp):
            return self._convert_boolop(expr)
        elif isinstance(expr, ast.IfExp):
            return self._convert_ifexp(expr)
        else:
            raise CompilationError(
                f"Unsupported expression type: {type(expr).__name__}",
                expr,
            )
    
    def _convert_constant(self, const: ast.Constant) -> IRConstant:
        """Convert constant literal."""
        value = const.value
        
        if isinstance(value, bool):
            type_ = self._type_registry.get(TypeKind.BOOL)
        elif isinstance(value, int):
            type_ = self._type_registry.get(TypeKind.INT64)
        elif isinstance(value, float):
            type_ = self._type_registry.get(TypeKind.FLOAT64)
        elif isinstance(value, complex):
            type_ = self._type_registry.get(TypeKind.COMPLEX128)
        elif isinstance(value, str):
            from .types import Type
            type_ = Type(TypeKind.STRING, size=4, alignment=4)
        else:
            raise TypeInferenceError(
                f"Unsupported constant type: {type(value)}",
                const,
            )
        
        return IRConstant(type_=type_, value=value, source=const)
    
    def _convert_name(self, name: ast.Name) -> IRVariable:
        """Convert variable reference."""
        assert self._scope is not None
        var_type = self._scope.lookup(name.id)
        
        if var_type is None:
            raise TypeInferenceError(
                f"Undefined variable: {name.id}",
                name,
            )
        
        return IRVariable(type_=var_type, name=name.id, source=name)
    
    def _convert_binop(self, binop: ast.BinOp) -> IRBinOp:
        """Convert binary operation."""
        left = self._convert_expression(binop.left)
        right = self._convert_expression(binop.right)
        
        if left.type_ is None or right.type_ is None:
            raise TypeInferenceError(
                "Cannot infer types for binary operation",
                binop,
            )
        
        if isinstance(left.type_, ArrayType) or isinstance(right.type_, ArrayType):
            raise TypeInferenceError(
                "Binary operations on arrays not yet supported",
                binop,
            )
        
        result_type = self._type_registry.promote(left.type_, right.type_)
        op = ast_binop_to_ir(binop.op)
        
        return IRBinOp(
            type_=result_type,
            op=op,
            left=left,
            right=right,
            source=binop,
        )
    
    def _convert_unaryop(self, unaryop: ast.UnaryOp) -> IRUnaryOp:
        """Convert unary operation."""
        operand = self._convert_expression(unaryop.operand)
        
        if operand.type_ is None:
            raise TypeInferenceError(
                "Cannot infer type for unary operation",
                unaryop,
            )
        
        op = ast_unaryop_to_ir(unaryop.op)
        
        return IRUnaryOp(
            type_=operand.type_,
            op=op,
            operand=operand,
            source=unaryop,
        )
    
    def _convert_compare(self, compare: ast.Compare) -> IRCompare:
        """Convert comparison operation."""
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise CompilationError(
                "Chained comparisons not supported",
                compare,
            )
        
        left = self._convert_expression(compare.left)
        right = self._convert_expression(compare.comparators[0])
        
        if left.type_ is None or right.type_ is None:
            raise TypeInferenceError(
                "Cannot infer types for comparison",
                compare,
            )
        
        op = ast_cmpop_to_ir(compare.ops[0])
        bool_type = self._type_registry.get(TypeKind.BOOL)
        
        return IRCompare(
            type_=bool_type,
            op=op,
            left=left,
            right=right,
            source=compare,
        )
    
    def _convert_boolop(self, boolop: ast.BoolOp) -> IRNode:
        """Convert boolean operation (and/or)."""
        bool_type = self._type_registry.get(TypeKind.BOOL)
        
        if len(boolop.values) < 2:
            raise CompilationError(
                "BoolOp must have at least 2 values",
                boolop,
            )
        
        result = self._convert_expression(boolop.values[0])
        
        for value in boolop.values[1:]:
            right = self._convert_expression(value)
            
            if isinstance(boolop.op, ast.And):
                op = BinOp.AND
            elif isinstance(boolop.op, ast.Or):
                op = BinOp.OR
            else:
                raise CompilationError(
                    f"Unsupported boolean operator: {type(boolop.op).__name__}",
                    boolop,
                )
            
            result = IRBinOp(
                type_=bool_type,
                op=op,
                left=result,
                right=right,
                source=boolop,
            )
        
        return result
    
    def _convert_ifexp(self, ifexp: ast.IfExp) -> IRNode:
        """Convert ternary conditional expression (a if condition else b)."""
        condition = self._convert_expression(ifexp.test)
        true_value = self._convert_expression(ifexp.body)
        false_value = self._convert_expression(ifexp.orelse)
        
        if true_value.type_ != false_value.type_:
            raise CompilationError(
                f"Type mismatch in ternary: {true_value.type_.kind} vs {false_value.type_.kind}",
                ifexp,
            )
        
        return IRIfExp(
            type_=true_value.type_,
            condition=condition,
            true_value=true_value,
            false_value=false_value,
            source=ifexp,
        )
    
    def _resolve_type_annotation(self, annotation: ast.expr) -> Union[Type, ArrayType]:
        """
        Resolve type annotation to Type or ArrayType.
        
        Args:
            annotation: AST type annotation
            
        Returns:
            Resolved type
            
        Raises:
            CompilationError: If annotation is invalid
        """
        if isinstance(annotation, ast.Name):
            type_name = annotation.id
            type_map = {
                'int8': TypeKind.INT8,
                'int16': TypeKind.INT16,
                'int32': TypeKind.INT32,
                'int64': TypeKind.INT64,
                'uint8': TypeKind.UINT8,
                'uint16': TypeKind.UINT16,
                'uint32': TypeKind.UINT32,
                'uint64': TypeKind.UINT64,
                'float32': TypeKind.FLOAT32,
                'float64': TypeKind.FLOAT64,
                'complex64': TypeKind.COMPLEX64,
                'complex128': TypeKind.COMPLEX128,
                'bool': TypeKind.BOOL,
                'int': TypeKind.INT64,
                'float': TypeKind.FLOAT64,
            }
            
            kind = type_map.get(type_name)
            if kind is None:
                raise CompilationError(
                    f"Unknown type: {type_name}",
                    annotation,
                )
            
            return self._type_registry.get(kind)
        
        raise CompilationError(
            f"Unsupported type annotation: {ast.dump(annotation)}",
            annotation,
        )
