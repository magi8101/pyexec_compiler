"""
LLVM IR code generation from typed IR.

Converts pyexec IR to LLVM IR for native code generation.
"""

from typing import Dict, List, Optional, Union
from llvmlite import ir

from .types import Type, ArrayType
from .ir import (
    IRNode, IRFunction, IRConstant, IRVariable, IRBinOp, IRUnaryOp,
    IRCompare, IRIfExp, IRAssign, IRReturn, IRIf, IRWhile, IRFor, IRBlock,
    BinOp, UnaryOp, CmpOp,
)
from .errors import CodeGenerationError


class LLVMCodeGenerator:
    """
    Generate LLVM IR from pyexec IR.
    
    Implements straightforward code generation without complex optimizations
    (LLVM optimization passes handle that).
    """
    
    def __init__(self) -> None:
        """Initialize code generator."""
        self._module: Optional[ir.Module] = None
        self._builder: Optional[ir.IRBuilder] = None
        self._func: Optional[ir.Function] = None
        self._variables: Dict[str, ir.AllocaInstr] = {}
        self._current_block: Optional[ir.Block] = None
    
    def generate_module(self, functions: List[IRFunction]) -> ir.Module:
        """
        Generate LLVM module from IR functions.
        
        Args:
            functions: List of IR functions
            
        Returns:
            LLVM module
        """
        self._module = ir.Module(name="pyexec_module")
        
        for func in functions:
            self._generate_function(func)
        
        return self._module
    
    def _generate_function(self, ir_func: IRFunction) -> None:
        """
        Generate LLVM function from IR function.
        
        Args:
            ir_func: IR function to compile
        """
        assert self._module is not None
        
        param_types = [
            self._type_to_llvm(ptype)
            for _, ptype in ir_func.params
        ]
        
        return_type = ir.VoidType() if ir_func.return_type is None else \
                     self._type_to_llvm(ir_func.return_type)
        
        func_type = ir.FunctionType(return_type, param_types)
        
        self._func = ir.Function(self._module, func_type, name=ir_func.name)
        
        for llvm_arg, (param_name, _) in zip(self._func.args, ir_func.params):
            llvm_arg.name = param_name
        
        entry_block = self._func.append_basic_block(name="entry")
        self._builder = ir.IRBuilder(entry_block)
        self._current_block = entry_block
        
        self._variables = {}
        for llvm_arg, (param_name, param_type) in zip(self._func.args, ir_func.params):
            alloca = self._builder.alloca(self._type_to_llvm(param_type), name=param_name)
            self._builder.store(llvm_arg, alloca)
            self._variables[param_name] = alloca
        
        for stmt in ir_func.body:
            self._generate_statement(stmt)
        
        # Check if the actual builder block is terminated (handles all control flow)
        if self._builder is not None and self._builder.block is not None:
            if not self._builder.block.is_terminated:
                if isinstance(return_type, ir.VoidType):
                    self._builder.ret_void()
                else:
                    raise CodeGenerationError(
                        f"Function {ir_func.name} missing return statement",
                        ir_func.source,
                    )
    
    def _generate_statement(self, stmt: IRNode) -> None:
        """
        Generate code for statement.
        
        Args:
            stmt: IR statement node
        """
        if isinstance(stmt, IRAssign):
            self._generate_assign(stmt)
        elif isinstance(stmt, IRReturn):
            self._generate_return(stmt)
        elif isinstance(stmt, IRIf):
            self._generate_if(stmt)
        elif isinstance(stmt, IRWhile):
            self._generate_while(stmt)
        elif isinstance(stmt, IRFor):
            self._generate_for(stmt)
        else:
            self._generate_expression(stmt)
    
    def _generate_assign(self, assign: IRAssign) -> None:
        """Generate assignment."""
        assert self._builder is not None
        
        value = self._generate_expression(assign.value)
        
        if assign.target not in self._variables:
            assert assign.value.type_ is not None
            alloca = self._builder.alloca(
                self._type_to_llvm(assign.value.type_),
                name=assign.target,
            )
            self._variables[assign.target] = alloca
        
        self._builder.store(value, self._variables[assign.target])
    
    def _generate_return(self, ret: IRReturn) -> None:
        """Generate return statement."""
        assert self._builder is not None
        
        if ret.value is None:
            self._builder.ret_void()
        else:
            value = self._generate_expression(ret.value)
            self._builder.ret(value)
    
    def _generate_if(self, if_stmt: IRIf) -> None:
        """Generate if statement."""
        assert self._builder is not None
        assert self._func is not None
        
        condition = self._generate_expression(if_stmt.condition)
        
        then_block = self._func.append_basic_block(name="if.then")
        else_block = self._func.append_basic_block(name="if.else")
        merge_block = self._func.append_basic_block(name="if.merge")
        
        self._builder.cbranch(condition, then_block, else_block)
        
        # Generate then branch
        self._builder.position_at_start(then_block)
        self._current_block = then_block
        for stmt in if_stmt.then_body:
            self._generate_statement(stmt)
        
        if not self._builder.block.is_terminated:
            self._builder.branch(merge_block)
        
        # Generate else branch
        self._builder.position_at_start(else_block)
        self._current_block = else_block
        for stmt in if_stmt.else_body:
            self._generate_statement(stmt)
        
        if not self._builder.block.is_terminated:
            self._builder.branch(merge_block)
        
        # Position at merge block
        self._builder.position_at_start(merge_block)
        self._current_block = merge_block
    
    def _generate_while(self, while_stmt: IRWhile) -> None:
        """Generate while loop."""
        assert self._builder is not None
        assert self._func is not None
        
        cond_block = self._func.append_basic_block(name="while.cond")
        body_block = self._func.append_basic_block(name="while.body")
        exit_block = self._func.append_basic_block(name="while.exit")
        
        self._builder.branch(cond_block)
        
        self._builder.position_at_start(cond_block)
        self._current_block = cond_block
        condition = self._generate_expression(while_stmt.condition)
        self._builder.cbranch(condition, body_block, exit_block)
        
        self._builder.position_at_start(body_block)
        self._current_block = body_block
        for stmt in while_stmt.body:
            self._generate_statement(stmt)
        # Check current block (may have changed due to nested control flow)
        if self._builder.block is not None and not self._builder.block.is_terminated:
            self._builder.branch(cond_block)
        
        self._builder.position_at_start(exit_block)
        self._current_block = exit_block
    
    def _generate_for(self, for_stmt: IRFor) -> None:
        """Generate for loop."""
        assert self._builder is not None
        assert self._func is not None
        
        start_val = self._generate_expression(for_stmt.start)
        stop_val = self._generate_expression(for_stmt.stop)
        step_val = self._generate_expression(for_stmt.step)
        
        if for_stmt.target not in self._variables:
            assert for_stmt.start.type_ is not None
            alloca = self._builder.alloca(
                self._type_to_llvm(for_stmt.start.type_),
                name=for_stmt.target,
            )
            self._variables[for_stmt.target] = alloca
        
        self._builder.store(start_val, self._variables[for_stmt.target])
        
        cond_block = self._func.append_basic_block(name="for.cond")
        body_block = self._func.append_basic_block(name="for.body")
        incr_block = self._func.append_basic_block(name="for.incr")
        exit_block = self._func.append_basic_block(name="for.exit")
        
        self._builder.branch(cond_block)
        
        self._builder.position_at_start(cond_block)
        self._current_block = cond_block
        current_val = self._builder.load(self._variables[for_stmt.target])
        condition = self._builder.icmp_signed('<', current_val, stop_val)
        self._builder.cbranch(condition, body_block, exit_block)
        
        self._builder.position_at_start(body_block)
        self._current_block = body_block
        for stmt in for_stmt.body:
            self._generate_statement(stmt)
        # Check current block (may have changed due to nested control flow)
        if self._builder.block is not None and not self._builder.block.is_terminated:
            self._builder.branch(incr_block)
        
        self._builder.position_at_start(incr_block)
        self._current_block = incr_block
        current_val = self._builder.load(self._variables[for_stmt.target])
        next_val = self._builder.add(current_val, step_val)
        self._builder.store(next_val, self._variables[for_stmt.target])
        self._builder.branch(cond_block)
        
        self._builder.position_at_start(exit_block)
        self._current_block = exit_block
    
    def _generate_expression(self, expr: IRNode) -> ir.Value:
        """
        Generate code for expression.
        
        Args:
            expr: IR expression node
            
        Returns:
            LLVM value
        """
        if isinstance(expr, IRConstant):
            return self._generate_constant(expr)
        elif isinstance(expr, IRVariable):
            return self._generate_variable(expr)
        elif isinstance(expr, IRBinOp):
            return self._generate_binop(expr)
        elif isinstance(expr, IRUnaryOp):
            return self._generate_unaryop(expr)
        elif isinstance(expr, IRCompare):
            return self._generate_compare(expr)
        elif isinstance(expr, IRIfExp):
            return self._generate_ifexp(expr)
        else:
            raise CodeGenerationError(
                f"Unsupported expression type: {type(expr).__name__}",
                expr.source,
            )
    
    def _generate_constant(self, const: IRConstant) -> ir.Value:
        """Generate constant value."""
        assert const.type_ is not None
        assert self._module is not None
        assert self._builder is not None
        
        from .types import TypeKind
        
        # Handle string constants specially
        if const.type_.kind == TypeKind.STRING:
            # Store string in global memory and return pointer
            string_bytes = const.value.encode('utf-8')
            string_len = len(string_bytes)
            
            # Create global string: [i32 length][bytes data]
            i8_array_type = ir.ArrayType(ir.IntType(8), string_len)
            i32_type = ir.IntType(32)
            
            # Define global string struct
            global_string = ir.GlobalVariable(
                self._module,
                ir.LiteralStructType([i32_type, i8_array_type]),
                name=f"str_{abs(hash(const.value)) % 1000000}"
            )
            global_string.linkage = "internal"
            global_string.global_constant = True
            global_string.initializer = ir.Constant(
                ir.LiteralStructType([i32_type, i8_array_type]),
                [
                    ir.Constant(i32_type, string_len),
                    ir.Constant(i8_array_type, bytearray(string_bytes))
                ]
            )
            
            # Return pointer to the struct (i32)
            return self._builder.ptrtoint(global_string, ir.IntType(32))
        
        llvm_type = self._type_to_llvm(const.type_)
        
        if isinstance(llvm_type, ir.IntType):
            if isinstance(const.value, bool):
                return ir.Constant(llvm_type, int(const.value))
            return ir.Constant(llvm_type, const.value)
        elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
            return ir.Constant(llvm_type, float(const.value))
        else:
            raise CodeGenerationError(
                f"Unsupported constant type: {llvm_type}",
                const.source,
            )
    
    def _generate_variable(self, var: IRVariable) -> ir.Value:
        """Generate variable load."""
        assert self._builder is not None
        
        if var.name not in self._variables:
            raise CodeGenerationError(
                f"Undefined variable: {var.name}",
                var.source,
            )
        
        return self._builder.load(self._variables[var.name], name=var.name)
    
    def _generate_binop(self, binop: IRBinOp) -> ir.Value:
        """Generate binary operation."""
        assert self._builder is not None
        
        left = self._generate_expression(binop.left)
        right = self._generate_expression(binop.right)
        
        is_float = isinstance(left.type, (ir.FloatType, ir.DoubleType))
        
        if binop.op == BinOp.ADD:
            return self._builder.fadd(left, right) if is_float else self._builder.add(left, right)
        elif binop.op == BinOp.SUB:
            return self._builder.fsub(left, right) if is_float else self._builder.sub(left, right)
        elif binop.op == BinOp.MUL:
            return self._builder.fmul(left, right) if is_float else self._builder.mul(left, right)
        elif binop.op == BinOp.DIV:
            return self._builder.fdiv(left, right) if is_float else self._builder.sdiv(left, right)
        elif binop.op == BinOp.FLOORDIV:
            if is_float:
                div = self._builder.fdiv(left, right)
                return self._builder.call(
                    self._get_or_declare_floor(),
                    [div],
                )
            return self._builder.sdiv(left, right)
        elif binop.op == BinOp.MOD:
            return self._builder.frem(left, right) if is_float else self._builder.srem(left, right)
        elif binop.op == BinOp.POW:
            return self._builder.call(
                self._get_or_declare_pow(is_float),
                [left, right],
            )
        elif binop.op == BinOp.LSHIFT:
            return self._builder.shl(left, right)
        elif binop.op == BinOp.RSHIFT:
            return self._builder.ashr(left, right)
        elif binop.op == BinOp.BITOR:
            return self._builder.or_(left, right)
        elif binop.op == BinOp.BITXOR:
            return self._builder.xor(left, right)
        elif binop.op == BinOp.BITAND:
            return self._builder.and_(left, right)
        elif binop.op == BinOp.AND:
            return self._builder.and_(left, right)
        elif binop.op == BinOp.OR:
            return self._builder.or_(left, right)
        else:
            raise CodeGenerationError(
                f"Unsupported binary operator: {binop.op}",
                binop.source,
            )
    
    def _generate_unaryop(self, unaryop: IRUnaryOp) -> ir.Value:
        """Generate unary operation."""
        assert self._builder is not None
        
        operand = self._generate_expression(unaryop.operand)
        is_float = isinstance(operand.type, (ir.FloatType, ir.DoubleType))
        
        if unaryop.op == UnaryOp.USUB:
            if is_float:
                return self._builder.fsub(
                    ir.Constant(operand.type, 0.0),
                    operand,
                )
            return self._builder.sub(
                ir.Constant(operand.type, 0),
                operand,
            )
        elif unaryop.op == UnaryOp.UADD:
            return operand
        elif unaryop.op == UnaryOp.NOT:
            return self._builder.not_(operand)
        elif unaryop.op == UnaryOp.INVERT:
            return self._builder.not_(operand)
        else:
            raise CodeGenerationError(
                f"Unsupported unary operator: {unaryop.op}",
                unaryop.source,
            )
    
    def _generate_compare(self, compare: IRCompare) -> ir.Value:
        """Generate comparison operation."""
        assert self._builder is not None
        
        left = self._generate_expression(compare.left)
        right = self._generate_expression(compare.right)
        
        is_float = isinstance(left.type, (ir.FloatType, ir.DoubleType))
        
        if is_float:
            op_map = {
                CmpOp.EQ: 'oeq',
                CmpOp.NEQ: 'one',
                CmpOp.LT: 'olt',
                CmpOp.LTE: 'ole',
                CmpOp.GT: 'ogt',
                CmpOp.GTE: 'oge',
            }
            return self._builder.fcmp_ordered(op_map[compare.op], left, right)
        else:
            op_map = {
                CmpOp.EQ: '==',
                CmpOp.NEQ: '!=',
                CmpOp.LT: '<',
                CmpOp.LTE: '<=',
                CmpOp.GT: '>',
                CmpOp.GTE: '>=',
            }
            return self._builder.icmp_signed(op_map[compare.op], left, right)
    
    def _generate_ifexp(self, ifexp: IRIfExp) -> ir.Value:
        """Generate ternary conditional expression."""
        assert self._builder is not None
        assert self._function is not None
        
        condition = self._generate_expression(ifexp.condition)
        
        true_block = self._function.append_basic_block()
        false_block = self._function.append_basic_block()
        merge_block = self._function.append_basic_block()
        
        self._builder.cbranch(condition, true_block, false_block)
        
        self._builder.position_at_start(true_block)
        true_val = self._generate_expression(ifexp.true_value)
        self._builder.branch(merge_block)
        true_block = self._builder.block
        
        self._builder.position_at_start(false_block)
        false_val = self._generate_expression(ifexp.false_value)
        self._builder.branch(merge_block)
        false_block = self._builder.block
        
        self._builder.position_at_start(merge_block)
        phi = self._builder.phi(true_val.type)
        phi.add_incoming(true_val, true_block)
        phi.add_incoming(false_val, false_block)
        
        return phi
    
    def _type_to_llvm(self, type_: Union[Type, ArrayType]) -> ir.Type:
        """Convert pyexec type to LLVM type."""
        if isinstance(type_, Type):
            return type_.to_llvm()
        elif isinstance(type_, ArrayType):
            return type_.to_llvm()
        else:
            raise CodeGenerationError(f"Unknown type: {type_}")
    
    def _get_or_declare_pow(self, is_float: bool) -> ir.Function:
        """Get or declare pow intrinsic."""
        assert self._module is not None
        
        func_name = "llvm.pow.f64" if is_float else "llvm.powi.f64.i32"
        
        if func_name in self._module.globals:
            return self._module.get_global(func_name)
        
        if is_float:
            func_type = ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()])
        else:
            func_type = ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.IntType(32)])
        
        return ir.Function(self._module, func_type, name=func_name)
    
    def _get_or_declare_floor(self) -> ir.Function:
        """Get or declare floor intrinsic."""
        assert self._module is not None
        
        func_name = "llvm.floor.f64"
        
        if func_name in self._module.globals:
            return self._module.get_global(func_name)
        
        func_type = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        return ir.Function(self._module, func_type, name=func_name)
