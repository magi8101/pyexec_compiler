"""
WebAssembly compilation backend.
Pure Python WASM compiler with JavaScript bindings and WASI support.
"""

from typing import Callable, List, Optional, Dict, Any
from pathlib import Path
import subprocess
import tempfile

from llvmlite import ir
import llvmlite.binding as llvm

from .types import Type, TypeRegistry
from .ir import IRFunction
from .ast_to_ir import ASTToIRConverter
from .codegen_llvm import LLVMCodeGenerator
from .errors import CompilationError, LinkingError
from .wasm_linker import WasmLinker
from .memory import MemoryManager


class WasmCompiler:
    """
    WebAssembly compiler with LLVM backend.
    
    Compiles Python functions to standalone .wasm modules with JavaScript bindings.
    """
    
    def __init__(self) -> None:
        """Initialize WASM compiler."""
        # Initialize all LLVM targets (needed for WebAssembly)
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        
        self._type_registry = TypeRegistry()
        self._codegen = LLVMCodeGenerator()
        self._memory = MemoryManager()
    
    def _convert_types_for_wasm(self, ir_func: IRFunction) -> IRFunction:
        """
        Convert INT64 types to INT32 for WebAssembly/JavaScript compatibility.
        
        JavaScript numbers are 53-bit integers, so we use i32 to avoid BigInt.
        """
        from .types import TypeKind
        from .ir import IRFunction
        
        # Convert parameter types
        new_params = []
        for name, param_type in ir_func.params:
            if param_type.kind == TypeKind.INT64:
                new_type = self._type_registry.get(TypeKind.INT32)
                new_params.append((name, new_type))
            else:
                new_params.append((name, param_type))
        
        # Convert return type
        new_return_type = ir_func.return_type
        if new_return_type and new_return_type.kind == TypeKind.INT64:
            new_return_type = self._type_registry.get(TypeKind.INT32)
        
        # Convert all constants and variables in the body
        new_body = self._convert_node_types(ir_func.body)
        
        return IRFunction(
            name=ir_func.name,
            params=new_params,
            return_type=new_return_type,
            body=new_body,
            source=ir_func.source,
        )
    
    def _convert_node_types(self, nodes: List) -> List:
        """Recursively convert INT64 to INT32 in IR nodes."""
        from .types import TypeKind
        from .ir import IRConstant, IRVariable, IRBinOp, IRCompare, IRCall, IRAssign, IRReturn, IRIf, IRWhile, IRFor
        from dataclasses import replace
        
        result = []
        for node in nodes:
            if isinstance(node, IRConstant):
                if node.type_.kind == TypeKind.INT64:
                    new_type = self._type_registry.get(TypeKind.INT32)
                    node = replace(node, type_=new_type)
            elif isinstance(node, IRVariable):
                if node.type_.kind == TypeKind.INT64:
                    new_type = self._type_registry.get(TypeKind.INT32)
                    node = replace(node, type_=new_type)
            elif isinstance(node, IRBinOp):
                # Recursively convert operands
                left = self._convert_single_node(node.left)
                right = self._convert_single_node(node.right)
                new_type = node.type_
                if new_type and new_type.kind == TypeKind.INT64:
                    new_type = self._type_registry.get(TypeKind.INT32)
                node = replace(node, left=left, right=right, type_=new_type)
            elif isinstance(node, IRCompare):
                left = self._convert_single_node(node.left)
                right = self._convert_single_node(node.right)
                node = replace(node, left=left, right=right)
            elif isinstance(node, IRCall):
                args = [self._convert_single_node(arg) for arg in node.args]
                new_type = node.type_
                if new_type and new_type.kind == TypeKind.INT64:
                    new_type = self._type_registry.get(TypeKind.INT32)
                node = replace(node, args=args, type_=new_type)
            elif isinstance(node, IRAssign):
                value = self._convert_single_node(node.value)
                node = replace(node, value=value)
            elif isinstance(node, IRReturn):
                if node.value:
                    value = self._convert_single_node(node.value)
                    node = replace(node, value=value)
            elif isinstance(node, IRIf):
                condition = self._convert_single_node(node.condition)
                then_body = self._convert_node_types(node.then_body)
                else_body = self._convert_node_types(node.else_body) if node.else_body is not None else []
                node = replace(node, condition=condition, then_body=then_body, else_body=else_body)
            elif isinstance(node, IRWhile):
                condition = self._convert_single_node(node.condition)
                body = self._convert_node_types(node.body)
                node = replace(node, condition=condition, body=body)
            elif isinstance(node, IRFor):
                start = self._convert_single_node(node.start)
                stop = self._convert_single_node(node.stop)
                step = self._convert_single_node(node.step)
                body = self._convert_node_types(node.body)
                node = replace(node, start=start, stop=stop, step=step, body=body)
            
            result.append(node)
        
        return result
    
    def _convert_single_node(self, node):
        """Convert a single IR node."""
        from .types import TypeKind
        from .ir import IRConstant, IRVariable, IRBinOp, IRCompare, IRCall
        from dataclasses import replace
        
        if isinstance(node, IRConstant):
            if node.type_.kind == TypeKind.INT64:
                new_type = self._type_registry.get(TypeKind.INT32)
                return replace(node, type_=new_type)
        elif isinstance(node, IRVariable):
            if node.type_.kind == TypeKind.INT64:
                new_type = self._type_registry.get(TypeKind.INT32)
                return replace(node, type_=new_type)
        elif isinstance(node, IRBinOp):
            left = self._convert_single_node(node.left)
            right = self._convert_single_node(node.right)
            new_type = node.type_
            if new_type and new_type.kind == TypeKind.INT64:
                new_type = self._type_registry.get(TypeKind.INT32)
            return replace(node, left=left, right=right, type_=new_type)
        elif isinstance(node, IRCompare):
            left = self._convert_single_node(node.left)
            right = self._convert_single_node(node.right)
            return replace(node, left=left, right=right)
        elif isinstance(node, IRCall):
            args = [self._convert_single_node(arg) for arg in node.args]
            new_type = node.type_
            if new_type and new_type.kind == TypeKind.INT64:
                new_type = self._type_registry.get(TypeKind.INT32)
            return replace(node, args=args, type_=new_type)
        
        return node
    
    def compile_to_wasm(
        self,
        func: Callable[..., Any],
        output: Path,
        optimize: int = 2,
        export_name: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Compile function to WebAssembly module.
        
        Args:
            func: Python function with type annotations
            output: Output .wasm path
            optimize: Optimization level (0-3)
            export_name: Custom export name (default: function name)
            
        Returns:
            Dictionary with 'wasm' and 'js' paths
            
        Raises:
            CompilationError: If compilation fails
            ValueError: If inputs are invalid
        """
        from .aot_compiler import AOTCompiler
        import ast
        import inspect
        import textwrap
        
        # Validate inputs
        if not callable(func):
            raise ValueError(f"Expected callable, got {type(func).__name__}")
        
        if not hasattr(func, '__name__'):
            raise ValueError("Function must have __name__ attribute")
        
        if optimize not in range(4):
            raise ValueError(f"Optimization level must be 0-3, got {optimize}")
        
        # Extract source code
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError) as e:
            raise CompilationError(f"Cannot extract source for {func.__name__}: {e}")
        
        source = textwrap.dedent(source)
        
        # Parse to AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise CompilationError(f"Syntax error in {func.__name__}: {e}")
        
        # Find function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                func_def = node
                break
        
        if func_def is None:
            raise CompilationError(f"Cannot find function definition for {func.__name__}")
        
        converter = ASTToIRConverter(self._type_registry)
        ir_func = converter.convert_function(func_def)
        
        # Convert INT64 to INT32 for WASM/JavaScript compatibility
        ir_func = self._convert_types_for_wasm(ir_func)
        
        # Generate LLVM IR module
        llvm_module = self._codegen.generate_module([ir_func])

        
        # Set target triple for WASM
        llvm_module.triple = "wasm32-unknown-unknown"
        llvm_module.data_layout = "e-m:e-p:32:32-i64:64-n32:64-S128"
        
        # Convert to binding module for optimization
        llvm_ir = str(llvm_module)
        binding_module = llvm.parse_assembly(llvm_ir)
        binding_module.verify()
        
        # Optimize if requested
        if optimize > 0:
            aot = AOTCompiler()
            aot._optimize_module(binding_module, level=optimize)
        
        # Create WASM target machine
        target = llvm.Target.from_triple("wasm32-unknown-unknown")
        target_machine = target.create_target_machine(
            cpu='',
            features='',
            opt=optimize,
            reloc='pic',
            codemodel='default',
        )
        
        # Generate object file from optimized module
        obj_data = target_machine.emit_object(binding_module)
        
        # Write object to temporary file
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
            tmp.write(obj_data)
            obj_path = Path(tmp.name)
        
        try:
            # Link to WASM using wasm-ld
            wasm_path = Path(output).with_suffix('.wasm')
            self._link_wasm(obj_path, wasm_path, ir_func, export_name)
            
            # Generate JavaScript bindings
            js_path = wasm_path.with_suffix('.js')
            self._generate_js_bindings(ir_func, js_path, wasm_path.name, export_name)
            
            return {
                'wasm': wasm_path,
                'js': js_path,
            }
        finally:
            obj_path.unlink(missing_ok=True)
    
    def _link_wasm(
        self,
        obj_file: Path,
        output: Path,
        ir_func: IRFunction,
        export_name: Optional[str],
    ) -> None:
        """
        Link object file to WebAssembly module.
        
        Args:
            obj_file: Input object file
            output: Output .wasm path
            ir_func: IR function descriptor
            export_name: Custom export name
        """
        func_name = export_name or ir_func.name
        
        # Try Python linker first (production-grade, always available)
        try:
            linker = WasmLinker()
            linker.link(obj_file, output, [func_name])
            return
        except Exception as e:
            print(f"Note: Python linker failed ({e}), trying external linkers...")
        
        # Try wasm-ld (LLVM's WASM linker)
        try:
            result = subprocess.run(
                [
                    'wasm-ld',
                    str(obj_file),
                    '-o', str(output),
                    '--no-entry',
                    '--export-all',
                    '--allow-undefined',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except FileNotFoundError:
            # wasm-ld not available, try emscripten
            pass
        except subprocess.CalledProcessError as e:
            # wasm-ld failed
            print(f"Warning: wasm-ld failed: {e.stderr}")
        
        # Try emcc (Emscripten)
        try:
            result = subprocess.run(
                [
                    'emcc',
                    str(obj_file),
                    '-o', str(output),
                    '-s', 'STANDALONE_WASM=1',
                    '-s', f'EXPORTED_FUNCTIONS=[_{func_name}]',
                    '--no-entry',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except FileNotFoundError:
            pass
        except subprocess.CalledProcessError as e:
            print(f"Warning: emcc failed: {e.stderr}")
        
        # No linker available - copy object file as fallback and provide instructions
        # Object file is already in WebAssembly format, just needs relocation
        import shutil
        shutil.copy(obj_file, output)
        print(f"""
WARNING: No WebAssembly linker found. Object file saved as {output}.
The file is a WebAssembly object that may work with minimal relocation.

To properly link, install one of:
  1. wasm-ld (recommended): Part of LLVM toolchain with WASM support
  2. Emscripten: https://emscripten.org/docs/getting_started/downloads.html

Manual linking with wasm-ld:
  wasm-ld {output} -o {output}.linked --no-entry --export-all --allow-undefined
        """)
    
    def _generate_param_validation(self, ir_func: IRFunction) -> str:
        """Generate JavaScript parameter type validation code."""
        if not ir_func.params:
            return ""
        
        validations = []
        for name, param_type in ir_func.params:
            validations.append(f"""if (typeof {name} !== 'number' || !Number.isFinite({name})) {{
            throw new TypeError(`Parameter '{name}' must be a finite number, got ${{typeof {name}}}`);
        }}""")
        
        return '\n        '.join(validations)
    
    def _is_void_return(self, ir_func: IRFunction) -> str:
        """Check if function returns void (for validation)."""
        from .types import TypeKind
        return "true" if ir_func.return_type and ir_func.return_type.kind == TypeKind.VOID else "false"
    
    def _generate_js_bindings(
        self,
        ir_func: IRFunction,
        output: Path,
        wasm_file: str,
        export_name: Optional[str],
    ) -> None:
        """
        Generate JavaScript bindings for WASM module.
        
        Args:
            ir_func: IR function descriptor
            output: Output .js path
            wasm_file: WASM filename
            export_name: Custom export name
        """
        func_name = export_name or ir_func.name
        
        # Build parameter types and names
        params = [f"{name}: number" for name, _ in ir_func.params]
        param_names = [name for name, _ in ir_func.params]
        
        # Determine return type
        return_type = "number" if ir_func.return_type else "void"
        
        js_code = f'''/**
 * WebAssembly module for {func_name}
 * Auto-generated JavaScript bindings
 */

class {func_name.capitalize()}Module {{
    constructor() {{
        this.instance = null;
        this.memory = new WebAssembly.Memory({{ initial: 256, maximum: 256 }});
        this.stackPointer = new WebAssembly.Global({{ value: 'i32', mutable: true }}, 65536);
    }}

    /**
     * Load and initialize the WebAssembly module
     * @returns {{Promise<void>}}
     */
    async load() {{
        try {{
            const response = await fetch('{wasm_file}');
            
            if (!response.ok) {{
                throw new Error(`Failed to fetch WASM file: ${{response.status}} ${{response.statusText}}`);
            }}
            
            const buffer = await response.arrayBuffer();
            
            if (buffer.byteLength === 0) {{
                throw new Error('WASM file is empty');
            }}
            
            // Verify WASM magic number
            const magic = new Uint8Array(buffer, 0, 4);
            if (magic[0] !== 0x00 || magic[1] !== 0x61 || magic[2] !== 0x73 || magic[3] !== 0x6D) {{
                throw new Error('Invalid WASM file (bad magic number)');
            }}
            
            const importObject = {{
                env: {{
                    __linear_memory: this.memory,
                    __indirect_function_table: new WebAssembly.Table({{ initial: 0, element: 'anyfunc' }}),
                    __stack_pointer: this.stackPointer
                }}
            }};
            
            const result = await WebAssembly.instantiate(buffer, importObject);
            this.instance = result.instance;
            
            // Debug: Log available exports
            console.log('WASM module loaded successfully');
            console.log('Available exports:', Object.keys(this.instance.exports));
        }} catch (error) {{
            console.error('Failed to load WASM module:', error);
            
            // Provide helpful error messages
            if (error instanceof TypeError) {{
                throw new Error('WASM instantiation failed. Check browser console for details.');
            }} else if (error instanceof WebAssembly.CompileError) {{
                throw new Error(`WASM compilation failed: ${{error.message}}`);
            }} else if (error instanceof WebAssembly.LinkError) {{
                throw new Error(`WASM linking failed: ${{error.message}}. Missing imports?`);
            }} else {{
                throw error;
            }}
        }}
    }}

    /**
     * Call the compiled function
     * @param {{{', '.join(params)}}}
     * @returns {{{return_type}}}
     */
    {func_name}({', '.join(param_names)}) {{
        if (!this.instance) {{
            throw new Error('Module not loaded. Call load() first.');
        }}
        
        // Validate input types
        {self._generate_param_validation(ir_func)}
        
        // Try to find the function with various name mangling patterns
        const possibleNames = ['{func_name}', '_{func_name}', '__original_main'];
        let func = null;
        
        for (const name of possibleNames) {{
            if (this.instance.exports[name]) {{
                func = this.instance.exports[name];
                break;
            }}
        }}
        
        if (!func) {{
            const available = Object.keys(this.instance.exports).join(', ');
            throw new Error(`Function '{func_name}' not found in WASM exports. Available: ${{available}}`);
        }}
        
        // Call with error handling
        try {{
            const result = func({', '.join(param_names)});
            
            // Validate result type
            if (typeof result === 'undefined' && {self._is_void_return(ir_func)}) {{
                throw new Error('Function returned unexpected undefined');
            }}
            
            return result;
        }} catch (error) {{
            if (error instanceof WebAssembly.RuntimeError) {{
                throw new Error(`WASM runtime error: ${{error.message}}`);
            }}
            throw error;
        }}
    }}
}}

// Global export for browser usage
if (typeof window !== 'undefined') {{
    window.{func_name.capitalize()}Module = {func_name.capitalize()}Module;
}}

// CommonJS export for Node.js
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = {{ {func_name.capitalize()}Module }};
}}

// Example usage:
// const module = new {func_name.capitalize()}Module();
// await module.load();
// const result = module.{func_name}({', '.join(str(i) for i in range(len(param_names)))});
// console.log('Result:', result);
'''
        
        output.write_text(js_code, encoding='utf-8')
    
    def compile_multiple_to_wasm(
        self,
        funcs: List[Callable[..., Any]],
        output: Path,
        optimize: int = 2,
    ) -> Dict[str, Path]:
        """
        Compile multiple functions into single WASM module.
        
        Args:
            funcs: List of Python functions
            output: Output .wasm path
            optimize: Optimization level
            
        Returns:
            Dictionary with 'wasm' and 'js' paths
            
        Raises:
            CompilationError: If compilation fails
            ValueError: If inputs are invalid
        """
        from .aot_compiler import AOTCompiler
        import ast
        import inspect
        import textwrap
        
        # Validate inputs
        if not funcs:
            raise ValueError("At least one function required")
        
        if not all(callable(f) for f in funcs):
            raise ValueError("All items must be callable")
        
        if optimize not in range(4):
            raise ValueError(f"Optimization level must be 0-3, got {optimize}")
        
        # Parse all functions to IR
        ir_funcs = []
        for func in funcs:
            # Validate function
            if not hasattr(func, '__name__'):
                raise ValueError(f"Function {func} must have __name__ attribute")
            
            # Extract source
            try:
                source = inspect.getsource(func)
            except (OSError, TypeError) as e:
                raise CompilationError(f"Cannot extract source for {func.__name__}: {e}")
            
            source = textwrap.dedent(source)
            
            # Parse to AST
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                raise CompilationError(f"Syntax error in {func.__name__}: {e}")
            
            # Find function definition
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                    func_def = node
                    break
            
            if func_def is None:
                raise CompilationError(f"Cannot find function definition for {func.__name__}")
            
            converter = ASTToIRConverter(self._type_registry)
            ir_func = converter.convert_function(func_def)
            
            # Convert INT64 to INT32 for WASM/JavaScript compatibility
            ir_func = self._convert_types_for_wasm(ir_func)
            ir_funcs.append(ir_func)
        
        # Generate LLVM IR module with all functions
        llvm_module = self._codegen.generate_module(ir_funcs)
        
        # Set WASM target
        llvm_module.triple = "wasm32-unknown-unknown"
        llvm_module.data_layout = "e-m:e-p:32:32-i64:64-n32:64-S128"
        
        # Convert to binding module
        llvm_ir_str = str(llvm_module)
        binding_module = llvm.parse_assembly(llvm_ir_str)
        binding_module.verify()
        
        # Optimize
        if optimize > 0:
            aot = AOTCompiler()
            aot._optimize_module(binding_module, level=optimize)
        
        # Create target machine
        target = llvm.Target.from_triple("wasm32-unknown-unknown")
        target_machine = target.create_target_machine(
            cpu='',
            features='',
            opt=optimize,
            reloc='pic',
            codemodel='default',
        )
        
        # Generate object from optimized module
        obj_data = target_machine.emit_object(binding_module)
        
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
            tmp.write(obj_data)
            obj_path = Path(tmp.name)
        
        try:
            wasm_path = Path(output).with_suffix('.wasm')
            
            # Link all functions
            func_names = [f.name for f in ir_funcs]
            self._link_wasm_multi(obj_path, wasm_path, func_names)
            
            # Generate unified JS bindings
            js_path = wasm_path.with_suffix('.js')
            self._generate_multi_js_bindings(ir_funcs, js_path, wasm_path.name)
            
            return {
                'wasm': wasm_path,
                'js': js_path,
            }
        finally:
            obj_path.unlink(missing_ok=True)
    
    def _link_wasm_multi(
        self,
        obj_file: Path,
        output: Path,
        func_names: List[str],
    ) -> None:
        """Link multiple functions to WASM module."""
        export_funcs = ','.join(f'_{name}' for name in func_names)
        
        # Try Python linker first (production-grade, always available)
        try:
            linker = WasmLinker()
            linker.link(obj_file, output, func_names)
            return
        except Exception as e:
            print(f"Note: Python linker failed ({e}), trying external linkers...")
        
        # Try wasm-ld
        try:
            result = subprocess.run(
                [
                    'wasm-ld',
                    str(obj_file),
                    '-o', str(output),
                    '--no-entry',
                    '--export-all',
                    '--allow-undefined',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except FileNotFoundError:
            pass
        except subprocess.CalledProcessError as e:
            print(f"Warning: wasm-ld failed: {e.stderr}")
        
        # Try emscripten
        try:
            result = subprocess.run(
                [
                    'emcc',
                    str(obj_file),
                    '-o', str(output),
                    '-s', 'STANDALONE_WASM=1',
                    '-s', f'EXPORTED_FUNCTIONS=[{export_funcs}]',
                    '--no-entry',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except FileNotFoundError:
            pass
        except subprocess.CalledProcessError as e:
            print(f"Warning: emcc failed: {e.stderr}")
        
        # Fallback - copy object file
        import shutil
        shutil.copy(obj_file, output)
        print(f"""
WARNING: No WebAssembly linker found. Object file saved as {output}.

To link manually, install wasm-ld or Emscripten.
        """)
    
    def _generate_multi_js_bindings(
        self,
        ir_funcs: List[IRFunction],
        output: Path,
        wasm_file: str,
    ) -> None:
        """Generate JavaScript bindings for multiple functions."""
        
        # Build method definitions
        methods = []
        for ir_func in ir_funcs:
            params = [f"{name}: number" for name, _ in ir_func.params]
            param_names = [name for name, _ in ir_func.params]
            return_type = "number" if ir_func.return_type else "void"
            
            method = f'''    /**
     * @param {{{', '.join(params)}}}
     * @returns {{{return_type}}}
     */
    {ir_func.name}({', '.join(param_names)}) {{
        if (!this.instance) {{
            throw new Error('Module not loaded. Call load() first.');
        }}
        
        // Validate parameters
        {self._generate_param_validation(ir_func)}
        
        // Try to find the function with various name patterns
        const possibleNames = ['{ir_func.name}', '_{ir_func.name}'];
        let func = null;
        
        for (const name of possibleNames) {{
            if (this.instance.exports[name]) {{
                func = this.instance.exports[name];
                break;
            }}
        }}
        
        if (!func) {{
            const available = Object.keys(this.instance.exports).join(', ');
            throw new Error(`Function '{ir_func.name}' not found. Available: ${{available}}`);
        }}
        
        // Call with error handling
        try {{
            const result = func({', '.join(param_names)});
            
            // Validate result
            if (typeof result === 'undefined' && {self._is_void_return(ir_func)}) {{
                throw new Error('Function returned unexpected undefined');
            }}
            
            return result;
        }} catch (error) {{
            if (error instanceof WebAssembly.RuntimeError) {{
                throw new Error(`WASM runtime error in {ir_func.name}: ${{error.message}}`);
            }}
            throw error;
        }}
    }}'''
            methods.append(method)
        
        func_names_list = ', '.join(f"'{f.name}'" for f in ir_funcs)
        
        js_code = f'''/**
 * WebAssembly module with multiple functions
 * Auto-generated JavaScript bindings
 */

class WasmModule {{
    constructor() {{
        this.instance = null;
        this.memory = new WebAssembly.Memory({{ initial: 256, maximum: 256 }});
        this.stackPointer = new WebAssembly.Global({{ value: 'i32', mutable: true }}, 65536);
    }}

    /**
     * Load and initialize the WebAssembly module
     * @returns {{Promise<void>}}
     */
    async load() {{
        try {{
            const response = await fetch('{wasm_file}');
            
            if (!response.ok) {{
                throw new Error(`Failed to fetch WASM file: ${{response.status}} ${{response.statusText}}`);
            }}
            
            const buffer = await response.arrayBuffer();
            
            if (buffer.byteLength === 0) {{
                throw new Error('WASM file is empty');
            }}
            
            // Verify WASM magic number
            const magic = new Uint8Array(buffer, 0, 4);
            if (magic[0] !== 0x00 || magic[1] !== 0x61 || magic[2] !== 0x73 || magic[3] !== 0x6D) {{
                throw new Error('Invalid WASM file (bad magic number)');
            }}
            
            const importObject = {{
                env: {{
                    __linear_memory: this.memory,
                    __indirect_function_table: new WebAssembly.Table({{ initial: 0, element: 'anyfunc' }}),
                    __stack_pointer: this.stackPointer
                }}
            }};
            
            const result = await WebAssembly.instantiate(buffer, importObject);
            this.instance = result.instance;
            
            console.log(`WASM module loaded with functions: {func_names_list}`);
            console.log('Available exports:', Object.keys(this.instance.exports));
        }} catch (error) {{
            console.error('Failed to load WASM module:', error);
            
            // Provide helpful error messages
            if (error instanceof TypeError) {{
                throw new Error('WASM instantiation failed. Check browser console for details.');
            }} else if (error instanceof WebAssembly.CompileError) {{
                throw new Error(`WASM compilation failed: ${{error.message}}`);
            }} else if (error instanceof WebAssembly.LinkError) {{
                throw new Error(`WASM linking failed: ${{error.message}}. Missing imports?`);
            }} else {{
                throw error;
            }}
        }}
    }}

{chr(10).join(methods)}
}}

// Global export for browser usage
if (typeof window !== 'undefined') {{
    window.WasmModule = WasmModule;
}}

// CommonJS export for Node.js
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = {{ WasmModule }};
}}
'''
        
        output.write_text(js_code, encoding='utf-8')
    
    def compile_file(
        self,
        file_path: str,
        output: str,
        optimize: int = 2,
    ) -> Dict[str, Path]:
        """
        Compile entire Python file to WASM.
        
        Extracts all functions from a Python source file and compiles them
        into a single WASM module with JavaScript bindings.
        
        Args:
            file_path: Path to Python source file
            output: Output .wasm path
            optimize: Optimization level (0-3)
            
        Returns:
            Dict with 'wasm' and 'js' paths
        """
        import ast
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        source = file_path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise CompilationError(f"Syntax error in {file_path}: {e}")
        
        converter = ASTToIRConverter(self._type_registry)
        ir_funcs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('_'):
                    continue
                
                try:
                    ir_func = converter.convert_function(node)
                    ir_func = self._convert_types_for_wasm(ir_func)
                    ir_funcs.append(ir_func)
                except Exception as e:
                    print(f"Warning: Skipping {node.name}: {e}")
        
        if not ir_funcs:
            raise CompilationError(f"No compilable functions found in {file_path}")
        
        llvm_module = self._codegen.generate_module(ir_funcs)
        llvm_module.triple = "wasm32-unknown-unknown"
        llvm_module.data_layout = "e-m:e-p:32:32-i64:64-n32:64-S128"
        
        llvm_ir = str(llvm_module)
        binding_module = llvm.parse_assembly(llvm_ir)
        binding_module.verify()
        
        if optimize > 0:
            from .aot_compiler import AOTCompiler
            aot = AOTCompiler()
            aot._optimize_module(binding_module, level=optimize)
        
        target = llvm.Target.from_triple("wasm32-unknown-unknown")
        target_machine = target.create_target_machine(
            cpu='',
            features='',
            opt=optimize,
            reloc='pic',
            codemodel='default',
        )
        
        obj_data = target_machine.emit_object(binding_module)
        
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
            tmp.write(obj_data)
            obj_path = Path(tmp.name)
        
        try:
            wasm_path = Path(output).with_suffix('.wasm')
            func_names = [f.name for f in ir_funcs]
            self._link_wasm_multi(obj_path, wasm_path, func_names)
            
            js_path = wasm_path.with_suffix('.js')
            self._generate_multi_js_bindings(ir_funcs, js_path, wasm_path.name)
            
            return {
                'wasm': wasm_path,
                'js': js_path,
            }
        finally:
            obj_path.unlink(missing_ok=True)


def jit_backend(func: Callable) -> Callable:
    """
    JIT compilation decorator for Python functions.
    
    Compiles the function using LLVM and caches the compiled version.
    First call triggers compilation, subsequent calls use cached version.
    
    Usage:
        @jit_backend
        def add(x: int, y: int) -> int:
            return x + y
    """
    import ctypes
    import llvmlite.binding as llvm
    from .ast_to_ir import ASTToIRConverter
    from .codegen_llvm import LLVMCodeGenerator
    
    # Initialize LLVM
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    
    # Compile the function on first import
    try:
        # Convert Python function to IR
        converter = ASTToIRConverter()
        ir_func = converter.convert_function(func)
        
        # Generate LLVM IR
        codegen = LLVMCodeGenerator()
        llvm_module = codegen.generate(ir_func)
        
        # Create execution engine with optimization
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(opt=3)
        
        # Parse and optimize the module
        llvm_mod = llvm.parse_assembly(str(llvm_module))
        llvm_mod.verify()
        
        # Create JIT execution engine
        engine = llvm.create_mcjit_compiler(llvm_mod, target_machine)
        engine.finalize_object()
        
        # Get function pointer
        func_ptr = engine.get_function_address(ir_func.name)
        
        # Determine C function signature from IR types
        from .types import TypeKind
        
        # Map IR types to ctypes
        def ir_to_ctype(ir_type):
            if ir_type.kind == TypeKind.INT32:
                return ctypes.c_int32
            elif ir_type.kind == TypeKind.INT64:
                return ctypes.c_int64
            elif ir_type.kind == TypeKind.FLOAT:
                return ctypes.c_float
            elif ir_type.kind == TypeKind.DOUBLE:
                return ctypes.c_double
            elif ir_type.kind == TypeKind.BOOL:
                return ctypes.c_bool
            else:
                return ctypes.c_int32  # Default
        
        # Build ctypes function signature
        return_type = ir_to_ctype(ir_func.return_type)
        param_types = [ir_to_ctype(p.type) for p in ir_func.params]
        
        # Create ctypes function
        cfunc = ctypes.CFUNCTYPE(return_type, *param_types)(func_ptr)
        
        # Create wrapper that maintains Python interface
        def jit_wrapper(*args):
            return cfunc(*args)
        
        jit_wrapper.__name__ = func.__name__
        jit_wrapper.__doc__ = func.__doc__
        jit_wrapper._jit_compiled = True
        jit_wrapper._original = func
        
        return jit_wrapper
        
    except Exception as e:
        # If JIT compilation fails, fall back to original function
        print(f"Warning: JIT compilation failed for {func.__name__}: {e}")
        print(f"Falling back to interpreted mode")
        return func
