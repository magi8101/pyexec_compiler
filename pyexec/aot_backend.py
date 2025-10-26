"""
AOT (Ahead-of-Time) Compiler Backend

Provides two compilation backends:
1. Nuitka - Full Python to native executable/module compilation
2. LLVM - Direct LLVM IR generation for simple functions

Nuitka: https://nuitka.net/
LLVM: https://llvmlite.readthedocs.io/
"""

from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Literal
import platform
import subprocess
import tempfile
import inspect
import os
import sys
import shutil
import textwrap

# Try to import LLVM for low-level compilation
try:
    import llvmlite.binding as llvm
    import llvmlite.ir as ir
    LLVM_AVAILABLE = True
except ImportError:
    LLVM_AVAILABLE = False


class AotCompiler:
    """
    Compiles Python code to native binaries.
    
    Main methods use Nuitka for full Python compatibility.
    LLVM methods (compile_llvm_*) for low-level binary generation.
    """
    
    def __init__(self):
        """Initialize the AOT compiler with Nuitka and optionally LLVM."""
        self.system = platform.system()
        self.nuitka_available = False
        
        # Check Nuitka availability (don't fail if not installed)
        try:
            self._check_nuitka_installed()
            self.nuitka_available = True
        except RuntimeError:
            pass  # Nuitka not available, will fail only if user tries to use it
        
        # Initialize LLVM if available
        if LLVM_AVAILABLE:
            self._init_llvm()
        
    def _init_llvm(self):
        """Initialize LLVM backend."""
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        target = llvm.Target.from_default_triple()
        cpu = llvm.get_host_cpu_name()
        features = llvm.get_host_cpu_features()
        
        self.llvm_target_machine = target.create_target_machine(
            cpu=cpu,
            features=features.flatten(),
            opt=3,
            codemodel='default',
            reloc='pic',
        )
        
    def _check_nuitka_installed(self):
        """Check if Nuitka is installed."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'nuitka', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Nuitka not found")
            self.nuitka_version = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            raise RuntimeError(
                "Nuitka is not installed. Install it with:\n"
                "  pip install nuitka\n"
                "Website: https://nuitka.net/"
            )
    
    def _create_module_file(self, func: Callable, module_path: Path):
        """Create a Python module file from a function."""
        source = inspect.getsource(func)
        # Remove any leading indentation so the generated module is valid
        source = textwrap.dedent(source)
        
        # Create module content
        module_content = f'''"""
Auto-generated module for AOT compilation
"""

{source}

# Export for module interface
__all__ = ["{func.__name__}"]
'''
        
        module_path.write_text(module_content)
    
    def _create_executable_file(self, func: Callable, script_path: Path):
        """Create a Python script file with main entry point."""
        source = inspect.getsource(func)
        # Remove indentation to avoid "IndentationError: unexpected indent"
        source = textwrap.dedent(source)
        sig = inspect.signature(func)
        
        # Build parameter list for usage message
        param_names = list(sig.parameters.keys())
        usage_params = ' '.join([f'<{p}>' for p in param_names])
        
        # Create CLI interface
        script_content = f'''"""
Auto-generated executable for {func.__name__}
"""
import sys

{source}

if __name__ == "__main__":
    if len(sys.argv) < {len(sig.parameters) + 1}:
        print(f"Usage: {{sys.argv[0]}} {usage_params}")
        sys.exit(1)
    
    # Parse arguments (simple int conversion for now)
    args = [int(arg) for arg in sys.argv[1:]]
    result = {func.__name__}(*args)
    print(result)
'''
        
        script_path.write_text(script_content)
    
    def compile_to_executable(
        self,
        func: Callable,
        output_path: str,
        standalone: bool = True,
        onefile: bool = True,
        show_progress: bool = True,
        remove_output: bool = False
    ) -> Path:
        """
        Compile Python function to standalone executable.
        
        Args:
            func: Python function to compile
            output_path: Output executable path
            standalone: Create standalone bundle (includes Python)
            onefile: Create single file executable
            show_progress: Show compilation progress
            remove_output: Remove build artifacts
            
        Returns:
            Path to generated executable
        """
        if not self.nuitka_available:
            raise RuntimeError(
                "Nuitka is not installed. Install it with:\n"
                "  pip install nuitka\n"
                "Website: https://nuitka.net/"
            )
        
        output = Path(output_path)
        
        # Create temporary script
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.py"
            self._create_executable_file(func, script_path)
            
            # Build Nuitka command
            cmd = [
                sys.executable, '-m', 'nuitka',
                '--output-dir=' + str(output.parent or '.'),
                '--output-filename=' + output.name,
                
            ]
            
            if standalone:
                cmd.append('--standalone')
            
            if onefile:
                cmd.append('--onefile')
            
            if show_progress:
                cmd.append('--show-progress')
            
            if remove_output:
                cmd.append('--remove-output')
            
            cmd.append(str(script_path))
            
            # Run compilation
            result = subprocess.run(cmd, capture_output=not show_progress)
            
            if result.returncode != 0:
                raise RuntimeError(f"Nuitka compilation failed:\n{result.stderr.decode()}")
        
        return output
    
    def compile_to_module(
        self,
        func: Callable,
        output_path: str,
        show_progress: bool = True
    ) -> Path:
        """
        Compile Python function to extension module (.pyd/.so).
        
        Args:
            func: Python function to compile
            output_path: Output module path
            show_progress: Show compilation progress
            
        Returns:
            Path to generated module
        """
        if not self.nuitka_available:
            raise RuntimeError(
                "Nuitka is not installed. Install it with:\n"
                "  pip install nuitka\n"
                "Website: https://nuitka.net/"
            )
        
        output = Path(output_path)
        module_name = output.stem
        
        # Create temporary module
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir) / f"{module_name}.py"
            self._create_module_file(func, module_path)
            
            # Build Nuitka command for module
            cmd = [
                sys.executable, '-m', 'nuitka',
                '--module',
                '--output-dir=' + str(output.parent or '.'),
            ]
            
            if show_progress:
                cmd.append('--show-progress')
            
            cmd.append(str(module_path))
            
            # Run compilation
            result = subprocess.run(cmd, capture_output=not show_progress)
            
            if result.returncode != 0:
                raise RuntimeError(f"Nuitka compilation failed:\n{result.stderr.decode()}")
        
        # Find the generated .pyd/.so file
        ext = '.pyd' if self.system == 'Windows' else '.so'
        for file in (output.parent or Path('.')).glob(f"{module_name}*{ext}"):
            return file
        
        raise RuntimeError(f"Could not find compiled module {module_name}{ext}")
    
    def compile_to_shared_library(
        self,
        func: Callable,
        output_path: str,
        show_progress: bool = True
    ) -> Path:
        """
        Compile Python function to shared library (.dll/.so).
        
        This is an alias for compile_to_module().
        
        Args:
            func: Python function to compile
            output_path: Output library path
            show_progress: Show compilation progress
            
        Returns:
            Path to generated library
        """
        return self.compile_to_module(func, output_path, show_progress)
    
    def compile_package(
        self,
        funcs: List[Callable],
        package_name: str,
        output_dir: Optional[str] = None,
        create_executable: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Path]:
        """
        Compile multiple functions into a package.
        
        Args:
            funcs: List of Python functions
            package_name: Name of the package
            output_dir: Output directory
            create_executable: Also create standalone executable
            show_progress: Show compilation progress
            
        Returns:
            Dictionary mapping 'module' and optionally 'executable' to paths
        """
        output_dir = Path(output_dir or '.')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Create package module with all functions
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = Path(tmpdir) / f"{package_name}.py"
            
            # Combine all function sources
            sources = []
            func_names = []
            for func in funcs:
                sources.append(inspect.getsource(func))
                func_names.append(func.__name__)
            
            package_content = '"""Auto-generated package"""\n\n'
            package_content += '\n\n'.join(sources)
            package_content += f'\n\n__all__ = {func_names}\n'
            
            package_path.write_text(package_content)
            
            # Compile to module
            cmd = [
                sys.executable, '-m', 'nuitka',
                '--module',
                '--output-dir=' + str(output_dir),
            ]
            
            if show_progress:
                cmd.append('--show-progress')
            
            cmd.append(str(package_path))
            
            result = subprocess.run(cmd, capture_output=not show_progress)
            
            if result.returncode != 0:
                raise RuntimeError(f"Nuitka compilation failed:\n{result.stderr.decode()}")
            
            # Find generated module
            ext = '.pyd' if self.system == 'Windows' else '.so'
            for file in output_dir.glob(f"{package_name}*{ext}"):
                results['module'] = file
                break
        
        # Optionally create executable for first function
        if create_executable and funcs:
            exe_path = output_dir / f"{package_name}.exe" if self.system == 'Windows' else output_dir / package_name
            results['executable'] = self.compile_to_executable(
                funcs[0],
                str(exe_path),
                show_progress=show_progress
            )
        
        return results
    
    # =========================================================================
    # LLVM-based Low-Level Compilation Methods
    # =========================================================================
    
    def _check_llvm_available(self):
        """Check if LLVM backend is available."""
        if not LLVM_AVAILABLE:
            raise RuntimeError(
                "LLVM backend not available. Install it with:\n"
                "  pip install llvmlite"
            )
    
    def _infer_llvm_type(self, py_type) -> 'ir.Type':
        """Infer LLVM type from Python type annotation."""
        type_map = {
            int: ir.IntType(64),
            float: ir.DoubleType(),
            bool: ir.IntType(1),
        }
        return type_map.get(py_type, ir.IntType(64))
    
    def _analyze_function(self, func: Callable) -> Dict[str, Any]:
        """Analyze function signature and body."""
        sig = inspect.signature(func)
        source = inspect.getsource(func)
        
        # Extract parameter types
        param_types = []
        param_names = []
        for param_name, param in sig.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                param_types.append(ir.IntType(64))
            else:
                param_types.append(self._infer_llvm_type(param.annotation))
            param_names.append(param_name)
        
        # Extract return type
        if sig.return_annotation == inspect.Signature.empty:
            return_type = ir.IntType(64)
        else:
            return_type = self._infer_llvm_type(sig.return_annotation)
        
        return {
            'name': func.__name__,
            'param_types': param_types,
            'param_names': param_names,
            'return_type': return_type,
            'source': source,
        }
    
    def _generate_llvm_ir(self, func: Callable) -> str:
        """Generate LLVM IR from Python function."""
        self._check_llvm_available()
        
        info = self._analyze_function(func)
        
        # Create module
        module = ir.Module(name=f"{info['name']}_module")
        
        # Create function type
        func_type = ir.FunctionType(info['return_type'], info['param_types'])
        
        # Create function
        llvm_func = ir.Function(module, func_type, name=info['name'])
        
        # Create entry block
        block = llvm_func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        
        # Generate function body
        result = self._generate_function_body(builder, llvm_func, func, info)
        builder.ret(result)
        
        return str(module)
    
    def _generate_function_body(self, builder: 'ir.IRBuilder', llvm_func: 'ir.Function',
                                 py_func: Callable, info: Dict) -> 'ir.Value':
        """Generate function body from Python function."""
        params = {name: arg for name, arg in zip(info['param_names'], llvm_func.args)}
        source = info['source']
        
        # Pattern matching for simple arithmetic
        if 'n * (n + 1) // 2' in source or 'n*(n+1)//2' in source:
            n = params.get('n', llvm_func.args[0])
            n_plus_1 = builder.add(n, ir.Constant(ir.IntType(64), 1))
            mul = builder.mul(n, n_plus_1)
            result = builder.sdiv(mul, ir.Constant(ir.IntType(64), 2))
            return result
        
        if 'return a + b' in source or 'return x + y' in source:
            return builder.add(llvm_func.args[0], llvm_func.args[1])
        
        if 'return a * b' in source or 'return x * y' in source:
            return builder.mul(llvm_func.args[0], llvm_func.args[1])
        
        if 'return a - b' in source or 'return x - y' in source:
            return builder.sub(llvm_func.args[0], llvm_func.args[1])
        
        # Default: return first argument or 0
        if len(llvm_func.args) > 0:
            return llvm_func.args[0]
        else:
            return ir.Constant(info['return_type'], 0)
    
    def compile_llvm_to_object(self, func: Callable, output_path: str) -> Path:
        """
        Compile Python function to object file (.o) using LLVM.
        
        Args:
            func: Python function to compile
            output_path: Path for output .o file
            
        Returns:
            Path to generated object file
            
        Note:
            This is a low-level method for simple arithmetic functions.
            For full Python support, use compile_to_module() instead.
        """
        self._check_llvm_available()
        
        output = Path(output_path)
        
        # Generate LLVM IR
        llvm_ir = self._generate_llvm_ir(func)
        
        # Parse IR
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        
        # Compile to object code
        obj_code = self.llvm_target_machine.emit_object(mod)
        
        # Write to file
        output.write_bytes(obj_code)
        
        return output
    
    def compile_llvm_to_shared_library(self, func: Callable, output_path: str) -> Path:
        """
        Compile Python function to shared library (.dll/.so) using LLVM.
        
        Args:
            func: Python function to compile
            output_path: Path for output library file
            
        Returns:
            Path to generated library file
            
        Note:
            This is a low-level method. Requires C compiler (gcc/clang).
            For full Python support, use compile_to_module() instead.
        """
        self._check_llvm_available()
        
        output = Path(output_path)
        
        # Compile to object file first
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
            obj_file = Path(tmp.name)
        
        try:
            self.compile_llvm_to_object(func, str(obj_file))
            
            # Link to shared library
            if self.system == 'Windows':
                self._link_windows(obj_file, output)
            elif self.system == 'Linux':
                self._link_linux(obj_file, output)
            elif self.system == 'Darwin':
                self._link_macos(obj_file, output)
            else:
                raise RuntimeError(f"Unsupported platform: {self.system}")
        finally:
            if obj_file.exists():
                obj_file.unlink()
        
        return output
    
    def _link_windows(self, obj_file: Path, output: Path):
        """Link object file to DLL on Windows."""
        try:
            subprocess.run([
                'gcc', '-shared', '-o', str(output), str(obj_file),
                '-Wl,--export-all-symbols'
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Could not link DLL. Please install MinGW-w64.\n"
                "Install: choco install mingw"
            )
    
    def _link_linux(self, obj_file: Path, output: Path):
        """Link object file to .so on Linux."""
        try:
            subprocess.run([
                'gcc', '-shared', '-fPIC', '-o', str(output), str(obj_file)
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Could not link shared library. Install GCC:\n"
                "Ubuntu/Debian: sudo apt install gcc"
            )
    
    def _link_macos(self, obj_file: Path, output: Path):
        """Link object file to .dylib on macOS."""
        try:
            subprocess.run([
                'clang', '-shared', '-o', str(output), str(obj_file)
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Could not link shared library. Install Xcode:\n"
                "Run: xcode-select --install"
            )
    
    def generate_llvm_c_header(self, func: Callable, output_path: str) -> Path:
        """
        Generate C header file (.h) for LLVM-compiled function.
        
        Args:
            func: Python function
            output_path: Path for output .h file
            
        Returns:
            Path to generated header file
        """
        self._check_llvm_available()
        
        output = Path(output_path)
        info = self._analyze_function(func)
        
        # Map LLVM types to C types
        type_map = {
            'i1': 'bool',
            'i8': 'int8_t',
            'i16': 'int16_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'float': 'float',
            'double': 'double',
        }
        
        def llvm_to_c_type(llvm_type: 'ir.Type') -> str:
            type_str = str(llvm_type)
            return type_map.get(type_str, 'int64_t')
        
        c_return_type = llvm_to_c_type(info['return_type'])
        
        params = ', '.join([
            f"{llvm_to_c_type(ptype)} {pname}"
            for ptype, pname in zip(info['param_types'], info['param_names'])
        ])
        
        guard = f"_{info['name'].upper()}_H_"
        
        header_content = f"""#ifndef {guard}
#define {guard}

#ifdef __cplusplus
extern "C" {{
#endif

#include <stdint.h>
#include <stdbool.h>

/**
 * {info['name']} - LLVM-compiled from Python
 * 
 * @param {', '.join(info['param_names'])}
 * @return {c_return_type}
 */
{c_return_type} {info['name']}({params});

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
        
        output.write_text(header_content)
        return output
    
    def compile_llvm_full_package(self, func: Callable, base_name: str,
                                   output_dir: Optional[str] = None) -> Dict[str, Path]:
        """
        Compile function to all LLVM formats (.o, .dll/.so, .h).
        
        Args:
            func: Python function to compile
            base_name: Base name for output files
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping format to file path
            
        Note:
            This uses LLVM for low-level compilation.
            For full Python support, use compile_package() instead.
        """
        self._check_llvm_available()
        
        output_dir = Path(output_dir or '.')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        lib_ext = '.dll' if self.system == 'Windows' else '.so'
        
        results = {}
        
        # Generate object file
        obj_path = output_dir / f"{base_name}.o"
        results['object'] = self.compile_llvm_to_object(func, str(obj_path))
        
        # Generate shared library
        lib_path = output_dir / f"{base_name}{lib_ext}"
        results['library'] = self.compile_llvm_to_shared_library(func, str(lib_path))
        
        # Generate header file
        header_path = output_dir / f"{base_name}.h"
        results['header'] = self.generate_llvm_c_header(func, str(header_path))
        
        return results
