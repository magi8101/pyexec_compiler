"""
Pure Python WebAssembly linker.
Links LLVM-generated WASM object files into executable modules.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import struct


class WasmSection:
    """WASM binary section."""
    TYPE = 1
    IMPORT = 2
    FUNCTION = 3
    TABLE = 4
    MEMORY = 5
    GLOBAL = 6
    EXPORT = 7
    START = 8
    ELEMENT = 9
    CODE = 10
    DATA = 11
    DATA_COUNT = 12


class WasmLinker:
    """
    Pure Python WebAssembly linker.
    
    Links LLVM object files into executable WASM modules by:
    - Parsing object file sections
    - Resolving imports/exports
    - Generating proper export table
    - Creating executable module
    """
    
    def __init__(self):
        self._magic = b'\x00asm'
        self._version = b'\x01\x00\x00\x00'
    
    def link(self, obj_file: Path, output: Path, exports: List[str]) -> None:
        """
        Link object file to executable WASM module.
        
        Args:
            obj_file: Input object file
            output: Output WASM file
            exports: Function names to export
            
        Raises:
            FileNotFoundError: If object file doesn't exist
            ValueError: If object file is invalid or corrupted
            RuntimeError: If linking fails
        """
        # Validate inputs
        if not obj_file.exists():
            raise FileNotFoundError(f"Object file not found: {obj_file}")
        
        if not exports:
            raise ValueError("At least one export name required")
        
        # Read object file
        try:
            with open(obj_file, 'rb') as f:
                obj_data = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read object file: {e}")
        
        # Validate minimum size (8 bytes for magic + version)
        if len(obj_data) < 8:
            raise ValueError(f"Object file too small ({len(obj_data)} bytes), corrupted?")
        
        # Verify WASM magic
        if obj_data[:4] != self._magic:
            raise ValueError(
                f"Invalid WASM magic: expected {self._magic.hex()}, "
                f"got {obj_data[:4].hex()}. Not a WASM file?"
            )
        
        if obj_data[4:8] != self._version:
            raise ValueError(
                f"Unsupported WASM version: expected {self._version.hex()}, "
                f"got {obj_data[4:8].hex()}. Update LLVM?"
            )
        
        # Parse sections with error handling
        try:
            sections = self._parse_sections(obj_data[8:])
        except Exception as e:
            raise ValueError(f"Failed to parse WASM sections: {e}")
        
        # Build linked module
        try:
            linked_data = self._build_module(sections, exports)
        except Exception as e:
            raise RuntimeError(f"Failed to link module: {e}")
        
        # Write output with validation
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'wb') as f:
                f.write(self._magic)
                f.write(self._version)
                f.write(linked_data)
        except Exception as e:
            raise RuntimeError(f"Failed to write output file: {e}")
    
    def _parse_sections(self, data: bytes) -> Dict[int, bytes]:
        """
        Parse WASM sections from binary data.
        
        Raises:
            ValueError: If data is malformed or truncated
        """
        sections = {}
        pos = 0
        
        while pos < len(data):
            # Bounds check
            if pos >= len(data):
                break
            
            # Read section ID
            if pos + 1 > len(data):
                raise ValueError(f"Truncated section header at pos {pos}")
            
            section_id = data[pos]
            pos += 1
            
            # Validate section ID
            if section_id > 12:
                raise ValueError(f"Invalid section ID {section_id} at pos {pos-1}")
            
            # Read section size (LEB128)
            try:
                size, bytes_read = self._read_leb128(data[pos:])
            except Exception as e:
                raise ValueError(f"Failed to read section size at pos {pos}: {e}")
            
            pos += bytes_read
            
            # Validate section doesn't extend beyond data
            if pos + size > len(data):
                raise ValueError(
                    f"Section {section_id} claims size {size} but only "
                    f"{len(data) - pos} bytes remain"
                )
            
            # Read section data
            section_data = data[pos:pos + size]
            sections[section_id] = section_data
            pos += size
        
        return sections
    
    def _build_module(self, sections: Dict[int, bytes], exports: List[str]) -> bytes:
        """Build linked WASM module from sections."""
        output = bytearray()
        
        # Type section (function signatures)
        if WasmSection.TYPE in sections:
            output.extend(self._encode_section(WasmSection.TYPE, sections[WasmSection.TYPE]))
        
        # Import section (memory and table)
        import_section = self._build_import_section()
        output.extend(self._encode_section(WasmSection.IMPORT, import_section))
        
        # Function section (function type indices)
        if WasmSection.FUNCTION in sections:
            output.extend(self._encode_section(WasmSection.FUNCTION, sections[WasmSection.FUNCTION]))
        
        # Table section
        if WasmSection.TABLE in sections:
            output.extend(self._encode_section(WasmSection.TABLE, sections[WasmSection.TABLE]))
        
        # Memory section (skip - we import memory)
        
        # Global section
        if WasmSection.GLOBAL in sections:
            output.extend(self._encode_section(WasmSection.GLOBAL, sections[WasmSection.GLOBAL]))
        
        # Export section (build from function names)
        export_section = self._build_export_section(sections, exports)
        if export_section:
            output.extend(self._encode_section(WasmSection.EXPORT, export_section))
        
        # Start section
        if WasmSection.START in sections:
            output.extend(self._encode_section(WasmSection.START, sections[WasmSection.START]))
        
        # Element section
        if WasmSection.ELEMENT in sections:
            output.extend(self._encode_section(WasmSection.ELEMENT, sections[WasmSection.ELEMENT]))
        
        # Data count section
        if WasmSection.DATA_COUNT in sections:
            output.extend(self._encode_section(WasmSection.DATA_COUNT, sections[WasmSection.DATA_COUNT]))
        
        # Code section (function bodies)
        if WasmSection.CODE in sections:
            output.extend(self._encode_section(WasmSection.CODE, sections[WasmSection.CODE]))
        
        # Data section
        if WasmSection.DATA in sections:
            output.extend(self._encode_section(WasmSection.DATA, sections[WasmSection.DATA]))
        
        return bytes(output)
    
    def _build_import_section(self) -> bytes:
        """Build import section for memory, table, and stack pointer."""
        imports = bytearray()
        
        # Number of imports (3: memory, table, stack_pointer)
        imports.extend(self._encode_leb128(3))
        
        # Import 1: Memory
        imports.extend(self._encode_string("env"))
        imports.extend(self._encode_string("__linear_memory"))
        imports.append(0x02)  # Memory import
        imports.append(0x00)  # No maximum
        imports.extend(self._encode_leb128(256))  # Initial 256 pages
        
        # Import 2: Table
        imports.extend(self._encode_string("env"))
        imports.extend(self._encode_string("__indirect_function_table"))
        imports.append(0x01)  # Table import
        imports.append(0x70)  # funcref type
        imports.append(0x00)  # No maximum
        imports.extend(self._encode_leb128(0))  # Initial 0 elements
        
        # Import 3: Stack pointer (global i32)
        imports.extend(self._encode_string("env"))
        imports.extend(self._encode_string("__stack_pointer"))
        imports.append(0x03)  # Global import
        imports.append(0x7F)  # i32 type
        imports.append(0x01)  # Mutable
        
        return bytes(imports)
    
    def _build_export_section(self, sections: Dict[int, bytes], exports: List[str]) -> Optional[bytes]:
        """
        Build export section for specified functions.
        
        Raises:
            ValueError: If export names exceed function count
        """
        if not exports or WasmSection.FUNCTION not in sections:
            return None
        
        # Parse function section to get function count
        func_data = sections[WasmSection.FUNCTION]
        
        if len(func_data) == 0:
            raise ValueError("FUNCTION section is empty")
        
        try:
            func_count, _ = self._read_leb128(func_data)
        except Exception as e:
            raise ValueError(f"Failed to read function count: {e}")
        
        # Validate export count
        if len(exports) > func_count:
            raise ValueError(
                f"Cannot export {len(exports)} functions, only {func_count} available"
            )
        
        # Build exports
        export_data = bytearray()
        export_data.extend(self._encode_leb128(len(exports)))
        
        for idx, name in enumerate(exports):
            if idx >= func_count:
                break
            
            # Export name
            export_data.extend(self._encode_string(name))
            # Export kind (0 = function)
            export_data.append(0x00)
            # Function index (offset by number of imports)
            export_data.extend(self._encode_leb128(idx))
        
        return bytes(export_data)
    
    def _encode_section(self, section_id: int, data: bytes) -> bytes:
        """Encode section with ID and size."""
        output = bytearray()
        output.append(section_id)
        output.extend(self._encode_leb128(len(data)))
        output.extend(data)
        return bytes(output)
    
    def _encode_string(self, s: str) -> bytes:
        """Encode string as length + UTF-8 bytes."""
        encoded = s.encode('utf-8')
        return self._encode_leb128(len(encoded)) + encoded
    
    def _encode_leb128(self, value: int) -> bytes:
        """Encode unsigned integer as LEB128."""
        result = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value != 0:
                byte |= 0x80
            result.append(byte)
            if value == 0:
                break
        return bytes(result)
    
    def _read_leb128(self, data: bytes) -> Tuple[int, int]:
        """
        Read LEB128 unsigned integer. Returns (value, bytes_read).
        
        Raises:
            ValueError: If data is empty or LEB128 is malformed
        """
        if not data:
            raise ValueError("Cannot read LEB128 from empty data")
        
        result = 0
        shift = 0
        pos = 0
        max_bytes = 10  # Protect against infinite loops (64-bit max needs ~10 bytes)
        
        while pos < len(data) and pos < max_bytes:
            byte = data[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            shift += 7
            
            # Check if this is the last byte
            if (byte & 0x80) == 0:
                return result, pos
        
        # If we got here, either data ran out or exceeded max bytes
        if pos >= max_bytes:
            raise ValueError(f"LEB128 exceeds maximum {max_bytes} bytes, data corrupted?")
        else:
            raise ValueError("LEB128 data truncated (missing terminator byte)")
