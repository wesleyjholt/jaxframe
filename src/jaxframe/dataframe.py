"""
A simple immutable DataFrame implementation using a dictionary of arrays and lists.
Optimized for JAX compatibility with fast constructors and minimal copying.
"""
from typing import Dict, Any, Union, List, Tuple, Optional, Literal
import numpy as np
import os
from enum import Enum

class ColumnType(Enum):
    """Column type enumeration for fast type tracking."""
    JAX_ARRAY = 'jax_array'
    NUMPY_ARRAY = 'array'
    LIST = 'list'


class DataFrame:
    """
    A simple immutable DataFrame that stores data as a dictionary of arrays and/or lists.
    
    All arrays and lists must have the same length. Once created, the DataFrame cannot be modified.
    """
    
    def __init__(self, 
                 data: Dict[str, Union[List, np.ndarray]], 
                 name: str = None,
                 column_types: Optional[Dict[str, Union[str, ColumnType]]] = None,
                 skip_validation: bool = False):
        """
        Initialize a DataFrame with optimized type detection and minimal copying.
        
        Args:
            data: Dictionary where keys are column names and values are arrays/lists
            name: Optional name for the DataFrame
            column_types: Pre-computed column types (skips expensive detection)
            skip_validation: Skip length validation for internal operations
                 
        Raises:
            ValueError: If arrays/lists have different lengths or if data is empty.
            TypeError: If data is not a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
            
        if not data:
            raise ValueError("Data dictionary cannot be empty")
        
        self._data = {}
        self._name = name
        self._columns = tuple(data.keys())
        
        # Fast path: use provided column types
        if column_types is not None:
            self._column_types = self._normalize_column_types(column_types)
            self._process_data_fast_path(data)
        else:
            # Optimized type detection
            self._column_types = {}
            self._process_data_with_detection(data)
        
        # Get length from first column
        self._length = self._get_length_fast()
        
        # Validate lengths only if requested
        if not skip_validation:
            self._validate_lengths()
    
    def _normalize_column_types(self, column_types: Dict[str, Union[str, ColumnType]]) -> Dict[str, str]:
        """Convert column types to string format."""
        normalized = {}
        for col, ctype in column_types.items():
            if isinstance(ctype, ColumnType):
                normalized[col] = ctype.value
            else:
                normalized[col] = ctype
        return normalized
    
    def _process_data_fast_path(self, data: Dict[str, Any]):
        """Fast data processing when types are known."""
        for column_name, values in data.items():
            col_type = self._column_types[column_name]
            
            if col_type == ColumnType.JAX_ARRAY.value:
                # JAX arrays are immutable - no copying needed
                self._data[column_name] = values
            elif col_type == ColumnType.NUMPY_ARRAY.value:
                # Only copy if necessary for mutable numpy arrays
                self._data[column_name] = values if hasattr(values, 'flags') and not values.flags.writeable else values.copy()
            else:  # LIST
                # Copy lists for safety
                self._data[column_name] = values.copy() if isinstance(values, list) else list(values)
    
    def _process_data_with_detection(self, data: Dict[str, Any]):
        """Optimized type detection using module checking."""
        for column_name, values in data.items():
            if not isinstance(column_name, str):
                raise TypeError("Column names must be strings")
            
            # Fast module-based type detection
            col_type, processed_values = self._detect_type_optimized(values)
            self._column_types[column_name] = col_type
            self._data[column_name] = processed_values
    
    def _detect_type_optimized(self, values) -> Tuple[str, Any]:
        """Optimized type detection using module checking instead of complex logic."""
        # Fast path: check module directly
        if hasattr(values, '__module__'):
            module_str = str(values.__module__)
            # Be more precise about JAX module detection
            if module_str.startswith('jax') or '.jax' in module_str:
                return ColumnType.JAX_ARRAY.value, values  # No copy for immutable JAX arrays
            elif 'numpy' in module_str:
                return ColumnType.NUMPY_ARRAY.value, values.copy()  # Copy numpy arrays
        
        # Handle lists
        if isinstance(values, list):
            # Check if list contains JAX elements (simplified check)
            if values and hasattr(values[0], '__module__') and 'jax' in str(values[0].__module__):
                try:
                    import jax.numpy as jnp
                    return ColumnType.JAX_ARRAY.value, jnp.array(values)
                except ImportError:
                    pass
            return ColumnType.LIST.value, values.copy()
        
        # Handle numpy arrays
        if isinstance(values, np.ndarray):
            return ColumnType.NUMPY_ARRAY.value, values.copy()
        
        # Fallback: try to detect JAX arrays with precise checks
        if hasattr(values, 'shape') and hasattr(values, 'dtype'):
            try:
                import jax
                # Only classify as JAX if it's actually a JAX type
                if isinstance(values, (jax.Array, jax.core.Tracer)):
                    return ColumnType.JAX_ARRAY.value, values
                # If has __array__ method, convert to numpy array
                elif hasattr(values, '__array__'):
                    return ColumnType.NUMPY_ARRAY.value, np.array(values)
            except ImportError:
                # If no JAX, treat as numpy array if it has __array__
                if hasattr(values, '__array__'):
                    return ColumnType.NUMPY_ARRAY.value, np.array(values)
        
        # Default: convert to list
        return ColumnType.LIST.value, list(values)
    
    def _get_length_fast(self) -> int:
        """Fast length detection from first column."""
        if not self._data:
            return 0
        
        first_value = next(iter(self._data.values()))
        if hasattr(first_value, 'shape'):
            return first_value.shape[0]
        else:
            return len(first_value)
    
    def _validate_lengths(self):
        """Validate all columns have same length."""
        lengths = set()
        for values in self._data.values():
            if hasattr(values, 'shape'):
                lengths.add(values.shape[0])
            else:
                lengths.add(len(values))
        
        if len(lengths) > 1:
            raise ValueError(f"All arrays and lists must have the same length. Got lengths: {list(lengths)}")
    
    # Fast constructors for common cases
    @classmethod
    def from_jax_arrays(cls, data: Dict[str, Any], name: str = None) -> 'DataFrame':
        """Ultra-fast constructor for pure JAX data."""
        column_types = {k: ColumnType.JAX_ARRAY for k in data.keys()}
        return cls(data, name=name, column_types=column_types, skip_validation=True)
    
    @classmethod
    def from_numpy_arrays(cls, data: Dict[str, np.ndarray], name: str = None) -> 'DataFrame':
        """Fast constructor for pure NumPy data."""
        column_types = {k: ColumnType.NUMPY_ARRAY for k in data.keys()}
        return cls(data, name=name, column_types=column_types, skip_validation=True)
    
    @classmethod
    def from_lists(cls, data: Dict[str, List], name: str = None) -> 'DataFrame':
        """Fast constructor for list data."""
        column_types = {k: ColumnType.LIST for k in data.keys()}
        return cls(data, name=name, column_types=column_types, skip_validation=True)
    
    @property
    def name(self) -> str:
        """Get the name of the DataFrame."""
        return self._name
    
    @property
    def columns(self) -> Tuple[str, ...]:
        """Get column names as an immutable tuple."""
        return self._columns
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the DataFrame as (rows, columns)."""
        return (self._length, len(self._columns))
    
    @property
    def column_types(self) -> Dict[str, str]:
        """Get the data types of columns ('list' or 'array')."""
        return self._column_types.copy()
    
    # JIT-friendly operations
    def to_jax_dict(self) -> Dict[str, Any]:
        """Extract only JAX arrays for JIT functions - zero copy."""
        return {k: v for k, v in self._data.items() 
                if self._column_types[k] == ColumnType.JAX_ARRAY.value}
    
    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """Extract only NumPy arrays - returns views when possible."""
        return {k: v.view() if hasattr(v, 'view') else v 
                for k, v in self._data.items()
                if self._column_types[k] == ColumnType.NUMPY_ARRAY.value}
    
    def get_jax_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Get specific JAX columns for JIT functions."""
        return {col: self._data[col] for col in columns 
                if col in self._data and self._column_types[col] == ColumnType.JAX_ARRAY.value}
    
    def dtypes(self) -> Dict[str, str]:
        """Get detailed data types for each column."""
        dtypes = {}
        for col_name, col_data in self._data.items():
            if hasattr(col_data, 'dtype'):
                # For numpy arrays or JAX arrays
                dtypes[col_name] = str(col_data.dtype)
            elif isinstance(col_data, list) and len(col_data) > 0:
                # For lists, get the type of the first element
                first_element = col_data[0]
                element_type = type(first_element).__name__
                dtypes[col_name] = f"list[{element_type}]"
            else:
                # Fallback
                dtypes[col_name] = type(col_data).__name__
        return dtypes
    
    def __len__(self) -> int:
        """Get the number of rows in the DataFrame."""
        return self._length
    
    def with_name(self, name: str) -> 'DataFrame':
        """
        Create a new DataFrame with the specified name.
        
        Args:
            name: The new name for the DataFrame
            
        Returns:
            New DataFrame with the same data but different name
        """
        return DataFrame(self._data, name=name)
    
    def __getitem__(self, key: str) -> Union[List[Any], np.ndarray, Any]:
        """
        Get a column by name with optimized copying strategy.
        
        Args:
            key: Column name
            
        Returns:
            Column data with minimal copying (views for arrays, copies for lists)
            
        Raises:
            KeyError: If column doesn't exist
        """
        if key not in self._data:
            raise KeyError(f"Column '{key}' not found. Available columns: {list(self._columns)}")
        
        # Optimized access with minimal copying
        col_type = self._column_types[key]
        value = self._data[key]
        
        if col_type == ColumnType.JAX_ARRAY.value:
            # JAX arrays are immutable - return directly
            return value
        elif col_type == ColumnType.NUMPY_ARRAY.value:  # 'array'
            # Return copy for numpy arrays to maintain immutability semantics
            return value.copy() if hasattr(value, 'copy') else value
        else:  # LIST
            # Copy lists for safety
            return value.copy() if isinstance(value, list) else list(value)
    
    def __contains__(self, key: str) -> bool:
        """Check if a column exists in the DataFrame."""
        return key in self._data
    
    def _parse_env_limit(self, env_name: str, default: int) -> Optional[int]:
        """Parse an environment variable into a non-negative integer.
        
        Negative values are treated as None meaning "no limit".
        """
        val = os.getenv(env_name)
        if val is None or val == "":
            return default
        try:
            n = int(val)
            if n < 0:
                return None  # unlimited
            return n
        except ValueError:
            return default
    
    def _truncate_str(self, s: str, limit: Optional[int]) -> str:
        """Truncate a string to limit characters with an ellipsis."""
        if limit is None or len(s) <= limit:
            return s
        return s[:limit] + "…"
    
    def _format_value(self, value: Any, str_limit: Optional[int]) -> str:
        """Format a value for display, handling different data types."""
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                formatted = f"{value:.3f}"
            else:
                formatted = str(value)
        else:
            # Handle numpy/JAX scalars
            try:
                import numpy as np
                if isinstance(value, np.integer):
                    formatted = str(int(value))
                elif isinstance(value, np.floating):
                    formatted = f"{float(value):.3f}"
                else:
                    # Handle JAX arrays/scalars
                    try:
                        import jax
                        import jax.core
                        if hasattr(value, 'dtype') and (
                            str(type(value)).startswith('<class \'jaxlib.') or
                            str(type(value).__module__).startswith('jax') or
                            (hasattr(value, '__module__') and str(value.__module__).startswith('jax'))):
                            
                            if isinstance(value, jax.core.Tracer):
                                dtype_str = str(value.dtype) if hasattr(value, 'dtype') else 'unknown'
                                formatted = f"<tracer:{dtype_str}>"
                            else:
                                try:
                                    scalar_value = float(value) if 'float' in str(value.dtype) else int(value)
                                    if isinstance(scalar_value, float):
                                        formatted = f"{scalar_value:.3f}"
                                    else:
                                        formatted = str(scalar_value)
                                except (jax.errors.ConcretizationTypeError, AttributeError):
                                    dtype_str = str(value.dtype) if hasattr(value, 'dtype') else 'unknown'
                                    formatted = f"<jax:{dtype_str}>"
                        else:
                            formatted = str(value)
                    except ImportError:
                        formatted = str(value)
            except ImportError:
                formatted = str(value)
        
        return self._truncate_str(formatted, str_limit)
    
    def _get_dtype_str(self, col_name: str) -> str:
        """Get a short dtype string for a column."""
        col_type = self._column_types[col_name]
        data = self._data[col_name]
        
        if col_type == ColumnType.JAX_ARRAY.value:
            try:
                import jax.numpy as jnp
                if hasattr(data, 'dtype'):
                    dtype = str(data.dtype)
                    # Simplify dtype names
                    if dtype.startswith('float'):
                        return 'f' + dtype.replace('float', '')
                    elif dtype.startswith('int'):
                        return 'i' + dtype.replace('int', '')
                    elif dtype.startswith('bool'):
                        return 'bool'
                    return dtype
                return 'jax'
            except ImportError:
                return 'jax'
        elif col_type == ColumnType.NUMPY_ARRAY.value:
            try:
                import numpy as np
                if hasattr(data, 'dtype'):
                    dtype = str(data.dtype)
                    # Simplify dtype names  
                    if dtype.startswith('float'):
                        return 'f' + dtype.replace('float', '')
                    elif dtype.startswith('int'):
                        return 'i' + dtype.replace('int', '')
                    elif dtype.startswith('bool'):
                        return 'bool'
                    return dtype
                return 'array'
            except ImportError:
                return 'array'
        else:  # LIST
            if len(data) > 0:
                first_val = data[0]
                if isinstance(first_val, str):
                    return 'str'
                elif isinstance(first_val, bool):
                    return 'bool'
                elif isinstance(first_val, int):
                    return 'i64'
                elif isinstance(first_val, float):
                    return 'f64'
            return 'list'
    
    def to_string(self,
                  max_rows: Optional[int] = None,
                  max_cols: Optional[int] = None, 
                  str_limit: Optional[int] = None,
                  show_shape: bool = True) -> str:
        """Return a Polars-like string representation of the DataFrame.
        
        Args:
            max_rows: Maximum rows to display (None = unlimited)
            max_cols: Maximum columns to display (None = unlimited)  
            str_limit: Maximum characters per cell (None = unlimited)
            show_shape: Whether to show shape information
            
        Returns:
            Formatted string representation
        """
        # Use environment variables or defaults
        if max_rows is None:
            max_rows = self._parse_env_limit("POLARS_FMT_MAX_ROWS", 10)
        if max_cols is None:
            max_cols = self._parse_env_limit("POLARS_FMT_MAX_COLS", 8)
        if str_limit is None:
            str_limit = self._parse_env_limit("POLARS_FMT_STR_LEN", 30)
            
        n_rows = self._length
        n_cols = len(self._columns)
        
        if n_rows == 0:
            if show_shape:
                return f"shape: (0, {n_cols})\n┌─┐\n│ │\n└─┘"
            return "┌─┐\n│ │\n└─┘"
        
        # Determine column display strategy
        if max_cols is None or n_cols <= max_cols:
            n_first_cols = n_cols
            n_last_cols = 0
            reduce_cols = False
        else:
            n_first_cols = (max_cols + 1) // 2
            n_last_cols = max_cols // 2
            reduce_cols = True
        
        # Prepare headers with dtype info
        headers = []
        dtypes = []
        selected_columns = []
        
        # First columns
        for col in self._columns[:n_first_cols]:
            headers.append(self._truncate_str(col, str_limit))
            dtypes.append(self._get_dtype_str(col))
            selected_columns.append(col)
        
        # Ellipsis column
        if reduce_cols:
            headers.append("…")
            dtypes.append("…")
            selected_columns.append("…")
        
        # Last columns
        for col in self._columns[n_cols - n_last_cols:]:
            headers.append(self._truncate_str(col, str_limit))
            dtypes.append(self._get_dtype_str(col))
            selected_columns.append(col)
        
        # Determine rows to display
        if max_rows is None or n_rows <= max_rows or max_rows <= 0:
            row_indices = list(range(n_rows))
            insert_ellipsis_row = False
        else:
            half = max_rows // 2
            rest = max_rows % 2
            top_range = list(range(half + rest))
            bottom_range = list(range(n_rows - half, n_rows))
            row_indices = top_range + [-1] + bottom_range
            insert_ellipsis_row = True
        
        # Build data matrix
        data_matrix = []
        for idx in row_indices:
            if idx == -1 and insert_ellipsis_row:
                data_matrix.append(["…"] * len(headers))
                continue
                
            row_data = []
            # First columns
            for col in self._columns[:n_first_cols]:
                value = self._data[col][idx]
                row_data.append(self._format_value(value, str_limit))
            
            # Ellipsis column
            if reduce_cols:
                row_data.append("…")
            
            # Last columns  
            for col in self._columns[n_cols - n_last_cols:]:
                value = self._data[col][idx]
                row_data.append(self._format_value(value, str_limit))
                
            data_matrix.append(row_data)
        
        # Calculate column widths
        col_widths = []
        for i in range(len(headers)):
            max_width = max(len(headers[i]), len(dtypes[i]))
            for row in data_matrix:
                max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)
        
        # Build the table
        lines = []
        
        if show_shape:
            lines.append(f"shape: ({n_rows}, {n_cols})")
        
        # Top border
        top_line = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
        lines.append(top_line)
        
        # Header row
        header_cells = [f" {headers[i].ljust(col_widths[i])} " for i in range(len(headers))]
        header_line = "│" + "│".join(header_cells) + "│"
        lines.append(header_line)
        
        # Dtype row
        dtype_cells = [f" {dtypes[i].ljust(col_widths[i])} " for i in range(len(dtypes))]
        dtype_line = "│" + "│".join(dtype_cells) + "│"
        lines.append(dtype_line)
        
        # Header separator
        sep_line = "╞" + "╪".join("═" * (w + 2) for w in col_widths) + "╡"
        lines.append(sep_line)
        
        # Data rows
        for row in data_matrix:
            data_cells = [f" {row[i].ljust(col_widths[i])} " for i in range(len(row))]
            data_line = "│" + "│".join(data_cells) + "│"
            lines.append(data_line)
        
        # Bottom border
        bottom_line = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
        lines.append(bottom_line)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation using Polars-like formatting."""
        return self.to_string()
    
    def __repr__(self) -> str:
        """String representation of the DataFrame."""
        if self._length == 0:
            name_part = f" '{self._name}'" if self._name else ""
            return f"DataFrame{name_part}(empty)"
        
        name_part = f" '{self._name}'" if self._name else ""
        lines = [f"DataFrame{name_part}({self.shape[0]} rows, {self.shape[1]} columns)"]
        lines.append("Columns: " + ", ".join(self._columns))
        
        # Add dtypes information
        dtypes = self.dtypes()
        dtype_strs = [f"{col}: {dtypes[col]}" for col in self._columns]
        lines.append("Dtypes: " + ", ".join(dtype_strs))
        
        # Show first few rows
        max_display_rows = 5
        for i in range(min(self._length, max_display_rows)):
            row_dict = {}
            for col in self._columns:
                value = self._data[col][i]
                # Format the value without quotes for numeric types
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        row_dict[col] = f"{value:.3f}"
                    else:
                        row_dict[col] = str(value)
                else:
                    # Check if it's a numpy scalar
                    try:
                        import numpy as np
                        if isinstance(value, np.integer):
                            row_dict[col] = str(int(value))
                        elif isinstance(value, np.floating):
                            row_dict[col] = f"{float(value):.3f}"
                        else:
                            # Check if it's a JAX array/scalar
                            try:
                                import jax
                                import jax.core
                                if hasattr(value, 'dtype') and (
                                    str(type(value)).startswith('<class \'jaxlib.') or
                                    str(type(value).__module__).startswith('jax') or
                                    (hasattr(value, '__module__') and str(value.__module__).startswith('jax'))):
                                    # Check if this is a tracer (from jit, grad, etc.)
                                    if isinstance(value, jax.core.Tracer):
                                        # Display tracer info instead of trying to extract value
                                        dtype_str = str(value.dtype) if hasattr(value, 'dtype') else 'unknown'
                                        shape_str = str(value.shape) if hasattr(value, 'shape') else '[]'
                                        row_dict[col] = f"<JAX tracer {dtype_str}{shape_str}>"
                                    else:
                                        try:
                                            # Extract the underlying value from JAX array/scalar
                                            scalar_value = float(value) if 'float' in str(value.dtype) else int(value)
                                            if isinstance(scalar_value, float):
                                                row_dict[col] = f"{scalar_value:.3f}"
                                            else:
                                                row_dict[col] = str(scalar_value)
                                        except jax.errors.ConcretizationTypeError:
                                            # Fallback for any JAX value that can't be concretized
                                            dtype_str = str(value.dtype) if hasattr(value, 'dtype') else 'unknown'
                                            shape_str = str(value.shape) if hasattr(value, 'shape') else '[]'
                                            row_dict[col] = f"<JAX value {dtype_str}{shape_str}>"
                                else:
                                    # For strings and other non-JAX types, keep quotes
                                    row_dict[col] = repr(value)
                            except ImportError:
                                # JAX not available, treat as regular value
                                row_dict[col] = repr(value)
                    except ImportError:
                        # Numpy not available, check JAX only
                        try:
                            import jax
                            import jax.core
                            if hasattr(value, 'dtype') and (
                                str(type(value)).startswith('<class \'jaxlib.') or
                                str(type(value).__module__).startswith('jax') or
                                (hasattr(value, '__module__') and str(value.__module__).startswith('jax'))):
                                # Check if this is a tracer (from jit, grad, etc.)
                                if isinstance(value, jax.core.Tracer):
                                    # Display tracer info instead of trying to extract value
                                    dtype_str = str(value.dtype) if hasattr(value, 'dtype') else 'unknown'
                                    shape_str = str(value.shape) if hasattr(value, 'shape') else '[]'
                                    row_dict[col] = f"<JAX tracer {dtype_str}{shape_str}>"
                                else:
                                    try:
                                        # Extract the underlying value from JAX array/scalar
                                        scalar_value = float(value) if 'float' in str(value.dtype) else int(value)
                                        if isinstance(scalar_value, float):
                                            row_dict[col] = f"{scalar_value:.3f}"
                                        else:
                                            row_dict[col] = str(scalar_value)
                                    except jax.errors.ConcretizationTypeError:
                                        # Fallback for any JAX value that can't be concretized
                                        dtype_str = str(value.dtype) if hasattr(value, 'dtype') else 'unknown'
                                        shape_str = str(value.shape) if hasattr(value, 'shape') else '[]'
                                        row_dict[col] = f"<JAX value {dtype_str}{shape_str}>"
                            else:
                                # For strings and other types, keep quotes
                                row_dict[col] = repr(value)
                        except ImportError:
                            # Neither numpy nor JAX available
                            row_dict[col] = repr(value)
            
            # Manually format the dictionary to control quote display
            items = []
            for col in self._columns:
                items.append(f"'{col}': {row_dict[col]}")
            lines.append(f"  [{i}]: {{{', '.join(items)}}}")
        
        if self._length > max_display_rows:
            lines.append(f"  ... ({self._length - max_display_rows} more rows)")
        
        return "\n".join(lines)
    
    def __eq__(self, other) -> bool:
        """Check equality with another DataFrame."""
        if not isinstance(other, DataFrame):
            return False
        
        if self._columns != other._columns:
            return False
        
        for col in self._columns:
            # Compare values regardless of whether they're stored as lists, numpy arrays, or JAX arrays
            self_values = self._data[col]
            other_values = other._data[col]
            
            # Handle different types for comparison - convert to comparable forms
            self_type = self._column_types[col]
            other_type = other._column_types[col]
            
            # Special handling for JAX arrays - they require JAX for comparison
            if self_type == 'jax_array' or other_type == 'jax_array':
                try:
                    import jax.numpy as jnp
                    # Convert both to JAX arrays for comparison
                    if self_type != 'jax_array':
                        self_values = jnp.array(self_values)
                    if other_type != 'jax_array':
                        other_values = jnp.array(other_values)
                    if not jnp.array_equal(self_values, other_values):
                        return False
                except ImportError:
                    # Fall back to numpy comparison if JAX not available
                    self_values = np.asarray(self_values)
                    other_values = np.asarray(other_values)
                    if not np.array_equal(self_values, other_values):
                        return False
            else:
                # For list and numpy array comparisons, convert both to numpy arrays
                if self_type == 'list':
                    self_values = np.asarray(self_values)
                if other_type == 'list':
                    other_values = np.asarray(other_values)
                if not np.array_equal(self_values, other_values):
                    return False
        
        return True
    
    def to_dict(self, copy: bool = True) -> Dict[str, Union[List[Any], np.ndarray, Any]]:
        """
        Convert DataFrame to dictionary with optimized copying.
        
        Args:
            copy: Whether to copy mutable data (default True for safety)
        
        Returns:
            Dictionary with data, minimal copying when copy=False
        """
        if not copy:
            # Return references directly - fastest but potentially unsafe
            return dict(self._data)
        
        # Optimized copying based on mutability
        result = {}
        for col in self._columns:
            col_type = self._column_types[col]
            value = self._data[col]
            
            if col_type == ColumnType.JAX_ARRAY.value:
                # JAX arrays are immutable - no copy needed
                result[col] = value
            elif col_type == ColumnType.NUMPY_ARRAY.value:
                # Copy numpy arrays for safety
                result[col] = value.copy() if hasattr(value, 'copy') else value
            else:  # LIST
                # Copy lists for safety
                result[col] = value.copy() if isinstance(value, list) else list(value)
        return result
    
    def get_row(self, index: int) -> Dict[str, Any]:
        """
        Get a single row by index.
        
        Args:
            index: Row index
            
        Returns:
            Dictionary mapping column names to values for the specified row
            
        Raises:
            IndexError: If index is out of bounds
        """
        if not 0 <= index < self._length:
            raise IndexError(f"Index {index} out of bounds for DataFrame with {self._length} rows")
        
        return {col: self._data[col][index] for col in self._columns}
    
    def select_columns(self, columns: List[str]) -> 'DataFrame':
        """
        Create a new DataFrame with only the specified columns.
        
        Args:
            columns: List of column names to select
            
        Returns:
            New DataFrame with selected columns, preserving original data types
            
        Raises:
            KeyError: If any column doesn't exist
        """
        missing_columns = [col for col in columns if col not in self._data]
        if missing_columns:
            raise KeyError(f"Columns not found: {missing_columns}")
        
        # Preserve original types when creating new DataFrame
        new_data = {}
        for col in columns:
            if self._column_types[col] == 'list':
                new_data[col] = self._data[col].copy()
            elif self._column_types[col] == 'jax_array':
                # JAX arrays are immutable, so we can use them directly
                new_data[col] = self._data[col]
            else:  # numpy array
                new_data[col] = self._data[col].copy()
        
        # Preserve the name in the new DataFrame
        new_name = f"{self._name}_selected" if self._name else None
        return DataFrame(new_data, name=new_name)
    
    def to_pandas(self):
        """
        Convert the DataFrame to a pandas DataFrame.

        Returns:
            pandas.DataFrame: A pandas DataFrame with the same data.
        """
        import pandas as pd

        data = {}
        for col in self._columns:
            if self._column_types[col] == 'list':
                data[col] = self._data[col].copy()
            else:  # array
                data[col] = self._data[col].copy()

        return pd.DataFrame(data)
    
    def join(self, other: 'DataFrame', on: Union[str, List[str]], 
             source: Union[str, List[str], None] = None, 
             target: Union[str, List[str], None] = None,
             how: str = 'inner') -> 'DataFrame':
        """
        Join this DataFrame with another DataFrame.
        
        Args:
            other: The DataFrame to join with
            on: The column name(s) to join on (must exist in both DataFrames).
               Can be a string for single column or list of strings for multi-column joins.
            source: The column name(s) in the other DataFrame to copy values from.
                   Can be a string for single column or list of strings for multiple columns.
                   Required for 'inner' joins, ignored for 'semi' joins.
            target: The name(s) for the new column(s) in the result. If None, 
                   defaults to '{other.name}/{source}' if other has a name,
                   otherwise just source. Can be a string for single column,
                   list of strings for multiple columns, or None for auto-naming.
                   Ignored for 'semi' joins.
            how: Type of join to perform. Options:
                - 'inner': Inner join that adds columns from other DataFrame (default)
                - 'semi': Semi-join that filters rows based on existence of matches
                         but only returns columns from this DataFrame
        
        Returns:
            A new DataFrame with the joined data
            
        Raises:
            KeyError: If on doesn't exist in both DataFrames or source column(s) 
                     don't exist in other DataFrame
            ValueError: If there are duplicate values in the on column of other DataFrame,
                       or if target list length doesn't match source list length,
                       or if how is not 'inner' or 'semi', or if source is required but not provided
        """
        from typing import Union, List
        
        # Validate how parameter
        if how not in ['inner', 'semi']:
            raise ValueError(f"'how' must be 'inner' or 'semi', got '{how}'")
        
        # For semi joins, source and target are ignored
        if how == 'semi':
            source_columns = []
            target_columns = []
        else:
            # For inner joins, source is required
            if source is None:
                raise ValueError("'source' parameter is required for inner joins")
            
            # Normalize inputs to lists
            if isinstance(source, str):
                source_columns = [source]
            else:
                source_columns = source
                
            if target is None:
                target_columns = None
            elif isinstance(target, str):
                target_columns = [target]
            else:
                target_columns = target
                
            # Validate target_columns length if provided
            if target_columns is not None and len(target_columns) != len(source_columns):
                raise ValueError(f"Length of target list ({len(target_columns)}) must match "
                               f"length of source list ({len(source_columns)})")
        
        # Normalize on parameter to list
        if isinstance(on, str):
            on_columns = [on]
        else:
            on_columns = on
            
        # Validate join columns exist
        for col in on_columns:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found in left DataFrame")
            if col not in other.columns:
                raise KeyError(f"Column '{col}' not found in right DataFrame")
        
        # For inner joins, validate source columns
        if how == 'inner':
            for src_col in source_columns:
                if src_col not in other._columns:
                    raise KeyError(f"Column '{src_col}' not found in other DataFrame")
        
        # Handle semi-join case (exists functionality)
        if how == 'semi':
            # Create lookup set from other DataFrame for existence check
            # For multi-column joins, create tuples of values
            if len(on_columns) == 1:
                col = on_columns[0]
                other_keys = set(other[col])
            else:
                # Create tuples for multi-column keys
                other_keys = set()
                for i in range(len(other)):
                    key_tuple = tuple(other[col][i] for col in on_columns)
                    other_keys.add(key_tuple)
            
            # Find matching rows in this DataFrame (preserving order, eliminating duplicates)
            matching_indices = []
            seen_keys = set()
            
            for i in range(len(self)):
                if len(on_columns) == 1:
                    col = on_columns[0]
                    key = self[col][i]
                else:
                    key = tuple(self[col][i] for col in on_columns)
                
                # Check if key exists in other DataFrame and we haven't seen this key yet
                if key in other_keys and key not in seen_keys:
                    matching_indices.append(i)
                    seen_keys.add(key)
            
            # Return subset of this DataFrame with only matching rows
            if not matching_indices:
                # Return empty DataFrame with same structure
                empty_data = {}
                for col in self.columns:
                    if self.column_types[col] == 'list':
                        empty_data[col] = []
                    elif self.column_types[col] == 'jax_array':
                        try:
                            import jax.numpy as jnp
                            empty_data[col] = jnp.array([], dtype=self._data[col].dtype)
                        except ImportError:
                            import numpy as np
                            empty_data[col] = np.array([], dtype=self._data[col].dtype)
                    else:  # numpy array
                        import numpy as np
                        empty_data[col] = np.array([], dtype=self._data[col].dtype)
                return DataFrame(empty_data, name=self.name)
            
            # Create new data dictionary with matching rows
            new_data = {}
            for col in self.columns:
                if self.column_types[col] == 'list':
                    new_data[col] = [self._data[col][i] for i in matching_indices]
                elif self.column_types[col] == 'jax_array':
                    # For JAX arrays, use advanced indexing
                    try:
                        import jax.numpy as jnp
                        new_data[col] = self._data[col][jnp.array(matching_indices)]
                    except ImportError:
                        # JAX not available, fall back to numpy
                        import numpy as np
                        new_data[col] = self._data[col][matching_indices]
                else:  # numpy array
                    import numpy as np
                    new_data[col] = self._data[col][matching_indices]
            
            return DataFrame(new_data, name=self.name)
        
        # Handle inner join case (original functionality)
        # Use single column for backward compatibility validation
        on_single = on_columns[0] if len(on_columns) == 1 else on_columns
        
        # Check for duplicates in the join column(s) of other DataFrame
        if len(on_columns) == 1:
            other_join_values = other[on_columns[0]]
            if len(set(other_join_values)) != len(other_join_values):
                raise ValueError(f"Duplicate values found in '{on_columns[0]}' column of other DataFrame")
        else:
            # For multi-column joins, check for duplicate key combinations
            other_key_tuples = []
            for i in range(len(other)):
                key_tuple = tuple(other[col][i] for col in on_columns)
                other_key_tuples.append(key_tuple)
            if len(set(other_key_tuples)) != len(other_key_tuples):
                raise ValueError(f"Duplicate key combinations found in join columns {on_columns} of other DataFrame")
        
        # Create lookup dictionaries from other DataFrame for each source column
        lookups = {}
        for src_col in source_columns:
            lookups[src_col] = {}
            for i in range(len(other)):
                if len(on_columns) == 1:
                    key = other._data[on_columns[0]][i]
                else:
                    key = tuple(other._data[col][i] for col in on_columns)
                value = other._data[src_col][i]
                lookups[src_col][key] = value
        
        # Determine target column names
        if target_columns is None:
            target_columns = []
            for src_col in source_columns:
                if other._name:
                    target_columns.append(f"{other._name}/{src_col}")
                else:
                    target_columns.append(src_col)
        
        # Create new data with joined columns
        new_data = {}
        
        # First, determine which rows from the left DataFrame should be included (inner join)
        valid_indices = []  # Indices of rows that will be included in the result
        
        # Find all indices where the join key exists in the right DataFrame
        for i in range(len(self)):
            if len(on_columns) == 1:
                key = self._data[on_columns[0]][i]
            else:
                key = tuple(self._data[col][i] for col in on_columns)
            
            # Check if key exists in lookup (use first source column's lookup for key existence check)
            if source_columns and key in lookups[source_columns[0]]:
                valid_indices.append(i)
        
        # Copy existing columns but only for valid indices
        for col in self._columns:
            if self._column_types[col] == 'list':
                new_data[col] = [self._data[col][i] for i in valid_indices]
            elif self._column_types[col] == 'jax_array':
                # For JAX arrays, use advanced indexing to avoid list conversion
                try:
                    import jax.numpy as jnp
                    if valid_indices:
                        new_data[col] = self._data[col][jnp.array(valid_indices)]
                    else:
                        # Empty result - create empty array with same dtype
                        new_data[col] = jnp.array([], dtype=self._data[col].dtype)
                except ImportError:
                    # JAX not available, fall back to numpy
                    import numpy as np
                    if valid_indices:
                        new_data[col] = self._data[col][valid_indices]
                    else:
                        new_data[col] = np.array([], dtype=self._data[col].dtype)
            else:  # numpy array
                import numpy as np
                if valid_indices:
                    new_data[col] = self._data[col][valid_indices]
                else:
                    new_data[col] = np.array([], dtype=self._data[col].dtype)
        
        # Add the joined columns
        for src_col, target_col in zip(source_columns, target_columns):
            lookup = lookups[src_col]
            joined_values = []
            
            # Only process valid indices
            for i in valid_indices:
                if len(on_columns) == 1:
                    key = self._data[on_columns[0]][i]
                else:
                    key = tuple(self._data[col][i] for col in on_columns)
                joined_values.append(lookup[key])
            
            # Preserve the type from the source column
            if other._column_types[src_col] == 'list':
                new_data[target_col] = joined_values
            elif other._column_types[src_col] == 'jax_array':
                # For JAX arrays, convert efficiently without intermediate list conversion
                try:
                    import jax.numpy as jnp
                    if joined_values:
                        new_data[target_col] = jnp.array(joined_values)
                    else:
                        # Empty result - create empty array with same dtype as source
                        new_data[target_col] = jnp.array([], dtype=other._data[src_col].dtype)
                except ImportError:
                    # JAX not available, fall back to numpy
                    import numpy as np
                    if joined_values:
                        new_data[target_col] = np.asarray(joined_values)
                    else:
                        new_data[target_col] = np.array([], dtype=other._data[src_col].dtype)
            else:  # numpy array
                import numpy as np
                if joined_values:
                    new_data[target_col] = np.asarray(joined_values)
                else:
                    new_data[target_col] = np.array([], dtype=other._data[src_col].dtype)
        
        # Create new DataFrame with updated name
        new_name = f"{self._name}_joined" if self._name else None
        return DataFrame(new_data, name=new_name)
    
    def add_column(self, column_name: str, values: Union[List, np.ndarray, Any], 
                   column_type: Optional[Union[str, ColumnType]] = None) -> 'DataFrame':
        """
        Add a new column to the DataFrame with optimized copying.
        
        Args:
            column_name: Name of the new column
            values: Values for the new column
            column_type: Optional pre-computed column type for speed
                   
        Returns:
            New DataFrame with the added column
            
        Raises:
            ValueError: If values length doesn't match DataFrame length or column already exists
        """
        if column_name in self._columns:
            raise ValueError(f"Column '{column_name}' already exists")
        
        # Check length compatibility (optimized)
        if hasattr(values, 'shape'):
            value_length = values.shape[0]
        elif hasattr(values, '__len__'):
            value_length = len(values)
        else:
            value_length = 1
            
        if value_length != self._length:
            raise ValueError(f"New column must have length {self._length}, got {value_length}")
        
        # Shallow copy existing data dict (references to immutable data)
        new_data = dict(self._data)
        new_data[column_name] = values
        
        # Determine new column type efficiently
        if column_type is not None:
            new_column_types = dict(self._column_types)
            if isinstance(column_type, ColumnType):
                new_column_types[column_name] = column_type.value
            else:
                new_column_types[column_name] = column_type
        else:
            # Fast type detection for single value
            detected_type, _ = self._detect_type_optimized(values)
            new_column_types = dict(self._column_types)
            new_column_types[column_name] = detected_type
        
        # Create new DataFrame with minimal validation
        return DataFrame(new_data, name=self._name, column_types=new_column_types, skip_validation=True)
    
    def remove_column(self, column_name: str) -> 'DataFrame':
        """
        Remove a column from the DataFrame, returning a new DataFrame.
        
        Args:
            column_name: Name of the column to remove
            
        Returns:
            New DataFrame without the specified column
            
        Raises:
            KeyError: If column doesn't exist
            ValueError: If removing the column would result in an empty DataFrame
        """
        if column_name not in self._columns:
            raise KeyError(f"Column '{column_name}' not found. Available columns: {list(self._columns)}")
        
        if len(self._columns) == 1:
            raise ValueError("Cannot remove the last column from DataFrame")
        
        # Create new data dictionary without the specified column
        new_data = {}
        
        for col in self._columns:
            if col != column_name:
                if self._column_types[col] == 'list':
                    new_data[col] = self._data[col].copy()
                elif self._column_types[col] == 'jax_array':
                    new_data[col] = self._data[col]
                else:  # numpy array
                    new_data[col] = self._data[col].copy()
        
        return DataFrame(new_data, name=self._name)
    
    def add_row(self, row_data: Dict[str, Any]) -> 'DataFrame':
        """
        Add a new row to the DataFrame, returning a new DataFrame.
        
        Args:
            row_data: Dictionary mapping column names to values for the new row
                     Must contain values for all existing columns
                     
        Returns:
            New DataFrame with the added row
            
        Raises:
            ValueError: If row_data doesn't contain all required columns or has extra columns
        """
        # Check that all columns are present
        missing_cols = set(self._columns) - set(row_data.keys())
        if missing_cols:
            raise ValueError(f"Missing values for columns: {missing_cols}")
        
        extra_cols = set(row_data.keys()) - set(self._columns)
        if extra_cols:
            raise ValueError(f"Extra columns not in DataFrame: {extra_cols}")
        
        # Create new data dictionary with extended columns
        new_data = {}
        
        for col in self._columns:
            if self._column_types[col] == 'list':
                new_values = self._data[col].copy()
                new_values.append(row_data[col])
                new_data[col] = new_values
            elif self._column_types[col] == 'jax_array':
                # For JAX arrays, we need to concatenate
                try:
                    import jax.numpy as jnp
                    # Convert single value to JAX array and concatenate
                    single_value = jnp.array([row_data[col]])
                    new_data[col] = jnp.concatenate([self._data[col], single_value])
                except ImportError:
                    # JAX not available, convert to numpy
                    import numpy as np
                    single_value = np.array([row_data[col]])
                    new_data[col] = np.concatenate([np.asarray(self._data[col]), single_value])
            else:  # numpy array
                import numpy as np
                single_value = np.array([row_data[col]])
                new_data[col] = np.concatenate([self._data[col], single_value])
        
        return DataFrame(new_data, name=self._name)
    
    def remove_row(self, index: int) -> 'DataFrame':
        """
        Remove a row at the specified index from the DataFrame, returning a new DataFrame.
        
        Args:
            index: Index of the row to remove (0-based)
                  
        Returns:
            New DataFrame without the specified row
            
        Raises:
            IndexError: If index is out of bounds
            ValueError: If removing the row would result in an empty DataFrame
        """
        if index < 0 or index >= self._length:
            raise IndexError(f"Row index {index} out of bounds for DataFrame with {self._length} rows")
        
        if self._length == 1:
            raise ValueError("Cannot remove the last row from DataFrame")
        
        # Create new data dictionary with the specified row removed
        new_data = {}
        
        for col in self._columns:
            if self._column_types[col] == 'list':
                new_values = self._data[col].copy()
                del new_values[index]
                new_data[col] = new_values
            elif self._column_types[col] == 'jax_array':
                # For JAX arrays, we need to use array slicing
                try:
                    import jax.numpy as jnp
                    if index == 0:
                        new_data[col] = self._data[col][1:]
                    elif index == self._length - 1:
                        new_data[col] = self._data[col][:-1]
                    else:
                        new_data[col] = jnp.concatenate([
                            self._data[col][:index],
                            self._data[col][index+1:]
                        ])
                except ImportError:
                    # JAX not available, convert to numpy
                    arr = np.asarray(self._data[col])
                    if index == 0:
                        new_data[col] = arr[1:]
                    elif index == self._length - 1:
                        new_data[col] = arr[:-1]
                    else:
                        new_data[col] = np.concatenate([arr[:index], arr[index+1:]])
            else:  # numpy array
                if index == 0:
                    new_data[col] = self._data[col][1:]
                elif index == self._length - 1:
                    new_data[col] = self._data[col][:-1]
                else:
                    new_data[col] = np.concatenate([
                        self._data[col][:index],
                        self._data[col][index+1:]
                    ])
        
        return DataFrame(new_data, name=self._name)
    
    def concat(self, other: 'DataFrame', axis: int = 0, ignore_index: bool = False) -> 'DataFrame':
        """
        Concatenate this DataFrame with another DataFrame.
        
        Args:
            other: DataFrame to concatenate with
            axis: Axis along which to concatenate. 0 for rows (vertical), 1 for columns (horizontal)
            ignore_index: If True, don't check for matching columns when axis=0
                         If False, require same columns for row concatenation
                         
        Returns:
            New DataFrame with concatenated data
            
        Raises:
            ValueError: If DataFrames are incompatible for concatenation
            TypeError: If other is not a DataFrame
        """
        if not isinstance(other, DataFrame):
            raise TypeError("Can only concatenate with another DataFrame")
        
        if axis == 0:
            # Row-wise concatenation (vertical stacking)
            return self._concat_rows(other, ignore_index)
        elif axis == 1:
            # Column-wise concatenation (horizontal stacking)
            return self._concat_columns(other)
        else:
            raise ValueError("axis must be 0 (rows) or 1 (columns)")
    
    def _concat_rows(self, other: 'DataFrame', ignore_index: bool = False) -> 'DataFrame':
        """Concatenate DataFrames row-wise (vertically)."""
        if not ignore_index:
            # Check that both DataFrames have the same columns
            if set(self._columns) != set(other._columns):
                raise ValueError(
                    f"DataFrames must have the same columns for row concatenation. "
                    f"Self: {set(self._columns)}, Other: {set(other._columns)}"
                )
        else:
            # Use intersection of columns
            common_columns = set(self._columns) & set(other._columns)
            if not common_columns:
                raise ValueError("No common columns found between DataFrames")
        
        new_data = {}
        columns_to_use = self._columns if not ignore_index else tuple(common_columns)
        
        for col in columns_to_use:
            if col not in other._columns:
                continue
                
            self_type = self._column_types[col]
            other_type = other._column_types[col]
            
            self_values = self._data[col]
            other_values = other._data[col]
            
            # Handle concatenation based on column types
            if self_type == 'list' and other_type == 'list':
                # Both lists: simple concatenation
                new_data[col] = self_values + other_values
            elif self_type == 'list' and other_type == 'array':
                # List + array: convert list to array and concatenate
                self_array = np.asarray(self_values)
                new_data[col] = np.concatenate([self_array, other_values])
            elif self_type == 'array' and other_type == 'list':
                # Array + list: convert list to array and concatenate
                other_array = np.asarray(other_values)
                new_data[col] = np.concatenate([self_values, other_array])
            elif self_type == 'array' and other_type == 'array':
                # Both arrays: numpy concatenation
                new_data[col] = np.concatenate([self_values, other_values])
            elif self_type == 'jax_array' or other_type == 'jax_array':
                # Handle JAX arrays
                try:
                    import jax.numpy as jnp
                    # Convert both to JAX arrays if needed
                    if self_type != 'jax_array':
                        self_jax = jnp.asarray(self_values)
                    else:
                        self_jax = self_values
                    
                    if other_type != 'jax_array':
                        other_jax = jnp.asarray(other_values)
                    else:
                        other_jax = other_values
                    
                    new_data[col] = jnp.concatenate([self_jax, other_jax])
                except ImportError:
                    # JAX not available, fall back to numpy
                    self_array = np.asarray(self_values)
                    other_array = np.asarray(other_values)
                    new_data[col] = np.concatenate([self_array, other_array])
        
        # Create new name
        new_name = None
        if self._name and other._name:
            new_name = f"{self._name}_concat_{other._name}"
        elif self._name:
            new_name = f"{self._name}_concat"
        elif other._name:
            new_name = f"concat_{other._name}"
        
        return DataFrame(new_data, name=new_name)
    
    def _concat_columns(self, other: 'DataFrame') -> 'DataFrame':
        """Concatenate DataFrames column-wise (horizontally)."""
        # Check that both DataFrames have the same number of rows
        if self._length != other._length:
            raise ValueError(
                f"DataFrames must have the same number of rows for column concatenation. "
                f"Self: {self._length} rows, Other: {other._length} rows"
            )
        
        # Check for column name conflicts
        common_columns = set(self._columns) & set(other._columns)
        if common_columns:
            raise ValueError(
                f"DataFrames have overlapping column names: {common_columns}. "
                f"Column names must be unique for horizontal concatenation."
            )
        
        new_data = {}
        
        # Copy all columns from self
        for col in self._columns:
            if self._column_types[col] == 'list':
                new_data[col] = self._data[col].copy()
            elif self._column_types[col] == 'jax_array':
                new_data[col] = self._data[col]
            else:  # numpy array
                new_data[col] = self._data[col].copy()
        
        # Copy all columns from other
        for col in other._columns:
            if other._column_types[col] == 'list':
                new_data[col] = other._data[col].copy()
            elif other._column_types[col] == 'jax_array':
                new_data[col] = other._data[col]
            else:  # numpy array
                new_data[col] = other._data[col].copy()
        
        # Create new name
        new_name = None
        if self._name and other._name:
            new_name = f"{self._name}_hconcat_{other._name}"
        elif self._name:
            new_name = f"{self._name}_hconcat"
        elif other._name:
            new_name = f"hconcat_{other._name}"
        
        return DataFrame(new_data, name=new_name)
    
    @staticmethod
    def concat_dataframes(dataframes: List['DataFrame'], axis: int = 0, ignore_index: bool = False) -> 'DataFrame':
        """
        Concatenate multiple DataFrames.
        
        Args:
            dataframes: List of DataFrames to concatenate
            axis: Axis along which to concatenate. 0 for rows (vertical), 1 for columns (horizontal)
            ignore_index: If True, don't check for matching columns when axis=0
                         
        Returns:
            New DataFrame with concatenated data
            
        Raises:
            ValueError: If DataFrames are incompatible for concatenation or list is empty
            TypeError: If any element is not a DataFrame
        """
        if not dataframes:
            raise ValueError("Cannot concatenate empty list of DataFrames")
        
        if len(dataframes) == 1:
            # Return a copy if only one DataFrame
            return DataFrame(dataframes[0].to_dict(), name=dataframes[0].name)
        
        # Validate all elements are DataFrames
        for i, df in enumerate(dataframes):
            if not isinstance(df, DataFrame):
                raise TypeError(f"Element at index {i} is not a DataFrame")
        
        # Start with the first DataFrame and concatenate the rest
        result = dataframes[0]
        for df in dataframes[1:]:
            result = result.concat(df, axis=axis, ignore_index=ignore_index)
        
        return result
    
    def is_valid_lookup_table(self, id_columns: Union[str, List[str]]) -> bool:
        """
        Check if the DataFrame is a valid lookup table (no duplicate keys).
        
        Args:
            id_columns: Column name(s) that form the lookup key
            
        Returns:
            True if no duplicate keys exist, False otherwise
        """
        if isinstance(id_columns, str):
            id_columns = [id_columns]
        
        # Check that all id_columns exist
        for col in id_columns:
            if col not in self._columns:
                raise KeyError(f"Column '{col}' not found. Available columns: {list(self._columns)}")
        
        # Get all key combinations
        keys_seen = set()
        for row_idx in range(len(self)):
            key_tuple = tuple(self[col][row_idx] for col in id_columns)
            if key_tuple in keys_seen:
                return False
            keys_seen.add(key_tuple)
        
        return True
    
    def update_lookup_table(self, other: 'DataFrame', id_columns: Union[str, List[str]], 
                           strict: bool = True) -> 'DataFrame':
        """
        Update this lookup table with rows from another DataFrame.
        
        Args:
            other: DataFrame containing rows to add/update
            id_columns: Column name(s) that form the lookup key
            strict: If True, raises error on value mismatches. If False, replaces with new values.
            
        Returns:
            New DataFrame with updated lookup table
            
        Raises:
            ValueError: If DataFrames have incompatible columns or if strict=True and values mismatch
            KeyError: If id_columns don't exist in both DataFrames
        """
        if isinstance(id_columns, str):
            id_columns = [id_columns]
            
        # Validate that both DataFrames have the required columns
        for col in id_columns:
            if col not in self._columns:
                raise KeyError(f"Column '{col}' not found in current DataFrame. Available: {list(self._columns)}")
            if col not in other._columns:
                raise KeyError(f"Column '{col}' not found in other DataFrame. Available: {list(other._columns)}")
        
        # Check that both DataFrames have the same set of columns
        if set(self._columns) != set(other._columns):
            raise ValueError(f"DataFrames must have the same columns. "
                           f"Current: {set(self._columns)}, Other: {set(other._columns)}")
        
        # Check that current DataFrame is a valid lookup table
        if not self.is_valid_lookup_table(id_columns):
            raise ValueError("Current DataFrame is not a valid lookup table (has duplicate keys)")
        
        # Check that other DataFrame is a valid lookup table
        if not other.is_valid_lookup_table(id_columns):
            raise ValueError("Other DataFrame is not a valid lookup table (has duplicate keys)")
        
        # Build index of existing keys in current DataFrame
        existing_keys = {}  # key_tuple -> row_index
        for row_idx in range(len(self)):
            key_tuple = tuple(self[col][row_idx] for col in id_columns)
            existing_keys[key_tuple] = row_idx
        
        # Start with a copy of current data, converting to lists for mutability
        new_data = {}
        for col in self._columns:
            if isinstance(self._data[col], list):
                new_data[col] = self._data[col].copy()
            else:
                # Convert arrays (numpy or JAX) to lists so we can append
                new_data[col] = list(self._data[col])
        
        # Process each row in the other DataFrame
        for other_row_idx in range(len(other)):
            key_tuple = tuple(other[col][other_row_idx] for col in id_columns)
            
            if key_tuple in existing_keys:
                # Key exists - check for value conflicts or replace
                existing_row_idx = existing_keys[key_tuple]
                
                if strict:
                    # Check all non-key columns for mismatches
                    for col in self._columns:
                        if col not in id_columns:
                            existing_val = self[col][existing_row_idx]
                            new_val = other[col][other_row_idx]
                            
                            # Handle different data types for comparison
                            if not self._values_equal(existing_val, new_val):
                                raise ValueError(f"Value mismatch for key {key_tuple} in column '{col}': "
                                               f"existing='{existing_val}', new='{new_val}'")
                else:
                    # Replace existing values with new ones
                    for col in self._columns:
                        if col not in id_columns:
                            new_data[col][existing_row_idx] = other[col][other_row_idx]
            else:
                # New key - add the row
                for col in self._columns:
                    new_data[col].append(other[col][other_row_idx])
        
        # Determine new name - inherit from original DataFrame
        new_name = self._name
        
        return DataFrame(new_data, name=new_name)
    
    def replace_lookup_table(self, other: 'DataFrame', id_columns: Union[str, List[str]]) -> 'DataFrame':
        """
        Update this lookup table with rows from another DataFrame, replacing conflicting values.
        
        This is a convenience method equivalent to update_lookup_table with strict=False.
        
        Args:
            other: DataFrame containing rows to add/update
            id_columns: Column name(s) that form the lookup key
            
        Returns:
            New DataFrame with updated lookup table where new values replace old ones
        """
        return self.update_lookup_table(other, id_columns, strict=False)
    
    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """
        Helper method to compare two values for equality, handling different data types.
        
        Args:
            val1: First value
            val2: Second value
            
        Returns:
            True if values are considered equal, False otherwise
        """
        try:
            # Handle JAX arrays
            import jax.numpy as jnp
            if hasattr(val1, 'shape') and hasattr(val2, 'shape'):
                return jnp.allclose(val1, val2, equal_nan=True)
        except ImportError:
            pass
        
        # Handle numpy arrays
        if hasattr(val1, 'shape') and hasattr(val2, 'shape'):
            return np.allclose(val1, val2, equal_nan=True)
        
        # Handle regular values
        if isinstance(val1, float) and isinstance(val2, float):
            return abs(val1 - val2) < 1e-10
        
        return val1 == val2