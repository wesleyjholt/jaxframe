"""
JIT-compatible printing functions for JAXFrame objects.

This module provides functions that can print DataFrame and MaskedArray information
within jax.jit-compiled functions using jax.debug.print.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Union


def _format_value_for_jit_print(value: Any) -> str:
    """
    Format a value for JIT printing, handling JAX tracers nicely.
    
    For JAX tracers, this extracts the dtype and shape to show clean output
    like 'f32[3]' instead of ugly tracer representations.
    
    Args:
        value: The value to format (could be a tracer, array, scalar, etc.)
        
    Returns:
        A clean string representation of the value
    """
    
    def extract_aval_info(obj):
        """Try multiple strategies to extract aval information from a JAX object."""
        # Strategy 1: Direct aval access
        if hasattr(obj, 'aval'):
            return obj.aval
        
        # Strategy 2: Through primal
        if hasattr(obj, 'primal'):
            if hasattr(obj.primal, 'aval'):
                return obj.primal.aval
            # Recursive check for nested primals
            return extract_aval_info(obj.primal)
        
        # Strategy 3: Direct shape/dtype access
        if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            class MockAval:
                def __init__(self, shape, dtype):
                    self.shape = shape
                    self.dtype = dtype
            return MockAval(obj.shape, obj.dtype)
        
        # Strategy 4: Look for _aval attribute (some tracers use this)
        if hasattr(obj, '_aval'):
            return obj._aval
            
        # Strategy 5: Check for tangent (in JVP tracers)
        if hasattr(obj, 'tangent'):
            return extract_aval_info(obj.tangent)
        
        return None
    
    # First, check if the string representation looks like a complex tracer
    # This is our "emergency catch-all" for tracers we might miss
    str_repr = str(value)
    if len(str_repr) > 100 and ('Traced<' in str_repr or 'JVPTrace' in str_repr or 'JaxprTrace' in str_repr):
        # This looks like a complex tracer that we need to handle
        aval = extract_aval_info(value)
        if aval is not None:
            dtype_str = str(aval.dtype)
            dtype_map = {
                'float32': 'f32', 'float64': 'f64',
                'int32': 'i32', 'int64': 'i64', 
                'bool': 'bool', 'complex64': 'c64', 'complex128': 'c128'
            }
            dtype_str = dtype_map.get(dtype_str, dtype_str)
            
            if aval.shape == ():
                return dtype_str  # scalar
            else:
                shape_str = 'x'.join(map(str, aval.shape))
                return f"{dtype_str}[{shape_str}]"
        else:
            # Absolute fallback - return generic tracer indicator
            return "<tracer>"
    
    # Check if this looks like any kind of JAX tracer
    is_likely_tracer = (
        # Check for 'Traced' in type name
        ('Traced' in str(type(value))) or
        # Check for common tracer attributes
        (hasattr(value, 'aval')) or
        (hasattr(value, 'primal')) or
        # Check for tracer-like class names
        ('Tracer' in str(type(value))) or
        ('JVP' in str(type(value))) or
        ('Jaxpr' in str(type(value)))
    )
    
    if is_likely_tracer:
        aval = extract_aval_info(value)
        
        if aval is not None:
            # Extract clean dtype and shape info from aval
            dtype_str = str(aval.dtype)
            
            # Clean up dtype strings (e.g., 'float32' -> 'f32')
            dtype_map = {
                'float32': 'f32', 'float64': 'f64',
                'int32': 'i32', 'int64': 'i64', 
                'bool': 'bool', 'complex64': 'c64', 'complex128': 'c128'
            }
            dtype_str = dtype_map.get(dtype_str, dtype_str)
            
            if aval.shape == ():
                return dtype_str  # scalar
            else:
                shape_str = 'x'.join(map(str, aval.shape))
                return f"{dtype_str}[{shape_str}]"
        else:
            # Fallback: if we can't extract aval info but it's clearly a tracer,
            # return a generic tracer indicator rather than the ugly string
            return "<tracer>"
    
    # For regular values, use the existing formatting logic
    if isinstance(value, float):
        return f"{value:.3f}"
    elif hasattr(value, 'item'):  # numpy scalar or JAX scalar (when not traced)
        try:
            return f"{value.item():.3f}" if isinstance(value.item(), float) else str(value.item())
        except (TypeError, ValueError):
            return str(value)
    else:
        return str(value)


def _should_quote_value(original_value: Any, formatted_value: str) -> bool:
    """
    Determine whether a value should be quoted in the dictionary representation.
    
    Args:
        original_value: The original value from the DataFrame
        formatted_value: The formatted string representation
        
    Returns:
        True if the value should be quoted, False otherwise
    """
    # Don't quote numeric-looking values or tracers
    if formatted_value in ['<tracer>']:
        return False
    
    # Don't quote if it looks like a tracer format (f32, i32[3], etc.)
    if any(formatted_value.startswith(dt) for dt in ['f32', 'f64', 'i32', 'i64', 'bool', 'c64', 'c128']):
        return False
    
    # Check the original value type for the most reliable determination
    if isinstance(original_value, (int, float, bool)):
        return False
    
    # Check if it's a numpy scalar of numeric type
    if hasattr(original_value, 'dtype'):
        dtype_str = str(original_value.dtype)
        if any(dt in dtype_str for dt in ['int', 'float', 'bool']):
            return False
    
    # Don't quote if it's a pure number (but only if original isn't a string)
    if not isinstance(original_value, str):
        try:
            float(formatted_value)
            return False
        except (ValueError, TypeError):
            pass
    
    # Don't quote boolean values (if they're actual booleans, not string representations)
    if isinstance(original_value, bool) or formatted_value.lower() in ['true', 'false']:
        return False
    
    # Quote everything else (strings, etc.)
    return True


def jit_print_dataframe(df: Any) -> None:
    """
    Print a DataFrame representation that works within jax.jit.
    
    Args:
        df: The DataFrame to print
        
    Note:
        This function mimics the regular DataFrame.__repr__ output but uses
        jax.debug.print to work within JIT-compiled functions. The DataFrame
        must be passed as a static argument to the JIT function.
        
    Example:
        ```python
        @jax.jit(static_argnames=['df'])
        def process_with_df(x, df):
            jit_print_dataframe(df)
            return x * 2
        ```
    """
    if not hasattr(df, '_data') or not hasattr(df, '_columns') or not hasattr(df, '_length'):
        jax.debug.print("Not a valid DataFrame object")
        return
    
    # Print header
    if df._length == 0:
        name_part = f" '{df._name}'" if hasattr(df, '_name') and df._name else ""
        jax.debug.print("DataFrame{}(empty)", name_part)
        return
    
    name_part = f" '{df._name}'" if hasattr(df, '_name') and df._name else ""
    jax.debug.print("DataFrame{}({} rows, {} columns)", 
                   name_part, df.shape[0], df.shape[1])
    
    # Print columns
    columns_str = ", ".join(df._columns)
    jax.debug.print("Columns: {}", columns_str)
    
    # Print first few rows (max 5, same as regular __repr__)
    max_display_rows = min(df._length, 5)
    
    for i in range(max_display_rows):
        row_pairs = []
        for col in df._columns:
            value = df._data[col][i]
            # Format the value nicely, handling JAX tracers
            formatted_value = _format_value_for_jit_print(value)
            
            # Determine if we need quotes around the value
            # Only add quotes for string values, not for numeric values or tracers
            if _should_quote_value(value, formatted_value):
                row_pairs.append(f"'{col}': '{formatted_value}'")
            else:
                row_pairs.append(f"'{col}': {formatted_value}")
        
        # Create the row representation
        row_dict_str = ", ".join(row_pairs)
        jax.debug.print("  [{}]: {{{}}}", i, row_dict_str)
    
    # Show continuation if there are more rows
    if df._length > max_display_rows:
        jax.debug.print("  ... ({} more rows)", df._length - max_display_rows)


def jit_print_masked_array(ma: Any) -> None:
    """
    Print a MaskedArray representation that works within jax.jit.
    
    Args:
        ma: The MaskedArray to print
        
    Note:
        This function mimics the regular MaskedArray.__repr__ output but uses
        jax.debug.print to work within JIT-compiled functions. The MaskedArray
        must be passed as a static argument to the JIT function.
        
    Example:
        ```python
        @jax.jit(static_argnames=['ma'])
        def process_with_ma(x, ma):
            jit_print_masked_array(ma)
            return x * ma.data
        ```
    """
    if not hasattr(ma, '_data') or not hasattr(ma, '_mask') or not hasattr(ma, '_index_df'):
        jax.debug.print("Not a valid MaskedArray object")
        return
    
    # Get basic info
    n_rows, n_cols = ma.shape
    
    # Calculate mask statistics (using numpy since mask is numpy array)
    n_valid = int(np.sum(ma._mask))
    n_total = int(ma._mask.size)
    valid_pct = (n_valid / n_total * 100) if n_total > 0 else 0
    
    # Print the same format as regular __repr__
    jax.debug.print("MaskedArray({} rows, {} columns)", n_rows, n_cols)
    jax.debug.print("Valid values: {}/{} ({:.1f}%)", n_valid, n_total, valid_pct)
    jax.debug.print("Index DataFrame: {} rows, {} columns", 
                   len(ma._index_df), len(ma._index_df.columns))


def jit_print_dataframe_data(data: Dict[str, Any], columns: List[str], length: int, 
                           name: str = None) -> None:
    """
    Print DataFrame data inside a JIT function using individual components.
    
    Args:
        data: Dictionary mapping column names to arrays
        columns: List of column names
        length: Number of rows
        name: Optional name for the DataFrame
        
    Note:
        This function allows printing DataFrame-like data when the DataFrame
        object itself cannot be passed as a static argument. All arguments
        must be JAX-traceable or static.
    """
    # Print header
    if length == 0:
        name_part = f" '{name}'" if name else ""
        jax.debug.print("DataFrame{}(empty)", name_part)
        return
    
    name_part = f" '{name}'" if name else ""
    jax.debug.print("DataFrame{}({} rows, {} columns)", 
                   name_part, length, len(columns))
    
    # Print columns
    columns_str = ", ".join(columns)
    jax.debug.print("Columns: {}", columns_str)
    
    # Print first few rows
    max_display_rows = min(length, 5)
    
    for i in range(max_display_rows):
        row_pairs = []
        for col in columns:
            value = data[col][i]
            # Format values using our helper function
            formatted_value = _format_value_for_jit_print(value)
            
            # Apply same quoting logic as main DataFrame function
            if _should_quote_value(value, formatted_value):
                row_pairs.append(f"'{col}': '{formatted_value}'")
            else:
                row_pairs.append(f"'{col}': {formatted_value}")
        
        row_dict_str = ", ".join(row_pairs)
        jax.debug.print("  [{}]: {{{}}}", i, row_dict_str)
    
    if length > max_display_rows:
        jax.debug.print("  ... ({} more rows)", length - max_display_rows)


def jit_print_masked_array_data(data: Any, mask: Any, rows: int, cols: int) -> None:
    """
    Print MaskedArray data inside a JIT function using individual components.
    
    Args:
        data: The JAX array containing data values
        mask: Boolean mask array (converted to JAX array)
        rows: Number of rows
        cols: Number of columns
        
    Note:
        This function allows printing MaskedArray-like data when the MaskedArray
        object itself cannot be passed as a static argument. The mask should be
        converted to a JAX array before calling this function.
    """
    jax.debug.print("MaskedArray({} rows, {} columns)", rows, cols)
    
    # Calculate mask statistics using JAX operations
    n_valid = jnp.sum(mask)
    n_total = rows * cols
    valid_pct = (n_valid / n_total * 100) if n_total > 0 else 0.0
    
    jax.debug.print("Valid values: {}/{} ({:.1f}%)", n_valid, n_total, valid_pct)
    
    # Show a sample of the data and mask
    max_display_rows = min(rows, 3)
    max_display_cols = min(cols, 3)
    
    if max_display_rows > 0 and max_display_cols > 0:
        jax.debug.print("Sample data (first {} rows, first {} columns):", 
                       max_display_rows, max_display_cols)
        
        # Print data sample
        jax.debug.print("Data:")
        for i in range(max_display_rows):
            jax.debug.print("  [{}]: {}", i, data[i, :max_display_cols])
        
        # Print mask sample
        jax.debug.print("Mask (True=valid, False=masked):")
        for i in range(max_display_rows):
            jax.debug.print("  [{}]: {}", i, mask[i, :max_display_cols])
