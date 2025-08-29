"""
JIT-compatible printing functions for JAXFrame objects.

This module provides functions that can print DataFrame and MaskedArray information
within jax.jit-compiled functions using jax.debug.print.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Union


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
        row_data = []
        for col in df._columns:
            value = df._data[col][i]
            # Format the value nicely (same as regular __repr__)
            if isinstance(value, float):
                row_data.append(f"{value:.3f}")
            else:
                row_data.append(str(value))
        
        # Create the row representation
        row_dict_str = ", ".join([f"'{col}': '{val}'" for col, val in zip(df._columns, row_data)])
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
        row_values = []
        for col in columns:
            value = data[col][i]
            # Format values - note: this is simplified since we can't easily
            # check types in JIT context
            row_values.append(str(value))
        
        row_dict_str = ", ".join([f"'{col}': '{val}'" for col, val in zip(columns, row_values)])
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
