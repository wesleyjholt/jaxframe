"""
Data transformation utilities for jaxframe DataFrames.

This module provides functions for reshaping and transforming DataFrames,
including wide-to-long format conversions with mask support.
"""

from typing import List, Tuple, Any, Union, Optional
import re
from .dataframe import DataFrame


def wide_to_long_masked(
    df: DataFrame, 
    id_columns: Union[str, List[str]], 
    var_pattern: str = r'([^$]+)\$(\d+)\$(value|mask)',
    var_name: str = 'variable',
    value_name: str = 'value'
) -> DataFrame:
    """
    Convert a wide format DataFrame to long format, applying masks to filter out invalid values.
    
    This function takes a DataFrame in wide format where columns follow the pattern
    'var$N$value' and 'var$N$mask', and converts it to long format. Only values
    where the corresponding mask is True are included in the output.
    
    Args:
        df: Input DataFrame in wide format
        id_columns: Column name(s) that identify each row (will be preserved in long format)
        var_pattern: Regex pattern to parse column names. Should have 3 groups:
                    (variable_name, time_index, column_type)
                    Default matches patterns like 'time$0$value', 'time$1$mask', etc.
        var_name: Name for the variable column in long format (default: 'variable')
        value_name: Name for the value column in long format (default: 'value')
    
    Returns:
        DataFrame in long format with id_columns, var_name, and value_name columns
        
    Example:
        >>> # Wide format with masked time series data
        >>> wide_df = DataFrame({
        ...     'sample_id': ['001', '002', '003'],
        ...     'time$0$value': [0.0, 0.1, 0.2],
        ...     'time$1$value': [0.0, 0.2, 0.4], 
        ...     'time$0$mask': [True, True, True],
        ...     'time$1$mask': [True, True, False]  # Last sample masked for time 1
        ... })
        >>> long_df = wide_to_long_masked(wide_df, 'sample_id')
        >>> # Result will have 5 rows (3 for time 0, 2 for time 1)
    """
    # Ensure id_columns is a list
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    
    # Parse column names to identify value and mask columns
    value_columns = {}  # {(var_name, time_index): column_name}
    mask_columns = {}   # {(var_name, time_index): column_name}
    
    pattern = re.compile(var_pattern)
    
    for col in df.columns:
        match = pattern.match(col)
        if match:
            var_name_part, time_index, col_type = match.groups()
            time_index = int(time_index)
            key = (var_name_part, time_index)
            
            if col_type == 'value':
                value_columns[key] = col
            elif col_type == 'mask':
                mask_columns[key] = col
    
    # Build long format data
    long_data = {col: [] for col in id_columns}
    long_data[var_name] = []
    long_data[value_name] = []
    
    # Get unique variable names and time indices
    all_keys = set(value_columns.keys()) | set(mask_columns.keys())
    var_names = sorted(set(key[0] for key in all_keys))
    
    # For each row in the original DataFrame
    for row_idx in range(len(df)):
        # For each variable and time combination
        for key in sorted(value_columns.keys()):
            var_name_part, time_index = key
            value_col = value_columns[key]
            mask_col = mask_columns.get(key)  # May not exist
            
            # Check if this value should be included (mask is True or doesn't exist)
            include_value = True
            if mask_col is not None:
                mask_value = df[mask_col][row_idx]
                include_value = bool(mask_value)
            
            if include_value:
                # Add this observation to the long format
                for id_col in id_columns:
                    long_data[id_col].append(df[id_col][row_idx])
                
                long_data[var_name].append(time_index)
                long_data[value_name].append(df[value_col][row_idx])
    
    return DataFrame(long_data)


def long_to_wide_masked(
    df: DataFrame,
    id_columns: Union[str, List[str]],
    value_column: str,
    var_column: Optional[str] = None,
    var_prefix: str = 'var',
    fill_type: Union[Any, str] = 0.0,
    mask_value: bool = False
) -> DataFrame:
    """
    Convert a long format DataFrame to wide format with mask columns.
    
    This is the inverse operation of wide_to_long_masked. It takes a DataFrame
    in long format and creates a wide format with both value and mask columns.
    
    Args:
        df: Input DataFrame in long format
        id_columns: Column name(s) that identify each entity
        value_column: Column containing the values
        var_column: Column containing the variable indices/names (optional).
                   If not provided, will assign indices 0, 1, 2, ... based on
                   the order of values within each ID group.
        var_prefix: Prefix for variable names in wide format (default: 'var')
        fill_type: Fill strategy for missing observations. Can be:
                  - A specific value (e.g., 0.0, -999)
                  - 'local_max': Use the maximum value for each ID
                  - 'global_max': Use the global maximum value across all data
        mask_value: Mask value for missing observations (default: False)
    
    Returns:
        DataFrame in wide format with var_prefix$N$value and var_prefix$N$mask columns
    """
    # Ensure id_columns is a list
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    
    # Get unique IDs
    unique_ids = []
    id_tuples_seen = set()
    
    for row_idx in range(len(df)):
        id_tuple = tuple(df[id_col][row_idx] for id_col in id_columns)
        if id_tuple not in id_tuples_seen:
            unique_ids.append(id_tuple)
            id_tuples_seen.add(id_tuple)
    
    # Calculate fill values based on fill_type
    fill_values = {}  # {id_tuple: fill_value}
    
    if fill_type == 'global_max':
        # Find global maximum across all values
        global_max = max(df[value_column])
        for id_tuple in unique_ids:
            fill_values[id_tuple] = global_max
    elif fill_type == 'local_max':
        # Find local maximum for each ID
        for id_tuple in unique_ids:
            id_values = []
            for row_idx in range(len(df)):
                row_id_tuple = tuple(df[id_col][row_idx] for id_col in id_columns)
                if row_id_tuple == id_tuple:
                    id_values.append(df[value_column][row_idx])
            fill_values[id_tuple] = max(id_values) if id_values else 0.0
    else:
        # Use the provided value for all IDs
        for id_tuple in unique_ids:
            fill_values[id_tuple] = fill_type
    
    # If var_column is provided, use it; otherwise infer indices
    if var_column is not None:
        # Use provided variable column
        unique_vars = sorted(set(df[var_column]))
    else:
        # Infer variable indices based on order within each ID group
        max_observations = 0
        for id_tuple in unique_ids:
            # Count observations for this ID
            count = 0
            for row_idx in range(len(df)):
                row_id_tuple = tuple(df[id_col][row_idx] for id_col in id_columns)
                if row_id_tuple == id_tuple:
                    count += 1
            max_observations = max(max_observations, count)
        
        unique_vars = list(range(max_observations))
    
    # Initialize wide format data
    wide_data = {col: [] for col in id_columns}
    
    # Create value and mask columns for each variable
    for var_idx in unique_vars:
        value_col_name = f"{var_prefix}${var_idx}$value"
        mask_col_name = f"{var_prefix}${var_idx}$mask"
        wide_data[value_col_name] = []
        wide_data[mask_col_name] = []
    
    # For each unique ID combination
    for id_tuple in unique_ids:
        # Add ID values
        for i, id_col in enumerate(id_columns):
            wide_data[id_col].append(id_tuple[i])
        
        # Find all observations for this ID in order
        id_observations = {}  # {var_idx: value}
        if var_column is not None:
            # Use provided variable column
            for row_idx in range(len(df)):
                row_id_tuple = tuple(df[id_col][row_idx] for id_col in id_columns)
                if row_id_tuple == id_tuple:
                    var_idx = df[var_column][row_idx]
                    value = df[value_column][row_idx]
                    id_observations[var_idx] = value
        else:
            # Infer variable indices based on order of appearance
            var_idx = 0
            for row_idx in range(len(df)):
                row_id_tuple = tuple(df[id_col][row_idx] for id_col in id_columns)
                if row_id_tuple == id_tuple:
                    value = df[value_column][row_idx]
                    id_observations[var_idx] = value
                    var_idx += 1
        
        # Fill in values and masks for each variable
        for var_idx in unique_vars:
            value_col_name = f"{var_prefix}${var_idx}$value"
            mask_col_name = f"{var_prefix}${var_idx}$mask"
            
            if var_idx in id_observations:
                # Observation exists
                wide_data[value_col_name].append(id_observations[var_idx])
                wide_data[mask_col_name].append(True)
            else:
                # Missing observation - use calculated fill value
                wide_data[value_col_name].append(fill_values[id_tuple])
                wide_data[mask_col_name].append(mask_value)
    
    return DataFrame(wide_data)


def wide_df_to_jax_arrays(
    df: DataFrame,
    id_columns: Union[str, List[str]],
    var_pattern: str = r'([^$]+)\$(\d+)\$value',
    sort_by_var_index: bool = True
) -> Tuple[Any, Any, DataFrame]:
    """
    Convert a wide format DataFrame to JAX arrays for values and masks.
    
    Args:
        df: Wide format DataFrame with columns like 'var$0$value', 'var$1$value', etc.
        id_columns: Column name(s) that identify each row (will be preserved)
        var_pattern: Regex pattern to extract variable indices from column names
        sort_by_var_index: Whether to sort columns by variable index (default: True)
    
    Returns:
        Tuple of (values_array, mask_array, id_dataframe) where:
        - values_array: JAX array of shape (n_rows, n_variables) 
        - mask_array: JAX array of shape (n_rows, n_variables) with boolean masks
        - id_dataframe: DataFrame containing only the ID columns
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required for this function. Install with: pip install jax")
    
    import re
    
    # Ensure id_columns is a list
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    
    # Find value and mask columns
    value_columns = []
    mask_columns = []
    var_indices = []
    
    pattern = re.compile(var_pattern)
    
    for col in df.columns:
        if col in id_columns:
            continue
            
        match = pattern.match(col)
        if match:
            var_name, var_index = match.groups()
            var_index = int(var_index)
            value_columns.append(col)
            var_indices.append(var_index)
            
            # Look for corresponding mask column
            mask_col = f"{var_name}${var_index}$mask"
            if mask_col in df.columns:
                mask_columns.append(mask_col)
            else:
                mask_columns.append(None)  # No mask column found
    
    if not value_columns:
        raise ValueError(f"No value columns found matching pattern: {var_pattern}")
    
    # Sort by variable index if requested
    if sort_by_var_index:
        sorted_data = sorted(zip(var_indices, value_columns, mask_columns))
        var_indices, value_columns, mask_columns = zip(*sorted_data)
    
    # Extract values and masks
    n_rows = len(df)
    n_vars = len(value_columns)
    
    values = jnp.zeros((n_rows, n_vars))
    masks = jnp.ones((n_rows, n_vars), dtype=bool)  # Default to True (unmasked)
    
    for i, (value_col, mask_col) in enumerate(zip(value_columns, mask_columns)):
        # Get values
        col_values = df[value_col]
        if hasattr(col_values, 'copy'):
            col_values = col_values.copy()
        values = values.at[:, i].set(jnp.array(col_values))
        
        # Get masks
        if mask_col is not None:
            col_masks = df[mask_col]
            if hasattr(col_masks, 'copy'):
                col_masks = col_masks.copy()
            masks = masks.at[:, i].set(jnp.array(col_masks))
    
    # Create ID DataFrame
    id_data = {col: df[col] for col in id_columns}
    id_df = DataFrame(id_data)
    
    return values, masks, id_df


def jax_arrays_to_wide_df(
    values_array: Any,
    mask_array: Any,
    id_dataframe: DataFrame,
    var_prefix: str = 'var'
) -> DataFrame:
    """
    Convert JAX arrays back to wide format DataFrame.
    
    Args:
        values_array: JAX array of shape (n_rows, n_variables) with values
        mask_array: JAX array of shape (n_rows, n_variables) with boolean masks
        id_dataframe: DataFrame containing ID columns
        var_prefix: Prefix for variable column names (default: 'var')
    
    Returns:
        Wide format DataFrame with value and mask columns
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required for this function. Install with: pip install jax")
    
    if values_array.shape != mask_array.shape:
        raise ValueError(f"Values and mask arrays must have same shape. "
                        f"Got {values_array.shape} and {mask_array.shape}")
    
    if values_array.shape[0] != len(id_dataframe):
        raise ValueError(f"Number of rows in arrays ({values_array.shape[0]}) "
                        f"must match ID DataFrame length ({len(id_dataframe)})")
    
    n_rows, n_vars = values_array.shape
    
    # Start with ID columns
    wide_data = {col: id_dataframe[col] for col in id_dataframe.columns}
    
    # Add value and mask columns
    for var_idx in range(n_vars):
        value_col_name = f"{var_prefix}${var_idx}$value"
        mask_col_name = f"{var_prefix}${var_idx}$mask"
        
        # Extract column data and convert to Python lists/arrays
        values_col = values_array[:, var_idx]
        masks_col = mask_array[:, var_idx]
        
        # Convert JAX arrays to regular arrays/lists for DataFrame
        wide_data[value_col_name] = [float(x) for x in values_col]
        wide_data[mask_col_name] = [bool(x) for x in masks_col]
    
    return DataFrame(wide_data)


def roundtrip_wide_jax_conversion(
    df: DataFrame,
    id_columns: Union[str, List[str]],
    var_pattern: str = r'([^$]+)\$(\d+)\$value',
    var_prefix: str = 'var'
) -> DataFrame:
    """
    Test roundtrip conversion: wide DataFrame -> JAX arrays -> wide DataFrame.
    
    This is a utility function for testing that the conversion process preserves data.
    
    Args:
        df: Original wide format DataFrame
        id_columns: Column name(s) that identify each row
        var_pattern: Regex pattern for value columns  
        var_prefix: Prefix for reconstructed column names
        
    Returns:
        Reconstructed wide format DataFrame
    """
    # Convert to JAX arrays
    values, masks, id_df = wide_df_to_jax_arrays(df, id_columns, var_pattern)
    
    # Convert back to DataFrame
    reconstructed = jax_arrays_to_wide_df(values, masks, id_df, var_prefix)
    
    return reconstructed
