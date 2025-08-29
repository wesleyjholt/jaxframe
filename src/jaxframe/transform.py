"""
Data transformation utilities for jaxframe DataFrames.

This module provides functions for reshaping and transforming DataFrames,
including wide-to-long format conversions with mask support.
"""

from typing import List, Tuple, Any, Union, Optional
import re
import numpy as np
from .dataframe import DataFrame
from .masked_array import MaskedArray


def wide_to_long_masked(
    df: Union[DataFrame, List[DataFrame]], 
    id_columns: Union[str, List[str]], 
    var_pattern: str = r'([^$]+)\$(\d+)\$(value|mask)',
    var_name: Union[str, List[str]] = 'variable',
    value_name: Union[str, List[str]] = 'value'
) -> DataFrame:
    """
    Convert wide format DataFrame(s) to long format, applying masks to filter out invalid values.
    
    This function can handle either:
    1. Single DataFrame conversion (original behavior)
    2. Multiple DataFrame conversion where each DataFrame represents different variables
    
    Args:
        df: Input DataFrame(s) in wide format. Can be:
            - Single DataFrame for original behavior
            - List of DataFrames for multi-variable conversion
        id_columns: Column name(s) that identify each row (will be preserved in long format)
        var_pattern: Regex pattern to parse column names. Should have 3 groups:
                    (variable_name, time_index, column_type)
                    Default matches patterns like 'time$0$value', 'time$1$mask', etc.
        var_name: Name(s) for the variable column(s) in long format. Can be:
                 - Single string for single DataFrame input
                 - List of strings for multiple DataFrame input (must match df list length)
        value_name: Name(s) for the value column(s) in long format. Can be:
                   - Single string for single DataFrame input  
                   - List of strings for multiple DataFrame input (must match df list length)
    
    Returns:
        DataFrame in long format with id_columns and the specified variable/value columns
        
    Example:
        Single DataFrame (original behavior):
        >>> wide_df = DataFrame({
        ...     'sample_id': ['001', '002', '003'],
        ...     'time$0$value': [0.0, 0.1, 0.2],
        ...     'time$1$value': [0.0, 0.2, 0.4], 
        ...     'time$0$mask': [True, True, True],
        ...     'time$1$mask': [True, True, False]
        ... })
        >>> long_df = wide_to_long_masked(wide_df, 'sample_id')
        
        Multiple DataFrames:
        >>> time_df = DataFrame({'sample_id': [...], 'time$0$value': [...], ...})
        >>> group_df = DataFrame({'sample_id': [...], 'group$0$value': [...], ...})
        >>> long_df = wide_to_long_masked([time_df, group_df], 'sample_id', 
        ...                              var_name=['variable', 'variable'], 
        ...                              value_name=['time_value', 'group_value'])
    """
    # Handle single DataFrame case (backward compatibility)
    if not isinstance(df, list):
        return _single_wide_to_long_masked(df, id_columns, var_pattern, var_name, value_name)
    
    # Handle multiple DataFrames case
    df_list = df
    
    # Validate inputs
    if isinstance(var_name, str):
        var_name = [var_name] * len(df_list)
    if isinstance(value_name, str):
        value_name = [value_name] * len(df_list)
        
    if len(var_name) != len(df_list):
        raise ValueError(f"var_name list length ({len(var_name)}) must match df list length ({len(df_list)})")
    if len(value_name) != len(df_list):
        raise ValueError(f"value_name list length ({len(value_name)}) must match df list length ({len(df_list)})")
    
    # Convert each DataFrame to long format separately
    long_dfs = []
    for i, (df_single, var_n, val_n) in enumerate(zip(df_list, var_name, value_name)):
        long_df = _single_wide_to_long_masked(df_single, id_columns, var_pattern, var_n, val_n)
        long_dfs.append(long_df)
    
    # Merge all long DataFrames on id_columns
    if len(long_dfs) == 1:
        return long_dfs[0]
    
    # Start with the first DataFrame and join the rest
    result = long_dfs[0]
    for i in range(1, len(long_dfs)):
        # Join on id_columns and var_name columns
        join_columns = id_columns if isinstance(id_columns, list) else [id_columns]
        join_columns = join_columns + [var_name[0]]  # Use first var_name as variable column
        
        # Get the value column from the next DataFrame
        value_col = value_name[i]
        result = result.join(long_dfs[i], on=join_columns, source=value_col)
    
    return result


def _single_wide_to_long_masked(
    df: DataFrame, 
    id_columns: Union[str, List[str]], 
    var_pattern: str = r'([^$]+)\$(\d+)\$(value|mask)',
    var_name: str = 'variable',
    value_name: str = 'value'
) -> DataFrame:
    """
    Convert a single wide format DataFrame to long format (internal helper function).
    
    This is the original implementation for single DataFrame conversion.
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
    value_column: Union[str, List[str]],
    var_column: Optional[Union[str, List[str]]] = None,
    var_prefix: Union[str, List[str]] = 'var',
    fill_type: Union[Any, str, List[Union[Any, str]]] = 0.0,
    mask_value: bool = False
) -> Union[DataFrame, List[DataFrame]]:
    """
    Convert a long format DataFrame to wide format with mask columns.
    
    This function can handle either:
    1. Single column conversion (original behavior) 
    2. Multiple column conversion where multiple variables are processed simultaneously
    
    Args:
        df: Input DataFrame in long format
        id_columns: Column name(s) that identify each entity
        value_column: Column name(s) containing the values. Can be:
                     - Single string for original behavior
                     - List of strings for multi-variable conversion
        var_column: Column name(s) containing the variable indices/names. Can be:
                   - Single string or None for original behavior
                   - List of strings/None for multi-variable conversion
                   If not provided, will assign indices based on order within each ID group.
        var_prefix: Prefix(es) for variable names in wide format. Can be:
                   - Single string for original behavior
                   - List of strings for multi-variable conversion
        fill_type: Fill strategy for missing observations. Can be:
                  - Single value/strategy for original behavior
                  - List of values/strategies for multi-variable conversion
                  Strategies: specific value, 'local_max', 'global_max'
        mask_value: Mask value for missing observations (default: False)
    
    Returns:
        DataFrame(s) in wide format:
        - Single DataFrame if single column input
        - List of DataFrames if multiple column input
    
    Notes:
        - For string dtypes, 'local_max' and 'global_max' fill_types will raise ValueError
        - All output DataFrames maintain consistent row ordering based on first value_column
    """
    # Handle single column case (backward compatibility)
    if isinstance(value_column, str):
        return _single_long_to_wide_masked(df, id_columns, value_column, var_column, 
                                         var_prefix, fill_type, mask_value)
    
    # Handle multiple columns case
    value_columns = value_column
    
    # Normalize inputs to lists
    if isinstance(var_column, str) or var_column is None:
        var_columns = [var_column] * len(value_columns)
    else:
        var_columns = var_column
        
    if isinstance(var_prefix, str):
        var_prefixes = [var_prefix] * len(value_columns)
    else:
        var_prefixes = var_prefix
        
    if not isinstance(fill_type, list):
        fill_types = [fill_type] * len(value_columns)
    else:
        fill_types = fill_type
    
    # Validate input lengths
    if len(var_columns) != len(value_columns):
        raise ValueError(f"var_column list length ({len(var_columns)}) must match value_column length ({len(value_columns)})")
    if len(var_prefixes) != len(value_columns):
        raise ValueError(f"var_prefix list length ({len(var_prefixes)}) must match value_column length ({len(value_columns)})")
    if len(fill_types) != len(value_columns):
        raise ValueError(f"fill_type list length ({len(fill_types)}) must match value_column length ({len(value_columns)})")
    
    # Check for invalid fill_types with string data
    for i, (val_col, fill_t) in enumerate(zip(value_columns, fill_types)):
        if fill_t in ['local_max', 'global_max']:
            # Check if this column contains string data
            sample_values = [df[val_col][j] for j in range(min(10, len(df)))]
            if any(isinstance(v, str) for v in sample_values):
                raise ValueError(f"fill_type '{fill_t}' not supported for string data in column '{val_col}'. "
                               f"Use a specific string value instead.")
    
    # Process each column and ensure consistent ordering
    # The ordering is determined by the first value column
    wide_dfs = []
    reference_ordering = None
    
    for i, (val_col, var_col, var_pref, fill_t) in enumerate(zip(value_columns, var_columns, var_prefixes, fill_types)):
        wide_df = _single_long_to_wide_masked(df, id_columns, val_col, var_col, 
                                            var_pref, fill_t, mask_value)
        
        if i == 0:
            # First DataFrame establishes the reference ordering
            if isinstance(id_columns, str):
                reference_ordering = [wide_df.get_row(j)[id_columns] for j in range(len(wide_df))]
            else:
                reference_ordering = [tuple(wide_df.get_row(j)[col] for col in id_columns) for j in range(len(wide_df))]
            wide_dfs.append(wide_df)
        else:
            # Reorder subsequent DataFrames to match the reference ordering
            reordered_df = _reorder_dataframe_by_ids(wide_df, id_columns, reference_ordering)
            wide_dfs.append(reordered_df)
    
    return wide_dfs


def _single_long_to_wide_masked(
    df: DataFrame,
    id_columns: Union[str, List[str]],
    value_column: str,
    var_column: Optional[str] = None,
    var_prefix: str = 'var',
    fill_type: Union[Any, str] = 0.0,
    mask_value: bool = False
) -> DataFrame:
    """
    Convert a single column from long format to wide format (internal helper function).
    
    This is the original implementation for single column conversion.
    """
    # Check for invalid fill_types with string data
    if fill_type in ['local_max', 'global_max']:
        # Check if this column contains string data
        sample_values = [df[value_column][j] for j in range(min(10, len(df)))]
        if any(isinstance(v, str) for v in sample_values):
            raise ValueError(f"fill_type '{fill_type}' not supported for string data in column '{value_column}'. "
                           f"Use a specific string value instead.")
    
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


def _reorder_dataframe_by_ids(df: DataFrame, id_columns: Union[str, List[str]], reference_ordering: List) -> DataFrame:
    """
    Reorder a DataFrame to match a reference ordering of ID values.
    
    Args:
        df: DataFrame to reorder
        id_columns: Column name(s) that identify each row
        reference_ordering: List of ID tuples in the desired order
        
    Returns:
        Reordered DataFrame
    """
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    
    # Create a mapping from ID tuple to row index in the original DataFrame
    id_to_row = {}
    for row_idx in range(len(df)):
        if len(id_columns) == 1:
            id_key = df[id_columns[0]][row_idx]
        else:
            id_key = tuple(df[id_col][row_idx] for id_col in id_columns)
        id_to_row[id_key] = row_idx
    
    # Build reordered data
    reordered_data = {col: [] for col in df.columns}
    
    for id_key in reference_ordering:
        if id_key in id_to_row:
            row_idx = id_to_row[id_key]
            for col in df.columns:
                reordered_data[col].append(df[col][row_idx])
        else:
            # This shouldn't happen if the DataFrames are consistent
            raise ValueError(f"ID {id_key} not found in DataFrame being reordered")
    
    return DataFrame(reordered_data)


def wide_df_to_masked_array(
    df: DataFrame,
    id_columns: Union[str, List[str]],
    var_pattern: str = r'([^$]+)\$(\d+)\$value',
    sort_by_var_index: bool = True
) -> MaskedArray:
    """
    Convert a wide format DataFrame to a MaskedArray.
    
    Args:
        df: Wide format DataFrame with columns like 'var$0$value', 'var$1$value', etc.
        id_columns: Column name(s) that identify each row (will be preserved)
        var_pattern: Regex pattern to extract variable indices from column names
        sort_by_var_index: Whether to sort columns by variable index (default: True)
    
    Returns:
        MaskedArray containing:
        - data: JAX array of shape (n_rows, n_variables) 
        - mask: NumPy array of shape (n_rows, n_variables) with boolean masks
        - index_df: DataFrame containing only the ID columns
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
    masks = np.ones((n_rows, n_vars), dtype=bool)  # Use numpy for masks to avoid traced boolean errors
    
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
            masks[:, i] = np.array(col_masks)  # Use numpy assignment for masks
    
    # Create ID DataFrame
    id_data = {col: df[col] for col in id_columns}
    id_df = DataFrame(id_data)
    
    return MaskedArray(data=values, mask=masks, index_df=id_df)


def masked_array_to_wide_df(
    masked_array: MaskedArray,
    var_prefix: str = 'var'
) -> DataFrame:
    """
    Convert a MaskedArray back to wide format DataFrame.
    
    Args:
        masked_array: MaskedArray containing data, mask, and index_df
        var_prefix: Prefix for variable column names (default: 'var')
    
    Returns:
        Wide format DataFrame with value and mask columns
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required for this function. Install with: pip install jax")
    
    values_array = masked_array.data
    mask_array = masked_array.mask
    id_dataframe = masked_array.index_df
    
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
        
        # Extract column data as JAX arrays to preserve computational graph
        values_col = values_array[:, var_idx]
        masks_col = mask_array[:, var_idx]
        
        # Keep values as JAX arrays to preserve computational graph
        # Keep masks as numpy arrays to avoid JAX compilation issues
        wide_data[value_col_name] = values_col
        wide_data[mask_col_name] = masks_col
    
    return DataFrame(wide_data)


def roundtrip_wide_jax_conversion(
    df: DataFrame,
    id_columns: Union[str, List[str]],
    var_pattern: str = r'([^$]+)\$(\d+)\$value',
    var_prefix: str = 'var'
) -> DataFrame:
    """
    Test roundtrip conversion: wide DataFrame -> MaskedArray -> wide DataFrame.
    
    This is a utility function for testing that the conversion process preserves data.
    
    Args:
        df: Original wide format DataFrame
        id_columns: Column name(s) that identify each row
        var_pattern: Regex pattern for value columns  
        var_prefix: Prefix for reconstructed column names
        
    Returns:
        Reconstructed wide format DataFrame
    """
    # Convert to MaskedArray
    masked_array = wide_df_to_masked_array(df, id_columns, var_pattern)
    
    # Convert back to DataFrame
    reconstructed = masked_array_to_wide_df(masked_array, var_prefix)
    
    return reconstructed
