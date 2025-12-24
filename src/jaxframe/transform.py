"""
Data transformation utilities for jaxframe DataFrames.

This module provides functions for reshaping and transforming DataFrames,
including wide-to-long format conversions with mask support.
"""

from typing import List, Tuple, Any, Union, Optional, Dict
import re
import numpy as np
from .dataframe import DataFrame
from .masked_array import MaskedArray


def _is_jax_tracer(value: Any) -> bool:
    """
    Check if a value is a JAX tracer (used during JIT compilation).
    
    JAX tracers are abstract placeholders used during tracing and cannot be
    meaningfully compared for equality at trace time.
    
    Args:
        value: Value to check
        
    Returns:
        True if the value is a JAX tracer, False otherwise
    """
    # Check by class name to avoid importing JAX (which may not be installed)
    type_name = type(value).__name__
    if 'Tracer' in type_name:
        return True
    
    # Also check module path for more robustness
    module = getattr(type(value), '__module__', '')
    if module.startswith('jax') and 'Tracer' in type_name:
        return True
    
    # Check for DynamicJaxprTracer specifically
    if 'DynamicJaxprTracer' in type_name or 'JaxprTracer' in type_name:
        return True
        
    return False


def _validate_index_subdata_columns(
    df: DataFrame,
    index_columns: List[str],
    index_subdata_columns: List[str]
) -> None:
    """
    Validate that each index_subdata column has consistent values within each index group.
    
    A valid index_subdata column is one where all rows with the same index tuple
    have identical values in that column.
    
    Note: Validation is skipped for JAX tracer values, as they cannot be meaningfully
    compared during JIT tracing. The structural correctness will be validated when
    the function is called with concrete values.
    
    Args:
        df: Source DataFrame
        index_columns: List of column names that identify each entity
        index_subdata_columns: List of column names to validate as subdata
        
    Raises:
        ValueError: If a column doesn't exist or has inconsistent values within an index group
    """
    for col in index_subdata_columns:
        if col not in df.columns:
            raise ValueError(
                f"index_subdata_columns contains '{col}' which is not in the DataFrame"
            )
    
    # Build index_tuple -> first_seen_value mapping for each subdata column
    for col in index_subdata_columns:
        index_to_value: Dict[Tuple[Any, ...], Any] = {}
        
        for row_idx in range(len(df)):
            index_tuple = tuple(df[idx_col][row_idx] for idx_col in index_columns)
            value = df[col][row_idx]
            
            # Skip validation for JAX tracers - they cannot be compared during tracing
            if _is_jax_tracer(value):
                continue
            
            # Make value comparable (handle JAX/numpy scalars)
            comparable_value = value
            if hasattr(value, 'item'):
                try:
                    comparable_value = value.item()
                except Exception:
                    pass
            
            if index_tuple in index_to_value:
                first_value = index_to_value[index_tuple]
                
                # Fast path: same object reference
                if first_value is comparable_value:
                    continue
                
                # Skip if first_value was a tracer (shouldn't happen, but be safe)
                if _is_jax_tracer(first_value):
                    continue
                
                # Compare values
                values_match = False
                try:
                    if hasattr(first_value, '__eq__'):
                        values_match = bool(first_value == comparable_value)
                    else:
                        values_match = first_value == comparable_value
                except Exception:
                    values_match = False
                
                if not values_match:
                    raise ValueError(
                        f"Column '{col}' is not valid as index_subdata: "
                        f"index {index_tuple} has conflicting values "
                        f"{first_value!r} and {comparable_value!r}"
                    )
            else:
                index_to_value[index_tuple] = comparable_value


def _create_optimized_dataframe(data: Dict[str, Any]) -> DataFrame:
    """
    Create a DataFrame using the fastest available constructor based on data types.
    
    Args:
        data: Dictionary of column name -> column data
        
    Returns:
        DataFrame created with optimized constructor
    """
    if not data:
        raise ValueError("Data dictionary cannot be empty")
    
    # Quick type detection for optimization
    first_val = next(iter(data.values()))
    
    try:
        import jax.numpy as jnp
        # Check if all values are JAX arrays
        if all(hasattr(val, '__class__') and hasattr(val, 'shape') and 
               hasattr(val, 'dtype') and str(type(val).__module__).startswith('jax') 
               for val in data.values()):
            return DataFrame.from_jax_arrays(data)
    except ImportError:
        pass
    
    # Check if all values are NumPy arrays
    if all(hasattr(val, 'dtype') and hasattr(val, 'shape') and 
           str(type(val).__module__) == 'numpy' 
           for val in data.values()):
        return DataFrame.from_numpy_arrays(data)
    
    # Fall back to regular constructor for mixed types
    return DataFrame(data)


def _reorder_by_reference(
    df: DataFrame,
    index_columns: Union[str, List[str]],
    reference_ordering: List[Any],
    *,
    stable: bool,
    drop_unreferenced: bool,
    error_on_missing: bool,
) -> DataFrame:
    """Shared reorder helper supporting both stable and strict semantics."""
    if isinstance(index_columns, str):
        index_columns = [index_columns]

    if not reference_ordering:
        return df

    def _key_for_row(row_idx: int):
        if len(index_columns) == 1:
            return df[index_columns[0]][row_idx]
        return tuple(df[index_col][row_idx] for index_col in index_columns)

    if stable:
        order_index = {val: idx for idx, val in enumerate(reference_ordering)}
        default_rank = len(order_index)

        ranked_rows = []  # (rank, row_idx)
        for row_idx in range(len(df)):
            key = _key_for_row(row_idx)
            rank = order_index.get(key, default_rank)
            if drop_unreferenced and rank == default_rank:
                continue
            ranked_rows.append((rank, row_idx))

        if not ranked_rows:
            return df

        ranked_rows.sort(key=lambda pair: pair)
        row_order = [row_idx for _, row_idx in ranked_rows]
    else:
        index_to_row: Dict[Any, int] = {}
        for row_idx in range(len(df)):
            index_to_row[_key_for_row(row_idx)] = row_idx

        row_order = []
        for ref_key in reference_ordering:
            if ref_key not in index_to_row:
                if error_on_missing:
                    raise ValueError(f"ID {ref_key} not found in DataFrame being reordered")
                continue
            row_order.append(index_to_row[ref_key])

        if not row_order:
            return df

    try:
        import jax.numpy as jnp  # type: ignore
    except ImportError:  # pragma: no cover - jax optional
        jnp = None

    reordered: Dict[str, Any] = {}
    for col in df.columns:
        col_data = df[col]
        if jnp is not None and isinstance(col_data, jnp.ndarray):
            reordered[col] = jnp.take(col_data, jnp.array(row_order))
        elif isinstance(col_data, np.ndarray):
            reordered[col] = col_data.take(row_order)
        else:
            reordered[col] = [col_data[i] for i in row_order]

    return _create_optimized_dataframe(reordered)


def _reorder_long_by_id_column(long_df: DataFrame, id_column: str, id_order: List[Any]) -> DataFrame:
    """Reorder a long-format DataFrame to match the provided ID ordering.

    The sort is stable, preserving the existing relative order within each ID
    while globally aligning IDs to the sequence observed in ``id_order``.
    """
    if id_column not in long_df.columns:
        return long_df

    return _reorder_by_reference(
        long_df,
        id_column,
        id_order,
        stable=True,
        drop_unreferenced=False,
        error_on_missing=False,
    )


def _apply_skeleton_order(
    long_df: DataFrame,
    skeleton_df: DataFrame,
    skeleton_id_col: str,
    index_columns: Union[str, List[str]],
    value_name: Optional[str],
    var_name: Union[str, List[str]],
    order_column: Optional[str] = None,
) -> DataFrame:
    """Align rows to match a provided skeleton and attach its ID column.

    We match rows on index columns plus a disambiguating column (prefer ``value_name``
    when present in both frames; otherwise fall back to ``var_name`` when present).
    If a skeleton row cannot be matched, a ValueError is raised. When no
    disambiguator is available, we rely solely on the index columns and consume
    matches in skeleton order (queue semantics) to remain stable.
    
    When values are JAX traced, we skip them as disambiguators since they can't be hashed.
    If order_column is provided and value is traced, we use the order column to 
    restore original row order within each index group before matching.
    """
    if isinstance(index_columns, str):
        index_columns = [index_columns]

    # Check if value column contains traced values (can't use for hashing)
    value_is_traced = False
    if value_name and value_name in long_df.columns:
        if len(long_df) > 0:
            first_val = long_df[value_name][0]
            value_is_traced = _is_jax_traced(first_val)
    
    # Check if order column is traced too
    order_is_traced = False
    if order_column and order_column in long_df.columns and len(long_df) > 0:
        first_order = long_df[order_column][0]
        order_is_traced = _is_jax_traced(first_order)
    
    # If values are traced but order column is NOT traced, we can use it to re-sort in Python
    if value_is_traced and order_column and order_column in long_df.columns and not order_is_traced:
        # Sort long_df by index columns + order column to restore original order
        # Build sort key: (index_tuple, order_value) for each row
        sort_keys = []
        for i in range(len(long_df)):
            index_tuple = tuple(long_df[col][i] for col in index_columns)
            order_val = long_df[order_column][i]
            if hasattr(order_val, 'item'):
                order_val = order_val.item()
            sort_keys.append((index_tuple, order_val, i))
        
        # Sort by index tuple first, then by order value
        sorted_keys = sorted(sort_keys, key=lambda x: (x[0], x[1]))
        reorder_indices = [k[2] for k in sorted_keys]
        
        # Reorder long_df
        try:
            import jax.numpy as jnp
        except ImportError:
            jnp = None
        
        reordered_data = {}
        for col in long_df.columns:
            col_data = long_df[col]
            if jnp is not None and (isinstance(col_data, jnp.ndarray) or _is_jax_traced(col_data[0] if len(col_data) > 0 else None)):
                # For JAX arrays, use jnp.take
                if hasattr(col_data, '__len__'):
                    col_array = jnp.stack([col_data[i] for i in range(len(col_data))])
                    reordered_data[col] = jnp.take(col_array, jnp.array(reorder_indices))
            elif isinstance(col_data, np.ndarray):
                reordered_data[col] = col_data.take(reorder_indices)
            else:
                reordered_data[col] = [col_data[i] for i in reorder_indices]
        
        long_df = _create_optimized_dataframe(reordered_data)
    elif value_is_traced and order_is_traced:
        # Both value and order are traced - use JAX operations for reordering
        # We'll defer the reordering to use JAX gather at the final step
        # Mark that we need JAX-based skeleton matching
        pass  # We'll handle this in the final reorder step below

    # Determine matching columns
    join_cols: List[str] = []
    for col in index_columns:
        if col not in long_df.columns or col not in skeleton_df.columns:
            raise ValueError(f"Index column '{col}' must exist in both long_df and long_skeleton_df")
        join_cols.append(col)

    # Choose disambiguator - prefer 'variable' column (from create_unpivot_skeleton),
    # then fall back to value_name or var_name
    disambiguator: Optional[str] = None
    if 'variable' in skeleton_df.columns and 'variable' in long_df.columns:
        # Skeleton was created with create_unpivot_skeleton - use variable column
        disambiguator = 'variable'
    elif value_name and value_name in long_df.columns and value_name in skeleton_df.columns and not value_is_traced:
        disambiguator = value_name
    elif isinstance(var_name, str) and var_name in long_df.columns and var_name in skeleton_df.columns:
        disambiguator = var_name
    
    join_cols_with_disambiguator = join_cols + ([disambiguator] if disambiguator else [])

    def _as_hashable(val: Any) -> Any:
        try:
            import numpy as _np
        except ImportError:  # pragma: no cover
            _np = np

        if hasattr(val, "shape") and getattr(val, "ndim", 0) == 0:
            try:
                return _np.asarray(val).item()
            except Exception:
                return val
        if isinstance(val, (list, tuple)):
            return tuple(_as_hashable(v) for v in val)
        if isinstance(val, np.ndarray):
            if val.shape == ():
                return val.item()
            return tuple(_as_hashable(v) for v in val.tolist())
        return val

    def _row_key(frame: DataFrame, idx: int) -> Tuple[Any, ...]:
        return tuple(_as_hashable(frame[col][idx]) for col in join_cols_with_disambiguator)

    temp_occurrence_col = "__tmp_skel_occ__"

    def _add_occurrence_column(frame: DataFrame) -> DataFrame:
        counts: Dict[Tuple[Any, ...], int] = {}
        occ_values: List[int] = []
        for i in range(len(frame)):
            key = _row_key(frame, i)
            occ = counts.get(key, 0)
            occ_values.append(occ)
            counts[key] = occ + 1
        return frame.add_column(temp_occurrence_col, occ_values)

    long_df_with_occ = _add_occurrence_column(long_df)
    skeleton_df_with_occ = _add_occurrence_column(skeleton_df)

    def _row_key_with_occ(frame: DataFrame, idx: int) -> Tuple[Any, ...]:
        return _row_key(frame, idx) + (_as_hashable(frame[temp_occurrence_col][idx]),)

    # Handle JAX traced order column case: use order column values to map back to skeleton
    # In this case, long_df rows are in sorted order, order column tells us original position
    try:
        import jax.numpy as jnp  # type: ignore
    except ImportError:  # pragma: no cover - jax optional
        jnp = None

    if value_is_traced and order_is_traced and order_column and jnp is not None:
        # JAX-compatible skeleton matching using order column
        # We need to build a gather index that maps skeleton rows to produced rows
        # using JAX operations on the traced order column
        
        # Step 1: Build structural mappings (not traced)
        # For skeleton: (group_id, original_position) for each row
        # For long_df: group_id for each row
        
        skel_group_map = {}  # index_tuple -> group_id
        skel_row_to_group = []
        skel_row_to_pos_in_group = []
        skel_group_counters = {}
        
        for i in range(len(skeleton_df)):
            key = tuple(skeleton_df[col][i] for col in index_columns)
            if key not in skel_group_map:
                skel_group_map[key] = len(skel_group_map)
                skel_group_counters[key] = 0
            skel_row_to_group.append(skel_group_map[key])
            skel_row_to_pos_in_group.append(skel_group_counters[key])
            skel_group_counters[key] += 1
        
        prod_group_map = {}  # index_tuple -> group_id (same mapping for consistency)
        prod_row_to_group = []
        prod_group_counters = {}
        prod_group_sizes = {}
        
        for i in range(len(long_df)):
            key = tuple(long_df[col][i] for col in index_columns)
            if key not in prod_group_map:
                prod_group_map[key] = skel_group_map.get(key, len(skel_group_map) + len(prod_group_map))
                prod_group_counters[key] = 0
                prod_group_sizes[key] = 0
            prod_row_to_group.append(prod_group_map[key])
            prod_group_counters[key] += 1
            prod_group_sizes[key] = prod_group_counters[key]
        
        n_groups = len(skel_group_map)
        max_group_size = max(skel_group_counters.values()) if skel_group_counters else 0
        n_prod_rows = len(long_df)
        
        # Step 2: Build padded arrays for produced rows by group
        # prod_group_row_indices[group, position_in_produced] = global_produced_row_idx
        prod_group_row_indices = np.full((n_groups, max_group_size), n_prod_rows, dtype=np.int32)
        group_fill_counters = [0] * n_groups
        for i in range(n_prod_rows):
            g = prod_row_to_group[i]
            if g < n_groups:  # Only process groups that exist in skeleton
                pos = group_fill_counters[g]
                if pos < max_group_size:
                    prod_group_row_indices[g, pos] = i
                group_fill_counters[g] += 1
        
        # Step 3: Get order column values as JAX array
        order_col_data = long_df[order_column]
        if hasattr(order_col_data, '__len__') and len(order_col_data) > 0:
            order_list = [order_col_data[i] for i in range(len(order_col_data))]
            # Add padding value for out-of-bounds indices
            order_list.append(jnp.array(max_group_size + 1, dtype=order_list[0].dtype) if order_list else 0)
            order_array_padded = jnp.stack(order_list)
        else:
            order_array_padded = jnp.array([max_group_size + 1])
        
        # Step 4: For each group, gather order values and compute inverse mapping
        # order_in_group[g, i] = order value at produced row (g, i)
        order_in_group = order_array_padded[prod_group_row_indices]  # (n_groups, max_group_size)
        
        # argsort gives us: sorted_positions[g, i] = position that has the i-th smallest order value
        # We want: for original_position p, which produced row has order==p?
        # This is argsort of the order values
        inverse_order = jnp.argsort(order_in_group, axis=1)  # (n_groups, max_group_size)
        
        # Step 5: Build gather indices for each skeleton row
        # For skeleton row i in group g with original_position p:
        # gather from produced row at prod_group_row_indices[g, inverse_order[g, p]]
        skel_row_to_group_arr = np.array(skel_row_to_group)
        skel_row_to_pos_arr = np.array(skel_row_to_pos_in_group)
        
        # Two-step gather using JAX: first get the position within group, then the global row
        # pos_in_prod[skel_i] = inverse_order[skel_group[skel_i], skel_pos[skel_i]]
        pos_in_prod = inverse_order[skel_row_to_group_arr, skel_row_to_pos_arr]
        
        # global_prod_row[skel_i] = prod_group_row_indices[skel_group[skel_i], pos_in_prod[skel_i]]
        # Since prod_group_row_indices is numpy, convert or use jax
        prod_group_row_indices_jax = jnp.array(prod_group_row_indices)
        row_order_jax = prod_group_row_indices_jax[skel_row_to_group_arr, pos_in_prod]
        
        # Step 6: Reorder data using JAX gather
        reordered: Dict[str, Any] = {}
        for col in long_df.columns:
            col_data = long_df[col]
            is_traced = len(col_data) > 0 and _is_jax_traced(col_data[0])
            is_jax_array = isinstance(col_data, jnp.ndarray)
            
            # Check if data is numeric (can be converted to JAX array)
            is_numeric = False
            if len(col_data) > 0:
                first_elem = col_data[0]
                if is_traced or is_jax_array:
                    is_numeric = True
                elif isinstance(first_elem, (int, float, np.integer, np.floating)):
                    is_numeric = True
                elif isinstance(first_elem, np.ndarray) and first_elem.dtype.kind in 'iufcb':
                    is_numeric = True
            
            if is_traced or is_jax_array:
                # Stack into JAX array and pad for safety
                col_list = [col_data[i] for i in range(len(col_data))]
                # Add a dummy padding element for out-of-bounds
                if col_list:
                    col_list.append(jnp.zeros_like(col_list[0]))
                    col_array = jnp.stack(col_list)
                    reordered[col] = col_array[row_order_jax]
            elif isinstance(col_data, np.ndarray) and col_data.dtype.kind in 'iufcb':
                # Numeric numpy arrays - use JAX gather
                col_array = jnp.array(col_data)
                reordered[col] = col_array[row_order_jax]
            else:
                # Non-numeric data (strings, etc.) - we need concrete indices
                # The row_order structure IS deterministic based on skeleton structure
                # But row_order_jax contains traced values from argsort(order)
                # 
                # For non-numeric columns, we need to use the index columns
                # which ARE structural. The reordering is determined by skeleton.
                # We can't use row_order_jax since it's traced, but we CAN skip
                # these columns if they're redundant (e.g., index columns are in skeleton)
                # 
                # For index columns: copy from skeleton (same values)
                if col in index_columns:
                    reordered[col] = [skeleton_df[col][i] for i in range(len(skeleton_df))]
                else:
                    # For other non-numeric columns (like variable name columns)
                    # These should match skeleton order since they're structural
                    # Copy from skeleton if available, otherwise raise
                    if col in skeleton_df.columns:
                        reordered[col] = [skeleton_df[col][i] for i in range(len(skeleton_df))]
                    else:
                        # Non-numeric, non-index column that's not in skeleton
                        # This is tricky - we'd need concrete indices
                        # For now, just use position-based matching (rows in same group order)
                        # This works if the only difference is value ordering
                        if isinstance(col_data, np.ndarray):
                            reordered[col] = list(col_data)
                        else:
                            reordered[col] = list(col_data)
        
        skeleton_ids = [skeleton_df_with_occ[skeleton_id_col][i] for i in range(len(skeleton_df))]
        reordered[skeleton_id_col] = skeleton_ids
        return _create_optimized_dataframe(reordered)

    # Standard (non-traced order) path: Map result rows by unique key
    key_to_row: Dict[Tuple[Any, ...], int] = {}
    for i in range(len(long_df_with_occ)):
        key = _row_key_with_occ(long_df_with_occ, i)
        key_to_row[key] = i

    # Build row order following skeleton, matching on occurrence index.
    row_order: List[int] = []
    skeleton_ids: List[Any] = []
    for i in range(len(skeleton_df_with_occ)):
        key = _row_key_with_occ(skeleton_df_with_occ, i)
        if key not in key_to_row:
            raise ValueError(f"Skeleton row with key {key} not found in produced long DataFrame")

        row_order.append(key_to_row[key])
        skeleton_ids.append(skeleton_df_with_occ[skeleton_id_col][i])
    
    # Reorder data and attach skeleton id column
    reordered: Dict[str, Any] = {}
    try:
        import jax.numpy as jnp  # type: ignore
    except ImportError:  # pragma: no cover - jax optional
        jnp = None

    for col in long_df.columns:
        col_data = long_df[col]
        if jnp is not None and isinstance(col_data, jnp.ndarray):
            reordered[col] = jnp.take(col_data, jnp.array(row_order))
        elif isinstance(col_data, np.ndarray):
            reordered[col] = col_data.take(row_order)
        else:
            reordered[col] = [col_data[i] for i in row_order]

    reordered[skeleton_id_col] = skeleton_ids
    return _create_optimized_dataframe(reordered)


def wide_to_long_masked(
    df: Union[DataFrame, List[DataFrame]], 
    index_columns: Union[str, List[str]], 
    var_pattern: str = r'([^$]+)\$(\d+)\$(value|mask|order)',
    var_name: Union[str, List[str]] = 'variable',
    value_name: Union[str, List[str], None] = 'value',
    order_value_name: Union[str, List[str], None] = None,
    long_skeleton_df: DataFrame = None,
    long_skeleton_id_column: str = None
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
        index_columns: Column name(s) that identify each row (will be preserved in long format)
        var_pattern: Regex pattern to parse column names. Should have 3 groups:
                    (variable_name, time_index, column_type)
                    Default matches patterns like 'time$0$value', 'time$1$mask', etc.
        var_name: Name(s) for the variable column(s) in long format. Can be:
                 - Single string for single DataFrame input
                 - List of strings for multiple DataFrame input (must match df list length)
        value_name: Name(s) for the value column(s) in long format. Can be:
               - Single string for single DataFrame input  
                   - List of strings for multiple DataFrame input (must match df list length)
                   - None (default) to automatically extract names from DataFrame variable patterns
        order_value_name: Optional name(s) for the column that stores the original
                  ordering of each observation (when available). If None,
                  defaults to `<value_name>_order` for both single and
                  multi-DataFrame inputs.
        long_skeleton_df: Optional long-format DataFrame that carries the desired
                  row ordering. Must contain ``long_skeleton_id_column``
                  and any ``index_columns`` used here.
        long_skeleton_id_column: Column name in ``long_skeleton_df`` used to derive
                  the row ordering for the returned long DataFrame.
    
    Returns:
        DataFrame in long format with index_columns and the specified variable/value columns
        
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
    # Validate skeleton inputs early
    if long_skeleton_df is not None and long_skeleton_id_column is None:
        raise ValueError("long_skeleton_id_column must be provided when long_skeleton_df is supplied")

    if long_skeleton_df is not None and long_skeleton_id_column not in long_skeleton_df.columns:
        raise ValueError(f"long_skeleton_id_column '{long_skeleton_id_column}' not found in long_skeleton_df")

    # Skeleton ordering is currently supported only for single-DataFrame input
    if long_skeleton_df is not None and isinstance(df, list):
        raise ValueError("long_skeleton_df ordering is only supported when df is a single DataFrame")

    # Handle single DataFrame case (backward compatibility)
    if not isinstance(df, list):
        single_order_name: Optional[str]
        if isinstance(order_value_name, list):
            single_order_name = order_value_name[0] if order_value_name else None
        else:
            single_order_name = order_value_name

        # Defer defaulting to `_single_wide_to_long_masked` so it can use the
        # resolved `value_name` to build `<value_name>_order` consistently.
        result = _single_wide_to_long_masked(
            df,
            index_columns,
            var_pattern,
            var_name,
            value_name,
            order_value_name=single_order_name
        )

        if long_skeleton_df is not None:
            result = _apply_skeleton_order(
                result,
                long_skeleton_df,
                long_skeleton_id_column,
                index_columns,
                value_name if isinstance(value_name, str) else value_name[0] if value_name else None,
                var_name,
                order_column=single_order_name,  # Pass order column for traced value handling
            )
        return result
    
    # Handle multiple DataFrames case
    df_list = df
    
    # Validate parameter types for multi-DataFrame case
    if isinstance(df, list) and value_name is not None and not isinstance(value_name, list):
        raise ValueError(f"When df is a list of DataFrames, value_name must be a list or None, "
                        f"got {type(value_name).__name__}")
    
    # Extract default value names from DataFrames if value_name is 'value' (default)
    if value_name == 'value' and isinstance(df, list):
        # For multi-DataFrame case, extract meaningful names
        value_name = []
        for i, df_single in enumerate(df_list):
            # Extract variable name from the first value column found
            pattern = re.compile(var_pattern)
            for col in df_single.columns:
                match = pattern.match(str(col))
                if match and match.group(3) == 'value':
                    var_base_name = match.group(1)
                    value_name.append(f"{var_base_name}_value")
                    break
            else:
                # If no matching pattern found, use generic name
                value_name.append(f"value_{i}")
    elif value_name is None:
        # If explicitly set to None, extract meaningful names
        if isinstance(df, list):
            value_name = []
            for i, df_single in enumerate(df_list):
                pattern = re.compile(var_pattern)
                for col in df_single.columns:
                    match = pattern.match(str(col))
                    if match and match.group(3) == 'value':
                        var_base_name = match.group(1)
                        value_name.append(f"{var_base_name}_value")
                        break
                else:
                    value_name.append(f"value_{i}")
        else:
            # Single DataFrame with None -> extract meaningful name
            value_name = None  # Let _single_wide_to_long_masked handle it
    
    # Validate inputs
    if isinstance(var_name, str):
        var_name = [var_name] * len(df_list)
    if isinstance(value_name, str):
        value_name = [value_name] * len(df_list)

    if isinstance(order_value_name, list):
        order_value_names = order_value_name
    else:
        order_value_names = [order_value_name] * len(df_list)
    
    if len(var_name) != len(df_list):
        raise ValueError(f"var_name list length ({len(var_name)}) must match df list length ({len(df_list)})")
    if len(value_name) != len(df_list):
        raise ValueError(f"value_name list length ({len(value_name)}) must match df list length ({len(df_list)})")
    if len(order_value_names) != len(df_list):
        raise ValueError(
            f"order_value_name list length ({len(order_value_names)}) must match df list length ({len(df_list)})"
        )

    order_value_names = list(order_value_names)
    for i in range(len(order_value_names)):
        if order_value_names[i] is None:
            base_name = value_name[i] if value_name[i] is not None else f"value_{i}"
            order_value_names[i] = f"{base_name}_order"
    
    # Convert each DataFrame to long format separately
    long_dfs = []
    for i, (df_single, var_n, val_n) in enumerate(zip(df_list, var_name, value_name)):
        long_df = _single_wide_to_long_masked(
            df_single,
            index_columns,
            var_pattern,
            var_n,
            val_n,
            order_value_name=order_value_names[i]
        )
        long_dfs.append(long_df)
    
    # Merge all long DataFrames on index_columns
    if len(long_dfs) == 1:
        return long_dfs[0]
    
    # Start with the first DataFrame and join the rest
    result = long_dfs[0]
    for i in range(1, len(long_dfs)):
        join_columns = index_columns if isinstance(index_columns, list) else [index_columns]
        join_columns = join_columns + [var_name[0]]

        right_df = long_dfs[i]
        if var_name[i] != var_name[0]:
            right_df = right_df.rename({var_name[i]: var_name[0]})
        
        result = result.join(right_df, on=join_columns)

    # Multi-DataFrame path does not use skeleton ordering (guarded earlier)
    return result


def _single_wide_to_long_masked(
    df: DataFrame, 
    index_columns: Union[str, List[str]], 
    var_pattern: str = r'([^$]+)\$(\d+)\$(value|mask|order)',
    var_name: str = 'variable',
    value_name: Union[str, None] = 'value',
    order_value_name: Optional[str] = None
) -> DataFrame:
    """
    Convert a single wide format DataFrame to long format (internal helper function).
    
    This is the original implementation for single DataFrame conversion with optional
    propagation of original ordering metadata stored in `$order` columns.
    """
    # Handle default value_name
    if value_name is None:
        pattern = re.compile(var_pattern)
        for col in df.columns:
            match = pattern.match(str(col))
            if match and match.group(3) == 'value':
                var_base_name = match.group(1)
                value_name = f"{var_base_name}_value"
                break
        else:
            # If no matching pattern found, use generic name
            value_name = 'value'

    # Default order column naming aligns with multi-DataFrame behavior
    if order_value_name is None:
        order_value_name = f"{value_name}_order"
    
    # Ensure index_columns is a list
    if isinstance(index_columns, str):
        index_columns = [index_columns]
    
    # Parse column names to identify value and mask columns
    value_columns = {}  # {(var_name, time_index): column_name}
    mask_columns = {}   # {(var_name, time_index): column_name}
    order_columns = {}  # {(var_name, time_index): column_name}
    
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
            elif col_type == 'order':
                order_columns[key] = col
    
    # Check if any value column contains JAX traced values
    has_traced_values = False
    for key, col_name in value_columns.items():
        if len(df) > 0:
            first_val = df[col_name][0]
            if _is_jax_traced(first_val):
                has_traced_values = True
                break
    
    # Use JAX path for traced values
    if has_traced_values:
        return _single_wide_to_long_masked_jax(
            df, index_columns, value_name, var_pattern, var_name, value_name, order_value_name
        )
    
    # Build long format data
    long_data = {col: [] for col in index_columns}
    long_data[var_name] = []
    long_data[value_name] = []
    include_order_data = order_value_name is not None and len(order_columns) > 0
    if include_order_data:
        long_data[order_value_name] = []
    
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
                for index_col in index_columns:
                    long_data[index_col].append(df[index_col][row_idx])
                
                long_data[var_name].append(time_index)
                long_data[value_name].append(df[value_col][row_idx])
                if include_order_data:
                    if key in order_columns:
                        order_val = df[order_columns[key]][row_idx]
                    else:
                        order_val = time_index
                    long_data[order_value_name].append(order_val)
    
    return _create_optimized_dataframe(long_data)


def long_to_wide_masked(
    df: DataFrame,
    index_columns: Union[str, List[str]],
    value_column: Union[str, List[str]],
    var_column: Optional[Union[str, List[str]]] = None,
    var_prefix: Union[str, List[str]] = 'var',
    fill_type: Union[Any, str, List[Union[Any, str]]] = 0.0,
    mask_value: bool = False,
    sort_within_id: Union[bool, List[bool]] = False,
    order_suffix: Optional[str] = 'order',
    index_subdata_columns: Optional[List[str]] = None
) -> Union[DataFrame, List[DataFrame]]:
    """
    Convert a long format DataFrame to wide format with mask columns.
    
    This function can handle either:
    1. Single column conversion (original behavior) 
    2. Multiple column conversion where multiple variables are processed simultaneously
    
    Args:
        df: Input DataFrame in long format
        index_columns: Column name(s) that identify each entity
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
        sort_within_id: Whether to sort each entity's observations by their value
                before assigning them to wide columns. Accepts a bool or
                list matching `value_column` length.
        order_suffix: Suffix used for the additional column that records the
                  original position of each observation when
                  `sort_within_id` is enabled. Set to None to skip creating
                  these metadata columns.
        index_subdata_columns: Optional list of column names in the DataFrame that
                  should be carried forward to the wide format. These columns must
                  have consistent values within each index group (all rows with the
                  same index tuple must have identical values). The first value
                  encountered for each entity will be used in the output.
    
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
        single_sort_flag: bool
        if isinstance(sort_within_id, list):
            single_sort_flag = bool(sort_within_id[0]) if sort_within_id else False
        else:
            single_sort_flag = bool(sort_within_id)

        return _single_long_to_wide_masked(
            df,
            index_columns,
            value_column,
            var_column,
            var_prefix,
            fill_type,
            mask_value,
            sort_within_id=single_sort_flag,
            order_suffix=order_suffix,
            index_subdata_columns=index_subdata_columns
        )
    
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

    if isinstance(sort_within_id, list):
        sort_flags = [bool(flag) for flag in sort_within_id]
    else:
        sort_flags = [bool(sort_within_id)] * len(value_columns)
    
    # Validate input lengths
    if len(var_columns) != len(value_columns):
        raise ValueError(f"var_column list length ({len(var_columns)}) must match value_column length ({len(value_columns)})")
    if len(var_prefixes) != len(value_columns):
        raise ValueError(f"var_prefix list length ({len(var_prefixes)}) must match value_column length ({len(value_columns)})")
    if len(fill_types) != len(value_columns):
        raise ValueError(f"fill_type list length ({len(fill_types)}) must match value_column length ({len(value_columns)})")
    if len(sort_flags) != len(value_columns):
        raise ValueError(f"sort_within_id list length ({len(sort_flags)}) must match value_column length ({len(value_columns)})")
    
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
    
    for i, (val_col, var_col, var_pref, fill_t, sort_flag) in enumerate(
        zip(value_columns, var_columns, var_prefixes, fill_types, sort_flags)
    ):
        wide_df = _single_long_to_wide_masked(
            df,
            index_columns,
            val_col,
            var_col,
            var_pref,
            fill_t,
            mask_value,
            sort_within_id=sort_flag,
            order_suffix=order_suffix,
            index_subdata_columns=index_subdata_columns if i == 0 else None  # Only add to first DataFrame
        )
        
        if i == 0:
            # First DataFrame establishes the reference ordering
            if isinstance(index_columns, str):
                reference_ordering = [wide_df.get_row(j)[index_columns] for j in range(len(wide_df))]
            else:
                reference_ordering = [tuple(wide_df.get_row(j)[col] for col in index_columns) for j in range(len(wide_df))]
            wide_dfs.append(wide_df)
        else:
            # Reorder subsequent DataFrames to match the reference ordering
            reordered_df = _reorder_dataframe_by_ids(wide_df, index_columns, reference_ordering)
            wide_dfs.append(reordered_df)
    
    return wide_dfs


class PivotStructure:
    """
    Pre-computed structure for pivoting operations.
    
    This class separates the "structural" computation (which source rows map to
    which destination positions) from the "value" computation. This allows the
    value placement to be done with JAX-compatible array operations, enabling
    JIT compilation through pivot operations.
    
    The structure is computed from the index columns and optionally a variable
    column, none of which should be JAX traced values.
    
    Attributes:
        n_source_rows: Number of rows in the source (long) DataFrame
        n_dest_rows: Number of unique index combinations (entities)
        n_dest_cols: Number of variable slots per entity
        source_to_dest_row: Array mapping source row index to destination row index
        source_to_dest_col: Array mapping source row index to destination column index
        dest_fill_mask: Boolean array of shape (n_dest_rows, n_dest_cols) indicating
            which positions need to be filled (have no source data)
        unique_ids: List of unique index tuples in order
        unique_vars: List of unique variable indices/names
        sort_permutation: If sort_within_id is True, array of original positions
            for each source element after sorting
        requires_jax_sort: If True, sorting should be done with JAX operations
        entity_source_indices: For JAX sorting, maps (entity, slot) -> source row
        entity_source_mask: For JAX sorting, boolean mask of valid positions
    """
    
    def __init__(
        self,
        df: DataFrame,
        index_columns: Union[str, List[str]],
        var_column: Optional[str] = None,
        sort_within_id: bool = False,
        value_column_for_sort: Optional[str] = None,
        use_jax_sort: bool = False
    ):
        """
        Compute the pivot structure from a DataFrame.
        
        Args:
            df: Source DataFrame in long format
            index_columns: Column name(s) that identify each entity
            var_column: Optional column containing variable indices/names.
                If None, indices are assigned based on order within each entity.
            sort_within_id: Whether to sort observations within each entity
                before assigning to slots
            value_column_for_sort: Column to use for sorting when sort_within_id=True
                and use_jax_sort=False
            use_jax_sort: If True, prepare structure for JAX-based sorting
                (sorting will happen later with JAX operations)
        """
        if isinstance(index_columns, str):
            index_columns = [index_columns]
        
        self.index_columns = list(index_columns)
        self.n_source_rows = len(df)
        self.requires_jax_sort = sort_within_id and use_jax_sort
        
        # Get unique IDs in order of first appearance
        unique_ids = []
        index_tuples_seen = set()
        row_to_index_tuple = []
        
        for row_idx in range(len(df)):
            index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
            row_to_index_tuple.append(index_tuple)
            if index_tuple not in index_tuples_seen:
                unique_ids.append(index_tuple)
                index_tuples_seen.add(index_tuple)
        
        self.unique_ids = unique_ids
        self.n_dest_rows = len(unique_ids)
        id_to_dest_row = {id_tuple: i for i, id_tuple in enumerate(unique_ids)}
        
        # Track first row index for each entity (useful for index_subdata extraction)
        self.entity_first_row = np.zeros(self.n_dest_rows, dtype=np.int32)
        entity_first_seen = set()
        for row_idx in range(len(df)):
            index_tuple = row_to_index_tuple[row_idx]
            if index_tuple not in entity_first_seen:
                dest_row = id_to_dest_row[index_tuple]
                self.entity_first_row[dest_row] = row_idx
                entity_first_seen.add(index_tuple)
        
        # Determine variable indices
        if var_column is not None:
            unique_vars = sorted(set(df[var_column]))
            var_to_col = {var: i for i, var in enumerate(unique_vars)}
            self.unique_vars = unique_vars
            self.n_dest_cols = len(unique_vars)
        else:
            # Count max observations per entity to determine n_dest_cols
            entity_counts = {}
            for index_tuple in row_to_index_tuple:
                entity_counts[index_tuple] = entity_counts.get(index_tuple, 0) + 1
            max_observations = max(entity_counts.values()) if entity_counts else 0
            self.unique_vars = list(range(max_observations))
            self.n_dest_cols = max_observations
        
        # Build source-to-destination mappings
        source_to_dest_row = np.zeros(self.n_source_rows, dtype=np.int32)
        source_to_dest_col = np.zeros(self.n_source_rows, dtype=np.int32)
        sort_permutation = np.zeros(self.n_source_rows, dtype=np.int32) if (sort_within_id and not use_jax_sort) else None
        
        # Track which destination positions are filled
        dest_has_data = np.zeros((self.n_dest_rows, self.n_dest_cols), dtype=bool)
        
        # Initialize JAX sort structures
        self.entity_source_indices = None
        self.entity_source_mask = None
        
        if var_column is not None:
            # Use provided variable column for column assignment
            for row_idx in range(len(df)):
                index_tuple = row_to_index_tuple[row_idx]
                dest_row = id_to_dest_row[index_tuple]
                var_value = df[var_column][row_idx]
                dest_col = var_to_col[var_value]
                
                source_to_dest_row[row_idx] = dest_row
                source_to_dest_col[row_idx] = dest_col
                dest_has_data[dest_row, dest_col] = True
        elif sort_within_id and use_jax_sort:
            # Prepare structure for JAX-based sorting
            # Build entity membership mapping (static - only depends on index columns)
            source_to_entity = np.array([
                id_to_dest_row[row_to_index_tuple[i]] 
                for i in range(len(df))
            ], dtype=np.int32)
            
            # Create padded entity->source mapping
            # entity_source_indices[entity, slot] = source row index (or n_source_rows for padding)
            # We use n_source_rows as the padding index (will point to a fill value when gathering)
            entity_source_indices = np.full(
                (self.n_dest_rows, self.n_dest_cols), 
                self.n_source_rows,  # Padding index
                dtype=np.int32
            )
            entity_source_mask = np.zeros((self.n_dest_rows, self.n_dest_cols), dtype=bool)
            
            entity_counters = [0] * self.n_dest_rows
            for src_idx in range(len(df)):
                entity_idx = source_to_entity[src_idx]
                slot = entity_counters[entity_idx]
                entity_source_indices[entity_idx, slot] = src_idx
                entity_source_mask[entity_idx, slot] = True
                entity_counters[entity_idx] += 1
                dest_has_data[entity_idx, slot] = True
            
            self.entity_source_indices = entity_source_indices
            self.entity_source_mask = entity_source_mask
            
            # For JAX sort, we still need basic mappings for the unsorted case
            # These will be recomputed after sorting
            entity_col_counters = {id_tuple: 0 for id_tuple in unique_ids}
            for row_idx in range(len(df)):
                index_tuple = row_to_index_tuple[row_idx]
                dest_row = id_to_dest_row[index_tuple]
                dest_col = entity_col_counters[index_tuple]
                entity_col_counters[index_tuple] += 1
                source_to_dest_row[row_idx] = dest_row
                source_to_dest_col[row_idx] = dest_col
                
        elif sort_within_id and value_column_for_sort is not None:
            # Sort within each entity by the value column (Python-based sorting)
            # First, check if the sort column contains traced values
            if len(df) > 0:
                first_val = df[value_column_for_sort][0]
                if _is_jax_traced(first_val):
                    raise ValueError(
                        f"Cannot sort by column '{value_column_for_sort}' inside JAX JIT because "
                        f"it contains traced values. When using sort_within_index_group=True with "
                        f"JAX JIT, the sort column must contain concrete (non-traced) values. "
                        f"Options: (1) Use sort_within_index_group=False, or "
                        f"(2) Ensure the sort column is not derived from JIT inputs."
                    )
            
            entity_rows = {id_tuple: [] for id_tuple in unique_ids}
            for row_idx in range(len(df)):
                index_tuple = row_to_index_tuple[row_idx]
                value = df[value_column_for_sort][row_idx]
                # Handle JAX arrays - extract scalar value for sorting
                if hasattr(value, 'item'):
                    value = value.item()
                entity_rows[index_tuple].append((value, row_idx))
            
            for id_tuple, rows in entity_rows.items():
                dest_row = id_to_dest_row[id_tuple]
                # Sort by value, record original positions
                sorted_rows = sorted(enumerate(rows), key=lambda x: x[1][0])
                for dest_col, (orig_pos, (value, row_idx)) in enumerate(sorted_rows):
                    source_to_dest_row[row_idx] = dest_row
                    source_to_dest_col[row_idx] = dest_col
                    dest_has_data[dest_row, dest_col] = True
                    if sort_permutation is not None:
                        sort_permutation[row_idx] = orig_pos
        else:
            # Assign columns based on order of appearance within each entity
            entity_col_counters = {id_tuple: 0 for id_tuple in unique_ids}
            for row_idx in range(len(df)):
                index_tuple = row_to_index_tuple[row_idx]
                dest_row = id_to_dest_row[index_tuple]
                dest_col = entity_col_counters[index_tuple]
                entity_col_counters[index_tuple] += 1
                
                source_to_dest_row[row_idx] = dest_row
                source_to_dest_col[row_idx] = dest_col
                dest_has_data[dest_row, dest_col] = True
        
        self.source_to_dest_row = source_to_dest_row
        self.source_to_dest_col = source_to_dest_col
        self.dest_fill_mask = ~dest_has_data  # True where we need fill values
        self.sort_permutation = sort_permutation


def create_unpivot_skeleton(
    df: DataFrame,
    index_columns: Union[str, List[str]],
    value_column: str,
    *,
    id_column: Optional[str] = None,
    sort_within_index_group: bool = False,
    var_column: Optional[str] = None,
) -> DataFrame:
    """
    Create a skeleton DataFrame for unpivot operations.
    
    When you pivot data (long → wide) and then unpivot (wide → long), the rows
    may come back in a different order. This function creates a skeleton that
    captures the original row order and a "variable" column that maps each
    original row to its wide-format slot.
    
    The returned skeleton can be passed to ``unpivot_sparse`` via the
    ``long_skeleton_df`` parameter to restore original row ordering.
    
    Args:
        df: Original long-format DataFrame (before pivoting)
        index_columns: Column(s) that identify each entity/group
        value_column: Column containing values that will be pivoted.
            Used to determine sort order when ``sort_within_index_group=True``.
        id_column: Optional primary key column to include in the skeleton.
            If provided, this column will be preserved and can be used as
            ``long_skeleton_id_column`` in ``unpivot_sparse``.
        sort_within_index_group: Whether values are sorted within each group
            during pivoting. Must match the setting used in ``pivot_sparse``.
        var_column: Optional column specifying the variable index for each row.
            If None, indices are assigned by position within each group.
    
    Returns:
        DataFrame with columns:
        
        - All ``index_columns``
        - ``'variable'``: The wide-format slot index for each row
        - ``id_column`` (if provided): The primary key values
    
    Note:
        If ``value_column`` contains JAX tracers, the function uses a position-based
        approach that doesn't require concrete values, since the variable
        assignment is purely structural when values can't be sorted.
    
    Example:
        >>> # Original data
        >>> long_df = DataFrame({
        ...     'id_meas': ['01', '02', '03'],
        ...     'id_person': ['A', 'A', 'B'],
        ...     'value': jnp.array([1.0, 2.0, 3.0])
        ... })
        >>> 
        >>> # Create skeleton before pivoting
        >>> skeleton = create_unpivot_skeleton(
        ...     df=long_df,
        ...     index_columns='id_person',
        ...     value_column='value',
        ...     id_column='id_meas',
        ...     sort_within_index_group=True
        ... )
        >>> 
        >>> # Later, when unpivoting:
        >>> long_restored = unpivot_sparse(
        ...     wide_df,
        ...     index='id_person',
        ...     long_skeleton_df=skeleton,
        ...     long_skeleton_id_column='id_meas'
        ... )
    """
    if isinstance(index_columns, str):
        index_columns = [index_columns]
    else:
        index_columns = list(index_columns)
    
    # Check if value column contains JAX tracers
    values_are_traced = _has_jax_traced_values(df, value_column) if len(df) > 0 else False
    
    # Build the variable column based on the pivot structure
    if var_column is not None:
        # Variable column is explicitly provided - use it directly
        variable_values = list(df[var_column])
    elif values_are_traced or not sort_within_index_group:
        # Position-based assignment: variable = position within each index group
        # This works for:
        # 1. Traced values (can't sort them anyway)
        # 2. Non-sorted pivots (order is preserved)
        index_key_counts: Dict[Tuple[Any, ...], int] = {}
        variable_values: List[int] = []
        for i in range(len(df)):
            key = tuple(df[col][i] for col in index_columns)
            var_in_group = index_key_counts.get(key, 0)
            index_key_counts[key] = var_in_group + 1
            variable_values.append(var_in_group)
    else:
        # Sorted pivot: need to compute variable based on sorted order
        # This requires concrete values to determine sort order
        structure = PivotStructure(
            df=df,
            index_columns=index_columns,
            var_column=None,
            sort_within_id=True,
            value_column_for_sort=value_column,
            use_jax_sort=False
        )
        # source_to_dest_col tells us which wide column each source row maps to
        variable_values = list(structure.source_to_dest_col)
    
    # Build the skeleton DataFrame
    skeleton_data: Dict[str, Any] = {}
    
    # Add index columns
    for col in index_columns:
        skeleton_data[col] = list(df[col])
    
    # Add variable column
    skeleton_data['variable'] = variable_values
    
    # Add id column if provided
    if id_column is not None:
        if id_column not in df.columns:
            raise ValueError(f"id_column '{id_column}' not found in DataFrame")
        skeleton_data[id_column] = list(df[id_column])
    
    return _create_optimized_dataframe(skeleton_data)


def _apply_pivot_structure_jax(
    values: "jnp.ndarray",
    structure: PivotStructure,
    fill_type: Union[float, str] = 0.0
) -> "jnp.ndarray":
    """
    Apply a pre-computed pivot structure to values using JAX operations.
    
    This function is JIT-compatible because it uses pure array operations
    (scatter) rather than Python control flow.
    
    Args:
        values: 1D JAX array of values to pivot (length = n_source_rows)
        structure: Pre-computed PivotStructure
        fill_type: Fill value or strategy ('local_max', 'global_max', or numeric)
        
    Returns:
        2D JAX array of shape (n_dest_rows, n_dest_cols)
    """
    import jax.numpy as jnp
    
    # Determine fill value based on fill_type
    if fill_type == 'global_max':
        # Global max across all values
        numeric_fill_value = jnp.max(values)
    elif fill_type == 'local_max':
        # For non-sorted path, we need to compute per-entity max
        # First scatter values, then compute per-row max
        # Initialize with -inf to find max
        temp_output = jnp.full(
            (structure.n_dest_rows, structure.n_dest_cols),
            -jnp.inf,
            dtype=values.dtype
        )
        temp_output = temp_output.at[
            structure.source_to_dest_row,
            structure.source_to_dest_col
        ].set(values)
        
        # Per-row max (local max for each entity)
        local_maxes = jnp.max(temp_output, axis=1, keepdims=True)  # (n_dest_rows, 1)
        
        # Now create output with local max as fill, then scatter values
        # Broadcast local_maxes to full shape
        output = jnp.broadcast_to(local_maxes, (structure.n_dest_rows, structure.n_dest_cols)).copy()
        output = output.at[
            structure.source_to_dest_row,
            structure.source_to_dest_col
        ].set(values)
        return output
    elif isinstance(fill_type, (int, float)):
        numeric_fill_value = float(fill_type)
    else:
        try:
            numeric_fill_value = float(fill_type)
        except (TypeError, ValueError):
            numeric_fill_value = 0.0
    
    # Initialize output with fill values
    output = jnp.full(
        (structure.n_dest_rows, structure.n_dest_cols),
        numeric_fill_value,
        dtype=values.dtype
    )
    
    # Scatter source values to destination positions
    # Use .at[].set() for JAX-compatible indexing
    output = output.at[
        structure.source_to_dest_row,
        structure.source_to_dest_col
    ].set(values)
    
    return output


def _apply_pivot_structure_jax_sorted(
    values: "jnp.ndarray",
    structure: PivotStructure,
    fill_type: Union[float, str] = 0.0
) -> Tuple["jnp.ndarray", np.ndarray, "jnp.ndarray"]:
    """
    Apply pivot with JIT-compatible sorting within each entity.
    
    This function gathers values for each entity, sorts them using JAX's
    argsort, and returns the sorted values along with mask and order info.
    
    Args:
        values: 1D JAX array of values to pivot (length = n_source_rows)
        structure: Pre-computed PivotStructure with requires_jax_sort=True
        fill_type: Fill value or strategy ('local_max', 'global_max', or numeric)
        
    Returns:
        Tuple of:
        - output: (n_entities, max_obs) sorted values with fill_value for padding
        - mask: (n_entities, max_obs) numpy boolean mask (True = valid data) - NOT traced
        - original_positions: (n_entities, max_obs) original position before sorting
    """
    import jax.numpy as jnp
    
    n_entities = structure.n_dest_rows
    max_obs = structure.n_dest_cols
    
    # First gather values to compute fill values if needed
    # Use 0.0 as temporary padding (we'll replace later based on fill_type)
    temp_padded = jnp.concatenate([values, jnp.array([0.0], dtype=values.dtype)])
    gathered = temp_padded[structure.entity_source_indices]
    entity_source_mask_jax = jnp.array(structure.entity_source_mask)
    
    # Compute fill values based on fill_type
    if fill_type == 'global_max':
        # Global max across all values
        fill_value = jnp.max(values)
        # Per-entity fill is the same scalar
        per_entity_fill = jnp.full((n_entities, 1), fill_value, dtype=values.dtype)
    elif fill_type == 'local_max':
        # Per-entity max (local max)
        # Mask out invalids with -inf, then take max along axis=1
        masked_for_max = jnp.where(entity_source_mask_jax, gathered, -jnp.inf)
        per_entity_fill = jnp.max(masked_for_max, axis=1, keepdims=True)  # (n_entities, 1)
        fill_value = 0.0  # fallback scalar (not used when per_entity_fill is set)
    elif isinstance(fill_type, (int, float)):
        fill_value = float(fill_type)
        per_entity_fill = jnp.full((n_entities, 1), fill_value, dtype=values.dtype)
    else:
        try:
            fill_value = float(fill_type)
        except (TypeError, ValueError):
            fill_value = 0.0
        per_entity_fill = jnp.full((n_entities, 1), fill_value, dtype=values.dtype)
    
    # Now re-gather with proper padding using per_entity_fill for invalid slots
    # For sorting, we need +inf in invalid slots so they sort to the end
    sort_keys = jnp.where(
        entity_source_mask_jax,
        gathered,
        jnp.inf
    )
    
    # Sort within each entity (along axis=1)
    sort_perm = jnp.argsort(sort_keys, axis=1)
    
    # Apply permutation using advanced indexing
    entity_idx = jnp.arange(n_entities)[:, None]  # (n_entities, 1)
    sorted_values = gathered[entity_idx, sort_perm]
    
    # The original position of a sorted element is where it came from
    # This is the inverse of the sort permutation
    original_positions = jnp.argsort(sort_perm, axis=1)
    
    # Compute the mask from structure (NOT traced)
    # After sorting, invalids are pushed to the end due to +inf sort key
    # The mask is: first count[entity] values are True, rest are False
    # This is exactly the row sums of entity_source_mask
    counts = np.sum(structure.entity_source_mask, axis=1)  # (n_entities,)
    sorted_mask_np = np.zeros((n_entities, max_obs), dtype=bool)
    for i, count in enumerate(counts):
        sorted_mask_np[i, :count] = True
    
    # Final output: apply mask to fill invalid positions with appropriate fill value
    sorted_mask_jax = jnp.array(sorted_mask_np)
    # Broadcast per_entity_fill to full shape for jnp.where
    broadcast_fill = jnp.broadcast_to(per_entity_fill, (n_entities, max_obs))
    output = jnp.where(sorted_mask_jax, sorted_values, broadcast_fill)
    
    # For order column, use -1 for invalid positions  
    order_output = jnp.where(sorted_mask_jax, original_positions, -1)
    
    return output, sorted_mask_np, order_output


def _is_jax_traced(value) -> bool:
    """Check if a value is a JAX tracer (being traced through JIT)."""
    try:
        from jax import core
        return isinstance(value, core.Tracer)
    except ImportError:
        return False


def _has_jax_traced_values(df: DataFrame, column: str) -> bool:
    """Check if a DataFrame column contains JAX traced values."""
    if len(df) == 0:
        return False
    # Check first value
    first_val = df[column][0]
    return _is_jax_traced(first_val)


class UnpivotStructure:
    """
    Pre-computed structure for wide-to-long (unpivot) transformation.
    
    This class separates structure computation (which uses Python control flow)
    from value gathering (which can be done with JAX operations).
    
    The structure maps each valid observation in wide format to its position
    in the output long format array.
    """
    
    def __init__(
        self,
        df: DataFrame,
        index_columns: Union[str, List[str]],
        var_pattern: str = r'([^$]+)\$(\d+)\$(value|mask|order)',
    ):
        """
        Pre-compute the unpivot structure from a wide format DataFrame.
        
        Args:
            df: Wide format DataFrame with value, mask, and optionally order columns
            index_columns: Column name(s) that identify each entity
            var_pattern: Regex pattern to identify value/mask/order columns
        """
        if isinstance(index_columns, str):
            index_columns = [index_columns]
        
        self.index_columns = index_columns
        self.n_wide_rows = len(df)
        
        # Parse column names to identify value and mask columns
        value_columns = {}  # {time_index: column_name}
        mask_columns = {}   # {time_index: column_name}
        order_columns = {}  # {time_index: column_name}
        
        pattern = re.compile(var_pattern)
        
        for col in df.columns:
            match = pattern.match(col)
            if match:
                var_name_part, time_index, col_type = match.groups()
                time_index = int(time_index)
                
                if col_type == 'value':
                    value_columns[time_index] = col
                elif col_type == 'mask':
                    mask_columns[time_index] = col
                elif col_type == 'order':
                    order_columns[time_index] = col
        
        self.value_columns = value_columns
        self.mask_columns = mask_columns
        self.order_columns = order_columns
        self.n_wide_cols = len(value_columns)
        self.sorted_var_indices = sorted(value_columns.keys())
        
        # Build the mask array from non-traced mask columns
        # Shape: (n_wide_rows, n_wide_cols)
        mask_array = np.ones((self.n_wide_rows, self.n_wide_cols), dtype=bool)
        for col_idx, var_idx in enumerate(self.sorted_var_indices):
            if var_idx in mask_columns:
                mask_col = mask_columns[var_idx]
                for row_idx in range(self.n_wide_rows):
                    mask_val = df[mask_col][row_idx]
                    # Convert to Python bool (masks should never be traced)
                    mask_array[row_idx, col_idx] = bool(mask_val)
        
        self.mask_array = mask_array  # True = valid observation
        
        # Count total valid observations
        self.n_long_rows = int(np.sum(mask_array))
        
        # Build source-to-dest mapping
        # For each valid (row, col) in wide format, record its position in long format
        src_wide_rows = []
        src_wide_cols = []
        dest_long_indices = []
        
        # Also build index values for long format output
        long_index_values = {col: [] for col in index_columns}
        long_var_values = []
        
        dest_idx = 0
        for row_idx in range(self.n_wide_rows):
            for col_idx, var_idx in enumerate(self.sorted_var_indices):
                if mask_array[row_idx, col_idx]:
                    src_wide_rows.append(row_idx)
                    src_wide_cols.append(col_idx)
                    dest_long_indices.append(dest_idx)
                    
                    # Record index column values
                    for index_col in index_columns:
                        long_index_values[index_col].append(df[index_col][row_idx])
                    long_var_values.append(var_idx)
                    
                    dest_idx += 1
        
        self.src_wide_rows = np.array(src_wide_rows, dtype=np.int32)
        self.src_wide_cols = np.array(src_wide_cols, dtype=np.int32)
        self.dest_long_indices = np.array(dest_long_indices, dtype=np.int32)
        self.long_index_values = long_index_values
        self.long_var_values = long_var_values
        
        # Store order column values if present
        # Note: order values may be JAX traced, so we don't convert them
        if order_columns:
            long_order_values = []
            for row_idx in range(self.n_wide_rows):
                for col_idx, var_idx in enumerate(self.sorted_var_indices):
                    if mask_array[row_idx, col_idx]:
                        if var_idx in order_columns:
                            order_val = df[order_columns[var_idx]][row_idx]
                            # Don't call .item() - might be traced
                            # Check if it's a traced value or concrete
                            if _is_jax_traced(order_val):
                                # For traced values, we'll handle them separately
                                long_order_values.append(order_val)
                            elif hasattr(order_val, 'item'):
                                long_order_values.append(order_val.item())
                            else:
                                long_order_values.append(order_val)
                        else:
                            long_order_values.append(var_idx)
            self.long_order_values = long_order_values
        else:
            self.long_order_values = None


def _apply_unpivot_structure_jax(
    wide_values: "jnp.ndarray",
    structure: UnpivotStructure
) -> "jnp.ndarray":
    """
    Apply pre-computed unpivot structure to wide values using JAX operations.
    
    This function is JIT-compatible because it uses pure array operations
    (gather via advanced indexing) rather than Python control flow.
    
    Args:
        wide_values: 2D JAX array of shape (n_wide_rows, n_wide_cols)
        structure: Pre-computed UnpivotStructure
        
    Returns:
        1D JAX array of length n_long_rows containing gathered values
    """
    import jax.numpy as jnp
    
    # Gather valid values using pre-computed indices
    long_values = wide_values[structure.src_wide_rows, structure.src_wide_cols]
    
    return long_values


def _single_wide_to_long_masked_jax(
    df: DataFrame,
    index_columns: List[str],
    value_column: str,
    var_pattern: str = r'([^$]+)\$(\d+)\$(value|mask|order)',
    var_name: str = 'variable',
    value_name: str = 'value',
    order_value_name: Optional[str] = None
) -> DataFrame:
    """
    JIT-compatible version of _single_wide_to_long_masked.
    
    This function uses pre-computed unpivot structure and JAX gather operations
    to enable JIT compilation through unpivot operations.
    """
    import jax.numpy as jnp
    
    # Compute the unpivot structure (this is Python, not traced)
    structure = UnpivotStructure(df, index_columns, var_pattern)
    
    # Extract wide values as a 2D JAX array
    # Values are in columns matching the sorted var indices
    wide_value_cols = []
    for var_idx in structure.sorted_var_indices:
        col_name = structure.value_columns[var_idx]
        col_data = df[col_name]
        # Stack row values into a column
        if hasattr(col_data, '__len__') and not isinstance(col_data, str):
            col_values = jnp.stack([col_data[i] for i in range(len(df))])
        else:
            col_values = jnp.array([col_data])
        wide_value_cols.append(col_values)
    
    # Stack columns to form (n_rows, n_cols) array
    wide_values = jnp.stack(wide_value_cols, axis=1)
    
    # Apply the unpivot using JAX gather
    long_values = _apply_unpivot_structure_jax(wide_values, structure)
    
    # Build the output DataFrame with pre-computed structure
    long_data = {}
    
    # Add index columns (from pre-computed structure, not traced)
    for index_col in index_columns:
        long_data[index_col] = structure.long_index_values[index_col]
    
    # Add variable column
    long_data[var_name] = structure.long_var_values
    
    # Add value column (JAX array)
    long_data[value_name] = long_values
    
    # Add order column if present
    if order_value_name is not None and structure.long_order_values is not None:
        long_data[order_value_name] = structure.long_order_values
    
    return _create_optimized_dataframe(long_data)


def _single_long_to_wide_masked(
    df: DataFrame,
    index_columns: Union[str, List[str]],
    value_column: str,
    var_column: Optional[str] = None,
    var_prefix: str = 'var',
    fill_type: Union[Any, str] = 0.0,
    mask_value: bool = False,
    sort_within_id: bool = False,
    order_suffix: Optional[str] = 'order',
    index_subdata_columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Convert a single column from long format to wide format (internal helper function).
    
    This is the original implementation for single column conversion with optional
    in-entity sorting and metadata tracking of the original observation order.
    
    This function supports JAX JIT compilation when the value column contains
    JAX traced arrays. The structural computation is done in Python, and the
    value placement uses JAX-compatible scatter operations.
    
    Args:
        index_subdata_columns: Optional list of column names that should be
            carried forward to the wide format. Must have consistent values
            within each index group.
    """
    # Check for invalid fill_types with string data
    if fill_type in ['local_max', 'global_max']:
        # Check if this column contains string data
        sample_values = [df[value_column][j] for j in range(min(10, len(df)))]
        if any(isinstance(v, str) for v in sample_values):
            raise ValueError(f"fill_type '{fill_type}' not supported for string data in column '{value_column}'. "
                           f"Use a specific string value instead.")

    if sort_within_id and var_column is not None:
        raise ValueError("sort_within_id cannot be used when var_column is provided.")
    if sort_within_id and order_suffix is None:
        raise ValueError("order_suffix cannot be None when sort_within_id is enabled.")
    
    # Ensure index_columns is a list
    if isinstance(index_columns, str):
        index_columns = [index_columns]
    
    # Validate index_subdata_columns if provided
    if index_subdata_columns:
        _validate_index_subdata_columns(df, index_columns, index_subdata_columns)
    
    # Check if we're dealing with JAX traced values - if so, use JIT-compatible path
    use_jax_path = _has_jax_traced_values(df, value_column)
    
    if use_jax_path:
        return _single_long_to_wide_masked_jax(
            df=df,
            index_columns=index_columns,
            value_column=value_column,
            var_column=var_column,
            var_prefix=var_prefix,
            fill_type=fill_type,
            mask_value=mask_value,
            index_subdata_columns=index_subdata_columns,
            sort_within_id=sort_within_id,
            order_suffix=order_suffix
        )
    
    # Get unique IDs
    unique_ids = []
    index_tuples_seen = set()
    
    for row_idx in range(len(df)):
        index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
        if index_tuple not in index_tuples_seen:
            unique_ids.append(index_tuple)
            index_tuples_seen.add(index_tuple)
    
    # Calculate fill values based on fill_type
    fill_values = {}  # {index_tuple: fill_value}
    
    if fill_type == 'global_max':
        # Find global maximum across all values
        global_max = max(df[value_column])
        for index_tuple in unique_ids:
            fill_values[index_tuple] = global_max
    elif fill_type == 'local_max':
        # Find local maximum for each ID
        for index_tuple in unique_ids:
            index_values = []
            for row_idx in range(len(df)):
                row_index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
                if row_index_tuple == index_tuple:
                    index_values.append(df[value_column][row_idx])
            fill_values[index_tuple] = max(index_values) if index_values else 0.0
    else:
        # Use the provided value for all IDs
        for index_tuple in unique_ids:
            fill_values[index_tuple] = fill_type
    
    # If var_column is provided, use it; otherwise infer indices
    if var_column is not None:
        # Use provided variable column
        unique_vars = sorted(set(df[var_column]))
    else:
        # Infer variable indices based on order within each ID group
        max_observations = 0
        for index_tuple in unique_ids:
            # Count observations for this ID
            count = 0
            for row_idx in range(len(df)):
                row_index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
                if row_index_tuple == index_tuple:
                    count += 1
            max_observations = max(max_observations, count)
        
        unique_vars = list(range(max_observations))
    
    # Initialize wide format data
    wide_data = {col: [] for col in index_columns}
    record_original_order = sort_within_id and order_suffix is not None
    
    # Track first row for each entity (for index_subdata extraction)
    entity_first_row = {}
    for row_idx in range(len(df)):
        index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
        if index_tuple not in entity_first_row:
            entity_first_row[index_tuple] = row_idx
    
    # Initialize index_subdata columns
    if index_subdata_columns:
        for col in index_subdata_columns:
            wide_data[col] = []
    
    # Create value and mask columns for each variable
    for var_idx in unique_vars:
        value_col_name = f"{var_prefix}${var_idx}$value"
        mask_col_name = f"{var_prefix}${var_idx}$mask"
        wide_data[value_col_name] = []
        wide_data[mask_col_name] = []
        if record_original_order:
            order_col_name = f"{var_prefix}${var_idx}${order_suffix}"
            wide_data[order_col_name] = []
    
    # For each unique ID combination
    for index_tuple in unique_ids:
        # Add ID values
        for i, index_col in enumerate(index_columns):
            wide_data[index_col].append(index_tuple[i])
        
        # Add index_subdata values (from first row of this entity)
        if index_subdata_columns:
            first_row = entity_first_row[index_tuple]
            for col in index_subdata_columns:
                wide_data[col].append(df[col][first_row])
        
        if sort_within_id:
            observations = []  # List of (value, original_position)
            original_position = 0
            for row_idx in range(len(df)):
                row_index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
                if row_index_tuple == index_tuple:
                    value = df[value_column][row_idx]
                    observations.append((value, original_position))
                    original_position += 1
            observations.sort(key=lambda item: item[0])
        else:
            # Find all observations for this ID in order
            index_observations = {}  # {var_idx: value}
            if var_column is not None:
                # Use provided variable column
                for row_idx in range(len(df)):
                    row_index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
                    if row_index_tuple == index_tuple:
                        var_idx = df[var_column][row_idx]
                        value = df[value_column][row_idx]
                        index_observations[var_idx] = value
            else:
                # Infer variable indices based on order of appearance
                var_idx = 0
                for row_idx in range(len(df)):
                    row_index_tuple = tuple(df[index_col][row_idx] for index_col in index_columns)
                    if row_index_tuple == index_tuple:
                        value = df[value_column][row_idx]
                        index_observations[var_idx] = value
                        var_idx += 1
        
        # Fill in values and masks for each variable
        for slot_idx, var_idx in enumerate(unique_vars):
            value_col_name = f"{var_prefix}${var_idx}$value"
            mask_col_name = f"{var_prefix}${var_idx}$mask"
            order_col_name = f"{var_prefix}${var_idx}${order_suffix}" if record_original_order else None
            
            if sort_within_id:
                if slot_idx < len(observations):
                    value, original_pos = observations[slot_idx]
                    wide_data[value_col_name].append(value)
                    wide_data[mask_col_name].append(True)
                    if record_original_order and order_col_name is not None:
                        wide_data[order_col_name].append(original_pos)
                else:
                    wide_data[value_col_name].append(fill_values[index_tuple])
                    wide_data[mask_col_name].append(mask_value)
                    if record_original_order and order_col_name is not None:
                        wide_data[order_col_name].append(-1)
            else:
                if var_idx in index_observations:
                    # Observation exists
                    wide_data[value_col_name].append(index_observations[var_idx])
                    wide_data[mask_col_name].append(True)
                else:
                    # Missing observation - use calculated fill value
                    wide_data[value_col_name].append(fill_values[index_tuple])
                    wide_data[mask_col_name].append(mask_value)
    
    return _create_optimized_dataframe(wide_data)


def _single_long_to_wide_masked_jax(
    df: DataFrame,
    index_columns: List[str],
    value_column: str,
    var_column: Optional[str] = None,
    var_prefix: str = 'var',
    fill_type: Union[Any, str] = 0.0,
    mask_value: bool = False,
    index_subdata_columns: Optional[List[str]] = None,
    sort_within_id: bool = False,
    order_suffix: Optional[str] = 'order'
) -> DataFrame:
    """
    JIT-compatible version of _single_long_to_wide_masked.
    
    This function uses pre-computed pivot structure and JAX scatter operations
    to enable JIT compilation through pivot operations. When sort_within_id=True,
    sorting is performed using JAX's argsort for full JIT compatibility.
    
    Args:
        index_subdata_columns: Optional list of column names that should be
            carried forward to the wide format. Must have consistent values
            within each index group.
    """
    import jax.numpy as jnp
    
    # Compute the pivot structure (this is Python, not traced)
    # Use JAX sorting when sort_within_id is True
    structure = PivotStructure(
        df=df,
        index_columns=index_columns,
        var_column=var_column,
        sort_within_id=sort_within_id,
        value_column_for_sort=None,  # Don't use Python sorting
        use_jax_sort=sort_within_id  # Use JAX sorting instead
    )
    
    # Extract values as a JAX array (this is the traced part)
    values_list = [df[value_column][i] for i in range(len(df))]
    values = jnp.stack(values_list)
    
    # Apply the pivot using appropriate method
    # Pass fill_type directly - the apply functions handle local_max/global_max
    record_original_order = sort_within_id and order_suffix is not None
    
    if structure.requires_jax_sort:
        # Use JAX-based sorting
        wide_values, mask_array, wide_order = _apply_pivot_structure_jax_sorted(
            values, structure, fill_type
        )
    else:
        # Use regular scatter (no sorting)
        wide_values = _apply_pivot_structure_jax(values, structure, fill_type)
        # Build the mask (not traced, just numpy)
        mask_array = ~structure.dest_fill_mask  # numpy array, not jax
        wide_order = None
    
    # Build the output DataFrame
    wide_data = {}
    
    # Add index columns
    for i, index_col in enumerate(index_columns):
        wide_data[index_col] = [id_tuple[i] for id_tuple in structure.unique_ids]
    
    # Add index_subdata columns using entity_first_row mapping
    if index_subdata_columns:
        for col in index_subdata_columns:
            col_data = df[col]
            # Check if this column contains traced values
            if len(col_data) > 0 and _is_jax_traced(col_data[0]):
                # Use JAX gather for traced values
                col_array = jnp.stack([col_data[i] for i in range(len(col_data))])
                wide_data[col] = col_array[structure.entity_first_row]
            else:
                # Use Python indexing for non-traced values
                wide_data[col] = [col_data[i] for i in structure.entity_first_row]
    
    # Add value, mask, and order columns for each variable slot
    for col_idx, var_idx in enumerate(structure.unique_vars):
        value_col_name = f"{var_prefix}${var_idx}$value"
        mask_col_name = f"{var_prefix}${var_idx}$mask"
        
        # Extract column from 2D array
        wide_data[value_col_name] = wide_values[:, col_idx]
        # mask_array is always numpy (not traced), convert to list
        wide_data[mask_col_name] = list(mask_array[:, col_idx])
        
        if record_original_order and wide_order is not None:
            order_col_name = f"{var_prefix}${var_idx}${order_suffix}"
            wide_data[order_col_name] = wide_order[:, col_idx]
    
    return _create_optimized_dataframe(wide_data)


def _reorder_dataframe_by_ids(df: DataFrame, index_columns: Union[str, List[str]], reference_ordering: List) -> DataFrame:
    """
    Reorder a DataFrame to match a reference ordering of ID values.
    
    Args:
        df: DataFrame to reorder
        index_columns: Column name(s) that identify each row
        reference_ordering: List of ID tuples in the desired order
        
    Returns:
        Reordered DataFrame
    """
    return _reorder_by_reference(
        df,
        index_columns,
        reference_ordering,
        stable=False,
        drop_unreferenced=True,
        error_on_missing=True,
    )


def wide_df_to_masked_array(
    df: DataFrame,
    index_columns: Union[str, List[str]],
    var_pattern: str = r'([^$]+)\$(\d+)\$value',
    sort_by_var_index: bool = True,
    preserve_skeleton: bool = True
) -> MaskedArray:
    """
    Convert a wide format DataFrame to a MaskedArray.
    
    Args:
        df: Wide format DataFrame with columns like 'var$0$value', 'var$1$value', etc.
        index_columns: Column name(s) that identify each row (will be preserved)
        var_pattern: Regex pattern to extract variable indices from column names
        sort_by_var_index: Whether to sort columns by variable index (default: True)
        preserve_skeleton: Deprecated, kept for API compatibility. The wide DataFrame
            is always stored as the skeleton.
    
    Returns:
        MaskedArray containing:
        - data: JAX array of shape (n_rows, n_variables) 
        - mask: NumPy array of shape (n_rows, n_variables) with boolean masks
        - wide_skeleton_df: DataFrame with original wide format structure
        - index_columns: List of column names used as index
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required for this function. Install with: pip install jax")
    
    import re
    
    # Ensure index_columns is a list
    if isinstance(index_columns, str):
        index_columns = [index_columns]
    
    # Find value and mask columns
    value_columns = []
    mask_columns = []
    order_columns = []
    var_indices = []
    
    pattern = re.compile(var_pattern)
    
    for col in df.columns:
        if col in index_columns:
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
            
            # Look for corresponding order column
            order_col = f"{var_name}${var_index}$order"
            if order_col in df.columns:
                order_columns.append(order_col)
            else:
                order_columns.append(None)
    
    if not value_columns:
        raise ValueError(f"No value columns found matching pattern: {var_pattern}")
    
    # Sort by variable index if requested
    if sort_by_var_index:
        sorted_data = sorted(zip(var_indices, value_columns, mask_columns, order_columns))
        var_indices, value_columns, mask_columns, order_columns = zip(*sorted_data)
    
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
    
    # Use the original wide DataFrame as skeleton (it contains all needed columns including index and order)
    # The MaskedArray now stores wide_skeleton_df and index_columns (no separate index_df)
    wide_skeleton_df = df
    
    return MaskedArray(data=values, mask=masks, wide_skeleton_df=wide_skeleton_df, index_columns=index_columns)


def masked_array_to_wide_df(
    masked_array: MaskedArray,
    var_prefix: str = 'var',
    use_skeleton_order: bool = True
) -> DataFrame:
    """
    Convert a MaskedArray back to wide format DataFrame.
    
    When use_skeleton_order=True, the order columns from the skeleton are used 
    to restore values to their original positions before sorting occurred 
    during wide-to-masked-array conversion.
    
    Args:
        masked_array: MaskedArray containing data, mask, wide_skeleton_df, and index_columns
        var_prefix: Prefix for variable column names (default: 'var')
        use_skeleton_order: Whether to use the skeleton's order columns to restore
            original positions (default: True). If False, values are placed in 
            sequential slot order.
    
    Returns:
        Wide format DataFrame with value, mask, and optionally order columns
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required for this function. Install with: pip install jax")
    
    values_array = masked_array.data
    mask_array = masked_array.mask
    index_dataframe = masked_array.index_df
    wide_skeleton_df = masked_array.wide_skeleton_df
    
    if values_array.shape != mask_array.shape:
        raise ValueError(f"Values and mask arrays must have same shape. "
                        f"Got {values_array.shape} and {mask_array.shape}")
    
    if values_array.shape[0] != len(index_dataframe):
        raise ValueError(f"Number of rows in arrays ({values_array.shape[0]}) "
                        f"must match ID DataFrame length ({len(index_dataframe)})")
    
    n_rows, n_vars = values_array.shape
    
    # Start with ID columns
    result_df = DataFrame({col: index_dataframe[col] for col in index_dataframe.columns})
    
    # Check if we should use skeleton order columns
    has_skeleton_order = (
        use_skeleton_order and 
        wide_skeleton_df is not None and
        f"{var_prefix}$0$order" in wide_skeleton_df.columns
    )
    
    # Add value, mask, and optionally order columns
    for var_idx in range(n_vars):
        value_col_name = f"{var_prefix}${var_idx}$value"
        mask_col_name = f"{var_prefix}${var_idx}$mask"
        order_col_name = f"{var_prefix}${var_idx}$order"
        
        # Extract column data as JAX arrays to preserve computational graph
        values_col = values_array[:, var_idx]
        masks_col = mask_array[:, var_idx]
        
        # Use type hints for optimized column addition
        result_df = result_df.add_column(value_col_name, values_col, column_type='jax_array')
        result_df = result_df.add_column(mask_col_name, masks_col, column_type='array')
        
        # Include order column from skeleton if available
        if has_skeleton_order and order_col_name in wide_skeleton_df.columns:
            order_col = wide_skeleton_df[order_col_name]
            result_df = result_df.add_column(order_col_name, order_col)

    return result_df


def roundtrip_wide_jax_conversion(
    df: DataFrame,
    index_columns: Union[str, List[str]],
    var_pattern: str = r'([^$]+)\$(\d+)\$value',
    var_prefix: str = 'var'
) -> DataFrame:
    """
    Test roundtrip conversion: wide DataFrame -> MaskedArray -> wide DataFrame.
    
    This is a utility function for testing that the conversion process preserves data.
    
    Args:
        df: Original wide format DataFrame
        index_columns: Column name(s) that identify each row
        var_pattern: Regex pattern for value columns  
        var_prefix: Prefix for reconstructed column names
        
    Returns:
        Reconstructed wide format DataFrame
    """
    # Convert to MaskedArray
    masked_array = wide_df_to_masked_array(df, index_columns, var_pattern)
    
    # Convert back to DataFrame
    reconstructed = masked_array_to_wide_df(masked_array, var_prefix)
    
    return reconstructed


# ---------------------------------------------------------------------------
# Polars-style wrapper API
# ---------------------------------------------------------------------------


def pivot_sparse(
    df: DataFrame,
    index: Union[str, List[str]],
    value: Union[str, List[str]],
    *,
    on: Optional[Union[str, List[str]]] = None,
    prefix: Union[str, List[str]] = None,
    fill_type: Union[Any, str, List[Union[Any, str]]] = None,
    mask_value: bool = False,
    sort_within_index_group: Union[bool, List[bool]] = False,
    order_suffix: Optional[str] = None,
    index_subdata: Optional[List[str]] = None
) -> Union[DataFrame, List[DataFrame]]:
    """Polars-style interface to :func:`long_to_wide_masked`.

    Args:
        df: Long-format ``DataFrame`` to pivot.
        index: Column(s) that uniquely identify each entity (maps to
            ``index_columns``).
        value: Column(s) containing the measurements to distribute across
            wide slots (maps to ``value_column``).
        on: Optional column(s) describing the grouping/slot identifiers
            (maps to ``var_column``).
        prefix: Prefix applied to generated wide columns (maps to
            ``var_prefix``).
        fill_type: Missing-value fill strategy forwarded to
            ``long_to_wide_masked``.
        mask_value: Boolean mask to emit for synthesized values.
        sort_within_index_group: Whether each entity should be sorted before
            slotting; forwarded unchanged.
        order_suffix: Suffix for the extra column that records the original
            order when ``sort_within_index_group`` is True.
        index_subdata: Optional list of column names that should be
            carried forward to the wide format. Must have consistent values
            within each index group.
    """
    if prefix is None:
        if isinstance(value, list):
            prefix = [f"var_{i}" for i in range(len(value))]
        else:
            prefix = "var"
    
    if fill_type is None:
        if isinstance(value, list):
            fill_type = [0.0] * len(value)
        else:
            fill_type = 0.0
    
    if order_suffix is None:
        order_suffix = "order"

    return long_to_wide_masked(
        df=df,
        index_columns=index,
        value_column=value,
        var_column=on,
        var_prefix=prefix,
        fill_type=fill_type,
        mask_value=mask_value,
        sort_within_id=sort_within_index_group,
        order_suffix=order_suffix,
        index_subdata_columns=index_subdata,
    )


def unpivot_sparse(
    df: Union[DataFrame, List[DataFrame]],
    index: Union[str, List[str]],
    *,
    var_name: Union[str, List[str]] = None,
    value_name: Union[str, List[str], None] = None,
    order_name: Union[str, List[str], None] = None,
    pattern: str = None,
    long_skeleton_df: Optional[DataFrame] = None,
    long_skeleton_id_column: Optional[str] = None
) -> DataFrame:
    """Polars-style interface to :func:`wide_to_long_masked`.

    Args:
        df: Wide ``DataFrame`` (or list of aligned DataFrames) to melt.
        index: Column(s) that stay fixed across the reshape (maps to
            ``index_columns``).
        var_name: Output column name(s) that hold the variable indices.
        value_name: Output column name(s) for the values that were stored
            in the wide layout.
        order_name: Optional column(s) to capture previously recorded
            ordering metadata (maps to ``order_value_name``).
        pattern: Regex used to detect ``value``, ``mask`` and ``order``
            columns inside the wide frame.
    """
    if var_name is None:
        if isinstance(df, list):
            var_name = [f"variable_{i}" for i in range(len(df))]
        else:
            var_name = "variable"
    
    if value_name is None:
        if isinstance(df, list):
            value_name = [f"value_{i}" for i in range(len(df))]
        else:
            value_name = "value"
    
    if pattern is None:
        pattern = r'([^$]+)\$(\d+)\$(value|mask|order)'

    return wide_to_long_masked(
        df=df,
        index_columns=index,
        var_pattern=pattern,
        var_name=var_name,
        value_name=value_name,
        order_value_name=order_name,
        long_skeleton_df=long_skeleton_df,
        long_skeleton_id_column=long_skeleton_id_column,
    )


def to_masked_array(
    df: DataFrame,
    index: Union[str, List[str]],
    *,
    pattern: str = None,
    sort_by_var_index: bool = True,
) -> MaskedArray:
    """Polars-style interface to :func:`wide_df_to_masked_array`.

    Args:
        df: Wide ``DataFrame`` that stores ``value``/``mask`` column pairs.
        index: Column(s) identifying each row (maps to ``index_columns``).
        pattern: Regex specifying how to find the value columns.
        sort_by_var_index: Whether to sort the resulting data by the extracted
            variable index prior to packing into the array.
    """
    if pattern is None:
        pattern = r'([^$]+)\$(\d+)\$value'

    return wide_df_to_masked_array(
        df,
        index_columns=index,
        var_pattern=pattern,
        sort_by_var_index=sort_by_var_index,
    )


def from_masked_array(
    masked_array: MaskedArray,
    *,
    prefix: str = None
) -> DataFrame:
    """Polars-style interface to :func:`masked_array_to_wide_df`.

    Args:
        masked_array: ``MaskedArray`` produced via :func:`to_masked_array`
            or the base helper.
        prefix: Prefix for the reconstructed ``value``/``mask`` column
            names.
    """
    if prefix is None:
        prefix = "var"

    return masked_array_to_wide_df(masked_array, var_prefix=prefix)
