"""
A simple immutable DataFrame implementation using a dictionary of arrays and lists.
Optimized for JAX compatibility with fast constructors and minimal copying.
"""
from typing import Dict, Any, Union, List, Tuple, Optional, Literal, Callable
import numpy as np
import os
from enum import Enum

class ColumnType(Enum):
    """Column type enumeration for fast type tracking."""
    JAX_ARRAY = 'jax_array'
    NUMPY_ARRAY = 'array'
    LIST = 'list'


class GroupBy:
    """
    GroupBy object for performing aggregations on grouped data.
    
    This is designed to be JAX-compatible (jittable and differentiable).
    Uses jax.ops.segment_* operations for efficient aggregations.
    """
    
    def __init__(self, df: 'DataFrame', by: Union[str, List[str]]):
        """
        Initialize a GroupBy object.
        
        Args:
            df: The DataFrame to group
            by: Column name(s) to group by
        """
        self._df = df
        self._by = [by] if isinstance(by, str) else list(by)
        
        # Validate that all group columns exist
        for col in self._by:
            if col not in df._columns:
                raise KeyError(f"Group column '{col}' not found in DataFrame")
        
        # Lazy computation - only compute groups when needed
        self._group_indices = None
        self._unique_groups = None
        self._num_groups = None
    
    def _compute_groups(self):
        """Compute group indices using JAX-compatible operations."""
        if self._group_indices is not None:
            return  # Already computed
        
        import jax.numpy as jnp
        
        # Get the grouping column(s) data
        if len(self._by) == 1:
            # Single column grouping
            group_col = self._df._data[self._by[0]]
            
            # Handle strings/lists specially - can't use JAX operations on them
            if isinstance(group_col, list):
                # Convert to numpy for finding unique values
                group_array = np.array(group_col)
                unique_vals, inverse_indices = np.unique(group_array, return_inverse=True)
                
                # Keep as list for result
                self._unique_groups = {self._by[0]: list(unique_vals)}
                self._group_indices = jnp.array(inverse_indices)
                self._num_groups = len(unique_vals)
            else:
                # Convert to JAX array if needed
                if isinstance(group_col, np.ndarray):
                    group_col = jnp.array(group_col)
                
                # Find unique groups and their indices
                unique_vals, inverse_indices = jnp.unique(group_col, return_inverse=True)
                
                self._unique_groups = {self._by[0]: unique_vals}
                self._group_indices = inverse_indices
                self._num_groups = len(unique_vals)
        
        else:
            # Multi-column grouping using prime number encoding
            # This ensures uniqueness for combinations of group values
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
            
            if len(self._by) > len(primes):
                raise ValueError(f"Maximum {len(primes)} grouping columns supported")
            
            # Create composite key using prime encoding
            composite = jnp.ones(self._df._length)
            
            # Store the mapping from composite key to group values
            col_unique_vals = []
            col_inverse_indices = []
            
            for i, col_name in enumerate(self._by):
                col_data = self._df._data[col_name]
                
                # Handle lists (strings) specially
                if isinstance(col_data, list):
                    col_array = np.array(col_data)
                    unique_vals, col_indices = np.unique(col_array, return_inverse=True)
                    col_indices = jnp.array(col_indices)
                    unique_vals = list(unique_vals)
                else:
                    # Convert to JAX array if needed
                    if isinstance(col_data, np.ndarray):
                        col_data = jnp.array(col_data)
                    
                    # Get unique values and map to indices
                    unique_vals, col_indices = jnp.unique(col_data, return_inverse=True)
                
                col_unique_vals.append(unique_vals)
                col_inverse_indices.append(col_indices)
                
                # Multiply composite key by prime^index
                composite = composite * (primes[i] ** col_indices)
            
            # Find unique composite keys
            unique_composites, inverse_indices = jnp.unique(composite, return_inverse=True)
            num_groups = len(unique_composites)
            
            # Now we need to reconstruct the unique group combinations
            # For each unique composite key, find the corresponding group values
            unique_groups_dict = {col_name: [] for col_name in self._by}
            
            for group_idx in range(num_groups):
                # Find first row belonging to this group
                mask = inverse_indices == group_idx
                first_idx = jnp.where(mask, jnp.arange(len(inverse_indices)), len(inverse_indices)).min()
                
                # Get the group values for this row
                for i, col_name in enumerate(self._by):
                    col_group_idx = int(col_inverse_indices[i][first_idx])
                    unique_groups_dict[col_name].append(col_unique_vals[i][col_group_idx])
            
            # Convert lists to arrays if they're numeric
            for col_name in self._by:
                vals = unique_groups_dict[col_name]
                if not isinstance(self._df._data[col_name], list):
                    unique_groups_dict[col_name] = jnp.array(vals)
            
            self._unique_groups = unique_groups_dict
            self._group_indices = inverse_indices
            self._num_groups = num_groups
    
    def agg(self, agg_dict: Dict[str, Union[Tuple[str, Callable], List[Tuple[str, Callable]]]]) -> 'DataFrame':
        """
        Perform aggregations on grouped data.
        
        Args:
            agg_dict: Dictionary mapping column names to aggregation function(s).
                     Functions must be tuples of (name, callable):
                     - name: String name for the aggregation (used in result column names)
                     - callable: JAX-compatible function that reduces a group to a scalar
                     
                     Examples:
                     - ('mean', jnp.mean)
                     - ('sum', jnp.sum)
                     - ('p90', lambda x: jnp.percentile(x, 90))
                     - Lists of tuples: [('mean', jnp.mean), ('std', jnp.std)]
                     
                     Custom functions receive the group data as a JAX array and should
                     return a scalar value.
                     
        Returns:
            New DataFrame with aggregated results
            
        Examples:
            >>> # Single named function
            >>> df.group_by('category').agg({'value': ('mean', jnp.mean)})
            
            >>> # Multiple named functions
            >>> df.group_by('category').agg({
            ...     'value': [('mean', jnp.mean), ('std', jnp.std)]
            ... })
            
            >>> # Multiple columns with named functions
            >>> df.group_by('category').agg({
            ...     'sales': [('mean', jnp.mean), ('sum', jnp.sum)],
            ...     'profit': [('mean', jnp.mean), ('max', jnp.max)]
            ... })
        """
        import jax.numpy as jnp
        from jax.ops import segment_sum
        
        # Compute groups if not already done
        self._compute_groups()
        
        # Result will contain group columns + aggregated columns
        result_data = {}
        
        # Add group columns
        for col_name, unique_vals in self._unique_groups.items():
            result_data[col_name] = unique_vals
        
        # Process each aggregation
        for col_name, agg_funcs in agg_dict.items():
            if col_name not in self._df._columns:
                raise KeyError(f"Column '{col_name}' not found in DataFrame")
            
            # Normalize to list of functions
            if isinstance(agg_funcs, tuple):
                agg_funcs = [agg_funcs]
            elif not isinstance(agg_funcs, list):
                raise TypeError(
                    f"Aggregation functions must be tuples (name, callable) or lists of tuples. "
                    f"Got {type(agg_funcs).__name__}. "
                    f"Example: ('mean', jnp.mean) or [('mean', jnp.mean), ('std', jnp.std)]"
                )
            
            # Get column data as JAX array
            col_data = self._df._data[col_name]
            if isinstance(col_data, list):
                col_data = jnp.array(col_data)
            elif isinstance(col_data, np.ndarray):
                col_data = jnp.array(col_data)
            
            # Apply each aggregation function
            for func_idx, agg_func in enumerate(agg_funcs):
                # Check if it's a named custom function (tuple)
                if isinstance(agg_func, tuple):
                    if len(agg_func) != 2:
                        raise ValueError(
                            f"Tuple aggregation must be (name, function), got tuple of length {len(agg_func)}"
                        )
                    
                    func_name, func = agg_func
                    
                    if not isinstance(func_name, str):
                        raise TypeError(
                            f"First element of tuple must be a string name, got {type(func_name).__name__}"
                        )
                    
                    if not callable(func):
                        raise TypeError(
                            f"Second element of tuple must be a callable, got {type(func).__name__}"
                        )
                    
                    # Apply custom aggregation function
                    result_col = self._apply_custom_agg(func, col_data, col_name)
                    result_name = f"{col_name}_{func_name}"
                
                else:
                    raise TypeError(
                        f"Aggregation function must be a tuple (name, callable), "
                        f"got {type(agg_func).__name__}. "
                        f"Bare callables are not allowed. Use tuples like ('mean', jnp.mean) instead."
                    )
                
                result_data[result_name] = result_col
        
        # Import DataFrame here to avoid circular import issues
        return DataFrame(result_data)
    
    def _apply_custom_agg(self, func: Callable, col_data: Any, col_name: str) -> Any:
        """
        Apply a custom aggregation function to each group.
        
        Args:
            func: Callable that takes a group array and returns a scalar
            col_data: Column data as JAX array
            col_name: Column name (for error messages)
            
        Returns:
            Array of aggregated values (one per group)
        """
        import jax.numpy as jnp
        from jax import core

        # When we're being traced (e.g. inside jit), only a subset of reductions are supported.
        is_traced = isinstance(col_data, core.Tracer)

        if is_traced:
            return self._apply_traced_agg(func, col_data, col_name)

        # Non-traced path: we can safely use boolean indexing for exact results
        result = []
        for group_idx in range(self._num_groups):
            mask = jnp.asarray(self._group_indices == group_idx)
            group_data = col_data[mask]

            try:
                agg_value = func(group_data)
                if hasattr(agg_value, 'shape') and agg_value.shape != ():
                    raise ValueError(
                        f"Custom aggregation function must return a scalar, "
                        f"got shape {agg_value.shape}"
                    )
                result.append(agg_value)
            except Exception as e:
                raise ValueError(
                    f"Error applying custom aggregation to column '{col_name}' "
                    f"for group {group_idx}: {str(e)}"
                )

        return jnp.array(result)

    def _apply_traced_agg(self, func: Callable, col_data: Any, col_name: str):
        """Apply aggregation while under JAX tracing (e.g. inside jit)."""
        import jax.numpy as jnp
        from jax.ops import segment_sum

        func_name = getattr(func, '__name__', None)

        if func_name == 'mean':
            sums = segment_sum(col_data, self._group_indices, self._num_groups)
            counts = segment_sum(
                jnp.ones(self._df._length, dtype=col_data.dtype),
                self._group_indices,
                self._num_groups,
            )
            return sums / counts

        if func_name == 'sum':
            return segment_sum(col_data, self._group_indices, self._num_groups)

        if func_name == 'min':
            return self._segment_min(col_data, self._group_indices, self._num_groups)

        if func_name == 'max':
            return self._segment_max(col_data, self._group_indices, self._num_groups)

        raise TypeError(
            "Aggregation function is not supported under JIT/tracing. "
            "Supported callables are jnp.mean, jnp.sum, jnp.min, and jnp.max. "
            f"Got function '{func_name or type(func).__name__}' for column '{col_name}'."
        )
    
    def _segment_min(self, data, segment_ids, num_segments):
        """Compute minimum per segment using JAX operations."""
        import jax.numpy as jnp
        from jax.ops import segment_sum
        
        # Initialize with large values
        result = jnp.full(num_segments, jnp.inf)
        
        # For each unique segment, find minimum
        for i in range(num_segments):
            mask = segment_ids == i
            if jnp.any(mask):
                result = result.at[i].set(jnp.min(jnp.where(mask, data, jnp.inf)))
        
        return result
    
    def _segment_max(self, data, segment_ids, num_segments):
        """Compute maximum per segment using JAX operations."""
        import jax.numpy as jnp
        from jax.ops import segment_sum
        
        # Initialize with small values
        result = jnp.full(num_segments, -jnp.inf)
        
        # For each unique segment, find maximum
        for i in range(num_segments):
            mask = segment_ids == i
            if jnp.any(mask):
                result = result.at[i].set(jnp.max(jnp.where(mask, data, -jnp.inf)))
        
        return result
    
    def apply(self, func: Callable, column: str, output_column: Optional[str] = None) -> 'DataFrame':
        """
        Apply a function to each group separately.
        
        This applies a JAX-compatible function to a column within each group,
        returning a DataFrame with the original row count but with the function
        applied per-group. This is different from agg() which reduces each group
        to a single value.
        
        Args:
            func: A callable that takes an array and returns a transformed array
                  of the same length. Should be JAX-compatible.
            column: Column name to apply the function to
            output_column: Optional name for output column. If not specified,
                          replaces the input column.
        
        Returns:
            New DataFrame with function applied per group (original row count preserved)
            
        Examples:
            >>> # Normalize values within each group
            >>> df.group_by('category').apply(
            ...     lambda x: (x - x.mean()) / x.std(),
            ...     'values',
            ...     output_column='normalized_values'
            ... )
            
            >>> # Rank within groups
            >>> df.group_by('department').apply(
            ...     lambda x: jnp.argsort(jnp.argsort(x)),
            ...     'salary',
            ...     output_column='salary_rank'
            ... )
        
        Note:
            - Function must accept and return array of same length
            - Each group is processed independently
            - Original DataFrame row order is preserved
            - All rows from original DataFrame are returned
        """
        import jax.numpy as jnp
        
        # Compute groups if not already done
        self._compute_groups()
        
        # Validate column exists
        if column not in self._df._columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        # Determine output column name
        if output_column is None:
            output_column = column
        
        # Get column data
        col_data = self._df._data[column]
        if isinstance(col_data, list):
            col_data = jnp.array(col_data)
        elif isinstance(col_data, np.ndarray):
            col_data = jnp.array(col_data)
        
        # Create result array (same length as original data)
        result = jnp.zeros_like(col_data)
        
        # Apply function to each group
        for group_idx in range(self._num_groups):
            # Get indices for this group using jnp.where for JIT compatibility
            indices = jnp.where(self._group_indices == group_idx, size=self._df._length)[0]
            
            # Extract group data
            group_data = col_data[indices]
            
            # Apply function
            transformed = func(group_data)
            
            # Validate output length
            if len(transformed) != len(group_data):
                raise ValueError(
                    f"Function must return array of same length as input. "
                    f"Expected {len(group_data)}, got {len(transformed)}"
                )
            
            # Put transformed data back in result using integer indexing
            result = result.at[indices].set(transformed)
        
        # Create new DataFrame with result
        new_data = self._df._data.copy()
        new_data[output_column] = result
        
        # Import DataFrame here to avoid issues
        return DataFrame(new_data, name=self._df._name)


class DataFrame:
    """
    A simple immutable DataFrame that stores data as a dictionary of arrays and/or lists.
    
    All arrays and lists must have the same length. Once created, the DataFrame cannot be modified.
    """
    
    def __init__(self, 
                 data: Dict[str, Union[List, np.ndarray]], 
                 name: str = None,
                 column_types: Optional[Dict[str, Union[str, ColumnType]]] = None,
                 skip_validation: bool = False,
                 categorical: Optional[Dict[str, bool]] = None):
        """
        Initialize a DataFrame with optimized type detection and minimal copying.
        
        Args:
            data: Dictionary where keys are column names and values are arrays/lists
            name: Optional name for the DataFrame
            column_types: Pre-computed column types (skips expensive detection)
            skip_validation: Skip length validation for internal operations
            categorical: Optional dict specifying which columns are categorical.
                        If not provided, inferred automatically based on dtype.
                        - Float dtypes: Always non-categorical (cannot be forced)
                        - String dtypes: Always categorical (cannot be forced)
                        - Int/bool dtypes: Categorical by default (can be changed)
                 
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
        
        # Initialize categorical tracking
        self._categorical = self._init_categorical(categorical)
    
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
        if os.environ.get("JAXFRAME_DEBUG_CHURN", "0") in {"1", "true", "True", "yes", "YES"}:
            maybe_arraylike = hasattr(values, "shape") or hasattr(values, "dtype") or hasattr(values, "__array__")
            if maybe_arraylike:
                mod = getattr(type(values), "__module__", "")
                typ = type(values).__name__
                try:
                    shape = getattr(values, "shape", None)
                except Exception:
                    shape = None
                print(f"[JAXFRAME-CHURN] Converting to list: type={typ} module={mod} shape={shape}")
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
    
    def _init_categorical(self, categorical: Optional[Dict[str, bool]] = None) -> Dict[str, bool]:
        """
        Initialize categorical tracking for each column.
        
        Args:
            categorical: Optional user-specified categorical flags
            
        Returns:
            Dictionary mapping column names to categorical status
            
        Rules:
        - Float-like dtypes: Always non-categorical (cannot be forced)
        - String-like dtypes: Always categorical (cannot be forced)
        - Int-like dtypes: Categorical by default, but can be forced either way
        - Bool dtypes: Categorical by default, but can be forced either way
        """
        result = {}
        
        for col_name in self._columns:
            col_data = self._data[col_name]
            
            # Determine the base dtype category
            dtype_category = self._get_dtype_category(col_name, col_data)
            
            # Check if user provided override
            if categorical is not None and col_name in categorical:
                user_preference = categorical[col_name]
                
                # Validate that override is allowed
                if dtype_category == 'float' and user_preference:
                    raise ValueError(
                        f"Cannot force float-like column '{col_name}' to be categorical. "
                        f"Float columns must be non-categorical."
                    )
                elif dtype_category == 'string' and not user_preference:
                    raise ValueError(
                        f"Cannot force string-like column '{col_name}' to be non-categorical. "
                        f"String columns must be categorical."
                    )
                else:
                    # Int/bool columns can be toggled - use user preference
                    result[col_name] = user_preference
            else:
                # Use default based on dtype
                if dtype_category == 'float':
                    result[col_name] = False  # Float = non-categorical
                elif dtype_category == 'string':
                    result[col_name] = True   # String = categorical
                elif dtype_category == 'int':
                    result[col_name] = True   # Int = categorical by default
                elif dtype_category == 'bool':
                    result[col_name] = True   # Bool = categorical
                else:
                    result[col_name] = True   # Unknown = categorical by default
        
        return result
    
    def _get_dtype_category(self, col_name: str, col_data: Any) -> str:
        """
        Determine the dtype category (float, int, string, bool, other).
        
        Args:
            col_name: Column name
            col_data: Column data
            
        Returns:
            One of: 'float', 'int', 'string', 'bool', 'other'
        """
        # Check array dtypes first
        if hasattr(col_data, 'dtype'):
            dtype_str = str(col_data.dtype)
            
            if 'float' in dtype_str or dtype_str in ['float16', 'float32', 'float64']:
                return 'float'
            elif 'int' in dtype_str or dtype_str in ['int8', 'int16', 'int32', 'int64', 
                                                       'uint8', 'uint16', 'uint32', 'uint64']:
                return 'int'
            elif 'bool' in dtype_str:
                return 'bool'
            elif 'str' in dtype_str or 'object' in dtype_str or 'unicode' in dtype_str or '<U' in dtype_str:
                return 'string'
            else:
                return 'other'
        
        # Check list dtypes by examining first element
        elif isinstance(col_data, list) and len(col_data) > 0:
            first_elem = col_data[0]
            
            if isinstance(first_elem, str):
                return 'string'
            elif isinstance(first_elem, bool):
                return 'bool'
            elif isinstance(first_elem, int) and not isinstance(first_elem, bool):
                return 'int'
            elif isinstance(first_elem, float):
                return 'float'
            else:
                return 'other'
        
        return 'other'
    
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
    
    @property
    def categorical(self) -> Dict[str, bool]:
        """
        Get categorical status for each column.
        
        Returns:
            Dictionary mapping column names to boolean indicating if categorical
        """
        return self._categorical.copy()
    
    def is_categorical(self, column: str) -> bool:
        """
        Check if a specific column is categorical.
        
        Args:
            column: Column name
            
        Returns:
            True if column is categorical, False otherwise
            
        Raises:
            KeyError: If column doesn't exist
        """
        if column not in self._columns:
            raise KeyError(f"Column '{column}' not found. Available: {list(self._columns)}")
        
        return self._categorical[column]
    
    def get_categorical_columns(self) -> List[str]:
        """
        Get list of all categorical columns.
        
        Returns:
            List of column names that are categorical
        """
        return [col for col in self._columns if self._categorical[col]]
    
    def get_non_categorical_columns(self) -> List[str]:
        """
        Get list of all non-categorical columns.
        
        Returns:
            List of column names that are non-categorical
        """
        return [col for col in self._columns if not self._categorical[col]]
    
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
    
    @property
    def dtypes(self) -> Dict[str, str]:
        """Get detailed data types for each column."""
        dtypes = {}
        for col_name, col_data in self._data.items():
            if hasattr(col_data, 'dtype'):
                # For numpy arrays or JAX arrays
                dtypes[col_name] = str(col_data.dtype)
            elif isinstance(col_data, list) and len(col_data) > 0:
                # For lists, get the type of the first element (not the container)
                first_element = col_data[0]
                element_type = type(first_element).__name__
                dtypes[col_name] = element_type
            else:
                # Fallback - empty list or other type
                dtypes[col_name] = 'object'
        return dtypes
    
    @property
    def schema(self) -> Dict[str, str]:
        """Get the schema of the DataFrame (column names to data types mapping).
        
        This is an alias for dtypes to match Polars API.
        """
        return self.dtypes
    
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
        return DataFrame(self._data, name=name, column_types=self._column_types, 
                        skip_validation=True, categorical=self._categorical)
    
    def as_categorical(self, columns: Union[str, List[str]]) -> 'DataFrame':
        """
        Mark columns as categorical.
        
        Args:
            columns: Column name(s) to mark as categorical
            
        Returns:
            New DataFrame with updated categorical flags
            
        Raises:
            ValueError: If trying to mark float-like columns as categorical
            KeyError: If column doesn't exist
            
        Examples:
            >>> df.as_categorical('user_id')
            >>> df.as_categorical(['user_id', 'product_id'])
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # Validate all columns exist
        for col in columns:
            if col not in self._columns:
                raise KeyError(f"Column '{col}' not found. Available: {list(self._columns)}")
        
        # Create new categorical dict
        new_categorical = self._categorical.copy()
        
        # Validate and update each column
        for col in columns:
            dtype_category = self._get_dtype_category(col, self._data[col])
            
            if dtype_category == 'float':
                raise ValueError(
                    f"Cannot mark float-like column '{col}' as categorical. "
                    f"Float columns must be non-categorical. "
                    f"Detected dtype: {self.dtypes[col]}"
                )
            
            new_categorical[col] = True
        
        # Create new DataFrame with updated categorical flags
        return DataFrame(
            self._data,
            name=self._name,
            column_types=self._column_types,
            skip_validation=True,
            categorical=new_categorical
        )
    
    def as_non_categorical(self, columns: Union[str, List[str]]) -> 'DataFrame':
        """
        Mark columns as non-categorical.
        
        Args:
            columns: Column name(s) to mark as non-categorical
            
        Returns:
            New DataFrame with updated categorical flags
            
        Raises:
            ValueError: If trying to mark string-like columns as non-categorical
            KeyError: If column doesn't exist
            
        Examples:
            >>> df.as_non_categorical('age')
            >>> df.as_non_categorical(['age', 'count'])
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # Validate all columns exist
        for col in columns:
            if col not in self._columns:
                raise KeyError(f"Column '{col}' not found. Available: {list(self._columns)}")
        
        # Create new categorical dict
        new_categorical = self._categorical.copy()
        
        # Validate and update each column
        for col in columns:
            dtype_category = self._get_dtype_category(col, self._data[col])
            
            if dtype_category == 'string':
                raise ValueError(
                    f"Cannot mark string-like column '{col}' as non-categorical. "
                    f"String columns must be categorical. "
                    f"Detected dtype: {self.dtypes[col]}"
                )
            
            new_categorical[col] = False
        
        # Create new DataFrame with updated categorical flags
        return DataFrame(
            self._data,
            name=self._name,
            column_types=self._column_types,
            skip_validation=True,
            categorical=new_categorical
        )
    
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
        table_str = self.to_string()
        if self.name is not None:
            return f"{self.name}:\n{table_str}"
        return table_str
    
    def __repr__(self) -> str:
        """String representation of the DataFrame."""
        if self._length == 0:
            name_part = f" '{self._name}'" if self._name else ""
            return f"DataFrame{name_part}(empty)"
        
        name_part = f" '{self._name}'" if self._name else ""
        lines = [f"DataFrame{name_part}({self.shape[0]} rows, {self.shape[1]} columns)"]
        lines.append("Columns: " + ", ".join(self._columns))
        
        # Add dtypes information
        dtypes = self.dtypes
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
    
    def join(self, other: 'DataFrame', 
             on: Union[str, List[str], None] = None,
             how: str = 'inner',
             *,
             left_on: Union[str, List[str], None] = None,
             right_on: Union[str, List[str], None] = None,
             suffix: str = '_right') -> 'DataFrame':
        """
        Join in SQL-like fashion.
        
        Args:
            other: DataFrame to join with
            on: Name(s) of the join columns in both DataFrames
            how: Join strategy. Options:
                - 'inner': Returns rows that have matching values in both tables (default)
                - 'left': Returns all rows from the left table, and matched rows from right table
                - 'full': Returns all rows when there is a match in either left or right table
                - 'semi': Filter rows that have a match in the right table
                - 'anti': Filter rows that do not have a match in the right table
                - 'cross': Returns the Cartesian product of rows from both tables
            left_on: Name(s) of the left join column(s). Cannot be used with 'on'
            right_on: Name(s) of the right join column(s). Cannot be used with 'on'
            suffix: Suffix to append to columns with a duplicate name
        
        Returns:
            A new DataFrame with the joined data
            
        Raises:
            ValueError: If join parameters are invalid or unsupported join type
            KeyError: If join columns don't exist in the DataFrames
        """
        from typing import Union, List
        
        # Validate how parameter
        valid_strategies = {'inner', 'left', 'full', 'semi', 'anti', 'cross'}
        if how not in valid_strategies:
            raise ValueError(f"'how' must be one of {valid_strategies}, got '{how}'")
        
        # Validate join column parameters
        if on is not None and (left_on is not None or right_on is not None):
            raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on'")
        
        if on is None and (left_on is None or right_on is None):
            raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")
        
        # Determine join columns
        if on is not None:
            # Both sides use the same column names
            left_columns = [on] if isinstance(on, str) else list(on)
            right_columns = left_columns.copy()
        else:
            # Different column names for each side
            left_columns = [left_on] if isinstance(left_on, str) else list(left_on)
            right_columns = [right_on] if isinstance(right_on, str) else list(right_on)
        
        if len(left_columns) != len(right_columns):
            raise ValueError("Left and right join columns must have the same length")
        
        # Validate join columns exist
        for col in left_columns:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found in left DataFrame")
        
        for col in right_columns:
            if col not in other.columns:
                raise KeyError(f"Column '{col}' not found in right DataFrame")
        
        # Check for unsupported join types
        if how in ['left', 'full', 'cross']:
            raise NotImplementedError(f"Join strategy '{how}' is not yet implemented in JAXFrame")
        
        # Determine which columns from the right DataFrame to include
        right_columns_to_add = []
        right_column_names = []
        
        for col in other.columns:
            if col not in right_columns or how in ['inner']:  # Always include non-join columns for inner joins
                # Handle column name conflicts with suffix
                new_col_name = col
                if col in self.columns and col not in left_columns:
                    new_col_name = f"{col}{suffix}"
                elif col in self.columns and col in left_columns:
                    # For join columns, don't duplicate unless it's a different name
                    if on is None:  # left_on != right_on case
                        new_col_name = f"{col}{suffix}"
                    else:
                        continue  # Skip duplicate join column for 'on' case
                
                right_columns_to_add.append(col)
                right_column_names.append(new_col_name)
        
        # Handle semi-join case
        if how == 'semi':
            return self._semi_join(other, left_columns, right_columns)
        
        # Handle anti-join case
        if how == 'anti':
            return self._anti_join(other, left_columns, right_columns)
        
        # Handle inner join case
        if how == 'inner':
            return self._inner_join(other, left_columns, right_columns, right_columns_to_add, right_column_names)
        
        # Should not reach here
        raise ValueError(f"Unsupported join strategy: {how}")
    
    def _semi_join(self, other: 'DataFrame', left_columns: List[str], right_columns: List[str]) -> 'DataFrame':
        """Perform a semi-join (filter rows that have a match in the right table)."""
        # Create lookup set from other DataFrame for existence check
        other_keys = set()
        for i in range(len(other)):
            if len(right_columns) == 1:
                key = other[right_columns[0]][i]
            else:
                key = tuple(other[right_columns[j]][i] for j in range(len(right_columns)))
            other_keys.add(key)
        
        # Find matching rows in this DataFrame
        matching_indices = []
        seen_keys = set()
        
        for i in range(len(self)):
            if len(left_columns) == 1:
                key = self[left_columns[0]][i]
            else:
                key = tuple(self[left_columns[j]][i] for j in range(len(left_columns)))
            
            # Check if key exists in other DataFrame and we haven't seen this key yet
            if key in other_keys and key not in seen_keys:
                matching_indices.append(i)
                seen_keys.add(key)
        
        # Return subset of this DataFrame with only matching rows
        return self._filter_by_indices(matching_indices)
    
    def _anti_join(self, other: 'DataFrame', left_columns: List[str], right_columns: List[str]) -> 'DataFrame':
        """Perform an anti-join (filter rows that do not have a match in the right table)."""
        # Create lookup set from other DataFrame
        other_keys = set()
        for i in range(len(other)):
            if len(right_columns) == 1:
                key = other[right_columns[0]][i]
            else:
                key = tuple(other[right_columns[j]][i] for j in range(len(right_columns)))
            other_keys.add(key)
        
        # Find non-matching rows in this DataFrame
        non_matching_indices = []
        
        for i in range(len(self)):
            if len(left_columns) == 1:
                key = self[left_columns[0]][i]
            else:
                key = tuple(self[left_columns[j]][i] for j in range(len(left_columns)))
            
            # Check if key does not exist in other DataFrame
            if key not in other_keys:
                non_matching_indices.append(i)
        
        # Return subset of this DataFrame with only non-matching rows
        return self._filter_by_indices(non_matching_indices)
    
    def _inner_join(self, other: 'DataFrame', left_columns: List[str], right_columns: List[str], 
                   right_columns_to_add: List[str], right_column_names: List[str]) -> 'DataFrame':
        """Perform an inner join."""
        # Build lookup from right DataFrame
        right_lookup = {}
        for i in range(len(other)):
            if len(right_columns) == 1:
                key = other[right_columns[0]][i]
            else:
                key = tuple(other[right_columns[j]][i] for j in range(len(right_columns)))
            
            # Store all column values for this key
            right_lookup[key] = {}
            for col in right_columns_to_add:
                right_lookup[key][col] = other[col][i]
        
        # Find matching rows and collect data
        result_data = {}
        
        # Initialize result columns from left DataFrame
        for col in self.columns:
            result_data[col] = []
        
        # Initialize result columns from right DataFrame
        for col_name in right_column_names:
            result_data[col_name] = []
        
        # Process each row from left DataFrame
        for i in range(len(self)):
            if len(left_columns) == 1:
                key = self[left_columns[0]][i]
            else:
                key = tuple(self[left_columns[j]][i] for j in range(len(left_columns)))
            
            # Check if key exists in right DataFrame
            if key in right_lookup:
                # Add left DataFrame values
                for col in self.columns:
                    result_data[col].append(self[col][i])
                
                # Add right DataFrame values
                for src_col, target_col in zip(right_columns_to_add, right_column_names):
                    result_data[target_col].append(right_lookup[key][src_col])
        
        return DataFrame(result_data, name=self.name)
    
    def _filter_by_indices(self, indices: List[int]) -> 'DataFrame':
        """Helper method to create a new DataFrame with only the specified row indices."""
        if not indices:
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
                new_data[col] = [self._data[col][i] for i in indices]
            elif self.column_types[col] == 'jax_array':
                # For JAX arrays, use advanced indexing
                try:
                    import jax.numpy as jnp
                    new_data[col] = self._data[col][jnp.array(indices)]
                except ImportError:
                    # JAX not available, fall back to numpy
                    import numpy as np
                    new_data[col] = self._data[col][np.array(indices)]
            else:  # numpy array
                import numpy as np
                new_data[col] = self._data[col][np.array(indices)]
        
        return DataFrame(new_data, name=self.name)
    
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
        Update this lookup table with rows from another DataFrame using UPSERT semantics.
        
        This method performs an "upsert" operation (UPDATE + INSERT) on the current DataFrame
        using another DataFrame as the source of updates. It maintains referential integrity
        by ensuring both DataFrames are valid lookup tables (no duplicate keys).
        
        For existing keys (UPDATE):
        - In strict mode: Validates that all non-key column values match exactly, raising
          an error if any conflicts are found
        - In non-strict mode: Replaces existing values with new values from the update DataFrame
        
        For new keys (INSERT):
        - Always adds new rows from the update DataFrame that don't exist in the current DataFrame
        
        This is conceptually similar to a database UPSERT operation or SQL's MERGE statement,
        but with stronger validation guarantees to ensure data integrity.
        
        Args:
            other: DataFrame containing rows to add/update. Must have identical column structure
                  to the current DataFrame and be a valid lookup table (no duplicate keys).
            id_columns: Column name(s) that form the lookup key. Can be a single column name
                       or a list of column names for composite keys. These columns uniquely
                       identify each row and are used to match rows between DataFrames.
            strict: If True (default), raises ValueError when existing key-value pairs have
                   conflicting non-key values. If False, replaces conflicting values with
                   new values from the update DataFrame.
                   
        Returns:
            New DataFrame with updated lookup table containing:
            - All rows from current DataFrame with non-conflicting keys
            - Updated values for existing keys (in non-strict mode) or validated matches (strict mode)
            - All new rows from the update DataFrame
            
        Raises:
            ValueError: If DataFrames have different column structures, if either DataFrame
                       contains duplicate keys, or if strict=True and value conflicts are detected.
            KeyError: If id_columns don't exist in both DataFrames.
            
        Examples:
            Basic single-column key update:
            >>> df = DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
            >>> updates = DataFrame({'id': [2, 4], 'value': [99, 40]})
            >>> result = df.update_lookup_table(updates, 'id', strict=False)
            # Result: id=[1,2,3,4], value=[10,99,30,40]
            
            Multi-column key with strict validation:
            >>> df = DataFrame({'region': ['US', 'EU'], 'product': ['A', 'B'], 'price': [100, 200]})
            >>> updates = DataFrame({'region': ['US', 'ASIA'], 'product': ['A', 'A'], 'price': [100, 150]})
            >>> result = df.update_lookup_table(updates, ['region', 'product'], strict=True)
            # Validates US-A price matches (100), adds ASIA-A as new row
            
        Note:
            Both DataFrames must be valid lookup tables (no duplicate key combinations).
            This method provides stronger guarantees than pandas merge or polars update
            by validating data integrity and preventing accidental data corruption.
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
    
    # Polars-compatible methods
    def vstack(self, other: 'DataFrame', *, in_place: bool = False) -> 'DataFrame':
        """
        Grow this DataFrame vertically by stacking a DataFrame to it.
        
        This is a Polars-compatible method that performs vertical concatenation.
        
        Args:
            other: DataFrame to stack
            in_place: If True, modifies this DataFrame in place (not supported, raises error)
            
        Returns:
            New DataFrame with other stacked vertically
            
        Raises:
            TypeError: If other is not a DataFrame
            ValueError: If DataFrames don't have the same columns
            NotImplementedError: If in_place=True (JAXFrame is immutable)
        """
        if in_place:
            raise NotImplementedError("JAXFrame DataFrames are immutable, in_place=True not supported")
        
        if not isinstance(other, DataFrame):
            raise TypeError("Can only vstack with another DataFrame")
        
        # Check that both DataFrames have the same columns
        if set(self._columns) != set(other._columns):
            raise ValueError(
                f"DataFrames must have the same columns for vstack. "
                f"Self: {set(self._columns)}, Other: {set(other._columns)}"
            )
        
        # Use existing row concatenation logic
        return self._concat_rows(other, ignore_index=False)
    
    def hstack(self, columns: Union[List, 'DataFrame'], *, in_place: bool = False) -> 'DataFrame':
        """
        Return a new DataFrame grown horizontally by stacking multiple Series to it.
        
        This is a Polars-compatible method that performs horizontal concatenation.
        Note: JAXFrame doesn't have Series objects, so this accepts a DataFrame or list of arrays.
        
        Args:
            columns: DataFrame or list of arrays/lists to stack horizontally
            in_place: If True, modifies this DataFrame in place (not supported, raises error)
            
        Returns:
            New DataFrame with columns stacked horizontally
            
        Raises:
            TypeError: If columns is not a DataFrame or list
            ValueError: If arrays have incompatible lengths
            NotImplementedError: If in_place=True (JAXFrame is immutable)
        """
        if in_place:
            raise NotImplementedError("JAXFrame DataFrames are immutable, in_place=True not supported")
        
        if isinstance(columns, DataFrame):
            # Use existing column concatenation logic
            return self._concat_columns(columns)
        elif isinstance(columns, list):
            # Convert list of arrays to DataFrame first
            if not columns:
                return self  # Nothing to stack
            
            # Create column names for the arrays
            new_data = {}
            for i, arr in enumerate(columns):
                col_name = f"column_{i}"
                # Ensure it doesn't conflict with existing columns
                while col_name in self._columns:
                    col_name = f"column_{i}_{len(new_data)}"
                new_data[col_name] = arr
            
            other_df = DataFrame(new_data)
            return self._concat_columns(other_df)
        else:
            raise TypeError("columns must be a DataFrame or list of arrays")
    
    def with_columns(self, *exprs, **named_exprs) -> 'DataFrame':
        """
        Add columns to this DataFrame.
        
        This is a Polars-compatible method. Note that JAXFrame doesn't support expressions,
        so this method accepts new column data directly.
        
        Args:
            *exprs: Column data as positional arguments (dict or key-value pairs)
            **named_exprs: Column data as keyword arguments (name=data)
            
        Returns:
            New DataFrame with the columns added
            
        Examples:
            >>> df.with_columns({'new_col': [1, 2, 3]})
            >>> df.with_columns(new_col=[1, 2, 3])
            >>> df.with_columns({'col1': [1, 2]}, col2=[3, 4])
        """
        new_data = self._data.copy()
        
        # Process positional arguments
        for expr in exprs:
            if isinstance(expr, dict):
                new_data.update(expr)
            else:
                raise TypeError("Positional arguments must be dictionaries mapping column names to data")
        
        # Process keyword arguments
        new_data.update(named_exprs)
        
        return DataFrame(new_data, name=self._name)
    
    def drop(self, *columns: str, strict: bool = True) -> 'DataFrame':
        """
        Remove columns from the dataframe.
        
        This is a Polars-compatible method.
        
        Args:
            *columns: Names of the columns to remove
            strict: If True, raise error if column doesn't exist
            
        Returns:
            New DataFrame with specified columns removed
            
        Raises:
            KeyError: If strict=True and a column doesn't exist
        """
        if not columns:
            return self  # Nothing to drop
        
        # Flatten if a single list was passed
        if len(columns) == 1 and isinstance(columns[0], (list, tuple)):
            columns = columns[0]
        
        new_data = {}
        dropped_columns = set(columns)
        
        # Check for non-existent columns if strict mode
        if strict:
            missing_cols = dropped_columns - set(self._columns)
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in DataFrame")
        
        # Copy all columns except the ones to drop
        for col_name, col_data in self._data.items():
            if col_name not in dropped_columns:
                new_data[col_name] = col_data
        
        return DataFrame(new_data, name=self._name)
    
    def filter(self, *predicates, **constraints) -> 'DataFrame':
        """
        Filter rows, retaining those that match the given predicate.
        
        This is a Polars-compatible method. Note that JAXFrame doesn't support expressions,
        so this method accepts boolean arrays/lists or simple column value constraints.
        
        Args:
            *predicates: Boolean arrays/lists indicating which rows to keep
            **constraints: Column filters using name=value syntax
            
        Returns:
            New DataFrame with filtered rows
            
        Examples:
            >>> df.filter([True, False, True])  # Keep rows 0 and 2
            >>> df.filter(name='Alice')  # Keep rows where name column equals 'Alice'
            >>> df.filter(age=25, city='NYC')  # Keep rows where age=25 AND city='NYC'
        """
        if not predicates and not constraints:
            return self  # No filtering
        
        # Start with all rows selected
        mask = np.ones(self._length, dtype=bool)
        
        # Apply predicates (boolean arrays)
        for predicate in predicates:
            if isinstance(predicate, (list, tuple)):
                predicate = np.array(predicate)
            
            if not isinstance(predicate, np.ndarray) or predicate.dtype != bool:
                raise TypeError("Predicates must be boolean arrays/lists")
            
            if len(predicate) != self._length:
                raise ValueError(f"Predicate length {len(predicate)} doesn't match DataFrame length {self._length}")
            
            mask = mask & predicate
        
        # Apply constraints (column=value filters)
        for col_name, value in constraints.items():
            if col_name not in self._columns:
                raise KeyError(f"Column '{col_name}' not found")
            
            col_data = self._data[col_name]
            
            # Create boolean mask for this constraint
            if isinstance(col_data, list):
                col_mask = np.array([x == value for x in col_data])
            else:
                col_mask = col_data == value
            
            mask = mask & col_mask
        
        # Apply the mask to filter rows
        new_data = {}
        for col_name, col_data in self._data.items():
            if isinstance(col_data, list):
                new_data[col_name] = [col_data[i] for i in range(len(col_data)) if mask[i]]
            else:
                new_data[col_name] = col_data[mask]
        
        return DataFrame(new_data, name=self._name)
    
    def group_by(self, by: Union[str, List[str]]) -> GroupBy:
        """
        Group the DataFrame by one or more columns.
        
        This is a Polars-compatible method that returns a GroupBy object.
        The GroupBy object can be used to perform aggregations on the grouped data.
        
        All operations are JAX-compatible (jittable and differentiable).
        
        Args:
            by: Column name(s) to group by. Can be a single string or list of strings.
            
        Returns:
            GroupBy object for performing aggregations
            
        Examples:
            >>> df.group_by('category').agg({'value': 'sum'})
            >>> df.group_by(['year', 'month']).agg({'sales': ['sum', 'mean'], 'count': 'count'})
        """
        return GroupBy(self, by)
    
    def apply(self, func: Callable, columns: Union[str, List[str]], 
              output_column: Optional[str] = None) -> 'DataFrame':
        """
        Apply a JAX-compatible function to one or more columns.
        
        This method allows you to apply custom transformations to columns while maintaining
        JAX compatibility (jittable and differentiable). The function should work with JAX arrays.
        
        Args:
            func: A callable that takes column data and returns transformed data.
                  Should be JAX-compatible (works with jnp arrays).
            columns: Column name(s) to apply the function to.
                    - If str: apply to single column, replace it
                    - If List[str]: apply to multiple columns, func receives them as separate args
            output_column: Optional name for output column. If not specified:
                          - Single column: replaces the input column
                          - Multiple columns: raises error (must specify output_column)
        
        Returns:
            New DataFrame with the function applied
            
        Examples:
            >>> # Apply to single column (in-place replacement)
            >>> df.apply(lambda x: x ** 2, 'values')
            
            >>> # Apply to single column with new name
            >>> df.apply(lambda x: x ** 2, 'values', output_column='values_squared')
            
            >>> # Apply to multiple columns
            >>> df.apply(lambda x, y: x + y, ['col1', 'col2'], output_column='sum')
            
            >>> # JAX-compatible function
            >>> import jax.numpy as jnp
            >>> df.apply(jnp.log, 'values', output_column='log_values')
            
        Note:
            - The function should work with JAX arrays for full JAX compatibility
            - For multiple columns, function receives them as separate positional arguments
            - Output must have same length as input columns
        """
        import jax.numpy as jnp
        
        # Normalize columns to list
        if isinstance(columns, str):
            columns = [columns]
            single_column = True
        else:
            columns = list(columns)
            single_column = False
        
        # Validate columns exist
        for col in columns:
            if col not in self._columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
        
        # Determine output column name
        if output_column is None:
            if single_column:
                output_column = columns[0]  # Replace the input column
            else:
                raise ValueError("output_column must be specified when applying to multiple columns")
        
        # Get column data
        col_data_list = []
        for col in columns:
            col_data = self._data[col]
            
            # Convert to JAX array if needed for consistency
            if isinstance(col_data, list):
                col_data = jnp.array(col_data)
            elif isinstance(col_data, np.ndarray):
                col_data = jnp.array(col_data)
            
            col_data_list.append(col_data)
        
        # Apply function
        if len(col_data_list) == 1:
            result = func(col_data_list[0])
        else:
            result = func(*col_data_list)
        
        # Validate result length
        # Check if result is scalar (0-dimensional)
        if not hasattr(result, 'shape'):
            # Not an array-like object
            raise ValueError(f"Function must return an array, got {type(result)}")
        elif result.shape == () or len(result.shape) == 0:
            # Scalar (0-dimensional array)
            raise ValueError(f"Function returned scalar value. Output must be an array with length {self._length}")
        elif result.shape[0] != self._length:
            # Array with wrong length
            raise ValueError(f"Function output length ({result.shape[0]}) must match DataFrame length ({self._length})")
        
        # Create new DataFrame with result
        new_data = self._data.copy()
        new_data[output_column] = result
        
        return DataFrame(new_data, name=self._name)

    def rename(self, mapping: Optional[Union[Dict[str, str], List[Tuple[str, str]], Tuple[Tuple[str, str], ...]]] = None, *, function: Optional[Callable[[str], str]] = None) -> 'DataFrame':
        """Rename columns using Polars-compatible semantics."""
        if mapping is None:
            rename_map: Dict[str, str] = {}
        elif isinstance(mapping, dict):
            rename_map = dict(mapping)
        elif isinstance(mapping, tuple) and len(mapping) == 2 and all(isinstance(item, str) for item in mapping):
            rename_map = {mapping[0]: mapping[1]}
        elif isinstance(mapping, (list, tuple)):
            try:
                rename_map = dict(mapping)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise TypeError("mapping must be a dict or a sequence of (old, new) pairs")
        else:
            raise TypeError("mapping must be a dict or a sequence of (old, new) pairs")

        for old_name, new_name in rename_map.items():
            if old_name not in self._columns:
                raise KeyError(f"Column '{old_name}' not found in DataFrame")
            if not isinstance(new_name, str):
                raise TypeError("New column names must be strings")

        if function is not None and not callable(function):
            raise TypeError("function must be callable")

        new_columns: List[str] = []
        for col in self._columns:
            updated_name = rename_map.get(col, col)
            if function is not None:
                updated_name = function(updated_name)
                if not isinstance(updated_name, str):
                    raise TypeError("function must return a string")
            new_columns.append(updated_name)

        if len(set(new_columns)) != len(new_columns):
            raise ValueError("Renaming columns produced duplicate column names")

        new_data = {new_col: self._data[old_col] for old_col, new_col in zip(self._columns, new_columns)}
        new_column_types = {new_col: self._column_types[old_col] for old_col, new_col in zip(self._columns, new_columns)}
        new_categorical = {new_col: self._categorical[old_col] for old_col, new_col in zip(self._columns, new_columns)}

        return DataFrame(new_data, name=self._name, column_types=new_column_types, skip_validation=True, categorical=new_categorical)
    
    def select(self, *columns: str) -> 'DataFrame':
        """
        Select specific columns from the DataFrame.
        
        This is a Polars-compatible method.
        
        Args:
            *columns: Names of columns to select
            
        Returns:
            New DataFrame with only the selected columns
            
        Examples:
            >>> df.select('name', 'age')
            >>> df.select(['name', 'age'])
        """
        # Flatten if a single list was passed
        if len(columns) == 1 and isinstance(columns[0], (list, tuple)):
            columns = columns[0]
        
        new_data = {}
        for col in columns:
            if col not in self._columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
            new_data[col] = self._data[col]
        
        return DataFrame(new_data, name=self._name)