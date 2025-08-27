"""
A simple immutable DataFrame implementation using a dictionary of arrays and lists.
"""
from typing import Dict, Any, Union, List, Tuple
import numpy as np


class DataFrame:
    """
    A simple immutable DataFrame that stores data as a dictionary of arrays and/or lists.
    
    All arrays and lists must have the same length. Once created, the DataFrame cannot be modified.
    """
    
    def __init__(self, data: Dict[str, Union[List, np.ndarray]], name: str = None):
        """
        Initialize a DataFrame with a dictionary of arrays and/or lists.
        
        Args:
            data: Dictionary where keys are column names and values are arrays/lists
                 of the same length. Lists will be preserved as lists, arrays as arrays.
            name: Optional name for the DataFrame. If not provided, defaults to None.
                 
        Raises:
            ValueError: If arrays/lists have different lengths or if data is empty.
            TypeError: If data is not a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
            
        if not data:
            raise ValueError("Data dictionary cannot be empty")
        
        # Store data preserving original types (lists vs arrays)
        self._data = {}
        self._column_types = {}  # Track whether each column is a list or array
        lengths = []
        
        for column_name, values in data.items():
            if not isinstance(column_name, str):
                raise TypeError("Column names must be strings")
            
            if isinstance(values, list):
                # Keep as list, but make it immutable by storing a copy
                self._data[column_name] = values.copy()
                self._column_types[column_name] = 'list'
                lengths.append(len(values))
            elif isinstance(values, np.ndarray):
                # Keep as numpy array
                self._data[column_name] = values.copy()
                self._column_types[column_name] = 'array'
                lengths.append(len(values))
            else:
                # Check if it's a JAX array
                try:
                    import jax.numpy as jnp
                    # Check if it's a JAX array by checking for JAX-specific attributes
                    if hasattr(values, 'shape') and hasattr(values, 'dtype') and hasattr(values, '__array__'):
                        # Check if it's actually a JAX array (not just array-like)
                        if hasattr(values, 'device') or str(type(values)).startswith('<class \'jaxlib.'):
                            # This is a JAX array - preserve it as-is
                            self._data[column_name] = values
                            self._column_types[column_name] = 'jax_array'
                            lengths.append(len(values))
                        else:
                            # This is some other array-like object, convert to numpy
                            # Use np.asarray to avoid copy warnings in numpy 2.0+
                            self._data[column_name] = np.asarray(values)
                            self._column_types[column_name] = 'array'
                            lengths.append(len(values))
                    else:
                        # Convert other iterables to list
                        converted_list = list(values)
                        self._data[column_name] = converted_list
                        self._column_types[column_name] = 'list'
                        lengths.append(len(converted_list))
                except ImportError:
                    # JAX not available, fall back to list conversion
                    converted_list = list(values)
                    self._data[column_name] = converted_list
                    self._column_types[column_name] = 'list'
                    lengths.append(len(converted_list))
        
        # Check that all arrays/lists have the same length
        if len(set(lengths)) > 1:
            raise ValueError(f"All arrays and lists must have the same length. Got lengths: {lengths}")
        
        self._length = lengths[0] if lengths else 0
        self._columns = tuple(self._data.keys())  # Immutable tuple of column names
        self._name = name  # Store the name attribute
    
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
    
    def __len__(self) -> int:
        """Get the number of rows in the DataFrame."""
        return self._length
    
    def __getitem__(self, key: str) -> Union[List[Any], np.ndarray, Any]:
        """
        Get a column by name.
        
        Args:
            key: Column name
            
        Returns:
            A copy of the column data as a list (if originally a list), 
            numpy array (if originally a numpy array), or JAX array (if originally a JAX array)
            
        Raises:
            KeyError: If column doesn't exist
        """
        if key not in self._data:
            raise KeyError(f"Column '{key}' not found. Available columns: {list(self._columns)}")
        
        # Return a copy to maintain immutability, preserving original type
        if self._column_types[key] == 'list':
            return self._data[key].copy()
        elif self._column_types[key] == 'jax_array':
            # For JAX arrays, we need to be careful about copying
            # JAX arrays are immutable, so we can return them directly
            return self._data[key]
        else:  # numpy array
            return self._data[key].copy()
    
    def __contains__(self, key: str) -> bool:
        """Check if a column exists in the DataFrame."""
        return key in self._data
    
    def __repr__(self) -> str:
        """String representation of the DataFrame."""
        if self._length == 0:
            name_part = f" '{self._name}'" if self._name else ""
            return f"DataFrame{name_part}(empty)"
        
        name_part = f" '{self._name}'" if self._name else ""
        lines = [f"DataFrame{name_part}({self.shape[0]} rows, {self.shape[1]} columns)"]
        lines.append("Columns: " + ", ".join(self._columns))
        
        # Show first few rows
        max_display_rows = 5
        for i in range(min(self._length, max_display_rows)):
            row_data = []
            for col in self._columns:
                value = self._data[col][i]
                # Format the value nicely
                if isinstance(value, float):
                    row_data.append(f"{value:.3f}")
                else:
                    row_data.append(str(value))
            lines.append(f"  [{i}]: {dict(zip(self._columns, row_data))}")
        
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
    
    def to_dict(self) -> Dict[str, Union[List[Any], np.ndarray, Any]]:
        """
        Convert DataFrame to dictionary of arrays and lists.
        
        Returns:
            Dictionary with copies of the internal data, preserving original types
        """
        result = {}
        for col in self._columns:
            if self._column_types[col] == 'list':
                result[col] = self._data[col].copy()
            elif self._column_types[col] == 'jax_array':
                # JAX arrays are immutable, so we can return them directly
                result[col] = self._data[col]
            else:  # numpy array
                result[col] = self._data[col].copy()
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
    
    def join_column(self, other: 'DataFrame', on_column: str, 
                   source_column: Union[str, List[str]], 
                   target_column: Union[str, List[str], None] = None) -> 'DataFrame':
        """
        Join this DataFrame with another DataFrame by adding column(s) from the other DataFrame.
        
        Args:
            other: The DataFrame to join with
            on_column: The column name to join on (must exist in both DataFrames)
            source_column: The column name(s) in the other DataFrame to copy values from.
                          Can be a string for single column or list of strings for multiple columns.
            target_column: The name(s) for the new column(s) in the result. If None, 
                          defaults to '{other.name}/{source_column}' if other has a name,
                          otherwise just source_column. Can be a string for single column,
                          list of strings for multiple columns, or None for auto-naming.
        
        Returns:
            A new DataFrame with the joined column(s) added
            
        Raises:
            KeyError: If on_column doesn't exist in both DataFrames or source_column(s) 
                     don't exist in other DataFrame
            ValueError: If there are duplicate values in the on_column of other DataFrame,
                       or if target_column list length doesn't match source_column list length
        """
        from typing import Union, List
        
        # Normalize inputs to lists
        if isinstance(source_column, str):
            source_columns = [source_column]
        else:
            source_columns = source_column
            
        if target_column is None:
            target_columns = None
        elif isinstance(target_column, str):
            target_columns = [target_column]
        else:
            target_columns = target_column
            
        # Validate target_columns length if provided
        if target_columns is not None and len(target_columns) != len(source_columns):
            raise ValueError(f"Length of target_column list ({len(target_columns)}) must match "
                           f"length of source_column list ({len(source_columns)})")
        
        # Validate inputs
        if on_column not in self._columns:
            raise KeyError(f"Column '{on_column}' not found in this DataFrame")
        if on_column not in other._columns:
            raise KeyError(f"Column '{on_column}' not found in other DataFrame")
        
        for src_col in source_columns:
            if src_col not in other._columns:
                raise KeyError(f"Column '{src_col}' not found in other DataFrame")
        
        # Check for duplicates in the join column of other DataFrame
        other_join_values = other[on_column]
        if len(set(other_join_values)) != len(other_join_values):
            raise ValueError(f"Duplicate values found in '{on_column}' column of other DataFrame")
        
        # Create lookup dictionaries from other DataFrame for each source column
        lookups = {}
        for src_col in source_columns:
            lookups[src_col] = {}
            for i in range(len(other)):
                key = other._data[on_column][i]
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
        
        # Copy existing columns
        for col in self._columns:
            if self._column_types[col] == 'list':
                new_data[col] = self._data[col].copy()
            elif self._column_types[col] == 'jax_array':
                # JAX arrays are immutable, so we can use them directly
                new_data[col] = self._data[col]
            else:  # numpy array
                new_data[col] = self._data[col].copy()
        
        # Add the joined columns
        self_join_values = self[on_column]
        
        for src_col, target_col in zip(source_columns, target_columns):
            joined_values = []
            lookup = lookups[src_col]
            
            for value in self_join_values:
                if value in lookup:
                    joined_values.append(lookup[value])
                else:
                    raise ValueError(f"Value '{value}' from '{on_column}' not found in other DataFrame")
            
            # Preserve the type from the source column
            if other._column_types[src_col] == 'list':
                new_data[target_col] = joined_values
            elif other._column_types[src_col] == 'jax_array':
                # For JAX arrays, we need to create a new JAX array from the joined values
                try:
                    import jax.numpy as jnp
                    new_data[target_col] = jnp.array(joined_values)
                except ImportError:
                    # JAX not available, fall back to numpy
                    import numpy as np
                    new_data[target_col] = np.asarray(joined_values)
            else:  # numpy array
                import numpy as np
                new_data[target_col] = np.asarray(joined_values)
        
        # Create new DataFrame with updated name
        new_name = f"{self._name}_joined" if self._name else None
        return DataFrame(new_data, name=new_name)
    
    def add_column(self, column_name: str, values: Union[List, np.ndarray, Any]) -> 'DataFrame':
        """
        Add a new column to the DataFrame, returning a new DataFrame.
        
        Args:
            column_name: Name of the new column
            values: Values for the new column (list, numpy array, or JAX array)
                   Must have the same length as existing columns
                   
        Returns:
            New DataFrame with the added column
            
        Raises:
            ValueError: If values length doesn't match DataFrame length or column already exists
        """
        if column_name in self._columns:
            raise ValueError(f"Column '{column_name}' already exists")
        
        # Check length compatibility
        if hasattr(values, '__len__') and len(values) != self._length:
            raise ValueError(f"New column must have length {self._length}, got {len(values)}")
        
        # Create new data dictionary
        new_data = {}
        
        # Copy existing columns
        for col in self._columns:
            if self._column_types[col] == 'list':
                new_data[col] = self._data[col].copy()
            elif self._column_types[col] == 'jax_array':
                new_data[col] = self._data[col]
            else:  # numpy array
                new_data[col] = self._data[col].copy()
        
        # Add the new column
        new_data[column_name] = values
        
        return DataFrame(new_data, name=self._name)
    
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