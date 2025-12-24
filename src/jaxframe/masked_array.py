"""
Masked array implementation for JAXFrame.

This module provides a MaskedArray class that combines JAX arrays with masks
and associated metadata in a single data structure.
"""

from typing import Any, Union, List
import numpy as np
from .dataframe import DataFrame


class MaskedArray:
    """
    A data structure that combines JAX arrays with boolean masks and metadata.
    
    This class encapsulates:
    - A JAX array containing the actual data values
    - A numpy boolean array serving as a mask (True = valid, False = masked)
    - A wide_skeleton_df that preserves the original wide format structure
      including order columns for restoring data to original positions
    - A list of index_columns names used to extract the index DataFrame from the skeleton
    
    This structure is particularly useful for wide-to-long format conversions
    where some observations may be missing or invalid.
    """
    
    def __init__(self, data: Any, mask: np.ndarray, 
                 wide_skeleton_df: DataFrame,
                 index_columns: Union[str, List[str]],
                 validate: bool = True):
        """
        Initialize a MaskedArray.
        
        Args:
            data: JAX array containing the data values
            mask: Numpy boolean array with same shape as data (True = valid, False = masked)
            wide_skeleton_df: DataFrame containing the original wide format structure
                including order columns (e.g., var$0$order, var$1$order) and index columns
            index_columns: Column name(s) in wide_skeleton_df that identify each row
            validate: Whether to validate inputs (default: True)
            
        Raises:
            ValueError: If data and mask shapes don't match, or if data rows don't match skeleton length
        """
        # Normalize index_columns to list
        if isinstance(index_columns, str):
            index_columns = [index_columns]
        
        if validate:
            try:
                import jax.numpy as jnp
            except ImportError:
                raise ImportError("JAX is required for MaskedArray. Install with: pip install jax")
            
            # Validate inputs
            if not hasattr(data, 'shape'):
                raise ValueError("Data must be a JAX array with a shape attribute")
            
            if not isinstance(mask, np.ndarray):
                raise ValueError("Mask must be a numpy array")
            
            if data.shape != mask.shape:
                raise ValueError(f"Data and mask must have the same shape. "
                               f"Got data: {data.shape}, mask: {mask.shape}")
            
            if not isinstance(wide_skeleton_df, DataFrame):
                raise ValueError("wide_skeleton_df must be a DataFrame")
            
            if data.shape[0] != len(wide_skeleton_df):
                raise ValueError(f"Number of data rows ({data.shape[0]}) "
                               f"must match wide_skeleton_df length ({len(wide_skeleton_df)})")
            
            # Validate index_columns exist in wide_skeleton_df
            for col in index_columns:
                if col not in wide_skeleton_df.columns:
                    raise ValueError(f"Index column '{col}' not found in wide_skeleton_df")
        
        self._data = data
        self._mask = mask
        self._wide_skeleton_df = wide_skeleton_df
        self._index_columns = list(index_columns)
    
    @classmethod
    def from_validated_components(cls, data: Any, mask: np.ndarray, 
                                   wide_skeleton_df: DataFrame,
                                   index_columns: Union[str, List[str]]) -> 'MaskedArray':
        """
        Create a MaskedArray from pre-validated components (faster constructor).
        
        Args:
            data: JAX array (pre-validated)
            mask: Numpy boolean array (pre-validated)
            wide_skeleton_df: DataFrame with wide format skeleton (pre-validated)
            index_columns: Column name(s) for index (pre-validated)
            
        Returns:
            New MaskedArray without validation overhead
            
        Note:
            This is a performance optimization for cases where components are known to be valid.
            Use regular constructor for safety if unsure about input validity.
        """
        return cls(data, mask, wide_skeleton_df, index_columns, validate=False)
    
    @classmethod
    def create_zeros_masked(cls, shape: tuple, wide_skeleton_df: DataFrame,
                           index_columns: Union[str, List[str]],
                           fill_value: float = 0.0, 
                           mask_value: bool = True) -> 'MaskedArray':
        """
        Create a MaskedArray filled with zeros and uniform mask values.
        
        Args:
            shape: Shape of the data/mask arrays
            wide_skeleton_df: DataFrame containing the wide format structure
            index_columns: Column name(s) in wide_skeleton_df for index
            fill_value: Value to fill the data array with (default: 0.0)
            mask_value: Value to fill the mask array with (default: True for valid)
            
        Returns:
            New MaskedArray with zeros data and uniform mask
            
        Raises:
            ValueError: If shape[0] doesn't match wide_skeleton_df length
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX is required for MaskedArray. Install with: pip install jax")
            
        if shape[0] != len(wide_skeleton_df):
            raise ValueError(f"Shape[0] ({shape[0]}) must match wide_skeleton_df length ({len(wide_skeleton_df)})")
        
        data = jnp.full(shape, fill_value, dtype=jnp.float32)
        mask = np.full(shape, mask_value, dtype=bool)
        
        return cls.from_validated_components(data, mask, wide_skeleton_df, index_columns)
    
    @property
    def data(self) -> Any:
        """Get the JAX array containing the data values."""
        return self._data
    
    @property
    def mask(self) -> np.ndarray:
        """Get the numpy boolean mask array."""
        return self._mask
    
    @property
    def index_columns(self) -> List[str]:
        """Get the list of column names used as index."""
        return self._index_columns
    
    @property
    def index_df(self) -> DataFrame:
        """Get the DataFrame containing index mappings and key values.
        
        This is derived from wide_skeleton_df using index_columns.
        """
        index_data = {col: self._wide_skeleton_df[col] for col in self._index_columns}
        return DataFrame(index_data)
    
    @property
    def wide_skeleton_df(self) -> DataFrame:
        """Get the wide format skeleton DataFrame with order columns."""
        return self._wide_skeleton_df
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the data/mask arrays."""
        return self._data.shape
    
    def __repr__(self) -> str:
        """String representation of the MaskedArray."""
        n_rows, n_cols = self.shape
        n_valid = np.sum(self._mask)
        n_total = self._mask.size
        valid_pct = (n_valid / n_total * 100) if n_total > 0 else 0
        
        return (f"MaskedArray({n_rows} rows, {n_cols} columns)\n"
                f"Valid values: {n_valid}/{n_total} ({valid_pct:.1f}%)\n"
                f"Index columns: {self._index_columns}")
    
    def __eq__(self, other) -> bool:
        """Check equality with another MaskedArray."""
        if not isinstance(other, MaskedArray):
            return False
        
        try:
            import jax.numpy as jnp
            
            # Compare data arrays
            data_equal = jnp.array_equal(self._data, other._data)
            
            # Compare mask arrays
            mask_equal = np.array_equal(self._mask, other._mask)
            
            # Compare index columns
            index_cols_equal = self._index_columns == other._index_columns
            
            # Compare index DataFrames (derived from skeleton)
            index_equal = self.index_df == other.index_df
            
            return data_equal and mask_equal and index_cols_equal and index_equal
            
        except ImportError:
            # Fallback comparison without JAX
            return False
    
    def copy(self, deep: bool = True) -> 'MaskedArray':
        """
        Create a copy of the MaskedArray.
        
        Args:
            deep: Whether to deep copy arrays (default: True).
                 For backward compatibility, creates explicit JAX array copy when deep=True.
                 
        Returns:
            New MaskedArray copy
        """
        # Create explicit copy of JAX array for backward compatibility
        if deep:
            try:
                import jax.numpy as jnp
                data_copy = jnp.array(self._data)
            except ImportError:
                data_copy = self._data
        else:
            data_copy = self._data
        
        # Copy mask array if requested
        mask_copy = self._mask.copy() if deep else self._mask
            
        return MaskedArray.from_validated_components(
            data=data_copy,
            mask=mask_copy,
            wide_skeleton_df=self._wide_skeleton_df,  # DataFrames are immutable in jaxframe
            index_columns=self._index_columns
        )
    
    def get_valid_data(self) -> Any:
        """Get only the valid (non-masked) data values as a 1D array."""
        try:
            import jax.numpy as jnp
            return self._data[self._mask]
        except ImportError:
            raise ImportError("JAX is required for get_valid_data()")
    
    def to_dict(self) -> dict:
        """Convert to a dictionary representation."""
        return {
            'data': self._data,
            'mask': self._mask,
            'wide_skeleton_df': self._wide_skeleton_df.to_dict(),
            'index_columns': self._index_columns,
            'shape': self.shape
        }
    
    def with_data(self, new_data: Any, validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated data (immutable operation).
        
        Args:
            new_data: New JAX array to use as data
            validate: Whether to validate shape compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated data
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        if validate:
            # Validate that new data has correct shape
            if not hasattr(new_data, 'shape'):
                raise ValueError("New data must be a JAX array with a shape attribute")
            
            # Check number of rows first (more specific error)
            if new_data.shape[0] != len(self._wide_skeleton_df):
                raise ValueError(f"Number of data rows ({new_data.shape[0]}) must match wide_skeleton_df length ({len(self._wide_skeleton_df)})")
            
            # Then check overall shape compatibility
            if new_data.shape != self._mask.shape:
                raise ValueError(f"New data shape {new_data.shape} must match mask shape {self._mask.shape}")
        
        return MaskedArray.from_validated_components(
            data=new_data,
            mask=self._mask.copy(),  # Copy the mask to ensure independence
            wide_skeleton_df=self._wide_skeleton_df,  # DataFrames are immutable in jaxframe
            index_columns=self._index_columns
        )
    
    def with_mask(self, new_mask: np.ndarray, validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated mask (immutable operation).
        
        Args:
            new_mask: New numpy boolean array to use as mask
            validate: Whether to validate shape compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated mask
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        if validate:
            if not isinstance(new_mask, np.ndarray):
                raise ValueError("New mask must be a numpy array")
            
            if new_mask.shape != self._data.shape:
                raise ValueError(f"New mask shape {new_mask.shape} must match data shape {self._data.shape}")
        
        # JAX arrays are immutable, so no need to copy
        return MaskedArray.from_validated_components(
            data=self._data,
            mask=new_mask,
            wide_skeleton_df=self._wide_skeleton_df,  # DataFrames are immutable in jaxframe
            index_columns=self._index_columns
        )
    
    def with_data_and_mask(self, new_data: Any, new_mask: np.ndarray, validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated data and mask (immutable operation).
        
        Args:
            new_data: New JAX array to use as data
            new_mask: New numpy boolean array to use as mask
            validate: Whether to validate shape compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated data and mask
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        if validate:
            # Validate data
            if not hasattr(new_data, 'shape'):
                raise ValueError("New data must be a JAX array with a shape attribute")
            
            # Validate mask
            if not isinstance(new_mask, np.ndarray):
                raise ValueError("New mask must be a numpy array")
            
            # Check number of rows first (more specific error)
            if new_data.shape[0] != len(self._wide_skeleton_df):
                raise ValueError(f"Number of data rows ({new_data.shape[0]}) must match wide_skeleton_df length ({len(self._wide_skeleton_df)})")
            
            # Then check shape compatibility between data and mask
            if new_data.shape != new_mask.shape:
                raise ValueError(f"New data shape {new_data.shape} must match new mask shape {new_mask.shape}")
        
        return MaskedArray.from_validated_components(
            data=new_data,
            mask=new_mask,
            wide_skeleton_df=self._wide_skeleton_df,  # DataFrames are immutable in jaxframe
            index_columns=self._index_columns
        )
    
    def with_index_columns(self, new_index_columns: Union[str, List[str]], validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated index columns (immutable operation).
        
        Args:
            new_index_columns: New column name(s) to use as index
            validate: Whether to validate compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated index columns
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        # Normalize to list
        if isinstance(new_index_columns, str):
            new_index_columns = [new_index_columns]
        
        if validate:
            # Check that all columns exist in wide_skeleton_df
            for col in new_index_columns:
                if col not in self._wide_skeleton_df.columns:
                    raise ValueError(f"Index column '{col}' not found in wide_skeleton_df")
        
        return MaskedArray.from_validated_components(
            data=self._data,
            mask=self._mask.copy(),
            wide_skeleton_df=self._wide_skeleton_df,
            index_columns=new_index_columns
        )
    
    def with_wide_skeleton_df(self, new_wide_skeleton_df: 'DataFrame', 
                               new_index_columns: Union[str, List[str]] = None,
                               validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated wide_skeleton_df (immutable operation).
        
        Args:
            new_wide_skeleton_df: New DataFrame to use as wide skeleton
            new_index_columns: New column names for index. If None, uses current index_columns.
            validate: Whether to validate compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated wide_skeleton_df
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        # Use current index_columns if not specified
        final_index_columns = new_index_columns if new_index_columns is not None else self._index_columns
        if isinstance(final_index_columns, str):
            final_index_columns = [final_index_columns]
        
        if validate:
            if not hasattr(new_wide_skeleton_df, 'columns'):
                raise ValueError("New wide_skeleton_df must be a DataFrame")
            
            if len(new_wide_skeleton_df) != self._data.shape[0]:
                raise ValueError(f"New wide_skeleton_df length ({len(new_wide_skeleton_df)}) must match data rows ({self._data.shape[0]})")
            
            # Check that index columns exist in new skeleton
            for col in final_index_columns:
                if col not in new_wide_skeleton_df.columns:
                    raise ValueError(f"Index column '{col}' not found in new wide_skeleton_df")
        
        return MaskedArray.from_validated_components(
            data=self._data,
            mask=self._mask.copy(),
            wide_skeleton_df=new_wide_skeleton_df,
            index_columns=final_index_columns
        )
    
    def with_all(self, new_data: Any = None, new_mask: np.ndarray = None, 
                 new_wide_skeleton_df: 'DataFrame' = None, 
                 new_index_columns: Union[str, List[str]] = None,
                 validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated data, mask, and/or skeleton (immutable operation).
        
        Args:
            new_data: New JAX array to use as data (optional, keeps current if None)
            new_mask: New numpy boolean array to use as mask (optional, keeps current if None)
            new_wide_skeleton_df: New DataFrame for wide skeleton (optional, keeps current if None)
            new_index_columns: New column names for index (optional, keeps current if None)
            validate: Whether to validate compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated components
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        # Use current values if not provided
        final_data = new_data if new_data is not None else self._data
        final_mask = new_mask if new_mask is not None else self._mask
        final_wide_skeleton_df = new_wide_skeleton_df if new_wide_skeleton_df is not None else self._wide_skeleton_df
        final_index_columns = new_index_columns if new_index_columns is not None else self._index_columns
        
        # Normalize index_columns to list
        if isinstance(final_index_columns, str):
            final_index_columns = [final_index_columns]
        
        if validate:
            # Validate data
            if not hasattr(final_data, 'shape'):
                raise ValueError("Data must be a JAX array with a shape attribute")
            
            # Validate mask
            if not isinstance(final_mask, np.ndarray):
                raise ValueError("Mask must be a numpy array")
            
            # Validate wide_skeleton_df
            if not hasattr(final_wide_skeleton_df, 'columns'):
                raise ValueError("wide_skeleton_df must be a DataFrame")
            
            # Check rows compatibility first (more specific error)
            if final_data.shape[0] != len(final_wide_skeleton_df):
                raise ValueError(f"Data rows ({final_data.shape[0]}) must match wide_skeleton_df length ({len(final_wide_skeleton_df)})")
            
            # Then check shape compatibility between data and mask
            if final_data.shape != final_mask.shape:
                raise ValueError(f"Data shape {final_data.shape} must match mask shape {final_mask.shape}")
            
            # Validate index columns exist in skeleton
            for col in final_index_columns:
                if col not in final_wide_skeleton_df.columns:
                    raise ValueError(f"Index column '{col}' not found in wide_skeleton_df")
        
        # Create copies to ensure independence if needed
        final_mask_copy = final_mask.copy() if new_mask is not None else final_mask
        
        return MaskedArray.from_validated_components(
            data=final_data,
            mask=final_mask_copy,
            wide_skeleton_df=final_wide_skeleton_df,
            index_columns=final_index_columns
        )
