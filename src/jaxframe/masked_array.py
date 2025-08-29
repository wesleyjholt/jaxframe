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
    - A DataFrame containing index mappings and associated key values
    
    This structure is particularly useful for wide-to-long format conversions
    where some observations may be missing or invalid.
    """
    
    def __init__(self, data: Any, mask: np.ndarray, index_df: DataFrame):
        """
        Initialize a MaskedArray.
        
        Args:
            data: JAX array containing the data values
            mask: Numpy boolean array with same shape as data (True = valid, False = masked)
            index_df: DataFrame containing index mappings and key values
            
        Raises:
            ValueError: If data and mask shapes don't match, or if data rows don't match index_df length
        """
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
        
        if not isinstance(index_df, DataFrame):
            raise ValueError("index_df must be a DataFrame")
        
        if data.shape[0] != len(index_df):
            raise ValueError(f"Number of data rows ({data.shape[0]}) "
                           f"must match index DataFrame length ({len(index_df)})")
        
        self._data = data
        self._mask = mask
        self._index_df = index_df
    
    @property
    def data(self) -> Any:
        """Get the JAX array containing the data values."""
        return self._data
    
    @property
    def mask(self) -> np.ndarray:
        """Get the numpy boolean mask array."""
        return self._mask
    
    @property
    def index_df(self) -> DataFrame:
        """Get the DataFrame containing index mappings and key values."""
        return self._index_df
    
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
                f"Index DataFrame: {len(self._index_df)} rows, "
                f"{len(self._index_df.columns)} columns")
    
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
            
            # Compare index DataFrames
            index_equal = self._index_df == other._index_df
            
            return data_equal and mask_equal and index_equal
            
        except ImportError:
            # Fallback comparison without JAX
            return False
    
    def __hash__(self) -> int:
        """
        Compute hash for the MaskedArray to enable use as static argument in JAX JIT.
        
        Returns:
            Hash value based on shape, mask pattern, and a sample of the data
        """
        # Hash based on shape
        shape_hash = hash(self.shape)
        
        # Hash based on mask pattern (sample for performance)
        mask_sample = self._mask.flatten()[:min(20, self._mask.size)]
        mask_hash = hash(tuple(mask_sample))
        
        # Hash based on data sample
        try:
            import jax.numpy as jnp
            data_flat = self._data.flatten()
            data_sample = data_flat[:min(10, data_flat.size)]
            # Convert JAX array to regular values for hashing
            data_values = [float(x) for x in data_sample]
            data_hash = hash(tuple(data_values))
        except (ImportError, TypeError):
            # Fallback if JAX not available or conversion fails
            data_hash = hash(str(self._data.shape))
        
        # Hash based on index DataFrame
        index_hash = hash(self._index_df)
        
        # Combine all hashes
        return hash((shape_hash, mask_hash, data_hash, index_hash))
    
    def copy(self) -> 'MaskedArray':
        """Create a copy of the MaskedArray."""
        try:
            import jax.numpy as jnp
            # JAX arrays are immutable, but create a copy for consistency
            data_copy = jnp.array(self._data)
        except ImportError:
            data_copy = self._data
            
        return MaskedArray(
            data=data_copy,
            mask=self._mask.copy(),
            index_df=self._index_df  # DataFrames are immutable in jaxframe
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
            'index_df': self._index_df.to_dict(),
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
            if new_data.shape[0] != len(self._index_df):
                raise ValueError(f"Number of data rows ({new_data.shape[0]}) must match index DataFrame length ({len(self._index_df)})")
            
            # Then check overall shape compatibility
            if new_data.shape != self._mask.shape:
                raise ValueError(f"New data shape {new_data.shape} must match mask shape {self._mask.shape}")
        
        return MaskedArray(
            data=new_data,
            mask=self._mask.copy(),  # Copy the mask to ensure independence
            index_df=self._index_df  # DataFrames are immutable in jaxframe
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
        
        try:
            import jax.numpy as jnp
            # JAX arrays are immutable, but create a copy for consistency
            data_copy = jnp.array(self._data)
        except ImportError:
            data_copy = self._data
        
        return MaskedArray(
            data=data_copy,
            mask=new_mask,
            index_df=self._index_df  # DataFrames are immutable in jaxframe
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
            if new_data.shape[0] != len(self._index_df):
                raise ValueError(f"Number of data rows ({new_data.shape[0]}) must match index DataFrame length ({len(self._index_df)})")
            
            # Then check shape compatibility between data and mask
            if new_data.shape != new_mask.shape:
                raise ValueError(f"New data shape {new_data.shape} must match new mask shape {new_mask.shape}")
        
        return MaskedArray(
            data=new_data,
            mask=new_mask,
            index_df=self._index_df  # DataFrames are immutable in jaxframe
        )
    
    def with_index_df(self, new_index_df: 'DataFrame', validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated index DataFrame (immutable operation).
        
        Args:
            new_index_df: New DataFrame to use as index
            validate: Whether to validate compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated index DataFrame
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        if validate:
            # Validate index_df type
            if not hasattr(new_index_df, 'columns'):  # Check if it's DataFrame-like
                raise ValueError("New index_df must be a DataFrame")
            
            # Check number of rows compatibility
            if len(new_index_df) != self._data.shape[0]:
                raise ValueError(f"New index DataFrame length ({len(new_index_df)}) must match data rows ({self._data.shape[0]})")
        
        try:
            import jax.numpy as jnp
            # JAX arrays are immutable, but create a copy for consistency
            data_copy = jnp.array(self._data)
        except ImportError:
            data_copy = self._data
        
        return MaskedArray(
            data=data_copy,
            mask=self._mask.copy(),
            index_df=new_index_df
        )
    
    def with_all(self, new_data: Any = None, new_mask: np.ndarray = None, 
                 new_index_df: 'DataFrame' = None, validate: bool = True) -> 'MaskedArray':
        """
        Create a new MaskedArray with updated data, mask, and/or index DataFrame (immutable operation).
        
        Args:
            new_data: New JAX array to use as data (optional, keeps current if None)
            new_mask: New numpy boolean array to use as mask (optional, keeps current if None)
            new_index_df: New DataFrame to use as index (optional, keeps current if None)
            validate: Whether to validate compatibility (default: True)
            
        Returns:
            New MaskedArray with the updated components
            
        Raises:
            ValueError: If validation fails and validate=True
        """
        # Use current values if not provided
        final_data = new_data if new_data is not None else self._data
        final_mask = new_mask if new_mask is not None else self._mask
        final_index_df = new_index_df if new_index_df is not None else self._index_df
        
        if validate:
            # Validate data
            if not hasattr(final_data, 'shape'):
                raise ValueError("Data must be a JAX array with a shape attribute")
            
            # Validate mask
            if not isinstance(final_mask, np.ndarray):
                raise ValueError("Mask must be a numpy array")
            
            # Validate index_df
            if not hasattr(final_index_df, 'columns'):
                raise ValueError("Index_df must be a DataFrame")
            
            # Check rows compatibility first (more specific error)
            if final_data.shape[0] != len(final_index_df):
                raise ValueError(f"Data rows ({final_data.shape[0]}) must match index DataFrame length ({len(final_index_df)})")
            
            # Then check shape compatibility between data and mask
            if final_data.shape != final_mask.shape:
                raise ValueError(f"Data shape {final_data.shape} must match mask shape {final_mask.shape}")
        
        # Create copies to ensure independence if needed
        try:
            import jax.numpy as jnp
            final_data_copy = jnp.array(final_data) if new_data is not None else final_data
        except ImportError:
            final_data_copy = final_data
        
        final_mask_copy = final_mask.copy() if new_mask is not None else final_mask
        
        return MaskedArray(
            data=final_data_copy,
            mask=final_mask_copy,
            index_df=final_index_df
        )
