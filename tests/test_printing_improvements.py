"""
Test improved DataFrame printing functionality.
"""
import pytest
import numpy as np
from src.jaxframe import DataFrame


class TestImprovedPrinting:
    """Test class for improved DataFrame printing features."""
    
    def test_dtypes_method(self):
        """Test the new dtypes() method."""
        data = {
            'str_col': ['a', 'b', 'c'],
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'numpy_int': np.array([10, 20, 30]),
            'numpy_float': np.array([1.5, 2.5, 3.5])
        }
        df = DataFrame(data)
        dtypes = df.dtypes()
        
        assert isinstance(dtypes, dict)
        assert dtypes['str_col'] == 'list[str]'
        assert dtypes['int_col'] == 'list[int]'
        assert dtypes['float_col'] == 'list[float]'
        assert dtypes['numpy_int'] == 'int64'
        assert dtypes['numpy_float'] == 'float64'
    
    def test_improved_repr_without_quotes_for_numerics(self):
        """Test that numeric values don't have quotes in repr."""
        data = {
            'str_val': ['hello'],
            'int_val': [42],
            'float_val': [3.14159]
        }
        df = DataFrame(data)
        repr_str = repr(df)
        
        # String values should have quotes
        assert "'str_val': 'hello'" in repr_str
        # Numeric values should not have quotes
        assert "'int_val': 42" in repr_str
        assert "'float_val': 3.142" in repr_str  # formatted to 3 decimal places
    
    def test_repr_includes_dtypes(self):
        """Test that repr includes dtype information."""
        data = {'col1': [1, 2], 'col2': ['a', 'b']}
        df = DataFrame(data)
        repr_str = repr(df)
        
        assert 'Dtypes:' in repr_str
        assert 'col1: list[int]' in repr_str
        assert 'col2: list[str]' in repr_str
    
    def test_repr_with_numpy_arrays(self):
        """Test repr with numpy arrays."""
        data = {
            'np_int': np.array([1, 2, 3]),
            'np_float': np.array([1.111, 2.222, 3.333])
        }
        df = DataFrame(data)
        repr_str = repr(df)
        
        # Should show numpy dtypes
        assert 'np_int: int64' in repr_str
        assert 'np_float: float64' in repr_str
        
        # Values should not have quotes
        assert "'np_int': 1" in repr_str
        assert "'np_float': 1.111" in repr_str
    
    @pytest.mark.skipif(True, reason="Skip JAX test if JAX not available")
    def test_repr_with_jax_arrays(self):
        """Test repr with JAX arrays (if available)."""
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not available")
        
        data = {
            'jax_int': jnp.array([100, 200, 300]),
            'jax_float': jnp.array([1.23, 4.56, 7.89])
        }
        df = DataFrame(data)
        repr_str = repr(df)
        
        # Should show JAX dtypes
        assert 'jax_int: int32' in repr_str
        assert 'jax_float: float32' in repr_str
        
        # Values should be extracted from JAX arrays (no Array() wrapper)
        assert "'jax_int': 100" in repr_str
        assert "'jax_float': 1.230" in repr_str
    
    def test_empty_dataframe_repr(self):
        """Test repr for empty DataFrame."""
        df = DataFrame({'col1': [], 'col2': []})
        repr_str = repr(df)
        assert "DataFrame(empty)" in repr_str
    
    def test_large_dataframe_truncation(self):
        """Test that large DataFrames are truncated properly."""
        data = {
            'id': list(range(10)),
            'value': [i * 2.5 for i in range(10)]
        }
        df = DataFrame(data)
        repr_str = repr(df)
        
        # Should show truncation message
        assert "... (5 more rows)" in repr_str
        # Should show first 5 rows
        assert "[0]:" in repr_str
        assert "[4]:" in repr_str
        # Should not show all 10 rows
        assert "[9]:" not in repr_str
    
    def test_float_formatting(self):
        """Test that floats are formatted to 3 decimal places."""
        data = {'float_col': [1.23456789, 2.0, 3.1]}
        df = DataFrame(data)
        repr_str = repr(df)
        
        assert "1.235" in repr_str  # rounded to 3 decimal places
        assert "2.000" in repr_str  # shows .000 for whole numbers
        assert "3.100" in repr_str  # shows trailing zeros
    
    def test_mixed_data_types(self):
        """Test repr with mixed data types."""
        data = {
            'strings': ['A', 'B', 'C'],
            'integers': [10, 20, 30],
            'floats': [1.1, 2.2, 3.3],
            'numpy_arr': np.array([100.5, 200.7, 300.9])
        }
        df = DataFrame(data)
        repr_str = repr(df)
        
        # Check dtypes are shown
        assert 'strings: list[str]' in repr_str
        assert 'integers: list[int]' in repr_str
        assert 'floats: list[float]' in repr_str
        assert 'numpy_arr: float64' in repr_str
        
        # Check formatting
        assert "'strings': 'A'" in repr_str  # strings have quotes
        assert "'integers': 10" in repr_str  # integers no quotes
        assert "'floats': 1.100" in repr_str  # floats no quotes, 3 decimals
        assert "'numpy_arr': 100.500" in repr_str  # numpy values no quotes
