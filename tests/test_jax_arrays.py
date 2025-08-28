#!/usr/bin/env python3
"""
Test script for JAX array handling in DataFrame class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.jaxframe.dataframe import DataFrame

# Import JAX only if available
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# @pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_jax_array_equality_with_other_types():
    """Test that JAX arrays compare equal with equivalent numpy arrays and lists."""
    import jax.numpy as jnp
    
    # Create DataFrame with JAX array
    data_jax = {'values': jnp.array([1, 2, 3])}
    df_jax = DataFrame(data_jax)
    
    # Create DataFrame with numpy array (same values)
    data_numpy = {'values': np.array([1, 2, 3])}
    df_numpy = DataFrame(data_numpy)
    
    # Create DataFrame with list (same values)
    data_list = {'values': [1, 2, 3]}
    df_list = DataFrame(data_list)
    
    # All should be equal despite different storage types
    assert df_jax == df_numpy
    assert df_jax == df_list
    assert df_numpy == df_list
    
    # Verify they have different storage types
    assert df_jax.column_types == {'values': 'jax_array'}
    assert df_numpy.column_types == {'values': 'array'}
    assert df_list.column_types == {'values': 'list'}


# @pytest.mark.skipif(jax_available, reason="Testing JAX fallback behavior")
def test_no_jax_fallback():
    """Test that JAX arrays are properly handled as JAX arrays, not converted to numpy."""
    
    # Create test data with JAX arrays
    data = {
        'list_col': [1, 2, 3, 4],
        'numpy_col': np.array([5, 6, 7, 8]),
        'jax_col': jnp.array([9.0, 10.0, 11.0, 12.0])
    }
    
    df = DataFrame(data)
    
    # Check that JAX array is stored as jax_array type, not converted to numpy
    assert df.column_types['jax_col'] == 'jax_array', "JAX array should be stored as 'jax_array' type"
    assert df.column_types['list_col'] == 'list', "List should be stored as 'list' type"
    assert df.column_types['numpy_col'] == 'array', "NumPy array should be stored as 'array' type"
    
    # Check that the JAX data is preserved as JAX array
    jax_data = df['jax_col']
    assert hasattr(jax_data, 'shape'), "Should be a JAX array with shape attribute"
    assert hasattr(jax_data, 'dtype'), "Should be a JAX array with dtype attribute"
    # Check that it's not a numpy array
    assert not isinstance(jax_data, np.ndarray), "JAX array should not be converted to numpy array"
    
    # Check values are preserved correctly
    expected_values = [9.0, 10.0, 11.0, 12.0]
    assert list(jax_data) == expected_values, "JAX array values should be preserved"
    
    # Check shape and length
    assert jax_data.shape == (4,), "JAX array should preserve correct shape"
    assert len(df) == 4, "DataFrame should have correct length"


# @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_array_to_dict():
    """Test that to_dict preserves JAX arrays as JAX arrays."""
    
    data = {
        'jax_float': jnp.array([1.0, 2.0, 3.0]),
        'jax_int': jnp.array([1, 2, 3]),
        'regular_list': [4, 5, 6]
    }
    
    df = DataFrame(data)
    result_dict = df.to_dict()
    
    # Check that JAX arrays remain as JAX arrays in to_dict
    assert hasattr(result_dict['jax_float'], 'shape'), "JAX float array should remain JAX array"
    assert hasattr(result_dict['jax_int'], 'shape'), "JAX int array should remain JAX array"
    assert not isinstance(result_dict['jax_float'], np.ndarray), "JAX array should not become numpy array"
    assert not isinstance(result_dict['jax_int'], np.ndarray), "JAX array should not become numpy array"
    assert isinstance(result_dict['regular_list'], list), "Regular list should remain a list"
    
    # Check values are preserved
    assert list(result_dict['jax_float']) == [1.0, 2.0, 3.0], "JAX float values should be preserved"
    assert list(result_dict['jax_int']) == [1, 2, 3], "JAX int values should be preserved"
    assert result_dict['regular_list'] == [4, 5, 6], "List values should be preserved"


# @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_array_join_column():
    """Test that join works correctly with JAX arrays."""
    
    # Create DataFrames with JAX arrays
    df1_data = {
        'id': ['A', 'B', 'C'],
        'value1': jnp.array([1.0, 2.0, 3.0])
    }
    df1 = DataFrame(df1_data, name="df1")
    
    df2_data = {
        'id': ['A', 'B', 'C'],
        'value2': jnp.array([10.0, 20.0, 30.0])
    }
    df2 = DataFrame(df2_data, name="df2")
    
    # Join JAX array column
    result = df1.join(df2, on='id', source='value2')
    
    # Check that the joined column is properly handled
    assert 'df2/value2' in result.columns, "Joined column should be present"
    assert result.column_types['df2/value2'] == 'jax_array', "Joined JAX array should remain jax_array type"
    
    # Check values
    joined_values = result['df2/value2']
    assert hasattr(joined_values, 'shape'), "Joined JAX array should remain JAX array"
    assert not isinstance(joined_values, np.ndarray), "Joined JAX array should not become numpy array"
    assert list(joined_values) == [10.0, 20.0, 30.0], "Joined values should be correct"


# @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_mixed_jax_numpy_arrays():
    """Test DataFrame with mixed JAX and NumPy arrays."""
    
    data = {
        'numpy_col': np.array([1, 2, 3]),
        'jax_col': jnp.array([4.0, 5.0, 6.0]),
        'list_col': [7, 8, 9]
    }
    
    df = DataFrame(data)
    
    # All should be properly classified
    assert df.column_types['numpy_col'] == 'array', "NumPy array should be array type"
    assert df.column_types['jax_col'] == 'jax_array', "JAX array should be jax_array type"
    assert df.column_types['list_col'] == 'list', "List should be list type"
    
    # Check that types are preserved
    assert isinstance(df['numpy_col'], np.ndarray), "NumPy column should remain numpy array"
    assert hasattr(df['jax_col'], 'shape'), "JAX column should remain JAX array"
    assert not isinstance(df['jax_col'], np.ndarray), "JAX column should not become numpy array"
    assert isinstance(df['list_col'], list), "List column should remain list"


# @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_array_dtypes():
    """Test that JAX arrays with different dtypes are handled correctly."""
    
    data = {
        'float32': jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        'int32': jnp.array([1, 2, 3], dtype=jnp.int32),
        'bool': jnp.array([True, False, True], dtype=jnp.bool_)
    }
    
    df = DataFrame(data)
    
    # Check that all are treated as JAX arrays
    for col in data.keys():
        assert df.column_types[col] == 'jax_array', f"JAX array column '{col}' should be jax_array type"
        assert hasattr(df[col], 'shape'), f"JAX array column '{col}' should remain JAX array"
        assert not isinstance(df[col], np.ndarray), f"JAX array column '{col}' should not become numpy array"
    
    # Check dtypes are preserved
    assert df['float32'].dtype == jnp.float32, "float32 dtype should be preserved"
    assert df['int32'].dtype == jnp.int32, "int32 dtype should be preserved"
    assert df['bool'].dtype == jnp.bool_, "bool dtype should be preserved"


def test_no_jax_fallback():
    """Test that the code works even when JAX is not available."""
    # This test doesn't require JAX and tests the fallback behavior
    
    # Create a simple array-like object that has the required attributes
    class MockArray:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data),)
            self.dtype = np.float64
        
        def __len__(self):
            return len(self.data)
        
        def __array__(self):
            return np.array(self.data)
        
        def __iter__(self):
            return iter(self.data)
    
    mock_array = MockArray([1.0, 2.0, 3.0])
    
    data = {
        'mock_array': mock_array,
        'regular_list': [4, 5, 6]
    }
    
    df = DataFrame(data)
    
    # Mock array should be converted to numpy array (since it doesn't have JAX-specific attributes)
    assert df.column_types['mock_array'] == 'array', "Mock array should be treated as array"
    assert isinstance(df['mock_array'], np.ndarray), "Mock array should become numpy array"
    assert list(df['mock_array']) == [1.0, 2.0, 3.0], "Mock array values should be preserved"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_add_remove_jax_arrays():
    """Test adding and removing columns/rows with JAX arrays."""
    import jax.numpy as jnp
    
    # Create DataFrame with JAX arrays
    data = {
        'names': ['Alice', 'Bob'],
        'values': jnp.array([1.0, 2.0]),
        'flags': jnp.array([True, False])
    }
    df = DataFrame(data)
    
    # Test adding JAX array column
    new_jax_col = jnp.array([10.0, 20.0])
    df_with_jax = df.add_column('new_jax', new_jax_col)
    
    assert 'new_jax' in df_with_jax.columns
    assert df_with_jax.column_types['new_jax'] == 'jax_array'
    assert jnp.array_equal(df_with_jax['new_jax'], new_jax_col)
    
    # Test adding row with JAX array columns
    new_row = {'names': 'Charlie', 'values': 3.0, 'flags': True, 'new_jax': 30.0}
    df_with_row = df_with_jax.add_row(new_row)
    
    assert df_with_row.shape == (3, 4)
    assert jnp.array_equal(df_with_row['values'], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.array_equal(df_with_row['flags'], jnp.array([True, False, True]))
    assert jnp.array_equal(df_with_row['new_jax'], jnp.array([10.0, 20.0, 30.0]))
    
    # Test removing row from JAX array columns
    df_no_middle = df_with_row.remove_row(1)
    
    assert df_no_middle.shape == (2, 4)
    assert jnp.array_equal(df_no_middle['values'], jnp.array([1.0, 3.0]))
    assert jnp.array_equal(df_no_middle['flags'], jnp.array([True, True]))
    assert jnp.array_equal(df_no_middle['new_jax'], jnp.array([10.0, 30.0]))
    
    # Test removing JAX array column
    df_no_jax = df_no_middle.remove_column('new_jax')
    
    assert 'new_jax' not in df_no_jax.columns
    assert df_no_jax.shape == (2, 3)
    assert jnp.array_equal(df_no_jax['values'], jnp.array([1.0, 3.0]))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_mixed_jax_operations():
    """Test mixed operations with JAX arrays, numpy arrays, and lists."""
    import jax.numpy as jnp
    
    # Create DataFrame with mixed types
    data = {
        'names': ['Alice', 'Bob'],  # list
        'ages': np.array([25, 30]),  # numpy array
        'scores': jnp.array([85.5, 92.0])  # JAX array
    }
    df = DataFrame(data)
    
    # Add mixed types
    df_extended = (df
                   .add_column('grades', ['A', 'A+'])  # list
                   .add_column('weights', np.array([0.8, 0.9]))  # numpy array
                   .add_column('factors', jnp.array([1.1, 1.2]))  # JAX array
                   .add_row({
                       'names': 'Charlie',
                       'ages': 35,
                       'scores': 78.5,
                       'grades': 'B+',
                       'weights': 0.7,
                       'factors': 1.0
                   }))
    
    # Verify types are preserved
    assert df_extended.column_types['names'] == 'list'
    assert df_extended.column_types['ages'] == 'array'
    assert df_extended.column_types['scores'] == 'jax_array'
    assert df_extended.column_types['grades'] == 'list'
    assert df_extended.column_types['weights'] == 'array'
    assert df_extended.column_types['factors'] == 'jax_array'
    
    # Verify values
    assert df_extended['names'] == ['Alice', 'Bob', 'Charlie']
    assert np.array_equal(df_extended['ages'], [25, 30, 35])
    assert jnp.array_equal(df_extended['scores'], jnp.array([85.5, 92.0, 78.5]))
    assert df_extended['grades'] == ['A', 'A+', 'B+']
    assert np.array_equal(df_extended['weights'], [0.8, 0.9, 0.7])
    assert jnp.array_equal(df_extended['factors'], jnp.array([1.1, 1.2, 1.0]))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")  
def test_jax_edge_cases_add_remove():
    """Test edge cases for JAX array add/remove operations."""
    import jax.numpy as jnp
    
    # Test with empty JAX arrays (edge case)
    empty_df = DataFrame({
        'empty_jax': jnp.array([]),
        'empty_list': []
    })
    
    assert empty_df.shape == (0, 2)
    assert empty_df.column_types['empty_jax'] == 'jax_array'
    
    # Test removing first/last elements from single-element JAX arrays
    single_df = DataFrame({
        'single_jax': jnp.array([42.0]),
        'single_list': ['item']
    })
    
    # Can't remove the last row
    with pytest.raises(ValueError, match="Cannot remove the last row"):
        single_df.remove_row(0)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_concat_jax_arrays():
    """Test concatenating DataFrames with JAX arrays."""
    import jax.numpy as jnp
    
    # Create DataFrames with JAX arrays
    df1 = DataFrame({
        'names': ['Alice', 'Bob'],
        'values': jnp.array([1.0, 2.0]),
        'flags': jnp.array([True, False])
    }, name="jax1")
    
    df2 = DataFrame({
        'names': ['Charlie', 'Diana'],
        'values': jnp.array([3.0, 4.0]),
        'flags': jnp.array([True, True])
    }, name="jax2")
    
    # Test row concatenation with JAX arrays
    result = df1.concat(df2, axis=0)
    
    assert result.shape == (4, 3)
    assert result['names'] == ['Alice', 'Bob', 'Charlie', 'Diana']
    assert jnp.array_equal(result['values'], jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert jnp.array_equal(result['flags'], jnp.array([True, False, True, True]))
    
    # Check that JAX array types are preserved
    assert result.column_types['values'] == 'jax_array'
    assert result.column_types['flags'] == 'jax_array'
    assert result.name == "jax1_concat_jax2"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_concat_mixed_jax_numpy():
    """Test concatenating DataFrames with mixed JAX and numpy arrays."""
    import jax.numpy as jnp
    
    # JAX DataFrame
    df_jax = DataFrame({
        'values': jnp.array([1.0, 2.0]),
        'names': ['A', 'B']
    })
    
    # NumPy DataFrame  
    df_numpy = DataFrame({
        'values': np.array([3.0, 4.0]),
        'names': ['C', 'D']
    })
    
    # List DataFrame
    df_list = DataFrame({
        'values': [5.0, 6.0],
        'names': ['E', 'F']
    })
    
    # Test JAX + NumPy concatenation
    result1 = df_jax.concat(df_numpy, axis=0)
    assert result1.column_types['values'] == 'jax_array'  # JAX takes precedence
    assert jnp.array_equal(result1['values'], jnp.array([1.0, 2.0, 3.0, 4.0]))
    
    # Test JAX + List concatenation
    result2 = df_jax.concat(df_list, axis=0)
    assert result2.column_types['values'] == 'jax_array'  # JAX takes precedence
    assert jnp.array_equal(result2['values'], jnp.array([1.0, 2.0, 5.0, 6.0]))
    
    # Test NumPy + JAX concatenation (reverse order)
    result3 = df_numpy.concat(df_jax, axis=0)
    assert result3.column_types['values'] == 'jax_array'  # JAX takes precedence
    assert jnp.array_equal(result3['values'], jnp.array([3.0, 4.0, 1.0, 2.0]))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_concat_jax_horizontal():
    """Test horizontal concatenation with JAX arrays."""
    import jax.numpy as jnp
    
    df1 = DataFrame({
        'names': ['Alice', 'Bob'],
        'jax_values': jnp.array([1.0, 2.0])
    })
    
    df2 = DataFrame({
        'numpy_values': np.array([10.0, 20.0]),
        'list_values': [100, 200]
    })
    
    result = df1.concat(df2, axis=1)
    
    assert result.shape == (2, 4)
    assert result.column_types['jax_values'] == 'jax_array'
    assert result.column_types['numpy_values'] == 'array'
    assert result.column_types['list_values'] == 'list'
    
    assert jnp.array_equal(result['jax_values'], jnp.array([1.0, 2.0]))
    assert np.array_equal(result['numpy_values'], np.array([10.0, 20.0]))
    assert result['list_values'] == [100, 200]


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_concat_multiple_jax_dataframes():
    """Test concatenating multiple DataFrames with JAX arrays using static method."""
    import jax.numpy as jnp
    
    dfs = []
    for i in range(3):
        df = DataFrame({
            'id': [f'id_{i}_0', f'id_{i}_1'],
            'values': jnp.array([i*10.0, i*10.0 + 1.0])
        }, name=f"df_{i}")
        dfs.append(df)
    
    result = DataFrame.concat_dataframes(dfs, axis=0)
    
    assert result.shape == (6, 2)
    expected_ids = ['id_0_0', 'id_0_1', 'id_1_0', 'id_1_1', 'id_2_0', 'id_2_1']
    expected_values = jnp.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
    
    assert result['id'] == expected_ids
    assert jnp.array_equal(result['values'], expected_values)
    assert result.column_types['values'] == 'jax_array'
