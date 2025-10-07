#!/usr/bin/env python3
"""
Comprehensive tests for Polars-compatible DataFrame methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.jaxframe.dataframe import DataFrame


class TestPolarsCompatibleMethods:
    """Test class for Polars-compatible DataFrame methods."""
    
    def test_vstack_basic(self):
        """Test basic vstack functionality."""
        df1 = DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        df2 = DataFrame({
            'a': [7, 8],
            'b': [9, 10]
        })
        
        result = df1.vstack(df2)
        
        assert result.shape == (5, 2)
        assert result['a'] == [1, 2, 3, 7, 8]
        assert result['b'] == [4, 5, 6, 9, 10]
    
    def test_vstack_mismatched_columns(self):
        """Test vstack with mismatched columns raises error."""
        df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = DataFrame({'a': [5, 6], 'c': [7, 8]})
        
        with pytest.raises(ValueError, match="DataFrames must have the same columns"):
            df1.vstack(df2)
    
    def test_vstack_in_place_error(self):
        """Test that vstack with in_place=True raises error."""
        df1 = DataFrame({'a': [1, 2]})
        df2 = DataFrame({'a': [3, 4]})
        
        with pytest.raises(NotImplementedError, match="immutable"):
            df1.vstack(df2, in_place=True)
    
    def test_vstack_type_error(self):
        """Test vstack with non-DataFrame raises TypeError."""
        df = DataFrame({'a': [1, 2]})
        
        with pytest.raises(TypeError, match="Can only vstack with another DataFrame"):
            df.vstack([1, 2, 3])
    
    def test_hstack_with_dataframe(self):
        """Test hstack with another DataFrame."""
        df1 = DataFrame({'a': [1, 2, 3]})
        df2 = DataFrame({'b': [4, 5, 6]})
        
        result = df1.hstack(df2)
        
        assert result.shape == (3, 2)
        assert result['a'] == [1, 2, 3]
        assert result['b'] == [4, 5, 6]
    
    def test_hstack_with_list_of_arrays(self):
        """Test hstack with list of arrays."""
        df = DataFrame({'a': [1, 2, 3]})
        
        result = df.hstack([[4, 5, 6], [7, 8, 9]])
        
        assert result.shape == (3, 3)
        assert result['a'] == [1, 2, 3]
        assert result['column_0'] == [4, 5, 6]
        assert result['column_1'] == [7, 8, 9]
    
    def test_hstack_empty_list(self):
        """Test hstack with empty list returns original DataFrame."""
        df = DataFrame({'a': [1, 2, 3]})
        result = df.hstack([])
        
        assert result == df
    
    def test_hstack_in_place_error(self):
        """Test that hstack with in_place=True raises error."""
        df1 = DataFrame({'a': [1, 2]})
        df2 = DataFrame({'b': [3, 4]})
        
        with pytest.raises(NotImplementedError, match="immutable"):
            df1.hstack(df2, in_place=True)
    
    def test_hstack_type_error(self):
        """Test hstack with invalid type raises TypeError."""
        df = DataFrame({'a': [1, 2]})
        
        with pytest.raises(TypeError, match="columns must be a DataFrame or list"):
            df.hstack("invalid")
    
    def test_with_columns_dict_positional(self):
        """Test with_columns using dictionary as positional argument."""
        df = DataFrame({'a': [1, 2, 3]})
        
        result = df.with_columns({'b': [4, 5, 6]})
        
        assert result.shape == (3, 2)
        assert result['a'] == [1, 2, 3]
        assert result['b'] == [4, 5, 6]
    
    def test_with_columns_keyword_args(self):
        """Test with_columns using keyword arguments."""
        df = DataFrame({'a': [1, 2, 3]})
        
        result = df.with_columns(b=[4, 5, 6], c=[7, 8, 9])
        
        assert result.shape == (3, 3)
        assert result['a'] == [1, 2, 3]
        assert result['b'] == [4, 5, 6]
        assert result['c'] == [7, 8, 9]
    
    def test_with_columns_mixed_args(self):
        """Test with_columns using both positional and keyword arguments."""
        df = DataFrame({'a': [1, 2, 3]})
        
        result = df.with_columns({'b': [4, 5, 6]}, c=[7, 8, 9])
        
        assert result.shape == (3, 3)
        assert result['a'] == [1, 2, 3]
        assert result['b'] == [4, 5, 6]
        assert result['c'] == [7, 8, 9]
    
    def test_with_columns_replace_existing(self):
        """Test with_columns replaces existing columns."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = df.with_columns(a=[10, 20, 30])
        
        assert result.shape == (3, 2)
        assert result['a'] == [10, 20, 30]
        assert result['b'] == [4, 5, 6]
    
    def test_with_columns_invalid_positional(self):
        """Test with_columns with invalid positional argument raises TypeError."""
        df = DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(TypeError, match="Positional arguments must be dictionaries"):
            df.with_columns([1, 2, 3])
    
    def test_drop_single_column(self):
        """Test dropping a single column."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        
        result = df.drop('b')
        
        assert result.shape == (3, 2)
        assert result.columns == ('a', 'c')
        assert result['a'] == [1, 2, 3]
        assert result['c'] == [7, 8, 9]
    
    def test_drop_multiple_columns(self):
        """Test dropping multiple columns."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        
        result = df.drop('a', 'c')
        
        assert result.shape == (3, 1)
        assert result.columns == ('b',)
        assert result['b'] == [4, 5, 6]
    
    def test_drop_list_of_columns(self):
        """Test dropping columns passed as a list."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        
        result = df.drop(['a', 'c'])
        
        assert result.shape == (3, 1)
        assert result.columns == ('b',)
        assert result['b'] == [4, 5, 6]
    
    def test_drop_nonexistent_column_strict(self):
        """Test dropping non-existent column with strict=True raises error."""
        df = DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(KeyError, match="not found in DataFrame"):
            df.drop('nonexistent')
    
    def test_drop_nonexistent_column_not_strict(self):
        """Test dropping non-existent column with strict=False."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = df.drop('nonexistent', 'b', strict=False)
        
        assert result.shape == (3, 1)
        assert result.columns == ('a',)
        assert result['a'] == [1, 2, 3]
    
    def test_drop_no_columns(self):
        """Test drop with no columns returns original DataFrame."""
        df = DataFrame({'a': [1, 2, 3]})
        result = df.drop()
        
        assert result == df
    
    def test_filter_with_boolean_array(self):
        """Test filter with boolean array predicate."""
        df = DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        
        result = df.filter([True, False, True, False])
        
        assert result.shape == (2, 2)
        assert result['a'] == [1, 3]
        assert result['b'] == [5, 7]
    
    def test_filter_with_constraints(self):
        """Test filter with column value constraints."""
        df = DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Alice'],
            'age': [25, 30, 35, 25],
            'city': ['NYC', 'LA', 'NYC', 'Chicago']
        })
        
        result = df.filter(name='Alice')
        
        assert result.shape == (2, 3)
        assert result['name'] == ['Alice', 'Alice']
        assert result['age'] == [25, 25]
        assert result['city'] == ['NYC', 'Chicago']
    
    def test_filter_with_multiple_constraints(self):
        """Test filter with multiple column constraints (AND logic)."""
        df = DataFrame({
            'name': ['Alice', 'Bob', 'Alice'],
            'age': [25, 30, 25],
            'city': ['NYC', 'LA', 'NYC']
        })
        
        result = df.filter(name='Alice', city='NYC')
        
        assert result.shape == (2, 3)
        assert result['name'] == ['Alice', 'Alice']
        assert result['age'] == [25, 25]
        assert result['city'] == ['NYC', 'NYC']
    
    def test_filter_with_boolean_and_constraints(self):
        """Test filter with both boolean array and constraints."""
        df = DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        
        result = df.filter([True, True, False], name='Alice')
        
        assert result.shape == (1, 2)
        assert result['name'] == ['Alice']
        assert result['age'] == [25]
    
    def test_filter_invalid_predicate_type(self):
        """Test filter with invalid predicate type raises TypeError."""
        df = DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(TypeError, match="Predicates must be boolean arrays"):
            df.filter([1, 2, 3])  # Not boolean
    
    def test_filter_wrong_predicate_length(self):
        """Test filter with wrong length predicate raises ValueError."""
        df = DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="doesn't match DataFrame length"):
            df.filter([True, False])  # Length 2, but DataFrame has 3 rows
    
    def test_filter_nonexistent_column_constraint(self):
        """Test filter with constraint on non-existent column raises KeyError."""
        df = DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            df.filter(nonexistent='value')
    
    def test_filter_no_predicates_or_constraints(self):
        """Test filter with no predicates or constraints returns original DataFrame."""
        df = DataFrame({'a': [1, 2, 3]})
        result = df.filter()
        
        assert result == df
    
    def test_filter_with_numpy_arrays(self):
        """Test filter works with DataFrames containing numpy arrays."""
        df = DataFrame({
            'values': np.array([1, 2, 3, 4]),
            'flags': np.array([True, False, True, False])
        })
        
        result = df.filter([True, True, False, False])
        
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result['values'], [1, 2])
        np.testing.assert_array_equal(result['flags'], [True, False])

    def test_all_methods_preserve_immutability(self):
        """Test that all new methods preserve DataFrame immutability."""
        original = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        original_data_copy = {k: v.copy() if hasattr(v, 'copy') else list(v) 
                             for k, v in original._data.items()}
        
        # Test vstack
        other = DataFrame({'a': [7, 8], 'b': [9, 10]})
        result1 = original.vstack(other)
        assert original._data.keys() == original_data_copy.keys()
        assert all(np.array_equal(original._data[k], original_data_copy[k]) 
                  for k in original._data.keys())
        
        # Test hstack
        result2 = original.hstack(DataFrame({'c': [7, 8, 9]}))
        assert original._data.keys() == original_data_copy.keys()
        
        # Test with_columns
        result3 = original.with_columns(c=[7, 8, 9])
        assert original._data.keys() == original_data_copy.keys()
        
        # Test drop
        result4 = original.drop('a')
        assert original._data.keys() == original_data_copy.keys()
        
        # Test filter
        result5 = original.filter([True, False, True])
        assert original._data.keys() == original_data_copy.keys()


class TestPolarsCompatibilityIntegration:
    """Integration tests comparing behavior with expected Polars-like patterns."""
    
    def test_method_chaining(self):
        """Test that methods can be chained like in Polars."""
        df = DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40],
            'salary': [50000, 60000, 70000, 80000]
        })
        
        # Calculate bonus manually since JAXFrame doesn't support expressions
        bonus = [s * 0.1 for s in df['salary']]
        
        result = (df
                 .with_columns(bonus=bonus)
                 .filter(age=25)
                 .drop('salary'))
        
        assert result.shape == (1, 3)
        assert result['name'] == ['Alice']
        assert result['age'] == [25]
        assert result['bonus'] == [5000.0]
    
    def test_complex_operations(self):
        """Test complex operations combining multiple methods."""
        df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = DataFrame({'a': [5, 6], 'b': [7, 8]})
        df3 = DataFrame({'c': [9, 10, 11, 12]})
        
        result = (df1
                 .vstack(df2)
                 .hstack(df3)
                 .with_columns(sum_col=[df1['a'][0] + df1['b'][0]] * 4)  # Simplified calculation
                 .filter([True, False, True, False])
                 .drop('b'))
        
        assert result.shape == (2, 3)
        assert 'a' in result.columns
        assert 'c' in result.columns
        assert 'sum_col' in result.columns
        assert 'b' not in result.columns


if __name__ == "__main__":
    pytest.main([__file__])