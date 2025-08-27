#!/usr/bin/env python3
"""
Test script for DataFrame functionality with mixed lists and arrays using pytest.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.jaxframe.dataframe import DataFrame


def test_mixed_types():
    """Test DataFrame with mixed list and array columns."""
    
    # Create DataFrame with mixed types
    data = {
        'names': ['Alice', 'Bob', 'Charlie'],  # List
        'ages': np.array([25, 30, 35]),        # NumPy array
        'scores': [85.5, 92.0, 78.5],         # List
        'weights': np.array([120.5, 180.0, 165.2])  # NumPy array
    }
    
    df = DataFrame(data)
    assert df.shape == (3, 4)
    
    # Check column types
    expected_types = {
        'names': 'list',
        'ages': 'array', 
        'scores': 'list',
        'weights': 'array'
    }
    assert df.column_types == expected_types
    
    # Test accessing columns preserves types
    names = df['names']
    ages = df['ages']
    scores = df['scores']
    weights = df['weights']
    
    assert isinstance(names, list)
    assert isinstance(ages, np.ndarray)
    assert isinstance(scores, list)
    assert isinstance(weights, np.ndarray)
    
    assert names == ['Alice', 'Bob', 'Charlie']
    assert np.array_equal(ages, [25, 30, 35])
    assert scores == [85.5, 92.0, 78.5]
    assert np.array_equal(weights, [120.5, 180.0, 165.2])
    
    # Test immutability for both types
    original_names = df['names'].copy()
    names_copy = df['names']
    names_copy[0] = 'Modified'
    current_names = df['names']
    assert original_names == current_names
    
    original_ages = df['ages'].copy()
    ages_copy = df['ages']
    ages_copy[0] = 999
    current_ages = df['ages']
    assert np.array_equal(original_ages, current_ages)


def test_all_lists():
    """Test DataFrame with all list columns."""
    
    data = {
        'a': [1, 2, 3],
        'b': [4.0, 5.0, 6.0],
        'c': ['x', 'y', 'z']
    }
    
    df = DataFrame(data)
    assert df.shape == (3, 3)
    
    expected_types = {'a': 'list', 'b': 'list', 'c': 'list'}
    assert df.column_types == expected_types
    
    # Verify all columns are lists
    for col in df.columns:
        col_data = df[col]
        assert isinstance(col_data, list)


def test_all_arrays():
    """Test DataFrame with all array columns."""
    
    data = {
        'a': np.array([1, 2, 3]),
        'b': np.array([4.0, 5.0, 6.0]),
        'c': np.array(['x', 'y', 'z'])
    }
    
    df = DataFrame(data)
    assert df.shape == (3, 3)
    
    expected_types = {'a': 'array', 'b': 'array', 'c': 'array'}
    assert df.column_types == expected_types
    
    # Verify all columns are arrays
    for col in df.columns:
        col_data = df[col]
        assert isinstance(col_data, np.ndarray)


def test_equality_mixed_types():
    """Test equality between DataFrames with different storage types but same data."""
    
    # Same data, different storage types
    data1 = {
        'a': [1, 2, 3],           # List
        'b': np.array([4, 5, 6])  # Array
    }
    
    data2 = {
        'a': np.array([1, 2, 3]), # Array (same values as list above)
        'b': [4, 5, 6]            # List (same values as array above)
    }
    
    df1 = DataFrame(data1)
    df2 = DataFrame(data2)
    
    # Verify different storage types
    assert df1.column_types == {'a': 'list', 'b': 'array'}
    assert df2.column_types == {'a': 'array', 'b': 'list'}
    
    # Should be equal despite different storage types
    assert df1 == df2


def test_column_selection():
    """Test column selection preserves types."""
    
    data = {
        'list_col': [1, 2, 3],
        'array_col': np.array([4, 5, 6]),
        'another_list': ['a', 'b', 'c'],
        'another_array': np.array([7.0, 8.0, 9.0])
    }
    
    df = DataFrame(data)
    
    expected_original = {
        'list_col': 'list', 
        'array_col': 'array', 
        'another_list': 'list', 
        'another_array': 'array'
    }
    assert df.column_types == expected_original
    
    # Select subset
    selected = df.select_columns(['list_col', 'another_array'])
    expected_selected = {'list_col': 'list', 'another_array': 'array'}
    assert selected.column_types == expected_selected
    
    # Verify types are preserved
    list_data = selected['list_col']
    array_data = selected['another_array']
    assert isinstance(list_data, list)
    assert isinstance(array_data, np.ndarray)


def test_to_dict_preserves_types():
    """Test that to_dict preserves original types."""
    
    data = {
        'list_data': [1, 2, 3],
        'array_data': np.array([4.0, 5.0, 6.0])
    }
    
    df = DataFrame(data)
    dict_output = df.to_dict()
    
    # Verify original types are preserved
    assert isinstance(dict_output['list_data'], list)
    assert isinstance(dict_output['array_data'], np.ndarray)
    
    # Verify values are correct
    assert dict_output['list_data'] == [1, 2, 3]
    assert np.array_equal(dict_output['array_data'], [4.0, 5.0, 6.0])
    
    # Verify copies were made (immutability)
    dict_output['list_data'][0] = 999
    assert df['list_data'][0] == 1  # Original unchanged


def test_conversion_of_other_iterables():
    """Test that other iterables get converted to lists."""
    
    data = {
        'tuple_data': (1, 2, 3),      # Tuple -> should become list
        'range_data': range(3),       # Range -> should become list
        'set_data': {1, 2, 3}         # Set -> should become list (order may vary)
    }
    
    df = DataFrame(data)
    
    # All should be converted to lists
    expected_types = {'tuple_data': 'list', 'range_data': 'list', 'set_data': 'list'}
    assert df.column_types == expected_types
    
    # Verify data was converted correctly
    assert df['tuple_data'] == [1, 2, 3]
    assert df['range_data'] == [0, 1, 2]
    
    # Set order is not guaranteed, but should contain same elements
    set_data = df['set_data']
    assert len(set_data) == 3
    assert set(set_data) == {1, 2, 3}
