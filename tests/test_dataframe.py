#!/usr/bin/env python3
"""
Test script for DataFrame class functionality using pytest.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.jaxframe.dataframe import DataFrame


def test_basic_functionality():
    """Test basic DataFrame functionality."""
    # Create a simple DataFrame
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': [85.5, 92.0, 78.5]
    }
    
    df = DataFrame(data)
    
    # Test properties
    assert df.shape == (3, 3)
    assert len(df) == 3
    assert df.columns == ('name', 'age', 'score')
    
    # Test column access
    names = df['name']
    ages = df['age']
    scores = df['score']
    
    assert isinstance(names, list)
    assert names == ['Alice', 'Bob', 'Charlie']
    assert isinstance(ages, list)
    assert ages == [25, 30, 35]
    assert isinstance(scores, list)
    assert scores == [85.5, 92.0, 78.5]
    
    # Test row access
    row0 = df.get_row(0)
    row1 = df.get_row(1)
    
    assert row0 == {'name': 'Alice', 'age': 25, 'score': 85.5}
    assert row1 == {'name': 'Bob', 'age': 30, 'score': 92.0}
    
    # Test column selection
    selected_df = df.select_columns(['name', 'score'])
    assert selected_df.shape == (3, 2)
    assert selected_df.columns == ('name', 'score')
    
    # Test immutability - try to modify returned array
    ages_copy = df['age']
    original_ages = ages_copy.copy()
    ages_copy[0] = 999  # This should not affect the DataFrame
    
    current_ages = df['age']
    assert current_ages == original_ages
    assert ages_copy != current_ages


def test_error_conditions():
    """Test error conditions."""
    
    # Test mismatched array lengths
    with pytest.raises(ValueError, match="All arrays and lists must have the same length"):
        DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5]  # Different length
        })
    
    # Test empty DataFrame
    with pytest.raises(ValueError, match="Data dictionary cannot be empty"):
        DataFrame({})
    
    # Test non-string column name
    with pytest.raises(TypeError, match="Column names must be strings"):
        DataFrame({1: [1, 2, 3]})
    
    # Test accessing non-existent column
    df = DataFrame({'a': [1, 2, 3]})
    with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
        df['nonexistent']
    
    # Test invalid row index
    with pytest.raises(IndexError, match="Index .* out of bounds"):
        df.get_row(10)
    
    # Test invalid column selection
    with pytest.raises(KeyError, match="Columns not found"):
        df.select_columns(['nonexistent'])


def test_equality():
    """Test DataFrame equality."""
    
    data1 = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    data2 = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    data3 = {'a': [1, 2, 3], 'b': [4, 5, 7]}  # Different values
    
    df1 = DataFrame(data1)
    df2 = DataFrame(data2)
    df3 = DataFrame(data3)
    
    assert df1 == df2  # Same data
    assert df1 != df3  # Different data
    assert df1 != 'not a dataframe'  # Different type


def test_all_arrays():
    """Test DataFrame with all NumPy arrays."""
    
    data = {
        'x': np.array([1, 2, 3]),
        'y': np.array([4.0, 5.0, 6.0]),
        'z': np.array(['a', 'b', 'c'])
    }
    
    df = DataFrame(data)
    assert df.shape == (3, 3)
    
    # Verify all columns are arrays
    for col in df.columns:
        col_data = df[col]
        assert isinstance(col_data, np.ndarray), f"Column {col} should be numpy array"


def test_contains():
    """Test the __contains__ method."""
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    assert 'a' in df
    assert 'b' in df
    assert 'c' not in df


def test_to_dict():
    """Test the to_dict method."""
    data = {'a': [1, 2, 3], 'b': np.array([4, 5, 6])}
    df = DataFrame(data)
    
    result = df.to_dict()
    
    # Should get copies, not original references
    assert isinstance(result['a'], list)
    assert isinstance(result['b'], np.ndarray)
    
    # Modify result shouldn't affect original
    result['a'][0] = 999
    assert df['a'][0] == 1  # Original unchanged


def test_to_pandas():
    """Test the to_pandas method."""
    pd = pytest.importorskip("pandas")  # Skip test if pandas not available
    
    # Test with mixed data types
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': np.array([85.5, 92.0, 78.5]),
        'active': [True, False, True]
    }
    
    df = DataFrame(data)
    pandas_df = df.to_pandas()
    
    # Check that it's a pandas DataFrame
    assert isinstance(pandas_df, pd.DataFrame)
    
    # Check shape and columns
    assert pandas_df.shape == (3, 4)
    assert list(pandas_df.columns) == ['name', 'age', 'score', 'active']
    
    # Check data values
    assert list(pandas_df['name']) == ['Alice', 'Bob', 'Charlie']
    assert list(pandas_df['age']) == [25, 30, 35]
    assert list(pandas_df['score']) == [85.5, 92.0, 78.5]
    assert list(pandas_df['active']) == [True, False, True]
    
    # Test with empty DataFrame
    empty_data = {'a': [], 'b': []}
    empty_df = DataFrame(empty_data)
    empty_pandas_df = empty_df.to_pandas()
    
    assert isinstance(empty_pandas_df, pd.DataFrame)
    assert empty_pandas_df.shape == (0, 2)
    assert list(empty_pandas_df.columns) == ['a', 'b']


def test_name_attribute():
    """Test the name attribute functionality."""
    # Test DataFrame without name
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    df_no_name = DataFrame(data)
    assert df_no_name.name is None
    
    # Test DataFrame with name
    df_with_name = DataFrame(data, name="test_df")
    assert df_with_name.name == "test_df"
    
    # Test that name appears in repr
    repr_str = str(df_with_name)
    assert "test_df" in repr_str
    
    # Test empty DataFrame with name
    empty_data = {'a': [], 'b': []}
    empty_df = DataFrame(empty_data, name="empty_df")
    assert empty_df.name == "empty_df"
    assert "empty_df" in str(empty_df)
    
    # Test select_columns preserves name
    selected_df = df_with_name.select_columns(['a'])
    assert selected_df.name == "test_df_selected"


def test_join_column():
    """Test the join_column method."""
    # Create first DataFrame (like sample_assay_df)
    df1_data = {
        'sample_id': ['001', '002', '003', '001', '002'],
        'assay_id': ['01', '01', '02', '02', '01']
    }
    df1 = DataFrame(df1_data, name="samples")
    
    # Create second DataFrame (like assay_params_df)
    df2_data = {
        'assay_id': ['01', '02'],
        'var': np.array([10, 20])
    }
    df2 = DataFrame(df2_data, name="params")
    
    # Test basic join
    result = df1.join_column(df2, on_column='assay_id', source_column='var')
    
    # Check that new column was added with correct name
    assert 'params/var' in result.columns
    assert result.shape == (5, 3)
    
    # Check values are correctly mapped
    expected_var_values = [10, 10, 20, 20, 10]  # Based on assay_id mapping
    assert list(result['params/var']) == expected_var_values
    
    # Check original columns are preserved
    assert list(result['sample_id']) == ['001', '002', '003', '001', '002']
    assert list(result['assay_id']) == ['01', '01', '02', '02', '01']
    
    # Test custom target column name
    result2 = df1.join_column(df2, on_column='assay_id', source_column='var', 
                             target_column='custom_var')
    assert 'custom_var' in result2.columns
    assert 'params/var' not in result2.columns
    
    # Test with DataFrame without name
    df2_no_name = DataFrame(df2_data)
    result3 = df1.join_column(df2_no_name, on_column='assay_id', source_column='var')
    assert 'var' in result3.columns  # Should use source_column name when no name
    
    # Test error conditions
    with pytest.raises(KeyError):
        df1.join_column(df2, on_column='nonexistent', source_column='var')
    
    with pytest.raises(KeyError):
        df1.join_column(df2, on_column='assay_id', source_column='nonexistent')
    
    # Test duplicate values in join column
    df2_duplicates = DataFrame({
        'assay_id': ['01', '01'],  # Duplicate
        'var': [10, 20]
    })
    with pytest.raises(ValueError, match="Duplicate values found"):
        df1.join_column(df2_duplicates, on_column='assay_id', source_column='var')
    
    # Test missing value in lookup
    df1_missing = DataFrame({
        'sample_id': ['001'],
        'assay_id': ['03']  # Not in df2
    })
    with pytest.raises(ValueError, match="not found in other DataFrame"):
        df1_missing.join_column(df2, on_column='assay_id', source_column='var')


def test_join_multiple_columns():
    """Test the join_column method with multiple columns."""
    # Create first DataFrame
    df1_data = {
        'sample_id': ['001', '002', '003'],
        'treatment_id': ['T1', 'T2', 'T1']
    }
    df1 = DataFrame(df1_data, name="samples")
    
    # Create second DataFrame with multiple columns to join
    df2_data = {
        'treatment_id': ['T1', 'T2'],
        'dose': np.array([10.0, 20.0]),
        'duration': [30, 60],
        'category': ['A', 'B']
    }
    df2 = DataFrame(df2_data, name="treatments")
    
    # Test joining multiple columns with automatic naming
    result = df1.join_column(df2, on_column='treatment_id', 
                            source_column=['dose', 'duration', 'category'])
    
    # Check that all new columns were added with correct names
    expected_new_columns = ['treatments/dose', 'treatments/duration', 'treatments/category']
    for col in expected_new_columns:
        assert col in result.columns, f"Missing column: {col}"
    
    # Check shape
    assert result.shape == (3, 5)  # 2 original + 3 new columns
    
    # Check values are correctly mapped
    expected_dose_values = [10.0, 20.0, 10.0]  # T1->10.0, T2->20.0, T1->10.0
    expected_duration_values = [30, 60, 30]    # T1->30, T2->60, T1->30
    expected_category_values = ['A', 'B', 'A']  # T1->A, T2->B, T1->A
    
    assert list(result['treatments/dose']) == expected_dose_values
    assert list(result['treatments/duration']) == expected_duration_values
    assert list(result['treatments/category']) == expected_category_values
    
    # Test joining multiple columns with custom naming
    result2 = df1.join_column(df2, on_column='treatment_id',
                             source_column=['dose', 'duration'],
                             target_column=['custom_dose', 'custom_duration'])
    
    assert 'custom_dose' in result2.columns
    assert 'custom_duration' in result2.columns
    assert 'treatments/dose' not in result2.columns
    assert 'treatments/duration' not in result2.columns
    
    # Check values are still correct
    assert list(result2['custom_dose']) == expected_dose_values
    assert list(result2['custom_duration']) == expected_duration_values
    
    # Test error: mismatched target_column length
    with pytest.raises(ValueError, match="Length of target_column list"):
        df1.join_column(df2, on_column='treatment_id',
                       source_column=['dose', 'duration'],
                       target_column=['only_one_name'])  # Should be 2 names
    
    # Test with DataFrame without name (should use source column names)
    df2_no_name = DataFrame(df2_data)
    result3 = df1.join_column(df2_no_name, on_column='treatment_id',
                             source_column=['dose', 'duration'])
    
    assert 'dose' in result3.columns
    assert 'duration' in result3.columns
    assert 'treatments/dose' not in result3.columns
    
    # Test that types are preserved
    assert isinstance(result['treatments/dose'][0], (float, np.floating))  # numpy array
    assert isinstance(result['treatments/duration'][0], int)  # list
    assert isinstance(result['treatments/category'][0], str)  # list


def test_add_column():
    """Test adding columns to DataFrame."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    }
    df = DataFrame(data)
    
    # Add a list column
    new_scores = [85.5, 92.0, 78.5]
    df_with_scores = df.add_column('score', new_scores)
    
    # Check that original DataFrame is unchanged
    assert 'score' not in df.columns
    assert df.shape == (3, 2)
    
    # Check that new DataFrame has the column
    assert 'score' in df_with_scores.columns
    assert df_with_scores.shape == (3, 3)
    assert df_with_scores['score'] == new_scores
    assert df_with_scores.column_types['score'] == 'list'
    
    # Add a numpy array column
    new_grades = np.array([90, 85, 88])
    df_with_grades = df_with_scores.add_column('grade', new_grades)
    
    assert df_with_grades.shape == (3, 4)
    assert np.array_equal(df_with_grades['grade'], new_grades)
    assert df_with_grades.column_types['grade'] == 'array'
    
    # Test error cases
    with pytest.raises(ValueError, match="Column 'name' already exists"):
        df.add_column('name', [1, 2, 3])
    
    with pytest.raises(ValueError, match="New column must have length 3"):
        df.add_column('new_col', [1, 2])  # Wrong length


def test_remove_column():
    """Test removing columns from DataFrame."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': [85.5, 92.0, 78.5]
    }
    df = DataFrame(data)
    
    # Remove a column
    df_no_score = df.remove_column('score')
    
    # Check that original DataFrame is unchanged
    assert 'score' in df.columns
    assert df.shape == (3, 3)
    
    # Check that new DataFrame doesn't have the column
    assert 'score' not in df_no_score.columns
    assert df_no_score.shape == (3, 2)
    assert df_no_score.columns == ('name', 'age')
    
    # Verify remaining data is correct
    assert df_no_score['name'] == df['name']
    assert df_no_score['age'] == df['age']
    
    # Test error cases
    with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
        df.remove_column('nonexistent')
    
    # Can't remove last column
    single_col_df = DataFrame({'name': ['Alice']})
    with pytest.raises(ValueError, match="Cannot remove the last column"):
        single_col_df.remove_column('name')


def test_add_row():
    """Test adding rows to DataFrame."""
    data = {
        'name': ['Alice', 'Bob'],
        'age': [25, 30],
        'score': np.array([85.5, 92.0])
    }
    df = DataFrame(data)
    
    # Add a row
    new_row = {'name': 'Charlie', 'age': 35, 'score': 78.5}
    df_with_row = df.add_row(new_row)
    
    # Check that original DataFrame is unchanged
    assert df.shape == (2, 3)
    assert len(df['name']) == 2
    
    # Check that new DataFrame has the row
    assert df_with_row.shape == (3, 3)
    assert df_with_row['name'] == ['Alice', 'Bob', 'Charlie']
    assert df_with_row['age'] == [25, 30, 35]
    assert np.array_equal(df_with_row['score'], [85.5, 92.0, 78.5])
    
    # Test that column types are preserved
    assert df_with_row.column_types['name'] == 'list'
    assert df_with_row.column_types['age'] == 'list'
    assert df_with_row.column_types['score'] == 'array'
    
    # Test error cases
    with pytest.raises(ValueError, match="Missing values for columns"):
        df.add_row({'name': 'Dave'})  # Missing age and score
    
    with pytest.raises(ValueError, match="Extra columns not in DataFrame"):
        df.add_row({'name': 'Dave', 'age': 40, 'score': 90, 'extra': 'value'})


def test_remove_row():
    """Test removing rows from DataFrame."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': np.array([85.5, 92.0, 78.5])
    }
    df = DataFrame(data)
    
    # Remove first row
    df_no_first = df.remove_row(0)
    
    # Check that original DataFrame is unchanged
    assert df.shape == (3, 3)
    assert df['name'] == ['Alice', 'Bob', 'Charlie']
    
    # Check that new DataFrame doesn't have the first row
    assert df_no_first.shape == (2, 3)
    assert df_no_first['name'] == ['Bob', 'Charlie']
    assert df_no_first['age'] == [30, 35]
    assert np.array_equal(df_no_first['score'], [92.0, 78.5])
    
    # Remove middle row
    df_no_middle = df.remove_row(1)
    assert df_no_middle.shape == (2, 3)
    assert df_no_middle['name'] == ['Alice', 'Charlie']
    assert df_no_middle['age'] == [25, 35]
    assert np.array_equal(df_no_middle['score'], [85.5, 78.5])
    
    # Remove last row
    df_no_last = df.remove_row(2)
    assert df_no_last.shape == (2, 3)
    assert df_no_last['name'] == ['Alice', 'Bob']
    assert df_no_last['age'] == [25, 30]
    assert np.array_equal(df_no_last['score'], [85.5, 92.0])
    
    # Test that column types are preserved
    assert df_no_first.column_types['name'] == 'list'
    assert df_no_first.column_types['age'] == 'list'
    assert df_no_first.column_types['score'] == 'array'
    
    # Test error cases
    with pytest.raises(IndexError, match="Row index 3 out of bounds"):
        df.remove_row(3)
    
    with pytest.raises(IndexError, match="Row index -1 out of bounds"):
        df.remove_row(-1)
    
    # Can't remove last row
    single_row_df = DataFrame({'name': ['Alice'], 'age': [25]})
    with pytest.raises(ValueError, match="Cannot remove the last row"):
        single_row_df.remove_row(0)


def test_add_remove_operations_chaining():
    """Test chaining add/remove operations."""
    data = {
        'name': ['Alice', 'Bob'],
        'age': [25, 30]
    }
    df = DataFrame(data)
    
    # Chain multiple operations
    result = (df
              .add_column('score', [85.5, 92.0])
              .add_row({'name': 'Charlie', 'age': 35, 'score': 78.5})
              .remove_column('age')
              .add_column('grade', ['A', 'A+', 'B+']))
    
    assert result.shape == (3, 3)
    assert result.columns == ('name', 'score', 'grade')
    assert result['name'] == ['Alice', 'Bob', 'Charlie']
    assert result['score'] == [85.5, 92.0, 78.5]
    assert result['grade'] == ['A', 'A+', 'B+']
    
    # Original DataFrame should be unchanged
    assert df.shape == (2, 2)
    assert df.columns == ('name', 'age')
