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
    """Test the join method."""
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
    result = df1.join(df2, on='assay_id', source='var')
    
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
    result2 = df1.join(df2, on='assay_id', source='var', 
                       target='custom_var')
    assert 'custom_var' in result2.columns
    assert 'params/var' not in result2.columns
    
    # Test with DataFrame without name
    df2_no_name = DataFrame(df2_data)
    result3 = df1.join(df2_no_name, on='assay_id', source='var')
    assert 'var' in result3.columns  # Should use source name when no name
    
    # Test error conditions
    with pytest.raises(KeyError):
        df1.join(df2, on='nonexistent', source='var')
    
    with pytest.raises(KeyError):
        df1.join(df2, on='assay_id', source='nonexistent')
    
    # Test duplicate values in join column
    df2_duplicates = DataFrame({
        'assay_id': ['01', '01'],  # Duplicate
        'var': [10, 20]
    })
    with pytest.raises(ValueError, match="Duplicate values found"):
        df1.join(df2_duplicates, on='assay_id', source='var')
    
    # Test inner join behavior: missing value in lookup should result in empty DataFrame
    df1_missing = DataFrame({
        'sample_id': ['001'],
        'assay_id': ['03']  # Not in df2
    })
    result_missing = df1_missing.join(df2, on='assay_id', source='var')
    assert len(result_missing) == 0, "Inner join with no matching keys should return empty DataFrame"
    assert result_missing.shape[1] == 3, "Should have all columns even when empty"  # sample_id, assay_id, params/var
    
    # Test partial match - some rows match, some don't
    df1_partial = DataFrame({
        'sample_id': ['001', '002', '003'],
        'assay_id': ['01', '03', '02']  # '03' is not in df2
    })
    result_partial = df1_partial.join(df2, on='assay_id', source='var')
    assert len(result_partial) == 2, "Should include only matching rows"
    assert list(result_partial['assay_id']) == ['01', '02'], "Should include only rows with matching keys"
    assert list(result_partial['params/var']) == [10, 20], "Should have correct joined values"


def test_join_multiple_columns():
    """Test the join method with multiple columns."""
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
    result = df1.join(df2, on='treatment_id', 
                      source=['dose', 'duration', 'category'])
    
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
    result2 = df1.join(df2, on='treatment_id',
                       source=['dose', 'duration'],
                       target=['custom_dose', 'custom_duration'])
    
    assert 'custom_dose' in result2.columns
    assert 'custom_duration' in result2.columns
    assert 'treatments/dose' not in result2.columns
    assert 'treatments/duration' not in result2.columns
    
    # Check values are still correct
    assert list(result2['custom_dose']) == expected_dose_values
    assert list(result2['custom_duration']) == expected_duration_values
    
    # Test error: mismatched target length
    with pytest.raises(ValueError, match="Length of target list"):
        df1.join(df2, on='treatment_id',
                 source=['dose', 'duration'],
                 target=['only_one_name'])  # Should be 2 names
    
    # Test with DataFrame without name (should use source column names)
    df2_no_name = DataFrame(df2_data)
    result3 = df1.join(df2_no_name, on='treatment_id',
                       source=['dose', 'duration'])
    
    assert 'dose' in result3.columns
    assert 'duration' in result3.columns
    assert 'treatments/dose' not in result3.columns
    
    # Test that types are preserved
    assert isinstance(result['treatments/dose'][0], (float, np.floating))  # numpy array
    assert isinstance(result['treatments/duration'][0], int)  # list
    assert isinstance(result['treatments/category'][0], str)  # list


def test_join_inner_join_behavior():
    """Test that join behaves like an inner join, excluding non-matching rows."""
    # Create test DataFrames with some matching and some non-matching keys
    left_data = {
        'id': ['A', 'B', 'C', 'D', 'E'],
        'left_value': [1, 2, 3, 4, 5]
    }
    left_df = DataFrame(left_data, name="left")
    
    right_data = {
        'id': ['B', 'D', 'F'],  # Only B and D match with left
        'right_value': [20, 40, 60]
    }
    right_df = DataFrame(right_data, name="right")
    
    # Perform inner join
    result = left_df.join(right_df, on='id', source='right_value')
    
    # Should only include rows where keys match
    assert len(result) == 2, f"Expected 2 rows, got {len(result)}"
    assert list(result['id']) == ['B', 'D'], f"Expected ['B', 'D'], got {list(result['id'])}"
    assert list(result['left_value']) == [2, 4], f"Expected [2, 4], got {list(result['left_value'])}"
    assert list(result['right/right_value']) == [20, 40], f"Expected [20, 40], got {list(result['right/right_value'])}"
    
    # Test with JAX arrays
    try:
        import jax.numpy as jnp
        
        left_jax_data = {
            'id': ['A', 'B', 'C'],
            'values': jnp.array([1.0, 2.0, 3.0])
        }
        left_jax_df = DataFrame(left_jax_data, name="left_jax")
        
        right_jax_data = {
            'id': ['B', 'D'],  # Only B matches
            'other_values': jnp.array([20.0, 40.0])
        }
        right_jax_df = DataFrame(right_jax_data, name="right_jax")
        
        result_jax = left_jax_df.join(right_jax_df, on='id', source='other_values')
        
        # Should only include the matching row
        assert len(result_jax) == 1, f"Expected 1 row, got {len(result_jax)}"
        assert result_jax['id'][0] == 'B', f"Expected 'B', got {result_jax['id'][0]}"
        assert abs(result_jax['values'][0] - 2.0) < 1e-10, f"Expected 2.0, got {result_jax['values'][0]}"
        assert abs(result_jax['right_jax/other_values'][0] - 20.0) < 1e-10, f"Expected 20.0, got {result_jax['right_jax/other_values'][0]}"
        
        # Verify JAX array types are preserved
        assert result_jax.column_types['values'] == 'jax_array'
        assert result_jax.column_types['right_jax/other_values'] == 'jax_array'
        
    except ImportError:
        pass  # Skip JAX test if not available


def test_exists_semi_join():
    """Test the exists method for semi-join operations."""
    # Create test DataFrames
    customers_data = {
        'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA']
    }
    customers = DataFrame(customers_data, name="customers")
    
    orders_data = {
        'order_id': ['O1', 'O2', 'O3', 'O4', 'O5'],
        'customer_id': ['C1', 'C1', 'C3', 'C3', 'C5'],  # C2 and C4 have no orders
        'amount': [100, 150, 200, 75, 300]
    }
    orders = DataFrame(orders_data, name="orders")
    
    # Test basic semi-join
    customers_with_orders = customers.exists(orders, on='customer_id')
    
    # Should only include customers who have orders: C1, C3, C5
    assert len(customers_with_orders) == 3, f"Expected 3 customers, got {len(customers_with_orders)}"
    
    expected_customers = ['C1', 'C3', 'C5']
    actual_customers = list(customers_with_orders['customer_id'])
    assert actual_customers == expected_customers, f"Expected {expected_customers}, got {actual_customers}"
    
    # Should preserve original column structure
    assert list(customers_with_orders.columns) == list(customers.columns)
    assert customers_with_orders.columns == ('customer_id', 'name', 'city')
    
    # Should not add any columns from orders DataFrame
    assert 'order_id' not in customers_with_orders.columns
    assert 'amount' not in customers_with_orders.columns
    
    # Check specific values
    assert customers_with_orders['name'][0] == 'Alice'  # C1
    assert customers_with_orders['name'][1] == 'Charlie'  # C3
    assert customers_with_orders['name'][2] == 'Eve'  # C5


def test_exists_duplicate_elimination():
    """Test that exists eliminates duplicates automatically."""
    # Create DataFrames where left has duplicates and right has multiple matches
    left_data = {
        'id': ['A', 'A', 'B', 'B', 'C'],  # Duplicates in left
        'value': [1, 1, 2, 2, 3]
    }
    left_df = DataFrame(left_data, name="left")
    
    right_data = {
        'id': ['A', 'A', 'A', 'B'],  # Multiple matches for A and B
        'other': [10, 20, 30, 40]
    }
    right_df = DataFrame(right_data, name="right")
    
    result = left_df.exists(right_df, on='id')
    
    # Should eliminate duplicates - only one row for each unique key
    assert len(result) == 2, f"Expected 2 unique rows, got {len(result)}"
    assert list(result['id']) == ['A', 'B'], f"Expected ['A', 'B'], got {list(result['id'])}"
    assert list(result['value']) == [1, 2], f"Expected [1, 2], got {list(result['value'])}"


def test_exists_multi_column():
    """Test exists with multi-column joins."""
    # Create test data for multi-column join
    users_data = {
        'user_id': ['U1', 'U2', 'U3', 'U4'],
        'region': ['US', 'US', 'EU', 'EU'],
        'active': [True, False, True, True]
    }
    users = DataFrame(users_data, name="users")
    
    sessions_data = {
        'session_id': ['S1', 'S2', 'S3'],
        'user_id': ['U1', 'U3', 'U1'],
        'region': ['US', 'EU', 'US'],
        'duration': [30, 45, 60]
    }
    sessions = DataFrame(sessions_data, name="sessions")
    
    # Find users who have sessions in their region
    active_users = users.exists(sessions, on=['user_id', 'region'])
    
    # Should find U1 (US) and U3 (EU), but not U2 or U4
    assert len(active_users) == 2, f"Expected 2 users, got {len(active_users)}"
    assert list(active_users['user_id']) == ['U1', 'U3']
    assert list(active_users['region']) == ['US', 'EU']


def test_exists_with_jax_arrays():
    """Test exists with JAX arrays."""
    try:
        import jax.numpy as jnp
        
        # Create DataFrames with JAX arrays
        left_data = {
            'id': ['A', 'B', 'C', 'D'],
            'values': jnp.array([1.0, 2.0, 3.0, 4.0])
        }
        left_df = DataFrame(left_data, name="left")
        
        right_data = {
            'id': ['B', 'D', 'E'],
            'other_values': jnp.array([20.0, 40.0, 50.0])
        }
        right_df = DataFrame(right_data, name="right")
        
        result = left_df.exists(right_df, on='id')
        
        # Should include B and D
        assert len(result) == 2, f"Expected 2 rows, got {len(result)}"
        assert list(result['id']) == ['B', 'D']
        assert abs(result['values'][0] - 2.0) < 1e-10
        assert abs(result['values'][1] - 4.0) < 1e-10
        
        # Verify JAX array type is preserved
        assert result.column_types['values'] == 'jax_array'
        
    except ImportError:
        pass  # Skip JAX test if not available


def test_exists_empty_result():
    """Test exists when no matches are found."""
    left_data = {
        'id': ['A', 'B', 'C'],
        'value': [1, 2, 3]
    }
    left_df = DataFrame(left_data, name="left")
    
    right_data = {
        'id': ['X', 'Y', 'Z'],
        'other': [10, 20, 30]
    }
    right_df = DataFrame(right_data, name="right")
    
    result = left_df.exists(right_df, on='id')
    
    # Should return empty DataFrame with same structure
    assert len(result) == 0, f"Expected empty result, got {len(result)} rows"
    assert list(result.columns) == list(left_df.columns)
    assert result.name == left_df.name


def test_exists_error_conditions():
    """Test error conditions for exists method."""
    left_data = {
        'id': ['A', 'B'],
        'value': [1, 2]
    }
    left_df = DataFrame(left_data)
    
    right_data = {
        'other_id': ['A', 'B'],
        'other_value': [10, 20]
    }
    right_df = DataFrame(right_data)
    
    # Test missing column in left DataFrame
    with pytest.raises(ValueError, match="Column 'missing' not found in left DataFrame"):
        left_df.exists(right_df, on='missing')
    
    # Test missing column in right DataFrame
    with pytest.raises(ValueError, match="Column 'id' not found in right DataFrame"):
        left_df.exists(right_df, on='id')


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


def test_concat_rows_same_columns():
    """Test concatenating DataFrames with same columns (vertical)."""
    # Create two DataFrames with identical columns
    df1 = DataFrame({
        'name': ['Alice', 'Bob'],
        'age': [25, 30],
        'score': np.array([85.5, 92.0])
    }, name="df1")
    
    df2 = DataFrame({
        'name': ['Charlie', 'Diana'],
        'age': [35, 28],
        'score': np.array([78.5, 95.0])
    }, name="df2")
    
    # Concatenate vertically
    result = df1.concat(df2, axis=0)
    
    # Check result
    assert result.shape == (4, 3)
    assert result.columns == ('name', 'age', 'score')
    assert result['name'] == ['Alice', 'Bob', 'Charlie', 'Diana']
    assert result['age'] == [25, 30, 35, 28]
    assert np.array_equal(result['score'], [85.5, 92.0, 78.5, 95.0])
    
    # Check name
    assert result.name == "df1_concat_df2"
    
    # Check that column types are preserved/promoted correctly
    assert result.column_types['name'] == 'list'
    assert result.column_types['age'] == 'list'
    assert result.column_types['score'] == 'array'
    
    # Original DataFrames should be unchanged
    assert df1.shape == (2, 3)
    assert df2.shape == (2, 3)


def test_concat_rows_different_columns():
    """Test concatenating DataFrames with different columns."""
    df1 = DataFrame({
        'name': ['Alice', 'Bob'],
        'age': [25, 30]
    })
    
    df2 = DataFrame({
        'name': ['Charlie', 'Diana'],
        'salary': [50000, 60000]
    })
    
    # Should fail without ignore_index
    with pytest.raises(ValueError, match="DataFrames must have the same columns"):
        df1.concat(df2, axis=0)
    
    # Should work with ignore_index=True (only common columns)
    result = df1.concat(df2, axis=0, ignore_index=True)
    assert result.shape == (4, 1)  # Only 'name' column is common
    assert result.columns == ('name',)
    assert result['name'] == ['Alice', 'Bob', 'Charlie', 'Diana']


def test_concat_columns_horizontal():
    """Test concatenating DataFrames horizontally."""
    df1 = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    }, name="personal")
    
    df2 = DataFrame({
        'salary': np.array([50000, 60000, 55000]),
        'department': ['Engineering', 'Marketing', 'Sales']
    }, name="work")
    
    # Concatenate horizontally
    result = df1.concat(df2, axis=1)
    
    # Check result
    assert result.shape == (3, 4)
    assert set(result.columns) == {'name', 'age', 'salary', 'department'}
    assert result['name'] == ['Alice', 'Bob', 'Charlie']
    assert result['age'] == [25, 30, 35]
    assert np.array_equal(result['salary'], [50000, 60000, 55000])
    assert result['department'] == ['Engineering', 'Marketing', 'Sales']
    
    # Check name
    assert result.name == "personal_hconcat_work"
    
    # Check that column types are preserved
    assert result.column_types['name'] == 'list'
    assert result.column_types['age'] == 'list'
    assert result.column_types['salary'] == 'array'
    assert result.column_types['department'] == 'list'


def test_concat_columns_errors():
    """Test error conditions for horizontal concatenation."""
    df1 = DataFrame({
        'name': ['Alice', 'Bob'],
        'age': [25, 30]
    })
    
    df2 = DataFrame({
        'name': ['Charlie'],  # Different number of rows
        'salary': [50000]
    })
    
    df3 = DataFrame({
        'age': [35, 40],  # Overlapping column name
        'salary': [50000, 60000]
    })
    
    # Different number of rows
    with pytest.raises(ValueError, match="DataFrames must have the same number of rows"):
        df1.concat(df2, axis=1)
    
    # Overlapping column names
    with pytest.raises(ValueError, match="DataFrames have overlapping column names"):
        df1.concat(df3, axis=1)


def test_concat_mixed_types():
    """Test concatenating DataFrames with mixed column types."""
    df1 = DataFrame({
        'names': ['Alice', 'Bob'],           # list
        'ages': [25, 30],                    # list  
        'scores': np.array([85.5, 92.0])     # numpy array
    })
    
    df2 = DataFrame({
        'names': np.array(['Charlie', 'Diana']),  # numpy array
        'ages': np.array([35, 28]),               # numpy array
        'scores': [78.5, 95.0]                    # list
    })
    
    # Concatenate - types should be promoted appropriately
    result = df1.concat(df2, axis=0)
    
    assert result.shape == (4, 3)
    # For arrays, we need to convert to list for comparison or use np.array_equal
    assert list(result['names']) == ['Alice', 'Bob', 'Charlie', 'Diana']
    assert list(result['ages']) == [25, 30, 35, 28]
    assert list(result['scores']) == [85.5, 92.0, 78.5, 95.0]
    
    # Check type promotion: list + array = array
    assert result.column_types['names'] == 'array'  # list + array = array
    assert result.column_types['ages'] == 'array'   # list + array = array  
    assert result.column_types['scores'] == 'array' # array + list = array


def test_concat_static_method():
    """Test the static concat_dataframes method."""
    df1 = DataFrame({'a': [1, 2], 'b': [3, 4]}, name="first")
    df2 = DataFrame({'a': [5, 6], 'b': [7, 8]}, name="second")
    df3 = DataFrame({'a': [9, 10], 'b': [11, 12]}, name="third")
    
    # Concatenate multiple DataFrames
    result = DataFrame.concat_dataframes([df1, df2, df3], axis=0)
    
    assert result.shape == (6, 2)
    assert result['a'] == [1, 2, 5, 6, 9, 10]
    assert result['b'] == [3, 4, 7, 8, 11, 12]
    
    # Test with single DataFrame
    single_result = DataFrame.concat_dataframes([df1])
    assert single_result.shape == df1.shape
    assert single_result == df1
    
    # Test error conditions
    with pytest.raises(ValueError, match="Cannot concatenate empty list"):
        DataFrame.concat_dataframes([])
    
    with pytest.raises(TypeError, match="Element at index 1 is not a DataFrame"):
        DataFrame.concat_dataframes([df1, "not a dataframe"])


def test_concat_invalid_axis():
    """Test concatenation with invalid axis parameter."""
    df1 = DataFrame({'a': [1, 2]})
    df2 = DataFrame({'a': [3, 4]})
    
    with pytest.raises(ValueError, match="axis must be 0 \\(rows\\) or 1 \\(columns\\)"):
        df1.concat(df2, axis=2)


def test_concat_non_dataframe():
    """Test concatenation with non-DataFrame object."""
    df1 = DataFrame({'a': [1, 2]})
    
    with pytest.raises(TypeError, match="Can only concatenate with another DataFrame"):
        df1.concat("not a dataframe")
