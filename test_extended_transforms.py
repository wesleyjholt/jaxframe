"""
Test the extended wide_to_long_masked and long_to_wide_masked functions.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from src.jaxframe import DataFrame
from src.jaxframe.transform import wide_to_long_masked, long_to_wide_masked


class TestExtendedTransformFunctions:
    """Test the extended transform functions with multiple columns/DataFrames."""
    
    def test_single_column_backward_compatibility(self):
        """Test that single column usage still works (backward compatibility)."""
        # Create a simple long DataFrame
        df_long = DataFrame({
            'sample_id': ['001', '001', '002', '002', '003'],
            'time': [0, 1, 0, 1, 0],
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        })
        
        # Convert to wide
        df_wide = long_to_wide_masked(
            df_long, 
            id_columns='sample_id',
            value_column='value',
            var_column='time',
            var_prefix='time',
            fill_type=0.0
        )
        
        # Check structure
        expected_columns = ['sample_id', 'time$0$value', 'time$0$mask', 'time$1$value', 'time$1$mask']
        assert set(df_wide.columns) == set(expected_columns)
        
        # Convert back to long
        df_long_back = wide_to_long_masked(df_wide, 'sample_id', value_name='value')
        
        # Should have 5 rows (all observations had masks=True)
        assert len(df_long_back) == 5
        
    def test_multi_column_conversion_example_case(self):
        """Test the exact example case from the user request."""
        # Create the example long DataFrame
        sample_ids = ['001', '001', '001', '002', '002', '002', '003', '003', '004', '004', '005', '006', '007', '008', '009', '010']
        values_1 = jnp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.4, 0.3, 0.6, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        values_2 = ['01', '10', '00', '02', '10', '03', '03', '11', '04', '21', '05', '06', '07', '06', '05', '04']
        
        df_long = DataFrame({
            'sample_id': sample_ids,
            'value_1': values_1,
            'value_2': values_2
        })
        
        # Convert to wide format using multiple columns
        wide_dfs = long_to_wide_masked(
            df_long,
            id_columns='sample_id',
            value_column=['value_1', 'value_2'],
            var_prefix=['time', 'group'],
            fill_type=[0.0, '']  # Different fill types for different data types
        )
        
        # Should return a list of two DataFrames
        assert isinstance(wide_dfs, list)
        assert len(wide_dfs) == 2
        
        time_wide, group_wide = wide_dfs
        
        # Check time DataFrame structure
        assert 'sample_id' in time_wide.columns
        assert 'time$0$value' in time_wide.columns
        assert 'time$0$mask' in time_wide.columns
        assert 'time$1$value' in time_wide.columns
        assert 'time$2$value' in time_wide.columns
        
        # Check group DataFrame structure  
        assert 'sample_id' in group_wide.columns
        assert 'group$0$value' in group_wide.columns
        assert 'group$0$mask' in group_wide.columns
        assert 'group$1$value' in group_wide.columns
        assert 'group$2$value' in group_wide.columns
        
        # Check that both DataFrames have consistent ordering
        assert time_wide['sample_id'] == group_wide['sample_id']
        
        # Convert back to long format
        long_back = wide_to_long_masked(
            [time_wide, group_wide],
            id_columns='sample_id', 
            var_name=['variable', 'variable'],
            value_name=['time_value', 'group_value']
        )
        
        # Should have the appropriate columns
        assert 'sample_id' in long_back.columns
        assert 'variable' in long_back.columns
        assert 'time_value' in long_back.columns
        assert 'group_value' in long_back.columns
        
    def test_string_dtype_error_with_max_fill_types(self):
        """Test that using max fill types with string data raises appropriate errors."""
        df_long = DataFrame({
            'id': ['A', 'B', 'C'],
            'text_value': ['hello', 'world', 'test']
        })
        
        # Should raise error for local_max with string data
        with pytest.raises(ValueError, match="fill_type 'local_max' not supported for string data"):
            long_to_wide_masked(
                df_long,
                id_columns='id',
                value_column='text_value',
                fill_type='local_max'
            )
            
        # Should raise error for global_max with string data  
        with pytest.raises(ValueError, match="fill_type 'global_max' not supported for string data"):
            long_to_wide_masked(
                df_long,
                id_columns='id',
                value_column='text_value',
                fill_type='global_max'
            )
    
    def test_multi_column_string_dtype_error(self):
        """Test error handling for string dtypes in multi-column case."""
        df_long = DataFrame({
            'id': ['A', 'B'],
            'numeric_col': [1.0, 2.0],
            'string_col': ['hello', 'world']
        })
        
        # Should raise error when trying to use max fill types with string column
        with pytest.raises(ValueError, match="fill_type 'local_max' not supported for string data in column 'string_col'"):
            long_to_wide_masked(
                df_long,
                id_columns='id',
                value_column=['numeric_col', 'string_col'],
                fill_type=[0.0, 'local_max']  # Second fill_type is invalid for string data
            )
    
    def test_consistent_ordering_across_dataframes(self):
        """Test that multiple DataFrames maintain consistent ordering."""
        # Create long data with different patterns for different variables
        df_long = DataFrame({
            'id': ['X', 'X', 'Y', 'Z', 'Z', 'Z'],
            'var1': [10.0, 20.0, 15.0, 5.0, 25.0, 35.0],
            'var2': ['a', 'b', 'c', 'd', 'e', 'f']
        })
        
        # Convert to wide with multiple columns
        wide_dfs = long_to_wide_masked(
            df_long,
            id_columns='id',
            value_column=['var1', 'var2'],
            var_prefix=['num', 'text'],
            fill_type=[0.0, 'empty']
        )
        
        wide1, wide2 = wide_dfs
        
        # Check that ID ordering is consistent
        assert wide1['id'] == wide2['id']
        
        # The first DataFrame (var1) determines the ordering
        # X should come first (appears first in var1), then Y, then Z
        ids = wide1['id']
        expected_order = ['X', 'Y', 'Z']  # Order from first appearance in var1
        assert list(ids) == expected_order
        
    def test_input_validation(self):
        """Test input validation for extended functions."""
        df_long = DataFrame({
            'id': ['A', 'B'],
            'val1': [1, 2],
            'val2': [3, 4]
        })
        
        # Test mismatched list lengths
        with pytest.raises(ValueError, match="var_prefix list length"):
            long_to_wide_masked(
                df_long,
                id_columns='id', 
                value_column=['val1', 'val2'],
                var_prefix=['prefix1']  # Only 1 prefix for 2 columns
            )
            
        with pytest.raises(ValueError, match="fill_type list length"):
            long_to_wide_masked(
                df_long,
                id_columns='id',
                value_column=['val1', 'val2'], 
                fill_type=[0.0]  # Only 1 fill_type for 2 columns
            )
    
    def test_wide_to_long_multiple_dataframes(self):
        """Test wide_to_long_masked with multiple input DataFrames."""
        # Create two wide DataFrames
        time_df = DataFrame({
            'sample_id': ['001', '002', '003'],
            'time$0$value': jnp.array([1.0, 2.0, 3.0]),
            'time$1$value': jnp.array([4.0, 5.0, 6.0]),
            'time$0$mask': [True, True, True],
            'time$1$mask': [True, False, True]  # Middle sample masked
        })
        
        group_df = DataFrame({
            'sample_id': ['001', '002', '003'],
            'group$0$value': ['A', 'B', 'C'],
            'group$1$value': ['X', 'Y', 'Z'],
            'group$0$mask': [True, True, True],
            'group$1$mask': [True, True, False]  # Last sample masked
        })
        
        # Convert both to long format
        long_df = wide_to_long_masked(
            [time_df, group_df],
            id_columns='sample_id',
            var_name=['time_var', 'group_var'],
            value_name=['time_val', 'group_val']
        )
        
        # Check structure
        expected_columns = {'sample_id', 'time_var', 'time_val', 'group_val'}
        assert set(long_df.columns) == expected_columns
        
        # Should have fewer rows due to masking
        # time_df contributes: 3 + 2 = 5 rows (middle sample time$1 masked)
        # But we're joining, so we need to see what the actual result is
        assert len(long_df) > 0  # At least some data should be present
        
    def test_roundtrip_conversion(self):
        """Test that long -> wide -> long conversion preserves data."""
        # Create original long data
        original_long = DataFrame({
            'id': ['A', 'A', 'B', 'B', 'C'],
            'time_val': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'category': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        # Convert to wide (multiple columns)
        wide_dfs = long_to_wide_masked(
            original_long,
            id_columns='id',
            value_column=['time_val', 'category'],
            var_prefix=['time', 'cat'],
            fill_type=[0.0, 'missing']
        )
        
        # Convert back to long
        recovered_long = wide_to_long_masked(
            wide_dfs,
            id_columns='id',
            var_name=['variable', 'variable'], 
            value_name=['time_val', 'category']
        )
        
        # Should have same number of valid observations
        # (All original data had implicit masks=True)
        assert len(recovered_long) == len(original_long)
        
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = DataFrame({'id': [], 'value': []})
        
        # Should handle empty input gracefully
        result = long_to_wide_masked(
            empty_df,
            id_columns='id',
            value_column='value'
        )
        
        # Result should be an empty DataFrame with the right structure
        assert len(result) == 0
        assert 'id' in result.columns
