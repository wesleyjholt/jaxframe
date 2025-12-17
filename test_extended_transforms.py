"""
Test the extended wide_to_long_masked and long_to_wide_masked functions.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from src.jaxframe import DataFrame
from src.jaxframe.transform import (
    wide_to_long_masked,
    long_to_wide_masked,
    wide_df_to_masked_array,
    masked_array_to_wide_df,
    pivot_sparse,
    unpivot_sparse,
    to_masked_array,
    from_masked_array,
)


def _assert_dataframes_equal(df_left: DataFrame, df_right: DataFrame):
    assert df_left.columns == df_right.columns
    for column in df_left.columns:
        left_col = df_left[column]
        right_col = df_right[column]
        if hasattr(left_col, '__array__') or hasattr(right_col, '__array__'):
            left_arr = np.array(left_col)
            right_arr = np.array(right_col)
            assert left_arr.shape == right_arr.shape
            assert np.allclose(left_arr, right_arr, equal_nan=True)
        else:
            assert list(left_col) == list(right_col)


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

    def test_sort_within_id_adds_original_order_metadata(self):
        """long_to_wide_masked should optionally sort values and retain original order."""
        long_df = DataFrame({
            'patient_id': ['P001', 'P001', 'P002', 'P002', 'P003', 'P003', 'P003'],
            'study_arm': ['control', 'control', 'treated', 'treated', 'control', 'control', 'control'],
            'temperature_f': jnp.array([98.6, 99.2, 97.9, 98.4, 99.1, 99.0, 99.3], dtype=jnp.float32)
        })

        wide_df = long_to_wide_masked(
            long_df,
            id_columns=['patient_id', 'study_arm'],
            value_column='temperature_f',
            var_prefix='temp',
            fill_type=jnp.nan,
            mask_value=False,
            sort_within_id=True
        )

        # Expect order-tracking columns
        assert 'temp$0$order' in wide_df.columns

        # Build reference ordering info
        observations_by_key = {}
        counters = {}
        for idx in range(len(long_df)):
            key = (long_df['patient_id'][idx], long_df['study_arm'][idx])
            position = counters.get(key, 0)
            counters[key] = position + 1
            value = float(long_df['temperature_f'][idx])
            observations_by_key.setdefault(key, []).append((position, value))
        sorted_by_value = {
            key: sorted([(val, pos) for pos, val in values], key=lambda x: x[0])
            for key, values in observations_by_key.items()
        }

        # Verify wide view reflects ascending order and tracks original positions
        temp_value_columns = [
            col for col in wide_df.columns if col.startswith('temp$') and col.endswith('$value')
        ]
        max_slots = len(temp_value_columns)

        for row_idx in range(len(wide_df)):
            key = (wide_df['patient_id'][row_idx], wide_df['study_arm'][row_idx])
            expected = sorted_by_value[key]
            for slot_idx, (expected_value, original_pos) in enumerate(expected):
                value_col = f"temp${slot_idx}$value"
                order_col = f"temp${slot_idx}$order"
                assert pytest.approx(float(wide_df[value_col][row_idx]), rel=1e-6) == expected_value
                assert int(wide_df[order_col][row_idx]) == original_pos

            # Slots beyond observed values should have mask=False and order=-1
            for slot_idx in range(len(expected), max_slots):
                mask_col = f"temp${slot_idx}$mask"
                order_col = f"temp${slot_idx}$order"
                assert wide_df[mask_col][row_idx] is False
                assert int(wide_df[order_col][row_idx]) == -1

        # Convert back to long format and ensure original ordering can be restored
        long_back = wide_to_long_masked(
            wide_df,
            id_columns=['patient_id', 'study_arm'],
            var_name='slot',
            value_name='temperature_f',
            order_value_name='original_order'
        )

        reconstructed = {}
        for idx in range(len(long_back)):
            key = (long_back['patient_id'][idx], long_back['study_arm'][idx])
            reconstructed.setdefault(key, []).append(
                (int(long_back['original_order'][idx]), float(long_back['temperature_f'][idx]))
            )

        for key, expected_values in observations_by_key.items():
            expected_sorted = sorted(expected_values, key=lambda x: x[0])
            actual_sorted = sorted(reconstructed[key], key=lambda x: x[0])
            assert len(expected_sorted) == len(actual_sorted)
            for (_, expected_value), (_, actual_value) in zip(expected_sorted, actual_sorted):
                assert pytest.approx(expected_value, rel=1e-6) == actual_value

    def test_sort_within_id_with_var_column_not_allowed(self):
        """sort_within_id should not accept explicit var_column assignments."""
        long_df = DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'visit': [0, 1, 0, 1],
            'value': jnp.array([2.0, 1.0, 4.0, 3.5])
        })

        with pytest.raises(ValueError, match="sort_within_id cannot be used when var_column is provided"):
            long_to_wide_masked(
                long_df,
                id_columns='id',
                value_column='value',
                var_column='visit',
                sort_within_id=True
            )

    def test_order_value_name_defaults_for_multiple_dataframes(self):
        """wide_to_long_masked should auto-name order columns for multiple inputs."""
        df_long = DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'metric1': jnp.array([2.0, 1.0, 5.0, 4.0]),
            'metric2': ['x', 'y', 'z', 'w']
        })

        metric1_wide = long_to_wide_masked(
            df_long,
            id_columns='id',
            value_column='metric1',
            var_prefix='m1',
            sort_within_id=True
        )

        metric2_wide = long_to_wide_masked(
            df_long,
            id_columns='id',
            value_column='metric2',
            var_prefix='m2'
        )

        combined = wide_to_long_masked(
            [metric1_wide, metric2_wide],
            id_columns='id',
            var_name=['slot', 'slot'],
            value_name=['metric1', 'metric2']
        )

        assert 'metric1_order' in combined.columns
        assert 'metric2_order' not in combined.columns
        assert len(combined) > 0  # At least some data should be present
        
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

    def test_pivot_sparse_wrapper_matches_core_function(self):
        df_long = DataFrame({
            'entity': ['A', 'A', 'B'],
            'category': ['x', 'y', 'x'],
            'value': jnp.array([1.0, 2.0, 3.0])
        })

        direct = long_to_wide_masked(
            df_long,
            id_columns='entity',
            value_column='value',
            var_column='category',
            var_prefix='cat'
        )
        wrapper = pivot_sparse(
            df_long,
            index='entity',
            value='value',
            on='category',
            prefix='cat'
        )

        _assert_dataframes_equal(direct, wrapper)

    def test_unpivot_sparse_wrapper_matches_core_function(self):
        wide_df = DataFrame({
            'entity': ['A', 'B'],
            'cat$0$value': jnp.array([1.0, 3.0]),
            'cat$0$mask': [True, True],
            'cat$1$value': jnp.array([2.0, 4.0]),
            'cat$1$mask': [True, True],
        })

        direct = wide_to_long_masked(
            wide_df,
            id_columns='entity',
            var_name='slot',
            value_name='val'
        )
        wrapper = unpivot_sparse(
            wide_df,
            index='entity',
            var_name='slot',
            value_name='val'
        )

        _assert_dataframes_equal(direct, wrapper)

    def test_masked_array_wrappers_match_core_functions(self):
        wide_df = DataFrame({
            'entity': ['A', 'B'],
            'cat$0$value': jnp.array([1.0, 3.0]),
            'cat$0$mask': [True, True],
            'cat$1$value': jnp.array([2.0, 4.0]),
            'cat$1$mask': [True, False],
        })

        direct_masked = wide_df_to_masked_array(
            wide_df,
            id_columns='entity'
        )
        wrapper_masked = to_masked_array(
            wide_df,
            index='entity'
        )

        assert np.allclose(direct_masked.data, wrapper_masked.data)
        assert np.array_equal(direct_masked.mask, wrapper_masked.mask)
        _assert_dataframes_equal(direct_masked.index_df, wrapper_masked.index_df)

        direct_roundtrip = masked_array_to_wide_df(direct_masked, var_prefix='cat')
        wrapper_roundtrip = from_masked_array(wrapper_masked, prefix='cat')
        _assert_dataframes_equal(direct_roundtrip, wrapper_roundtrip)
