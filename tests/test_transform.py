"""
Test cases for the transform module functionality.
"""

import pytest
import numpy as np
from jaxframe import DataFrame, wide_to_long_masked, long_to_wide_masked
from jaxframe import wide_df_to_masked_array, masked_array_to_wide_df, roundtrip_wide_jax_conversion, MaskedArray
from jaxframe.transform import _apply_skeleton_order

# Skip JAX tests if JAX is not available
try:
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    jax_available = False
    jnp = None


def test_wide_to_long_basic():
    """Test basic wide to long conversion with masks."""
    # Create test data similar to the example
    wide_data = {
        'sample_id': ['001', '002', '003'],
        'time$0$value': [0.0, 0.1, 0.2],
        'time$1$value': [0.0, 0.2, 0.4],
        'time$0$mask': [True, True, True],
        'time$1$mask': [True, True, False]  # Last sample masked for time 1
    }
    
    wide_df = DataFrame(wide_data, name="wide_test")
    long_df = wide_to_long_masked(wide_df, 'sample_id')
    
    # Should have 5 rows (3 for time 0, 2 for time 1)
    assert len(long_df) == 5
    assert long_df.columns == ('sample_id', 'variable', 'value')
    
    # Check specific values
    long_dict = long_df.to_dict()
    expected_sample_ids = ['001', '001', '002', '002', '003']  # 003 time 1 is masked out
    expected_variables = [0, 1, 0, 1, 0]
    expected_values = [0.0, 0.0, 0.1, 0.2, 0.2]
    
    assert long_dict['sample_id'] == expected_sample_ids
    assert long_dict['variable'] == expected_variables
    assert long_dict['value'] == expected_values


def test_wide_to_long_multiple_index_columns():
    """Test wide to long with multiple index columns."""
    wide_data = {
        'sample_id': ['001', '002'],
        'batch_id': ['A', 'B'],
        'exp$0$value': [1.0, 2.0],
        'exp$1$value': [1.5, 2.5],
        'exp$0$mask': [True, True],
        'exp$1$mask': [True, False]  # Second sample masked for exp 1
    }
    
    wide_df = DataFrame(wide_data)
    long_df = wide_to_long_masked(wide_df, ['sample_id', 'batch_id'])
    
    assert len(long_df) == 3  # 2 for exp 0, 1 for exp 1
    assert 'sample_id' in long_df.columns
    assert 'batch_id' in long_df.columns
    assert 'variable' in long_df.columns
    assert 'value' in long_df.columns


def test_wide_to_long_custom_names():
    """Test wide to long with custom column names."""
    wide_data = {
        'id': ['A', 'B'],
        'test$0$value': [10, 20],
        'test$1$value': [15, 25],
        'test$0$mask': [True, True],
        'test$1$mask': [True, False]
    }
    
    wide_df = DataFrame(wide_data)
    long_df = wide_to_long_masked(
        wide_df, 'id', 
        var_name='timepoint', 
        value_name='measurement'
    )
    
    assert 'timepoint' in long_df.columns
    assert 'measurement' in long_df.columns
    assert len(long_df) == 3


def test_wide_to_long_respects_skeleton_order():
    """long_skeleton_df controls final row ordering via long_skeleton_id_column."""
    wide_data = {
        'sample_id': ['b', 'a'],
        'time$0$value': [0.0, 1.0],
        'time$1$value': [0.5, 1.5],
        'time$0$mask': [True, True],
        'time$1$mask': [True, True],
    }

    wide_df = DataFrame(wide_data)
    # Skeleton must have same row count as long output, with matching occurrence counts
    # Long output will be: b(var=0), b(var=1), a(var=0), a(var=1)
    # Skeleton reorders to: a first, then b
    skeleton_df = DataFrame({'sample_id': ['a', 'a', 'b', 'b']})

    long_df = wide_to_long_masked(
        wide_df,
        'sample_id',
        long_skeleton_df=skeleton_df,
        long_skeleton_id_column='sample_id',
    )

    result = long_df.to_dict()
    # Expect rows to follow skeleton sample_id order while preserving time ordering per ID
    assert result['sample_id'] == ['a', 'a', 'b', 'b']
    assert result['variable'] == [0, 1, 0, 1]
    assert result['value'] == [1.0, 1.5, 0.0, 0.5]


def test_apply_skeleton_order_without_disambiguator_queue():
    """When no disambiguator exists, skeleton rows consume one matching row each in order."""
    long_df = DataFrame({'id': ['a', 'a', 'a'], 'value': [1, 2, 3]})
    skeleton_df = DataFrame({'id': ['a', 'a', 'a'], 'sid': [10, 11, 12]})

    result = _apply_skeleton_order(
        long_df,
        skeleton_df,
        'sid',
        index_columns='id',
        value_name=None,
        var_name='variable',
    )

    res = result.to_dict()
    assert res['id'] == ['a', 'a', 'a']
    assert res['value'] == [1, 2, 3]
    assert res['sid'] == [10, 11, 12]


def test_apply_skeleton_order_without_disambiguator_missing_key():
    """Missing skeleton keys should raise a ValueError when no disambiguator is used."""
    long_df = DataFrame({'id': ['a'], 'value': [1]})
    skeleton_df = DataFrame({'id': ['b'], 'sid': [99]})

    with pytest.raises(ValueError):
        _apply_skeleton_order(
            long_df,
            skeleton_df,
            'sid',
            index_columns='id',
            value_name=None,
            var_name='variable',
        )


def test_apply_skeleton_order_without_disambiguator_reorders():
    """Without a disambiguator, skeleton order should override original order."""
    # Original order alternates ids; skeleton groups all 'a' rows first.
    long_df = DataFrame({'id': ['b', 'a', 'b', 'a'], 'value': [1, 2, 3, 4]})
    skeleton_df = DataFrame({'id': ['a', 'a', 'b', 'b'], 'sid': [10, 11, 12, 13]})

    result = _apply_skeleton_order(
        long_df,
        skeleton_df,
        'sid',
        index_columns='id',
        value_name=None,
        var_name='variable',
    )

    res = result.to_dict()
    # Expect rows ordered by skeleton: all 'a' rows first, then 'b' rows.
    assert res['id'] == ['a', 'a', 'b', 'b']
    assert res['value'] == [2, 4, 1, 3]
    assert res['sid'] == [10, 11, 12, 13]


def test_apply_skeleton_order_missing_matching_occurrence():
    """Extra skeleton occurrence for a key should raise when no matching row exists."""
    long_df = DataFrame({'id': ['a', 'a'], 'value': [1, 2]})
    skeleton_df = DataFrame({'id': ['a', 'a', 'a'], 'sid': [10, 11, 12]})

    with pytest.raises(ValueError):
        _apply_skeleton_order(
            long_df,
            skeleton_df,
            'sid',
            index_columns='id',
            value_name=None,
            var_name='variable',
        )


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_wide_to_long_skeleton_with_time_ordering():
    import jax.numpy as jnp

    weight_meas = DataFrame({
        'id_weight_meas': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
        'id_person': ['01', '01', '02', '02', '02', '02', '03', '04', '04'],
        'id_operator': ['01', '01', '01', '01', '01', '01', '02', '02', '02'],
        'time': jnp.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
        'value': jnp.array([170.0, 200.0, 150.0, 190.0, 195.0, 180.0, 160.0, 175.0, 165.0]),
    })

    wide_df = long_to_wide_masked(
        df=weight_meas,
        index_columns=["id_person", "id_operator"],
        value_column="time",
        mask_value=False,
        sort_within_id=True,
    )

    long_df = wide_to_long_masked(
        df=wide_df,
        index_columns=["id_person", "id_operator"],
        value_name="time",
        long_skeleton_df=weight_meas,
        long_skeleton_id_column="id_weight_meas",
    )

    result = long_df.to_dict()
    assert result['id_weight_meas'] == ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    assert list(result['time']) == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]
    assert result['id_person'] == ['01', '01', '02', '02', '02', '02', '03', '04', '04']
    assert result['id_operator'] == ['01', '01', '01', '01', '01', '01', '02', '02', '02']


def test_wide_to_long_no_masks():
    """Test wide to long when no mask columns exist."""
    wide_data = {
        'sample_id': ['001', '002'],
        'time$0$value': [0.0, 0.1],
        'time$1$value': [0.5, 0.6]
    }
    
    wide_df = DataFrame(wide_data)
    long_df = wide_to_long_masked(wide_df, 'sample_id')
    
    # Should include all values since no masks
    assert len(long_df) == 4  # 2 samples * 2 timepoints
    
    long_dict = long_df.to_dict()
    expected_sample_ids = ['001', '001', '002', '002']
    expected_variables = [0, 1, 0, 1]
    expected_values = [0.0, 0.5, 0.1, 0.6]
    
    assert long_dict['sample_id'] == expected_sample_ids
    assert long_dict['variable'] == expected_variables
    assert long_dict['value'] == expected_values

def test_wide_to_long_order_naming_single_default():
    """Order column should default to <value_name>_order for single DataFrame."""
    wide_data = {
        'sample_id': ['001', '002'],
        'time$0$value': [0.0, 1.0],
        'time$1$value': [0.5, 1.5],
        'time$0$mask': [True, True],
        'time$1$mask': [True, True],
        'time$0$order': [10, 20],
        'time$1$order': [11, 21],
    }

    wide_df = DataFrame(wide_data)
    long_df = wide_to_long_masked(wide_df, 'sample_id')

    assert 'value_order' in long_df.columns
    assert 'original_order' not in long_df.columns
    # Validate values preserve order payload
    order_values = long_df.to_dict()['value_order']
    # time 0 then time 1 for each row because masks all True
    assert order_values == [10, 11, 20, 21]


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_wide_to_long_with_jax_arrays():
    """Test wide to long conversion with JAX arrays."""
    wide_data = {
        'sample_id': ['001', '002', '003'],
        'time$0$value': jnp.array([0.0, 0.1, 0.2]),
        'time$1$value': jnp.array([0.0, 0.2, 0.4]),
        'time$0$mask': jnp.array([True, True, True]),
        'time$1$mask': jnp.array([True, True, False])
    }
    
    wide_df = DataFrame(wide_data)
    long_df = wide_to_long_masked(wide_df, 'sample_id')
    
    assert len(long_df) == 5
    # Values should now be preserved as JAX arrays for computational efficiency
    long_dict = long_df.to_dict()
    assert long_df.column_types['value'] == 'jax_array'
    # Verify it's a proper 1D JAX array, not a list of individual JAX arrays
    assert isinstance(long_dict['value'], jnp.ndarray)
    assert long_dict['value'].ndim == 1


def test_long_to_wide_basic():
    """Test basic long to wide conversion."""
    long_data = {
        'sample_id': ['001', '001', '002', '002', '003'],
        'variable': [0, 1, 0, 1, 0],
        'value': [0.0, 0.5, 0.1, 0.6, 0.2]
    }
    
    long_df = DataFrame(long_data)
    wide_df = long_to_wide_masked(long_df, 'sample_id', 'value', var_column='variable')
    
    assert len(wide_df) == 3  # 3 unique sample_ids
    assert 'sample_id' in wide_df.columns
    assert 'var$0$value' in wide_df.columns
    assert 'var$1$value' in wide_df.columns
    assert 'var$0$mask' in wide_df.columns
    assert 'var$1$mask' in wide_df.columns
    
    wide_dict = wide_df.to_dict()
    # Sample 003 should have mask=False for var$1 since it's missing
    expected_masks_1 = [True, True, False]
    assert wide_dict['var$1$mask'] == expected_masks_1


def test_long_to_wide_no_var_column():
    """Test long to wide conversion without specifying var_column."""
    long_data = {
        'sample_id': ['001', '001', '002', '002', '003'],
        'value': [0.0, 0.5, 0.1, 0.6, 0.2]
    }
    
    long_df = DataFrame(long_data)
    wide_df = long_to_wide_masked(long_df, 'sample_id', 'value')  # No var_column
    
    assert len(wide_df) == 3  # 3 unique sample_ids
    assert 'sample_id' in wide_df.columns
    assert 'var$0$value' in wide_df.columns
    assert 'var$1$value' in wide_df.columns
    assert 'var$0$mask' in wide_df.columns
    assert 'var$1$mask' in wide_df.columns
    
    wide_dict = wide_df.to_dict()
    
    # Sample 001: should have values [0.0, 0.5] based on order
    sample_001_idx = wide_dict['sample_id'].index('001')
    assert wide_dict['var$0$value'][sample_001_idx] == 0.0
    assert wide_dict['var$1$value'][sample_001_idx] == 0.5
    assert wide_dict['var$0$mask'][sample_001_idx] == True
    assert wide_dict['var$1$mask'][sample_001_idx] == True
    
    # Sample 003: should have only first value [0.2], second should be masked
    sample_003_idx = wide_dict['sample_id'].index('003')
    assert wide_dict['var$0$value'][sample_003_idx] == 0.2
    assert wide_dict['var$0$mask'][sample_003_idx] == True
    assert wide_dict['var$1$value'][sample_003_idx] == 0.0  # default fill_type
    assert wide_dict['var$1$mask'][sample_003_idx] == False  # masked


def test_long_to_wide_fill_modes():
    """Test different fill modes for missing values."""
    long_data = {
        'sample_id': ['A', 'A', 'B', 'B', 'C'],  # C has only one observation
        'value': [1.0, 3.0, 2.0, 5.0, 4.0]       # Global max: 5.0, A max: 3.0, B max: 5.0, C max: 4.0
    }
    
    long_df = DataFrame(long_data)
    
    # Test default fill (0.0)
    wide_default = long_to_wide_masked(long_df, 'sample_id', 'value')
    wide_dict = wide_default.to_dict()
    c_idx = wide_dict['sample_id'].index('C')
    assert wide_dict['var$1$value'][c_idx] == 0.0  # Default fill
    
    # Test custom fill value
    wide_custom = long_to_wide_masked(long_df, 'sample_id', 'value', fill_type=-999)
    wide_dict = wide_custom.to_dict()
    c_idx = wide_dict['sample_id'].index('C')
    assert wide_dict['var$1$value'][c_idx] == -999  # Custom fill
    
    # Test global_max fill
    wide_global = long_to_wide_masked(long_df, 'sample_id', 'value', fill_type='global_max')
    wide_dict = wide_global.to_dict()
    c_idx = wide_dict['sample_id'].index('C')
    assert wide_dict['var$1$value'][c_idx] == 5.0  # Global max
    
    # Test local_max fill
    wide_local = long_to_wide_masked(long_df, 'sample_id', 'value', fill_type='local_max')
    wide_dict = wide_local.to_dict()
    
    # Check each ID gets its own local max as fill value
    a_idx = wide_dict['sample_id'].index('A')
    b_idx = wide_dict['sample_id'].index('B')
    c_idx = wide_dict['sample_id'].index('C')
    
    # A and B should have all observations present (no missing values)
    assert wide_dict['var$1$mask'][a_idx] == True  # A has second observation
    assert wide_dict['var$1$mask'][b_idx] == True  # B has second observation
    
    # C should have local max (4.0) as fill for missing second observation
    assert wide_dict['var$1$value'][c_idx] == 4.0  # C's local max
    assert wide_dict['var$1$mask'][c_idx] == False  # Missing observation


def test_long_to_wide_custom_prefix():
    """Test long to wide with custom variable prefix."""
    long_data = {
        'id': ['A', 'A', 'B'],
        'time': [0, 1, 0],
        'measurement': [10.0, 15.0, 20.0]
    }
    
    long_df = DataFrame(long_data)
    wide_df = long_to_wide_masked(
        long_df, 'id', 'measurement', 
        var_column='time',
        var_prefix='timepoint'
    )
    
    assert 'timepoint$0$value' in wide_df.columns
    assert 'timepoint$1$value' in wide_df.columns
    assert 'timepoint$0$mask' in wide_df.columns
    assert 'timepoint$1$mask' in wide_df.columns


def test_round_trip_conversion():
    """Test that wide->long->wide conversion preserves data structure."""
    # Start with wide format
    original_wide_data = {
        'sample_id': ['001', '002', '003'],
        'time$0$value': [0.0, 0.1, 0.2],
        'time$1$value': [0.0, 0.2, 0.4],
        'time$0$mask': [True, True, True],
        'time$1$mask': [True, True, False]
    }
    
    original_wide_df = DataFrame(original_wide_data)
    
    # Convert to long
    long_df = wide_to_long_masked(original_wide_df, 'sample_id', var_name='variable', value_name='value')
    
    # Convert back to wide
    reconstructed_wide_df = long_to_wide_masked(
        long_df, 'sample_id', 'value',
        var_column='variable',
        var_prefix='time'
    )
    
    # Should have same number of rows
    assert len(reconstructed_wide_df) == len(original_wide_df)
    
    # Should have same columns
    original_cols = set(original_wide_df.columns)
    reconstructed_cols = set(reconstructed_wide_df.columns)
    assert original_cols == reconstructed_cols
    
    # Check that the mask values are preserved
    orig_dict = original_wide_df.to_dict()
    recon_dict = reconstructed_wide_df.to_dict()
    
    assert orig_dict['time$0$mask'] == recon_dict['time$0$mask']
    assert orig_dict['time$1$mask'] == recon_dict['time$1$mask']


def test_wide_to_long_real_example():
    """Test with the exact example from the user request."""
    # Create the exact wide format from the example
    wide_data = {
        'sample_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
        'time$0$value': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'time$1$value': [0.0, 0.2, 0.4, 0.6, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'time$2$value': [0.0, 0.3, 0.4, 0.6, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'time$0$mask': [True] * 10,  # All True
        'time$1$mask': [True, True, True, True, False, False, False, False, False, False],
        'time$2$mask': [True, True, False, False, False, False, False, False, False, False]
    }
    
    wide_df = DataFrame(wide_data)
    long_df = wide_to_long_masked(wide_df, 'sample_id', value_name='value')
    
    # Expected result based on the user's example
    expected_tuples = [
        ('001', 0, 0.0), ('001', 1, 0.0), ('001', 2, 0.0),
        ('002', 0, 0.1), ('002', 1, 0.2), ('002', 2, 0.3),
        ('003', 0, 0.2), ('003', 1, 0.4),
        ('004', 0, 0.3), ('004', 1, 0.6),
        ('005', 0, 0.4),
        ('006', 0, 0.5),
        ('007', 0, 0.6),
        ('008', 0, 0.7),
        ('009', 0, 0.8),
        ('010', 0, 0.9)
    ]
    
    long_dict = long_df.to_dict()
    
    # Check that we have the right number of observations
    assert len(long_df) == len(expected_tuples)
    
    # Check each observation
    for i, (exp_sample, exp_var, exp_value) in enumerate(expected_tuples):
        assert long_dict['sample_id'][i] == exp_sample
        assert long_dict['variable'][i] == exp_var
        assert abs(long_dict['value'][i] - exp_value) < 1e-10  # Float comparison


def test_wide_to_long_order_naming_multi_default():
    """Order columns should default to <value_name>_order for multi-DataFrame input."""
    time_df = DataFrame({
        'id': ['A', 'B'],
        'time$0$value': [1.0, 2.0],
        'time$1$value': [1.5, 2.5],
        'time$0$mask': [True, True],
        'time$1$mask': [True, True],
        'time$0$order': [0, 0],
        'time$1$order': [1, 1],
    })

    group_df = DataFrame({
        'id': ['A', 'B'],
        'group$0$value': [10.0, 20.0],
        'group$1$value': [15.0, 25.0],
        'group$0$mask': [True, True],
        'group$1$mask': [True, True],
        'group$0$order': [5, 6],
        'group$1$order': [7, 8],
    })

    long_df = wide_to_long_masked(
        [time_df, group_df],
        'id',
        var_name=['variable', 'variable'],
        value_name=['time_value', 'group_value'],
    )

    # Both order columns should be present and follow <value_name>_order
    assert 'time_value_order' in long_df.columns
    assert 'group_value_order' in long_df.columns
    # No legacy name should appear
    assert 'original_order' not in long_df.columns

    data = long_df.to_dict()
    # Rows are time values then group values per join logic; just ensure payloads align
    assert data['time_value_order'] == [0, 1, 0, 1]
    assert data['group_value_order'] == [5, 7, 6, 8]


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_wide_df_to_masked_array():
    """Test converting wide DataFrame to MaskedArray."""
    import jax.numpy as jnp
    
    # Create test wide DataFrame
    wide_data = {
        'sample_id': ['A', 'B', 'C'],
        'var$0$value': [1.0, 4.0, 7.0],
        'var$1$value': [2.0, 5.0, 8.0], 
        'var$2$value': [3.0, 6.0, 9.0],
        'var$0$mask': [True, True, False],
        'var$1$mask': [True, False, True],
        'var$2$mask': [False, True, True]
    }
    wide_df = DataFrame(wide_data)
    
    # Convert to MaskedArray
    masked_array = wide_df_to_masked_array(wide_df, 'sample_id')
    
    # Check that it's a MaskedArray instance
    assert isinstance(masked_array, MaskedArray)
    
    # Check shapes
    assert masked_array.shape == (3, 3)  # 3 rows, 3 variables
    assert masked_array.data.shape == (3, 3)
    assert masked_array.mask.shape == (3, 3)
    assert len(masked_array.index_df) == 3
    assert masked_array.index_df.columns == ('sample_id',)
    
    # Check values (should be sorted by variable index)
    expected_values = jnp.array([
        [1.0, 2.0, 3.0],  # Row A
        [4.0, 5.0, 6.0],  # Row B  
        [7.0, 8.0, 9.0]   # Row C
    ])
    assert jnp.allclose(masked_array.data, expected_values)
    
    # Check masks
    expected_masks = jnp.array([
        [True, True, False],   # Row A
        [True, False, True],   # Row B
        [False, True, True]    # Row C
    ])
    assert jnp.array_equal(masked_array.mask, expected_masks)
    
    # Check ID DataFrame
    id_dict = masked_array.index_df.to_dict()
    assert id_dict['sample_id'] == ['A', 'B', 'C']


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_masked_array_to_wide_df():
    """Test converting MaskedArray back to wide DataFrame."""
    import jax.numpy as jnp
    import numpy as np
    
    # Create test arrays
    values = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    masks = np.array([
        [True, True, False],
        [True, False, True], 
        [False, True, True]
    ])
    
    # Create wide_skeleton_df with index and placeholder columns
    wide_skeleton_df = DataFrame({
        'sample_id': ['A', 'B', 'C'],
        'var$0$value': [0.0, 0.0, 0.0],
        'var$0$mask': [True, True, True],
        'var$1$value': [0.0, 0.0, 0.0],
        'var$1$mask': [True, True, True],
        'var$2$value': [0.0, 0.0, 0.0],
        'var$2$mask': [True, True, True]
    })
    
    # Create MaskedArray
    masked_array = MaskedArray(data=values, mask=masks, wide_skeleton_df=wide_skeleton_df,
                               index_columns='sample_id')
    
    # Convert to wide DataFrame
    wide_df = masked_array_to_wide_df(masked_array)
    
    # Check structure
    expected_columns = [
        'sample_id', 
        'var$0$value', 'var$0$mask',
        'var$1$value', 'var$1$mask', 
        'var$2$value', 'var$2$mask'
    ]
    assert set(wide_df.columns) == set(expected_columns)
    assert len(wide_df) == 3
    
    # Check values
    wide_dict = wide_df.to_dict()
    assert wide_dict['sample_id'] == ['A', 'B', 'C']
    
    # Check JAX array values using jnp.array_equal
    assert jnp.array_equal(wide_dict['var$0$value'], jnp.array([1.0, 4.0, 7.0]))
    assert jnp.array_equal(wide_dict['var$1$value'], jnp.array([2.0, 5.0, 8.0]))
    assert jnp.array_equal(wide_dict['var$2$value'], jnp.array([3.0, 6.0, 9.0]))
    
    # Check numpy array masks using np.array_equal
    assert np.array_equal(wide_dict['var$0$mask'], np.array([True, True, False]))
    assert np.array_equal(wide_dict['var$1$mask'], np.array([True, False, True]))
    assert np.array_equal(wide_dict['var$2$mask'], np.array([False, True, True]))
    
    # Check that values are JAX arrays and masks are numpy arrays
    assert wide_df.column_types['var$0$value'] == 'jax_array'
    assert wide_df.column_types['var$0$mask'] == 'array'


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_roundtrip_wide_jax_conversion():
    """Test roundtrip conversion preserves data."""
    import jax.numpy as jnp
    
    # Create original wide DataFrame
    original_data = {
        'id1': ['X', 'Y'],
        'id2': [1, 2], 
        'time$0$value': [10.0, 30.0],
        'time$1$value': [20.0, 40.0],
        'time$0$mask': [True, False],
        'time$1$mask': [False, True]
    }
    original_df = DataFrame(original_data)
    
    # Roundtrip conversion
    reconstructed_df = roundtrip_wide_jax_conversion(
        original_df, 
        index_columns=['id1', 'id2'],
        var_pattern=r'([^$]+)\$(\d+)\$value',
        var_prefix='time'
    )
    
    # Check that structure is preserved
    assert set(original_df.columns) == set(reconstructed_df.columns)
    assert len(original_df) == len(reconstructed_df)
    
    # Check that data is preserved (allowing for small floating point differences)
    orig_dict = original_df.to_dict()
    recon_dict = reconstructed_df.to_dict()
    
    for col in original_df.columns:
        if 'value' in col or 'mask' in col:
            # For value and mask columns, compare using JAX array equality
            if isinstance(recon_dict[col], jnp.ndarray):
                # Reconstructed column is JAX array, convert original for comparison
                expected = jnp.array(orig_dict[col])
                assert jnp.array_equal(expected, recon_dict[col])
            else:
                # Both should be lists/arrays - compare element by element
                assert len(orig_dict[col]) == len(recon_dict[col])
                for i in range(len(orig_dict[col])):
                    if 'value' in col:
                        assert abs(orig_dict[col][i] - recon_dict[col][i]) < 1e-10
                    else:
                        assert orig_dict[col][i] == recon_dict[col][i]
        else:
            # Check exact equality for index columns
            assert orig_dict[col] == recon_dict[col]


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_wide_df_to_masked_array_preserves_skeleton():
    """Test that wide_df_to_masked_array stores the skeleton."""
    import jax.numpy as jnp
    
    # Create wide DataFrame with order columns (as produced by long_to_wide_masked with sorting)
    wide_data = {
        'id': ['A', 'B'],
        'var$0$value': [1.0, 3.0],
        'var$0$mask': [True, True],
        'var$0$order': [1, 0],  # Original positions before sorting
        'var$1$value': [2.0, 4.0],
        'var$1$mask': [True, True],
        'var$1$order': [0, 1],
    }
    wide_df = DataFrame(wide_data)
    
    # Convert to masked array with skeleton preservation
    masked_array = wide_df_to_masked_array(wide_df, 'id', preserve_skeleton=True)
    
    assert masked_array.wide_skeleton_df is not None
    assert 'var$0$order' in masked_array.wide_skeleton_df.columns
    assert masked_array.wide_skeleton_df['var$0$order'] == [1, 0]


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_wide_df_to_masked_array_preserve_skeleton_deprecated():
    """Test that preserve_skeleton=False is ignored (deprecated parameter)."""
    import jax.numpy as jnp
    
    wide_data = {
        'id': ['A'],
        'var$0$value': [1.0],
        'var$0$mask': [True],
    }
    wide_df = DataFrame(wide_data)
    
    # preserve_skeleton=False is deprecated and ignored - skeleton is always stored
    masked_array = wide_df_to_masked_array(wide_df, 'id', preserve_skeleton=False)
    
    # Skeleton is always stored now (needed to derive index_df)
    assert masked_array.wide_skeleton_df is not None
    assert masked_array.index_columns == ['id']


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_masked_array_to_wide_df_includes_order_columns():
    """Test that order columns from skeleton are included in output."""
    import jax.numpy as jnp
    
    # Create wide DataFrame with order columns
    wide_data = {
        'id': ['A', 'B'],
        'var$0$value': [1.0, 3.0],
        'var$0$mask': [True, True],
        'var$0$order': [1, 0],
        'var$1$value': [2.0, 4.0],
        'var$1$mask': [True, False],
        'var$1$order': [0, 1],
    }
    wide_df = DataFrame(wide_data)
    
    # Convert to masked array (preserves skeleton)
    masked_array = wide_df_to_masked_array(wide_df, 'id')
    
    # Convert back to wide df
    reconstructed = masked_array_to_wide_df(masked_array, var_prefix='var')
    
    # Check that order columns are restored
    assert 'var$0$order' in reconstructed.columns
    assert 'var$1$order' in reconstructed.columns
    assert reconstructed['var$0$order'] == [1, 0]
    assert reconstructed['var$1$order'] == [0, 1]


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_masked_array_to_wide_df_without_skeleton_order():
    """Test output without using skeleton order."""
    import jax.numpy as jnp
    
    # Create wide DataFrame with order columns
    wide_data = {
        'id': ['A'],
        'var$0$value': [1.0],
        'var$0$mask': [True],
        'var$0$order': [5],
    }
    wide_df = DataFrame(wide_data)
    
    masked_array = wide_df_to_masked_array(wide_df, 'id')
    
    # Convert with use_skeleton_order=False
    reconstructed = masked_array_to_wide_df(masked_array, var_prefix='var', use_skeleton_order=False)
    
    # Order columns should NOT be included
    assert 'var$0$order' not in reconstructed.columns


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_roundtrip_with_sorting_preserves_order_info():
    """Test full roundtrip: long -> wide (sorted) -> masked array -> wide."""
    import jax.numpy as jnp
    
    # Create a long df with unsorted times
    long_df = DataFrame({
        'id': ['a', 'a', 'a', 'b', 'b'],
        'time': [0.5, 0.1, 0.3, 0.2, 0.4],
    })
    
    # Convert to wide with sorting (generates order columns)
    wide_df = long_to_wide_masked(
        long_df,
        index_columns='id',
        value_column='time',
        sort_within_id=True,
    )
    
    # Wide df should have order columns
    assert 'var$0$order' in wide_df.columns
    original_order = list(wide_df['var$0$order'])
    
    # Convert to masked array
    ma = wide_df_to_masked_array(wide_df, 'id')
    
    # Convert back
    reconstructed = masked_array_to_wide_df(ma, var_prefix='var')
    
    # Order columns should be preserved
    assert 'var$0$order' in reconstructed.columns
    assert list(reconstructed['var$0$order']) == original_order


@pytest.mark.skipif(not jax_available, reason="JAX not available") 
def test_wide_df_to_masked_array_missing_masks():
    """Test conversion when some mask columns are missing."""
    import jax.numpy as jnp
    
    # Create DataFrame with missing mask columns
    wide_data = {
        'sample_id': ['A', 'B'],
        'var$0$value': [1.0, 2.0],
        'var$1$value': [3.0, 4.0],
        'var$0$mask': [True, False],  # Only var$0 has mask
        # var$1$mask is missing
    }
    wide_df = DataFrame(wide_data)
    
    # Convert to MaskedArray
    masked_array = wide_df_to_masked_array(wide_df, 'sample_id')
    
    # Check that missing masks default to True
    expected_masks = jnp.array([
        [True, True],   # A: var$0 has mask=True, var$1 defaults to True
        [False, True]   # B: var$0 has mask=False, var$1 defaults to True  
    ])
    assert jnp.array_equal(masked_array.mask, expected_masks)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_jax_computational_graph_preservation():
    """Test that JAX computational graph is preserved through conversion."""
    import jax.numpy as jnp
    from jax import grad
    
    # Create a function that uses the DataFrame conversion
    def compute_loss(params):
        # Create wide DataFrame with JAX arrays that depend on params
        wide_data = {
            'sample_id': ['A', 'B'],
            'var$0$value': params * jnp.array([1.0, 2.0]),
            'var$0$mask': jnp.array([True, True]),
            'var$1$value': params * jnp.array([3.0, 4.0]),
            'var$1$mask': jnp.array([True, True])
        }
        df = DataFrame(wide_data)
        
        # Convert to MaskedArray
        masked_array = wide_df_to_masked_array(df, ['sample_id'])
        
        # Convert back to wide DataFrame
        reconstructed_df = masked_array_to_wide_df(masked_array, 'var')
        
        # Get values and compute a loss
        val0 = reconstructed_df['var$0$value']
        val1 = reconstructed_df['var$1$value']
        
        # Compute sum of squares loss
        loss = jnp.sum(val0**2) + jnp.sum(val1**2)
        return loss
    
    # Test that we can compute gradients (meaning computational graph is preserved)
    params = 2.0
    loss_fn = compute_loss
    grad_fn = grad(loss_fn)
    
    # This should work if the computational graph is preserved
    gradient = grad_fn(params)
    
    # Verify the gradient is correct
    # loss = sum((params * [1,2])^2) + sum((params * [3,4])^2)
    #      = params^2 * (1 + 4 + 9 + 16) = params^2 * 30
    # dloss/dparams = 2 * params * 30 = 60 * params
    expected_gradient = 60.0 * params
    assert abs(gradient - expected_gradient) < 1e-6
    
    print(f"✓ JAX computational graph preserved! Gradient: {gradient}")


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_jax_values_numpy_masks_preserved():
    """Test that values remain JAX arrays and masks remain numpy arrays."""
    import jax.numpy as jnp
    import numpy as np
    
    # Create input arrays
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    masks = np.array([[True, True], [False, True]])  # Use numpy for masks
    wide_skeleton_df = DataFrame({
        'sample_id': ['A', 'B'],
        'test$0$value': [0.0, 0.0],
        'test$0$mask': [True, True],
        'test$1$value': [0.0, 0.0],
        'test$1$mask': [True, True]
    })
    
    # Create MaskedArray and convert to DataFrame
    masked_array = MaskedArray(data=values, mask=masks, wide_skeleton_df=wide_skeleton_df,
                               index_columns='sample_id')
    wide_df = masked_array_to_wide_df(masked_array, 'test')
    
    # Check that the values are JAX arrays, masks are numpy arrays
    val0 = wide_df['test$0$value']
    val1 = wide_df['test$1$value']
    mask0 = wide_df['test$0$mask']
    mask1 = wide_df['test$1$mask']
    
    assert isinstance(val0, jnp.ndarray), f"Expected JAX array, got {type(val0)}"
    assert isinstance(val1, jnp.ndarray), f"Expected JAX array, got {type(val1)}"
    assert isinstance(mask0, np.ndarray), f"Expected numpy array, got {type(mask0)}"
    assert isinstance(mask1, np.ndarray), f"Expected numpy array, got {type(mask1)}"
    
    # Check that DataFrame recognizes them correctly
    assert wide_df.column_types['test$0$value'] == 'jax_array'
    assert wide_df.column_types['test$1$value'] == 'jax_array'
    assert wide_df.column_types['test$0$mask'] == 'array'
    assert wide_df.column_types['test$1$mask'] == 'array'
    
    # Test that we can do JAX operations on them
    sum_val0 = jnp.sum(val0)
    assert abs(sum_val0 - 4.0) < 1e-6
    
    print("✓ JAX values and numpy masks preserved correctly in DataFrame!")


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_transform_functions_create_proper_jax_arrays():
    """Test that transform functions create proper 1D JAX arrays instead of lists of individual JAX arrays."""
    
    # Test wide_to_long_masked
    wide_data = {
        'id': ['A', 'B', 'C'],
        'var$0$value': jnp.array([1.0, 2.0, 3.0]),
        'var$1$value': jnp.array([4.0, 5.0, 6.0]),
        'var$0$mask': jnp.array([True, True, True]),
        'var$1$mask': jnp.array([True, True, False])
    }
    
    wide_df = DataFrame(wide_data)
    long_df = wide_to_long_masked(wide_df, 'id')
    
    # Check that the value column is a proper JAX array, not a list
    assert long_df.column_types['value'] == 'jax_array'
    value_col = long_df.to_dict()['value']
    assert isinstance(value_col, jnp.ndarray)
    assert value_col.ndim == 1
    assert len(value_col) == 5  # Should have 5 values
    
    # Test long_to_wide_masked with mixed values (JAX elements + fill values)
    long_data = {
        'id': ['A', 'A', 'B'],
        'variable': [0, 1, 0],
        'value': jnp.array([1.0, 2.0, 3.0])
    }
    
    long_df2 = DataFrame(long_data)
    wide_df2 = long_to_wide_masked(long_df2, 'id', 'value', 'variable', 'var')
    
    # Check that all value columns are proper JAX arrays
    for col_name in wide_df2.columns:
        if 'value' in col_name:
            assert wide_df2.column_types[col_name] == 'jax_array'
            col_data = wide_df2.to_dict()[col_name]
            assert isinstance(col_data, jnp.ndarray)
            assert col_data.ndim == 1

# ===== JIT Compatibility Tests =====

@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_sparse_jit_basic():
    """Test that pivot_sparse works under JAX JIT compilation."""
    from jax import jit
    from jaxframe import pivot_sparse, to_masked_array
    
    # Create table with non-traced columns for structure
    table = DataFrame({
        'id': ['A', 'A', 'B', 'B', 'C'],
        'time': [0, 1, 0, 1, 0],  # Python list - not traced
        'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # JAX array - will be traced
    })
    
    def pivot_fn(x):
        df = table.add_column('traced_value', x)
        wide_df = pivot_sparse(
            df=df,
            index=['id'],
            value='value',  # Pivot the non-traced value column
            on=None,
            mask_value=False,
            sort_within_index_group=False,
            prefix='var',
            fill_type=0.0
        )
        masked_array = to_masked_array(wide_df, index=['id'])
        return masked_array.data
    
    values = table['value']
    result = jit(pivot_fn)(values)
    
    # Check output shape (3 entities, 2 max observations)
    assert result.shape == (3, 2)
    # Check values are correctly placed
    expected = jnp.array([
        [1.0, 2.0],  # A: values at time 0, 1
        [3.0, 4.0],  # B: values at time 0, 1
        [5.0, 0.0]   # C: value at time 0, fill at time 1
    ])
    assert jnp.allclose(result, expected)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_sparse_jit_with_multi_index():
    """Test pivot_sparse JIT with multiple index columns."""
    from jax import jit
    from jaxframe import pivot_sparse, to_masked_array
    
    table = DataFrame({
        'id_person': ['01', '01', '02', '02'],
        'id_group': ['A', 'A', 'A', 'B'],
        'obs': [0, 1, 0, 0],  # Python ints - not traced
        'value': jnp.array([10.0, 20.0, 30.0, 40.0])
    })
    
    def pivot_fn(x):
        df = table.add_column('traced', x)
        wide_df = pivot_sparse(
            df=df,
            index=['id_person', 'id_group'],
            value='value',
            on=None,
            sort_within_index_group=False,
            prefix='var',
            fill_type=-1.0
        )
        masked_array = to_masked_array(wide_df, index=['id_person', 'id_group'])
        return masked_array.data
    
    values = table['value']
    result = jit(pivot_fn)(values)
    
    # 3 unique (person, group) combinations, max 2 observations per combo
    assert result.shape == (3, 2)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_sparse_jit_gradients():
    """Test that gradients flow through pivot_sparse."""
    from jax import jit, grad
    from jaxframe import pivot_sparse, to_masked_array
    
    table = DataFrame({
        'id': ['A', 'A', 'B'],
        'idx': [0, 1, 0],  # Non-traced index values
    })
    
    def loss_fn(x):
        # Add traced values and pivot them
        df = table.add_column('value', x)
        wide_df = pivot_sparse(
            df=df,
            index=['id'],
            value='value',  # This is now the traced column
            sort_within_index_group=False,
            prefix='var',
            fill_type=0.0
        )
        masked_array = to_masked_array(wide_df, index=['id'])
        # Sum of squared values
        return jnp.sum(masked_array.data ** 2)
    
    values = jnp.array([1.0, 2.0, 3.0])
    
    # Compute gradient
    grad_fn = jit(grad(loss_fn))
    grads = grad_fn(values)
    
    # Gradients should be 2 * values (derivative of sum of squares)
    expected_grads = 2 * values
    assert jnp.allclose(grads, expected_grads)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_sparse_jit_sort_traced_values():
    """Test that sorting works with traced values using JAX sorting."""
    from jax import jit
    from jaxframe import pivot_sparse, to_masked_array
    
    # The values we're sorting are traced
    table = DataFrame({
        'id': ['A', 'A', 'A'],
    })
    
    def pivot_fn(x):
        # Add traced values and pivot/sort them
        df = table.add_column('value', x)
        wide_df = pivot_sparse(
            df=df,
            index=['id'],
            value='value',
            sort_within_index_group=True,  # Now works with JAX sorting!
            prefix='var',
            fill_type=0.0
        )
        masked_array = to_masked_array(wide_df, index=['id'])
        return masked_array.data
    
    # Unsorted values
    values = jnp.array([30.0, 10.0, 20.0])
    result = jit(pivot_fn)(values)
    
    # After sorting: 10.0, 20.0, 30.0
    expected = jnp.array([[10.0, 20.0, 30.0]])
    assert jnp.allclose(result, expected)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_sparse_jit_sort_with_separate_arrays():
    """Test sorting works when sort column uses non-traced values."""
    from jax import jit
    from jaxframe import pivot_sparse, to_masked_array
    
    # Use Python list for time (won't become traced)
    # The value column we pivot is also a Python list (non-traced)
    # The traced input is a separate column
    table = DataFrame({
        'id': ['A', 'A', 'A'],
        'time': [2.0, 1.0, 0.0],  # Python list - will be pivoted and sorted
    })
    
    def pivot_fn(x):
        # Add traced values as a separate column (not used for pivot)
        df = table.add_column('traced', x)
        wide_df = pivot_sparse(
            df=df,
            index=['id'],
            value='time',  # Pivot the non-traced time column
            sort_within_index_group=True,  # Sort by time (which is concrete)
            prefix='var',
            fill_type=0.0
        )
        masked_array = to_masked_array(wide_df, index=['id'])
        return masked_array.data
    
    # Pass any JAX array as input - it's used for add_column but not pivoted
    dummy_input = jnp.array([100.0, 200.0, 300.0])
    result = jit(pivot_fn)(dummy_input)
    
    # After sorting by time value: 0.0, 1.0, 2.0
    expected = jnp.array([[0.0, 1.0, 2.0]])
    assert jnp.allclose(result, expected)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_sparse_jit_local_max_fill():
    """Test local_max fill_type works correctly with JIT - fills with per-entity max."""
    from jax import jit
    from jaxframe import pivot_sparse, to_masked_array
    
    # Different entities with different max values
    table = DataFrame({
        'id': ['A', 'A', 'B', 'B', 'B'],
    })
    
    def pivot_fn(x):
        df = table.add_column('value', x)
        wide_df = pivot_sparse(
            df=df,
            index=['id'],
            value='value',
            sort_within_index_group=True,
            prefix='var',
            fill_type='local_max'  # Fill with per-entity max
        )
        masked_array = to_masked_array(wide_df, index=['id'])
        return masked_array.data
    
    # A: [10, 20] -> max=20, B: [5, 15, 25] -> max=25
    values = jnp.array([10.0, 20.0, 5.0, 15.0, 25.0])
    
    result_jit = jit(pivot_fn)(values)
    result_no_jit = pivot_fn(values)
    
    # After sorting: A=[10, 20, fill=20], B=[5, 15, 25]
    expected = jnp.array([
        [10.0, 20.0, 20.0],  # A: sorted, 3rd slot filled with local max (20)
        [5.0, 15.0, 25.0],   # B: sorted, no fill needed
    ])
    
    assert jnp.allclose(result_jit, expected), f"JIT result: {result_jit}"
    assert jnp.allclose(result_no_jit, expected), f"No-JIT result: {result_no_jit}"
    assert jnp.allclose(result_jit, result_no_jit), "JIT and non-JIT should match"


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_sparse_jit_global_max_fill():
    """Test global_max fill_type works correctly with JIT - fills with global max."""
    from jax import jit
    from jaxframe import pivot_sparse, to_masked_array
    
    # Different entities with different max values
    table = DataFrame({
        'id': ['A', 'A', 'B', 'B', 'B'],
    })
    
    def pivot_fn(x):
        df = table.add_column('value', x)
        wide_df = pivot_sparse(
            df=df,
            index=['id'],
            value='value',
            sort_within_index_group=True,
            prefix='var',
            fill_type='global_max'  # Fill with global max across all values
        )
        masked_array = to_masked_array(wide_df, index=['id'])
        return masked_array.data
    
    # A: [10, 20], B: [5, 15, 25] -> global max = 25
    values = jnp.array([10.0, 20.0, 5.0, 15.0, 25.0])
    
    result_jit = jit(pivot_fn)(values)
    result_no_jit = pivot_fn(values)
    
    # After sorting: A=[10, 20, fill=25], B=[5, 15, 25]
    expected = jnp.array([
        [10.0, 20.0, 25.0],  # A: sorted, 3rd slot filled with global max (25)
        [5.0, 15.0, 25.0],   # B: sorted, no fill needed
    ])
    
    assert jnp.allclose(result_jit, expected), f"JIT result: {result_jit}"
    assert jnp.allclose(result_no_jit, expected), f"No-JIT result: {result_no_jit}"
    assert jnp.allclose(result_jit, result_no_jit), "JIT and non-JIT should match"


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_structure_basic():
    """Test PivotStructure correctly computes source-to-dest mappings."""
    from jaxframe.transform import PivotStructure
    
    df = DataFrame({
        'id': ['A', 'A', 'B', 'C', 'C', 'C'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    structure = PivotStructure(df, index_columns='id')
    
    assert structure.n_source_rows == 6
    assert structure.n_dest_rows == 3  # A, B, C
    assert structure.n_dest_cols == 3  # Max 3 observations per entity (C has 3)
    assert structure.unique_ids == [('A',), ('B',), ('C',)]
    
    # Check mappings
    # A's rows (0, 1) -> dest row 0, cols 0, 1
    assert structure.source_to_dest_row[0] == 0
    assert structure.source_to_dest_col[0] == 0
    assert structure.source_to_dest_row[1] == 0
    assert structure.source_to_dest_col[1] == 1
    
    # B's row (2) -> dest row 1, col 0
    assert structure.source_to_dest_row[2] == 1
    assert structure.source_to_dest_col[2] == 0
    
    # C's rows (3, 4, 5) -> dest row 2, cols 0, 1, 2
    assert structure.source_to_dest_row[3] == 2
    assert structure.source_to_dest_col[3] == 0


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_apply_pivot_structure_jax():
    """Test _apply_pivot_structure_jax correctly scatters values."""
    from jaxframe.transform import PivotStructure, _apply_pivot_structure_jax
    
    df = DataFrame({
        'id': ['A', 'A', 'B'],
        'value': [10, 20, 30]
    })
    
    structure = PivotStructure(df, index_columns='id')
    values = jnp.array([10.0, 20.0, 30.0])
    
    result = _apply_pivot_structure_jax(values, structure, fill_type=-1.0)
    
    expected = jnp.array([
        [10.0, 20.0],  # A
        [30.0, -1.0]   # B (second slot filled)
    ])
    assert jnp.allclose(result, expected)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_unpivot_sparse_jit_roundtrip():
    """Test that unpivot_sparse works with JIT through a full pivot/unpivot roundtrip."""
    from functools import partial
    from jax import jit
    from jaxframe import pivot_sparse, unpivot_sparse, to_masked_array, from_masked_array, MaskedArray
    
    # Create source data with multiple observations per entity
    source_table = DataFrame({
        'id_meas': ['01', '02', '03', '04', '05'],
        'id_person': ['A', 'A', 'A', 'B', 'B'],
        'time': [2.0, 1.0, 0.0, 1.0, 0.0],  # Unsorted
    })
    
    _index_columns = ['id_person']
    _value_column = 'time'
    
    def pivot_fn(x):
        df = source_table.add_column('VALUE', x)
        _wide_df = pivot_sparse(
            df=df,
            index=_index_columns,
            value='VALUE',
            sort_within_index_group=True,
            prefix='VALUE',
            fill_type='local_max'
        )
        _masked_array = to_masked_array(
            _wide_df,
            index=_index_columns,
            sort_by_var_index=True,
        )
        return _masked_array.data

    def unpivot_fn(x, pivot_input):
        _long_df = source_table.add_column('VALUE', pivot_input)

        _wide_df = pivot_sparse(
            df=_long_df,
            index=_index_columns,
            value='VALUE',
            sort_within_index_group=True,
            prefix='VALUE',
            fill_type='local_max'
        )

        _masked_array = to_masked_array(
            _wide_df,
            index=_index_columns,
            sort_by_var_index=True,
        )

        _masked = MaskedArray(
            data=x,
            mask=_masked_array.mask,
            wide_skeleton_df=_masked_array.wide_skeleton_df,
            index_columns=_index_columns,
            validate=False
        )

        _wide_df_2 = from_masked_array(_masked, prefix='VALUE')

        _long_df_2 = unpivot_sparse(
            df=_wide_df_2,
            index=_index_columns,
            value_name='VALUE',
            order_name='order',
            long_skeleton_df=_long_df,
            long_skeleton_id_column='id_meas'
        )

        return jnp.asarray(_long_df_2['VALUE'])

    # Test values (will be sorted during pivot)
    values = jnp.array([12.0, 11.0, 10.0, 21.0, 20.0])
    
    # Pivot with JIT
    pivoted_jit = jit(pivot_fn)(values)
    pivoted_no_jit = pivot_fn(values)
    assert jnp.allclose(pivoted_jit, pivoted_no_jit), "Pivot JIT/no-JIT mismatch"
    
    # Unpivot with JIT should recover original values
    result_jit = jit(partial(unpivot_fn, pivot_input=values))(pivoted_jit)
    result_no_jit = unpivot_fn(pivoted_no_jit, values)
    
    assert jnp.allclose(result_jit, values), f"JIT roundtrip failed: {result_jit} != {values}"
    assert jnp.allclose(result_no_jit, values), f"No-JIT roundtrip failed: {result_no_jit} != {values}"
    assert jnp.allclose(result_jit, result_no_jit), "JIT/no-JIT unpivot mismatch"


# ---------------------------------------------------------------------------
# Tests for index_subdata_columns feature
# ---------------------------------------------------------------------------


def test_pivot_with_index_subdata_columns_basic():
    """Test basic index_subdata_columns propagation to wide format."""
    df = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'category': ['X', 'X', 'Y', 'Y'],  # Valid: same within each id
        'value': [1.0, 2.0, 3.0, 4.0],
    })
    
    wide = long_to_wide_masked(
        df, 'id', 'value', 
        index_subdata_columns=['category']
    )
    
    assert 'category' in wide.columns
    assert list(wide['category']) == ['X', 'Y']
    assert list(wide['id']) == ['a', 'b']


def test_pivot_with_multiple_index_subdata_columns():
    """Test multiple index_subdata columns."""
    df = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'group': ['G1', 'G1', 'G2', 'G2'],
        'weight': [0.5, 0.5, 0.8, 0.8],
        'value': [1.0, 2.0, 3.0, 4.0],
    })
    
    wide = long_to_wide_masked(
        df, 'id', 'value',
        index_subdata_columns=['group', 'weight']
    )
    
    assert list(wide['group']) == ['G1', 'G2']
    assert list(wide['weight']) == [0.5, 0.8]


def test_pivot_index_subdata_invalid_column_not_found():
    """Test error when index_subdata column doesn't exist."""
    df = DataFrame({'id': ['a', 'b'], 'value': [1, 2]})
    
    with pytest.raises(ValueError, match="not in the DataFrame"):
        long_to_wide_masked(df, 'id', 'value', index_subdata_columns=['missing'])


def test_pivot_index_subdata_invalid_inconsistent_values():
    """Test error when index_subdata has different values for same index."""
    df = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'category': ['X', 'Z', 'Y', 'Y'],  # Invalid: 'a' has both 'X' and 'Z'
        'value': [1.0, 2.0, 3.0, 4.0],
    })
    
    with pytest.raises(ValueError, match="conflicting values"):
        long_to_wide_masked(df, 'id', 'value', index_subdata_columns=['category'])


def test_pivot_index_subdata_with_sorting():
    """Test index_subdata works correctly with sort_within_id."""
    df = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'meta': ['M1', 'M1', 'M2', 'M2'],
        'value': [2.0, 1.0, 4.0, 3.0],  # Will be sorted
    })
    
    wide = long_to_wide_masked(
        df, 'id', 'value',
        sort_within_id=True,
        index_subdata_columns=['meta']
    )
    
    assert list(wide['meta']) == ['M1', 'M2']


def test_pivot_index_subdata_with_pivot_sparse():
    """Test index_subdata with pivot_sparse wrapper."""
    from jaxframe.transform import pivot_sparse
    
    df = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'category': ['X', 'X', 'Y', 'Y'],
        'value': [1.0, 2.0, 3.0, 4.0],
    })
    
    wide = pivot_sparse(
        df, index='id', value='value',
        index_subdata=['category']
    )
    
    assert 'category' in wide.columns
    assert list(wide['category']) == ['X', 'Y']


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_index_subdata_jax_jit():
    """Test index_subdata with JAX JIT compilation."""
    from jaxframe.transform import pivot_sparse
    import jax
    
    source_table = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'weight': [0.5, 0.5, 0.8, 0.8],
    })
    
    @jax.jit
    def pivot_fn(values):
        df = source_table.add_column('value', values)
        wide = pivot_sparse(
            df, index='id', value='value',
            index_subdata=['weight']
        )
        return wide['var$0$value']
    
    values = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = pivot_fn(values)
    
    # Should get first value per entity
    assert jnp.allclose(result, jnp.array([1.0, 3.0]))


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_index_subdata_with_jax_subdata_column():
    """Test when the subdata column itself contains JAX arrays."""
    df = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'weight': jnp.array([0.5, 0.5, 0.8, 0.8]),  # JAX array
        'value': jnp.array([1.0, 2.0, 3.0, 4.0]),
    })
    
    wide = long_to_wide_masked(
        df, 'id', 'value',
        index_subdata_columns=['weight']
    )
    
    # weight column should be preserved correctly
    weight_values = wide['weight']
    assert jnp.allclose(jnp.array([weight_values[0], weight_values[1]]), jnp.array([0.5, 0.8]))


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_pivot_index_subdata_jax_jit_with_sorting():
    """Test index_subdata with JAX JIT and sorting enabled."""
    from jaxframe.transform import pivot_sparse
    import jax
    
    source_table = DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'meta': ['M1', 'M1', 'M2', 'M2'],
    })
    
    @jax.jit
    def pivot_fn(values):
        df = source_table.add_column('value', values)
        wide = pivot_sparse(
            df, index='id', value='value',
            sort_within_index_group=True,
            index_subdata=['meta']
        )
        return wide['var$0$value']
    
    # Values will be sorted within each entity
    values = jnp.array([2.0, 1.0, 4.0, 3.0])
    result = pivot_fn(values)
    
    # After sorting: a=[1.0, 2.0], b=[3.0, 4.0], first slot values are 1.0 and 3.0
    assert jnp.allclose(result, jnp.array([1.0, 3.0]))