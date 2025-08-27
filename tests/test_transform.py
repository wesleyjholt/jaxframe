"""
Test cases for the transform module functionality.
"""

import pytest
import numpy as np
from jaxframe import DataFrame, wide_to_long_masked, long_to_wide_masked
from jaxframe import wide_df_to_jax_arrays, jax_arrays_to_wide_df, roundtrip_wide_jax_conversion

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


def test_wide_to_long_multiple_id_columns():
    """Test wide to long with multiple ID columns."""
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
    # Values should be converted to regular Python types for the long format
    long_dict = long_df.to_dict()
    assert isinstance(long_dict['value'], list)


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


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_wide_df_to_jax_arrays():
    """Test converting wide DataFrame to JAX arrays."""
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
    
    # Convert to JAX arrays
    values, masks, id_df = wide_df_to_jax_arrays(wide_df, 'sample_id')
    
    # Check shapes
    assert values.shape == (3, 3)  # 3 rows, 3 variables
    assert masks.shape == (3, 3)
    assert len(id_df) == 3
    assert id_df.columns == ('sample_id',)
    
    # Check values (should be sorted by variable index)
    expected_values = jnp.array([
        [1.0, 2.0, 3.0],  # Row A
        [4.0, 5.0, 6.0],  # Row B  
        [7.0, 8.0, 9.0]   # Row C
    ])
    assert jnp.allclose(values, expected_values)
    
    # Check masks
    expected_masks = jnp.array([
        [True, True, False],   # Row A
        [True, False, True],   # Row B
        [False, True, True]    # Row C
    ])
    assert jnp.array_equal(masks, expected_masks)
    
    # Check ID DataFrame
    id_dict = id_df.to_dict()
    assert id_dict['sample_id'] == ['A', 'B', 'C']


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_jax_arrays_to_wide_df():
    """Test converting JAX arrays back to wide DataFrame."""
    import jax.numpy as jnp
    
    # Create test arrays
    values = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    masks = jnp.array([
        [True, True, False],
        [True, False, True], 
        [False, True, True]
    ])
    
    # Create ID DataFrame
    id_data = {'sample_id': ['A', 'B', 'C']}
    id_df = DataFrame(id_data)
    
    # Convert to wide DataFrame
    wide_df = jax_arrays_to_wide_df(values, masks, id_df)
    
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
    assert wide_dict['var$0$value'] == [1.0, 4.0, 7.0]
    assert wide_dict['var$1$value'] == [2.0, 5.0, 8.0]
    assert wide_dict['var$2$value'] == [3.0, 6.0, 9.0]
    assert wide_dict['var$0$mask'] == [True, True, False]
    assert wide_dict['var$1$mask'] == [True, False, True]
    assert wide_dict['var$2$mask'] == [False, True, True]


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
        id_columns=['id1', 'id2'],
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
        if 'value' in col:
            # Check floating point values with tolerance
            for i in range(len(orig_dict[col])):
                assert abs(orig_dict[col][i] - recon_dict[col][i]) < 1e-10
        else:
            # Check exact equality for other columns
            assert orig_dict[col] == recon_dict[col]


@pytest.mark.skipif(not jax_available, reason="JAX not available") 
def test_wide_df_to_jax_arrays_missing_masks():
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
    
    # Convert to JAX arrays
    values, masks, id_df = wide_df_to_jax_arrays(wide_df, 'sample_id')
    
    # Check that missing masks default to True
    expected_masks = jnp.array([
        [True, True],   # A: var$0 has mask=True, var$1 defaults to True
        [False, True]   # B: var$0 has mask=False, var$1 defaults to True  
    ])
    assert jnp.array_equal(masks, expected_masks)
