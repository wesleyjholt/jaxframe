"""
Tests for MaskedArray class.
"""

import pytest
import numpy as np
from jaxframe import DataFrame, MaskedArray

# Check if JAX is available
try:
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    jax_available = False


def make_skeleton_df(index_data: dict, n_vars: int = 2) -> DataFrame:
    """Helper to create a skeleton DataFrame with index columns and placeholder value/mask columns."""
    skeleton_data = dict(index_data)
    n_rows = len(next(iter(index_data.values())))
    for i in range(n_vars):
        skeleton_data[f'var${i}$value'] = [0.0] * n_rows
        skeleton_data[f'var${i}$mask'] = [True] * n_rows
    return DataFrame(skeleton_data)


@pytest.mark.skipif(not jax_available, reason="JAX not available")
class TestMaskedArray:
    """Test the MaskedArray class functionality."""
    
    def test_masked_array_creation(self):
        """Test basic MaskedArray creation."""
        # Create test data
        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = np.array([[True, False], [True, True], [False, True]])
        index_data = {
            'sample_id': ['A', 'B', 'C'],
            'group': [1, 1, 2]
        }
        skeleton_df = make_skeleton_df(index_data, n_vars=2)
        
        # Create MaskedArray
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns=['sample_id', 'group'])
        
        # Test basic properties
        assert jnp.array_equal(masked_array.data, data)
        assert np.array_equal(masked_array.mask, mask)
        assert masked_array.index_columns == ['sample_id', 'group']
        assert masked_array.index_df == DataFrame(index_data)  
        assert masked_array.shape == (3, 2)
    
    def test_masked_array_validation(self):
        """Test validation during MaskedArray creation."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        # Valid creation should work
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns='id')
        assert masked_array.shape == (2, 2)
        
        # Test shape mismatch between data and mask
        wrong_mask = np.array([[True, False, True]])  # Wrong shape
        with pytest.raises(ValueError, match="Data and mask must have the same shape"):
            MaskedArray(data=data, mask=wrong_mask, wide_skeleton_df=skeleton_df,
                       index_columns='id')
        
        # Test shape mismatch between data and skeleton_df
        wrong_skeleton = make_skeleton_df({'id': ['A']}, n_vars=2)  # Wrong length
        with pytest.raises(ValueError, match="Number of data rows \\(2\\) must match wide_skeleton_df length \\(1\\)"):
            MaskedArray(data=data, mask=mask, wide_skeleton_df=wrong_skeleton,
                       index_columns='id')
    
    def test_masked_array_equality(self):
        """Test MaskedArray equality comparison."""
        data1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask1 = np.array([[True, False], [True, True]])
        skeleton1 = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        data2 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask2 = np.array([[True, False], [True, True]])
        skeleton2 = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        masked_array1 = MaskedArray(data=data1, mask=mask1, wide_skeleton_df=skeleton1,
                                    index_columns='id')
        masked_array2 = MaskedArray(data=data2, mask=mask2, wide_skeleton_df=skeleton2,
                                    index_columns='id')
        
        # Should be equal
        assert masked_array1 == masked_array2
        
        # Different data
        data3 = jnp.array([[1.0, 2.0], [3.0, 5.0]])  # Changed last value
        masked_array3 = MaskedArray(data=data3, mask=mask1, wide_skeleton_df=skeleton1,
                                    index_columns='id')
        assert masked_array1 != masked_array3
        
        # Different mask
        mask3 = np.array([[True, True], [True, True]])  # Changed mask
        masked_array4 = MaskedArray(data=data1, mask=mask3, wide_skeleton_df=skeleton1,
                                    index_columns='id')
        assert masked_array1 != masked_array4
        
        # Different index_df (via different skeleton)
        skeleton3 = make_skeleton_df({'id': ['A', 'C']}, n_vars=2)  # Changed second ID
        masked_array5 = MaskedArray(data=data1, mask=mask1, wide_skeleton_df=skeleton3,
                                    index_columns='id')
        assert masked_array1 != masked_array5
    
    def test_masked_array_copy(self):
        """Test MaskedArray copy functionality."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                               index_columns='id')
        copy = original.copy()
        
        # Should be equal but not the same object (except for immutable DataFrame)
        assert original == copy
        assert original is not copy
        assert original.data is not copy.data
        assert original.mask is not copy.mask
        # Note: DataFrames are immutable in jaxframe, so same reference is expected
    
    def test_get_valid_data(self):
        """Test extracting valid (non-masked) data."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = np.array([[True, False, True], [False, True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=3)
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns='id')
        valid_data = masked_array.get_valid_data()
        
        # Should extract values where mask is True
        expected = jnp.array([1.0, 3.0, 5.0, 6.0])  # Valid values
        assert jnp.array_equal(valid_data, expected)
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B'], 'group': [1, 2]}, n_vars=2)
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns=['id', 'group'])
        result = masked_array.to_dict()
        
        # Check structure
        assert 'data' in result
        assert 'mask' in result
        assert 'wide_skeleton_df' in result
        assert 'index_columns' in result
        assert 'shape' in result
        
        # Check values
        assert jnp.array_equal(result['data'], data)
        assert np.array_equal(result['mask'], mask)
        assert result['index_columns'] == ['id', 'group']
        assert result['shape'] == (2, 2)
    
    def test_string_representation(self):
        """Test string representation of MaskedArray."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = np.array([[True, False, True], [False, True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=3)
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns='id')
        str_repr = str(masked_array)
        
        # Check that key information is present
        assert "MaskedArray" in str_repr
        assert "2 rows, 3 columns" in str_repr  # Updated to match actual format
        assert "Valid values: 4" in str_repr  # 4 valid values out of 6
        assert "Index columns: ['id']" in str_repr
    
    def test_masked_array_with_single_column_index(self):
        """Test MaskedArray with single column index DataFrame."""
        data = jnp.array([[1.0], [2.0], [3.0]])
        mask = np.array([[True], [False], [True]])
        skeleton_df = make_skeleton_df({'sample': ['X', 'Y', 'Z']}, n_vars=1)
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns='sample')
        
        assert masked_array.shape == (3, 1)
        assert len(masked_array.get_valid_data()) == 2  # Two valid elements
    
    def test_masked_array_with_multi_column_index(self):
        """Test MaskedArray with multi-column index DataFrame."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, True], [False, True]])
        skeleton_df = make_skeleton_df({
            'patient_id': ['P001', 'P002'],
            'visit_id': ['V1', 'V2'],
            'treatment': ['A', 'B']
        }, n_vars=2)
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns=['patient_id', 'visit_id', 'treatment'])
        
        assert masked_array.shape == (2, 2)
        assert len(masked_array.index_df.columns) == 3
        assert len(masked_array.get_valid_data()) == 3  # Three valid elements
    
    def test_masked_array_empty_data(self):
        """Test MaskedArray with empty data."""
        data = jnp.array([]).reshape(0, 2)
        mask = np.array([]).reshape(0, 2).astype(bool)
        skeleton_df = DataFrame({'id': [], 'var$0$value': [], 'var$0$mask': [],
                                  'var$1$value': [], 'var$1$mask': []})
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns='id')
        
        assert masked_array.shape == (0, 2)
        assert len(masked_array.get_valid_data()) == 0
    
    def test_masked_array_all_masked(self):
        """Test MaskedArray where all data is masked."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[False, False], [False, False]])  # All False = all masked
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns='id')
        
        assert masked_array.shape == (2, 2)
        assert len(masked_array.get_valid_data()) == 0  # No valid data
    
    def test_masked_array_all_valid(self):
        """Test MaskedArray where all data is valid."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, True], [True, True]])  # All True = all valid
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                                   index_columns='id')
        
        assert masked_array.shape == (2, 2)
        assert len(masked_array.get_valid_data()) == 4  # All data valid
        
        # Check that valid data matches original data (flattened)
        expected_valid = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert jnp.array_equal(masked_array.get_valid_data(), expected_valid)
    
    def test_with_data_method(self):
        """Test with_data method for immutable data updates."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                               index_columns='id')
        
        # Valid update should create new MaskedArray
        new_data = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        updated = original.with_data(new_data)
        
        # Original should be unchanged
        assert jnp.array_equal(original.data, data)
        
        # New array should have updated data
        assert jnp.array_equal(updated.data, new_data)
        assert np.array_equal(updated.mask, mask)  # Mask should be copied
        assert updated.index_df == DataFrame({'id': ['A', 'B']})  # Same index
        
        # Objects should be different
        assert original is not updated
        
        # Invalid shape should fail  
        wrong_shape_data = jnp.array([[1.0, 2.0, 3.0]])  # Wrong shape
        with pytest.raises(ValueError, match="Number of data rows .* must match wide_skeleton_df length"):
            original.with_data(wrong_shape_data)
        
        # Wrong number of columns should also fail
        wrong_cols_data = jnp.array([[1.0], [2.0]])  # Right rows, wrong columns  
        with pytest.raises(ValueError, match="New data shape .* must match mask shape"):
            original.with_data(wrong_cols_data)
    
    def test_with_mask_method(self):
        """Test with_mask method for immutable mask updates."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                               index_columns='id')
        
        # Valid update should create new MaskedArray
        new_mask = np.array([[False, True], [False, False]])
        updated = original.with_mask(new_mask)
        
        # Original should be unchanged
        assert np.array_equal(original.mask, mask)
        
        # New array should have updated mask
        assert np.array_equal(updated.mask, new_mask)
        assert jnp.array_equal(updated.data, data)  # Data should be copied
        assert updated.index_df == DataFrame({'id': ['A', 'B']})  # Same index
        
        # Objects should be different
        assert original is not updated
        
        # Invalid shape should fail
        wrong_shape_mask = np.array([[True, False, True]])  # Wrong shape
        with pytest.raises(ValueError, match="New mask shape .* must match data shape"):
            original.with_mask(wrong_shape_mask)
        
        # Non-numpy array should fail
        with pytest.raises(ValueError, match="New mask must be a numpy array"):
            original.with_mask([[True, False], [True, True]])  # List instead of numpy array
    
    def test_with_data_and_mask_method(self):
        """Test with_data_and_mask method for immutable updates."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=2)
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                               index_columns='id')
        
        # Valid update should create new MaskedArray
        new_data = jnp.array([[100.0, 200.0], [300.0, 400.0]])
        new_mask = np.array([[False, False], [True, False]])
        updated = original.with_data_and_mask(new_data, new_mask)
        
        # Original should be unchanged
        assert jnp.array_equal(original.data, data)
        assert np.array_equal(original.mask, mask)
        
        # New array should have updated data and mask
        assert jnp.array_equal(updated.data, new_data)
        assert np.array_equal(updated.mask, new_mask)
        assert updated.index_df == DataFrame({'id': ['A', 'B']})  # Same index
        
        # Objects should be different
        assert original is not updated
        
        # Test validation errors
        wrong_data = jnp.array([[1.0]])  # Wrong shape
        with pytest.raises(ValueError, match="Number of data rows .* must match wide_skeleton_df length"):
            original.with_data_and_mask(wrong_data, new_mask)
        
        wrong_mask = np.array([[True]])  # Wrong shape
        with pytest.raises(ValueError, match="New data shape .* must match new mask shape"):
            original.with_data_and_mask(new_data, wrong_mask)
    
    def test_immutable_property_modification_preserves_functionality(self):
        """Test that immutable operations preserve functionality."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = np.array([[True, False, True], [False, True, True]])
        skeleton_df = make_skeleton_df({'id': ['A', 'B']}, n_vars=3)
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                               index_columns='id')
        
        # Create modified version
        new_data = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        new_mask = np.array([[False, True, False], [True, False, True]])
        modified = original.with_data_and_mask(new_data, new_mask)
        
        # Test that get_valid_data works correctly on both
        original_valid = original.get_valid_data()
        expected_original = jnp.array([1.0, 3.0, 5.0, 6.0])
        assert jnp.array_equal(original_valid, expected_original)
        
        modified_valid = modified.get_valid_data()
        expected_modified = jnp.array([20.0, 40.0, 60.0])
        assert jnp.array_equal(modified_valid, expected_modified)
        
        # Test that string representation works on both
        original_str = str(original)
        modified_str = str(modified)
        assert "MaskedArray" in original_str
        assert "MaskedArray" in modified_str
        assert "4/6 (66.7%)" in original_str  # 4 valid out of 6 total
        assert "3/6 (50.0%)" in modified_str  # 3 valid out of 6 total
        
        # Test that copy still works on both
        original_copy = original.copy()
        modified_copy = modified.copy()
        assert original == original_copy
        assert modified == modified_copy
    
    def test_with_index_columns_method(self):
        """Test with_index_columns method for immutable index column updates."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = DataFrame({
            'id': ['A', 'B'],
            'group': [1, 2],
            'var$0$value': [0.0, 0.0],
            'var$0$mask': [True, True],
            'var$1$value': [0.0, 0.0],
            'var$1$mask': [True, True]
        })
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                               index_columns='id')
        
        # Valid update should create new MaskedArray with different index columns
        updated = original.with_index_columns(['id', 'group'])
        
        # Original should be unchanged
        assert original.index_columns == ['id']
        
        # New array should have updated index_columns
        assert updated.index_columns == ['id', 'group']
        assert jnp.array_equal(updated.data, data)  # Data unchanged
        assert np.array_equal(updated.mask, mask)  # Mask copied
        
        # Objects should be different
        assert original is not updated
        
        # Invalid column name should fail
        with pytest.raises(ValueError, match="Index column .* not found in wide_skeleton_df"):
            original.with_index_columns(['nonexistent'])
    
    def test_with_all_method(self):
        """Test with_all method for updating multiple components at once."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        skeleton_df = DataFrame({
            'id': ['A', 'B'],
            'group': [1, 2],
            'var$0$value': [0.0, 0.0],
            'var$0$mask': [True, True],
            'var$1$value': [0.0, 0.0],
            'var$1$mask': [True, True]
        })
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=skeleton_df,
                               index_columns='id')
        
        # Test updating data and mask
        new_data = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        new_mask = np.array([[False, True], [False, False]])
        
        updated = original.with_all(new_data=new_data, new_mask=new_mask)
        
        # Original should be unchanged
        assert jnp.array_equal(original.data, data)
        assert np.array_equal(original.mask, mask)
        
        # New array should have all updates
        assert jnp.array_equal(updated.data, new_data)
        assert np.array_equal(updated.mask, new_mask)
        assert updated.index_columns == ['id']  # Unchanged
        
        # Test updating only data
        updated_data_only = original.with_all(new_data=new_data)
        assert jnp.array_equal(updated_data_only.data, new_data)
        assert np.array_equal(updated_data_only.mask, mask)  # Unchanged
        
        # Test updating only mask
        updated_mask_only = original.with_all(new_mask=new_mask)
        assert jnp.array_equal(updated_mask_only.data, data)  # Unchanged
        assert np.array_equal(updated_mask_only.mask, new_mask)
        
        # Test updating index_columns
        updated_index = original.with_all(new_index_columns=['id', 'group'])
        assert updated_index.index_columns == ['id', 'group']
        assert jnp.array_equal(updated_index.data, data)  # Unchanged
        
        # Test validation errors
        wrong_shape_data = jnp.array([[1.0]])  # Wrong shape
        with pytest.raises(ValueError, match="Data rows .* must match wide_skeleton_df length"):
            original.with_all(new_data=wrong_shape_data)
        
        incompatible_mask = np.array([[True]])  # Wrong shape
        with pytest.raises(ValueError, match="Data shape .* must match mask shape"):
            original.with_all(new_mask=incompatible_mask)

    def test_wide_skeleton_df_with_order_columns(self):
        """Test MaskedArray creation with wide_skeleton_df containing order columns."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, True], [True, False]])
        
        # Create a skeleton with order columns
        wide_skeleton_df = DataFrame({
            'id': ['A', 'B'],
            'var$0$value': [1.0, 3.0],
            'var$0$mask': [True, True],
            'var$0$order': [1, 0],  # Original positions before sorting
            'var$1$value': [2.0, 4.0],
            'var$1$mask': [True, False],
            'var$1$order': [0, 1],
        })
        
        # Create MaskedArray with skeleton
        masked_array = MaskedArray(data=data, mask=mask, wide_skeleton_df=wide_skeleton_df,
                                   index_columns='id')
        
        assert masked_array.wide_skeleton_df is not None
        assert 'var$0$order' in masked_array.wide_skeleton_df.columns
        assert masked_array.wide_skeleton_df['var$0$order'] == [1, 0]
        assert 'var$0$order' in masked_array.wide_skeleton_df.columns
        assert masked_array.wide_skeleton_df['var$0$order'] == [1, 0]
    
    def test_wide_skeleton_df_validation(self):
        """Test validation of wide_skeleton_df length."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, True], [True, True]])
        
        # Skeleton with wrong length should fail
        wrong_length_skeleton = make_skeleton_df({'id': ['A']}, n_vars=2)
        with pytest.raises(ValueError, match="Number of data rows .* must match wide_skeleton_df length"):
            MaskedArray(data=data, mask=mask, wide_skeleton_df=wrong_length_skeleton,
                       index_columns='id')
    
    def test_wide_skeleton_df_preserved_in_copy(self):
        """Test that wide_skeleton_df is preserved when copying."""
        data = jnp.array([[1.0, 2.0]])
        mask = np.array([[True, True]])
        wide_skeleton_df = DataFrame({
            'id': ['A'],
            'var$0$value': [0.0],
            'var$0$mask': [True],
            'var$0$order': [0],
            'var$1$value': [0.0],
            'var$1$mask': [True]
        })
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=wide_skeleton_df,
                               index_columns='id')
        copied = original.copy()
        
        assert copied.wide_skeleton_df is not None
        assert copied.wide_skeleton_df['var$0$order'] == [0]
    
    def test_wide_skeleton_df_preserved_in_with_methods(self):
        """Test that wide_skeleton_df is preserved through with_* methods."""
        data = jnp.array([[1.0, 2.0]])
        mask = np.array([[True, True]])
        wide_skeleton_df = DataFrame({
            'id': ['A'],
            'var$0$value': [0.0],
            'var$0$mask': [True],
            'var$0$order': [0],
            'var$1$value': [0.0],
            'var$1$mask': [True]
        })
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=wide_skeleton_df,
                               index_columns='id')
        
        # with_data should preserve skeleton
        new_data = jnp.array([[5.0, 6.0]])
        updated = original.with_data(new_data)
        assert updated.wide_skeleton_df is not None
        
        # with_mask should preserve skeleton
        new_mask = np.array([[False, True]])
        updated = original.with_mask(new_mask)
        assert updated.wide_skeleton_df is not None
        
        # with_data_and_mask should preserve skeleton
        updated = original.with_data_and_mask(new_data, new_mask)
        assert updated.wide_skeleton_df is not None
    
    def test_with_wide_skeleton_df(self):
        """Test the with_wide_skeleton_df method."""
        data = jnp.array([[1.0, 2.0]])
        mask = np.array([[True, True]])
        wide_skeleton_df = DataFrame({
            'id': ['A'],
            'var$0$value': [0.0],
            'var$0$mask': [True],
            'var$1$value': [0.0],
            'var$1$mask': [True]
        })
        
        original = MaskedArray(data=data, mask=mask, wide_skeleton_df=wide_skeleton_df,
                               index_columns='id')
        
        # Update skeleton with order columns
        new_skeleton = DataFrame({
            'id': ['A'],
            'var$0$value': [0.0],
            'var$0$mask': [True],
            'var$0$order': [5],
            'var$1$value': [0.0],
            'var$1$mask': [True]
        })
        updated = original.with_wide_skeleton_df(new_skeleton)
        
        assert updated.wide_skeleton_df is not None
        assert updated.wide_skeleton_df['var$0$order'] == [5]
        
        # Original should be unchanged (immutability)
        assert 'var$0$order' not in original.wide_skeleton_df.columns
    
    def test_to_dict_includes_skeleton(self):
        """Test that to_dict includes wide_skeleton_df."""
        data = jnp.array([[1.0]])
        mask = np.array([[True]])
        wide_skeleton_df = DataFrame({
            'id': ['A'],
            'order': [0],
            'var$0$value': [0.0],
            'var$0$mask': [True]
        })
        
        ma = MaskedArray(data=data, mask=mask, wide_skeleton_df=wide_skeleton_df,
                        index_columns='id')
        
        result = ma.to_dict()
        assert 'wide_skeleton_df' in result
        assert result['wide_skeleton_df']['order'] == [0]
        assert result['index_columns'] == ['id']
