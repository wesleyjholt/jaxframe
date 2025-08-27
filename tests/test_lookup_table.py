"""
Test cases for the lookup table functionality in DataFrame.
"""

import pytest
import numpy as np
from jaxframe import DataFrame

# Skip JAX tests if JAX is not available
try:
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    jax_available = False
    jnp = None


def test_is_valid_lookup_table_single_column():
    """Test is_valid_lookup_table with single column keys."""
    # Valid lookup table
    valid_data = {
        'id': ['A', 'B', 'C'],
        'value': [1, 2, 3]
    }
    valid_df = DataFrame(valid_data)
    assert valid_df.is_valid_lookup_table('id') == True
    
    # Invalid lookup table (duplicates)
    invalid_data = {
        'id': ['A', 'B', 'A'],
        'value': [1, 2, 3]
    }
    invalid_df = DataFrame(invalid_data)
    assert invalid_df.is_valid_lookup_table('id') == False


def test_is_valid_lookup_table_multi_column():
    """Test is_valid_lookup_table with multi-column keys."""
    # Valid multi-column lookup table
    valid_data = {
        'region': ['US', 'US', 'EU'],
        'product': ['A', 'B', 'A'],
        'price': [10, 20, 15]
    }
    valid_df = DataFrame(valid_data)
    assert valid_df.is_valid_lookup_table(['region', 'product']) == True
    
    # Invalid multi-column lookup table
    invalid_data = {
        'region': ['US', 'US', 'US'],
        'product': ['A', 'B', 'A'],  # Duplicate (US, A)
        'price': [10, 20, 15]
    }
    invalid_df = DataFrame(invalid_data)
    assert invalid_df.is_valid_lookup_table(['region', 'product']) == False


def test_is_valid_lookup_table_errors():
    """Test error conditions for is_valid_lookup_table."""
    df = DataFrame({'id': ['A', 'B'], 'value': [1, 2]})
    
    with pytest.raises(KeyError):
        df.is_valid_lookup_table('nonexistent')


def test_update_lookup_table_strict_add_new():
    """Test adding new rows with strict=True."""
    base_df = DataFrame({
        'id': ['A', 'B'],
        'name': ['Alice', 'Bob'],
        'value': [1.0, 2.0]
    })
    
    new_df = DataFrame({
        'id': ['C', 'D'],
        'name': ['Charlie', 'David'],
        'value': [3.0, 4.0]
    })
    
    result = base_df.update_lookup_table(new_df, 'id', strict=True)
    
    assert len(result) == 4
    assert set(result['id']) == {'A', 'B', 'C', 'D'}


def test_update_lookup_table_strict_matching_values():
    """Test updating with matching values in strict mode."""
    base_df = DataFrame({
        'id': ['A', 'B'],
        'name': ['Alice', 'Bob'],
        'value': [1.0, 2.0]
    })
    
    update_df = DataFrame({
        'id': ['A', 'C'],
        'name': ['Alice', 'Charlie'],  # Same value for A
        'value': [1.0, 3.0]
    })
    
    result = base_df.update_lookup_table(update_df, 'id', strict=True)
    
    assert len(result) == 3
    assert 'C' in result['id']


def test_update_lookup_table_strict_conflict():
    """Test that strict mode raises error on value conflicts."""
    base_df = DataFrame({
        'id': ['A', 'B'],
        'name': ['Alice', 'Bob'],
        'value': [1.0, 2.0]
    })
    
    conflict_df = DataFrame({
        'id': ['A'],
        'name': ['Alice_DIFFERENT'],  # Conflict!
        'value': [1.0]
    })
    
    with pytest.raises(ValueError, match="Value mismatch"):
        base_df.update_lookup_table(conflict_df, 'id', strict=True)


def test_update_lookup_table_non_strict():
    """Test update_lookup_table with strict=False (replacement mode)."""
    base_df = DataFrame({
        'id': ['A', 'B'],
        'name': ['Alice', 'Bob'],
        'value': [1.0, 2.0]
    })
    
    replacement_df = DataFrame({
        'id': ['A', 'C'],
        'name': ['Alice_NEW', 'Charlie'],
        'value': [10.0, 3.0]
    })
    
    result = base_df.update_lookup_table(replacement_df, 'id', strict=False)
    
    assert len(result) == 3
    # Find Alice's row and check it was updated
    alice_idx = next(i for i, id_val in enumerate(result['id']) if id_val == 'A')
    assert result['name'][alice_idx] == 'Alice_NEW'
    assert result['value'][alice_idx] == 10.0


def test_replace_lookup_table():
    """Test the replace_lookup_table convenience method."""
    base_df = DataFrame({
        'id': ['A', 'B'],
        'value': [1, 2]
    })
    
    replacement_df = DataFrame({
        'id': ['A', 'C'],
        'value': [10, 3]
    })
    
    result1 = base_df.replace_lookup_table(replacement_df, 'id')
    result2 = base_df.update_lookup_table(replacement_df, 'id', strict=False)
    
    # Should be equivalent
    assert len(result1) == len(result2)
    assert set(result1['id']) == set(result2['id'])


def test_update_lookup_table_multi_column():
    """Test lookup table updates with multi-column keys."""
    base_df = DataFrame({
        'region': ['US', 'EU'],
        'product': ['A', 'A'],
        'price': [10.0, 15.0]
    })
    
    new_df = DataFrame({
        'region': ['US', 'ASIA'],
        'product': ['A', 'A'],
        'price': [10.0, 12.0]  # Same for US, new for ASIA
    })
    
    result = base_df.update_lookup_table(new_df, ['region', 'product'], strict=True)
    
    assert len(result) == 3  # US+A, EU+A, ASIA+A


def test_update_lookup_table_errors():
    """Test error conditions for update_lookup_table."""
    df1 = DataFrame({'id': ['A'], 'value': [1]})
    
    # Missing column in other DataFrame
    df2 = DataFrame({'different_id': ['B'], 'value': [2]})
    with pytest.raises(KeyError):
        df1.update_lookup_table(df2, 'id')
    
    # Incompatible columns
    df3 = DataFrame({'id': ['B'], 'different_col': [2]})
    with pytest.raises(ValueError, match="same columns"):
        df1.update_lookup_table(df3, 'id')
    
    # Invalid base lookup table
    invalid_base = DataFrame({'id': ['A', 'A'], 'value': [1, 2]})  # Duplicates
    valid_other = DataFrame({'id': ['B'], 'value': [3]})
    with pytest.raises(ValueError, match="not a valid lookup table"):
        invalid_base.update_lookup_table(valid_other, 'id')


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_update_lookup_table_jax_arrays():
    """Test lookup table operations with JAX arrays."""
    base_df = DataFrame({
        'id': ['A', 'B'],
        'vector': [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])],
        'scalar': [jnp.array(10.0), jnp.array(20.0)]
    })
    
    # Same values for A, new entry for C
    new_df = DataFrame({
        'id': ['A', 'C'],
        'vector': [jnp.array([1.0, 2.0]), jnp.array([5.0, 6.0])],
        'scalar': [jnp.array(10.0), jnp.array(30.0)]
    })
    
    result = base_df.update_lookup_table(new_df, 'id', strict=True)
    assert len(result) == 3
    
    # Test conflict detection with JAX arrays
    conflict_df = DataFrame({
        'id': ['A'],
        'vector': [jnp.array([1.0, 2.1])],  # Slightly different
        'scalar': [jnp.array(10.0)]
    })
    
    with pytest.raises(ValueError, match="Value mismatch"):
        base_df.update_lookup_table(conflict_df, 'id', strict=True)


def test_values_equal_helper():
    """Test the _values_equal helper method."""
    df = DataFrame({'id': ['A'], 'value': [1]})
    
    # Test regular values
    assert df._values_equal(1, 1) == True
    assert df._values_equal(1, 2) == False
    
    # Test floats
    assert df._values_equal(1.0, 1.00000000001) == True   # Within tolerance (1e-11)
    assert df._values_equal(1.0, 1.000000001) == False    # Outside tolerance (1e-9)
    
    # Test numpy arrays
    assert df._values_equal(np.array([1, 2]), np.array([1, 2])) == True
    assert df._values_equal(np.array([1, 2]), np.array([1, 3])) == False


@pytest.mark.skipif(not jax_available, reason="JAX not available")
def test_values_equal_jax():
    """Test _values_equal with JAX arrays."""
    df = DataFrame({'id': ['A'], 'value': [1]})
    
    # Test JAX arrays
    assert df._values_equal(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0])) == True
    assert df._values_equal(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1])) == False
