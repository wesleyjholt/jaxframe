#!/usr/bin/env python3
"""
Test script for the lookup table functionality in DataFrame.
"""

import numpy as np
import jax.numpy as jnp
from jaxframe import DataFrame

def test_is_valid_lookup_table():
    """Test the is_valid_lookup_table method."""
    print("=== Testing is_valid_lookup_table ===")
    
    # Test valid lookup table (no duplicates)
    valid_data = {
        'id': ['A', 'B', 'C'],
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [1, 2, 3]
    }
    valid_df = DataFrame(valid_data, name="valid_lookup")
    
    assert valid_df.is_valid_lookup_table('id') == True
    print("✓ Valid single-column lookup table detected correctly")
    
    # Test invalid lookup table (has duplicates)
    invalid_data = {
        'id': ['A', 'B', 'A'],  # Duplicate 'A'
        'name': ['Alice', 'Bob', 'Alice2'],
        'value': [1, 2, 3]
    }
    invalid_df = DataFrame(invalid_data, name="invalid_lookup")
    
    assert invalid_df.is_valid_lookup_table('id') == False
    print("✓ Invalid single-column lookup table detected correctly")
    
    # Test multi-column keys
    multi_key_data = {
        'region': ['US', 'US', 'EU', 'EU'],
        'product': ['A', 'B', 'A', 'B'],
        'price': [10, 20, 15, 25]
    }
    multi_key_df = DataFrame(multi_key_data, name="multi_key_lookup")
    
    assert multi_key_df.is_valid_lookup_table(['region', 'product']) == True
    print("✓ Valid multi-column lookup table detected correctly")
    
    # Test invalid multi-column keys
    invalid_multi_data = {
        'region': ['US', 'US', 'US'],  # Duplicate (US, A)
        'product': ['A', 'B', 'A'],
        'price': [10, 20, 15]
    }
    invalid_multi_df = DataFrame(invalid_multi_data, name="invalid_multi_lookup")
    
    assert invalid_multi_df.is_valid_lookup_table(['region', 'product']) == False
    print("✓ Invalid multi-column lookup table detected correctly")
    
    # Test error on missing column
    try:
        valid_df.is_valid_lookup_table('nonexistent')
        assert False, "Should have raised KeyError"
    except KeyError:
        print("✓ KeyError raised for missing column")

def test_update_lookup_table_strict():
    """Test the update_lookup_table method with strict=True."""
    print("\n=== Testing update_lookup_table (strict=True) ===")
    
    # Base lookup table
    base_data = {
        'id': ['A', 'B'],
        'name': ['Alice', 'Bob'],
        'value': [1.0, 2.0]
    }
    base_df = DataFrame(base_data, name="base_lookup")
    
    # Test adding new rows (should work)
    new_data = {
        'id': ['C', 'D'],
        'name': ['Charlie', 'David'],
        'value': [3.0, 4.0]
    }
    new_df = DataFrame(new_data, name="new_data")
    
    updated_df = base_df.update_lookup_table(new_df, 'id', strict=True)
    
    assert len(updated_df) == 4
    assert set(updated_df['id']) == {'A', 'B', 'C', 'D'}
    print("✓ Successfully added new rows")
    
    # Test updating existing rows with same values (should work)
    same_data = {
        'id': ['A', 'E'],
        'name': ['Alice', 'Eve'],  # Same value for A, new row E
        'value': [1.0, 5.0]
    }
    same_df = DataFrame(same_data, name="same_values")
    
    updated_df2 = base_df.update_lookup_table(same_df, 'id', strict=True)
    
    assert len(updated_df2) == 3
    assert updated_df2['name'][0] == 'Alice'  # Should remain unchanged
    assert 'E' in updated_df2['id']
    print("✓ Successfully handled matching values and added new row")
    
    # Test conflicting values (should raise error)
    conflict_data = {
        'id': ['A'],
        'name': ['Alice_DIFFERENT'],  # Conflict!
        'value': [1.0]
    }
    conflict_df = DataFrame(conflict_data, name="conflict_data")
    
    try:
        base_df.update_lookup_table(conflict_df, 'id', strict=True)
        assert False, "Should have raised ValueError for conflict"
    except ValueError as e:
        assert "Value mismatch" in str(e)
        print("✓ ValueError raised for conflicting values")

def test_update_lookup_table_non_strict():
    """Test the update_lookup_table method with strict=False."""
    print("\n=== Testing update_lookup_table (strict=False) ===")
    
    # Base lookup table
    base_data = {
        'id': ['A', 'B'],
        'name': ['Alice', 'Bob'],
        'value': [1.0, 2.0]
    }
    base_df = DataFrame(base_data, name="base_lookup")
    
    # Test replacing existing values
    replacement_data = {
        'id': ['A', 'C'],
        'name': ['Alice_NEW', 'Charlie'],  # Replace Alice's name
        'value': [10.0, 3.0]  # Replace Alice's value, add Charlie
    }
    replacement_df = DataFrame(replacement_data, name="replacement_data")
    
    updated_df = base_df.update_lookup_table(replacement_df, 'id', strict=False)
    
    assert len(updated_df) == 3
    # Find Alice's row
    alice_idx = None
    for i, id_val in enumerate(updated_df['id']):
        if id_val == 'A':
            alice_idx = i
            break
    
    assert alice_idx is not None
    assert updated_df['name'][alice_idx] == 'Alice_NEW'
    assert updated_df['value'][alice_idx] == 10.0
    assert 'C' in updated_df['id']
    print("✓ Successfully replaced existing values and added new row")

def test_replace_lookup_table():
    """Test the replace_lookup_table convenience method."""
    print("\n=== Testing replace_lookup_table ===")
    
    # Base lookup table
    base_data = {
        'id': ['A', 'B'],
        'name': ['Alice', 'Bob'],
        'value': [1.0, 2.0]
    }
    base_df = DataFrame(base_data, name="base_lookup")
    
    # Replacement data
    replacement_data = {
        'id': ['A', 'C'],
        'name': ['Alice_REPLACED', 'Charlie'],
        'value': [100.0, 3.0]
    }
    replacement_df = DataFrame(replacement_data, name="replacement_data")
    
    updated_df = base_df.replace_lookup_table(replacement_df, 'id')
    
    # Should be equivalent to update_lookup_table with strict=False
    expected_df = base_df.update_lookup_table(replacement_df, 'id', strict=False)
    
    assert len(updated_df) == len(expected_df)
    assert set(updated_df['id']) == set(expected_df['id'])
    print("✓ replace_lookup_table works as expected")

def test_multi_column_lookup():
    """Test lookup table operations with multi-column keys."""
    print("\n=== Testing multi-column lookup operations ===")
    
    # Base lookup table with composite keys
    base_data = {
        'region': ['US', 'EU'],
        'product': ['A', 'A'],
        'price': [10.0, 15.0],
        'currency': ['USD', 'EUR']
    }
    base_df = DataFrame(base_data, name="pricing_table")
    
    # New data with overlapping and new keys
    new_data = {
        'region': ['US', 'EU', 'ASIA'],  # US+A exists, EU+A exists, ASIA+A is new
        'product': ['A', 'A', 'A'],
        'price': [10.0, 16.0, 12.0],  # Same for US, different for EU, new for ASIA
        'currency': ['USD', 'EUR', 'YEN']  # Same for US, same for EU, new for ASIA
    }
    new_df = DataFrame(new_data, name="new_pricing")
    
    # Test strict mode (should fail due to EU price difference)
    try:
        base_df.update_lookup_table(new_df, ['region', 'product'], strict=True)
        assert False, "Should have failed due to price mismatch for EU"
    except ValueError:
        print("✓ Strict mode correctly detected price mismatch for EU")
    
    # Test non-strict mode (should replace EU price)
    updated_df = base_df.update_lookup_table(new_df, ['region', 'product'], strict=False)
    
    assert len(updated_df) == 3  # US+A, EU+A, ASIA+A
    
    # Check that EU price was updated
    eu_row_idx = None
    for i in range(len(updated_df)):
        if updated_df['region'][i] == 'EU' and updated_df['product'][i] == 'A':
            eu_row_idx = i
            break
    
    assert eu_row_idx is not None
    assert updated_df['price'][eu_row_idx] == 16.0
    print("✓ Multi-column lookup update worked correctly")

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== Testing edge cases ===")
    
    # Test with different column orders
    df1_data = {
        'id': ['A'],
        'name': ['Alice'],
        'value': [1.0]
    }
    df1 = DataFrame(df1_data, name="df1")
    
    df2_data = {
        'value': [2.0],  # Different order
        'id': ['B'],
        'name': ['Bob']
    }
    df2 = DataFrame(df2_data, name="df2")
    
    updated_df = df1.update_lookup_table(df2, 'id')
    assert len(updated_df) == 2
    print("✓ Different column orders handled correctly")
    
    # Test with incompatible columns
    df3_data = {
        'id': ['C'],
        'different_col': ['Charlie']  # Missing 'name' and 'value'
    }
    df3 = DataFrame(df3_data, name="df3")
    
    try:
        df1.update_lookup_table(df3, 'id')
        assert False, "Should have failed due to incompatible columns"
    except ValueError:
        print("✓ Incompatible columns error raised correctly")
    
    # Test with invalid lookup table (duplicates in base)
    invalid_base_data = {
        'id': ['A', 'A'],  # Duplicate
        'name': ['Alice1', 'Alice2'],
        'value': [1.0, 2.0]
    }
    invalid_base_df = DataFrame(invalid_base_data, name="invalid_base")
    
    try:
        invalid_base_df.update_lookup_table(df2, 'id')
        assert False, "Should have failed due to invalid base lookup table"
    except ValueError:
        print("✓ Invalid base lookup table error raised correctly")

def test_with_jax_arrays():
    """Test lookup table operations with JAX arrays."""
    print("\n=== Testing with JAX arrays ===")
    
    # Base lookup table with JAX arrays
    base_data = {
        'id': ['A', 'B'],
        'vector': [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])],
        'scalar': [jnp.array(10.0), jnp.array(20.0)]
    }
    base_df = DataFrame(base_data, name="jax_lookup")
    
    # New data with JAX arrays
    new_data = {
        'id': ['A', 'C'],  # A exists (same values), C is new
        'vector': [jnp.array([1.0, 2.0]), jnp.array([5.0, 6.0])],
        'scalar': [jnp.array(10.0), jnp.array(30.0)]
    }
    new_df = DataFrame(new_data, name="new_jax_data")
    
    # Test strict mode (should work since values match for A)
    updated_df = base_df.update_lookup_table(new_df, 'id', strict=True)
    
    assert len(updated_df) == 3
    print("✓ JAX array lookup operations work correctly")
    
    # Test with slightly different values (should fail in strict mode)
    different_data = {
        'id': ['A'],
        'vector': [jnp.array([1.0, 2.1])],  # Slightly different
        'scalar': [jnp.array(10.0)]
    }
    different_df = DataFrame(different_data, name="different_jax_data")
    
    try:
        base_df.update_lookup_table(different_df, 'id', strict=True)
        assert False, "Should have failed due to JAX array value mismatch"
    except ValueError:
        print("✓ JAX array value mismatch detected correctly")

if __name__ == "__main__":
    test_is_valid_lookup_table()
    test_update_lookup_table_strict()
    test_update_lookup_table_non_strict()
    test_replace_lookup_table()
    test_multi_column_lookup()
    test_edge_cases()
    test_with_jax_arrays()
    
    print("\n🎉 All lookup table tests passed!")
