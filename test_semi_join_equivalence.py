#!/usr/bin/env python3
"""Test script to verify that join(how='semi') works identically to the old exists method."""

from jaxframe import DataFrame
import jax.numpy as jnp
import numpy as np

def test_semi_join_equivalence():
    """Test that join with how='semi' produces the same results as the old exists method would have."""
    
    print("Testing semi-join equivalence...")
    
    # Create test DataFrames
    customers_data = {
        'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA'],
        'scores': jnp.array([85.5, 92.0, 78.5, 88.2, 91.1])
    }
    customers = DataFrame(customers_data, name="customers")
    
    orders_data = {
        'order_id': ['O1', 'O2', 'O3', 'O4', 'O5'],
        'customer_id': ['C1', 'C1', 'C3', 'C3', 'C5'],  # C2 and C4 have no orders
        'amount': jnp.array([100.0, 150.0, 200.0, 75.0, 300.0])
    }
    orders = DataFrame(orders_data, name="orders")
    
    # Test 1: Basic semi-join
    print("Test 1: Basic semi-join")
    result = customers.join(orders, on='customer_id', how='semi')
    
    # Should only include customers who have orders: C1, C3, C5
    assert len(result) == 3, f"Expected 3 customers, got {len(result)}"
    expected_customers = ['C1', 'C3', 'C5']
    actual_customers = list(result['customer_id'])
    assert actual_customers == expected_customers, f"Expected {expected_customers}, got {actual_customers}"
    
    # Should preserve original column structure
    assert list(result.columns) == list(customers.columns)
    assert 'order_id' not in result.columns
    assert 'amount' not in result.columns
    
    # Verify JAX array type is preserved
    assert result.column_types['scores'] == 'jax_array'
    print("✓ Basic semi-join test passed")
    
    # Test 2: Multi-column semi-join
    print("Test 2: Multi-column semi-join")
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
    
    active_users = users.join(sessions, on=['user_id', 'region'], how='semi')
    assert len(active_users) == 2
    assert list(active_users['user_id']) == ['U1', 'U3']
    assert list(active_users['region']) == ['US', 'EU']
    print("✓ Multi-column semi-join test passed")
    
    # Test 3: Duplicate elimination
    print("Test 3: Duplicate elimination")
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
    
    dedup_result = left_df.join(right_df, on='id', how='semi')
    assert len(dedup_result) == 2
    assert list(dedup_result['id']) == ['A', 'B']
    assert list(dedup_result['value']) == [1, 2]
    print("✓ Duplicate elimination test passed")
    
    # Test 4: Empty result
    print("Test 4: Empty result")
    no_match_left = DataFrame({'id': ['X', 'Y'], 'value': [1, 2]})
    no_match_right = DataFrame({'id': ['A', 'B'], 'other': [10, 20]})
    
    empty_result = no_match_left.join(no_match_right, on='id', how='semi')
    assert len(empty_result) == 0
    assert list(empty_result.columns) == list(no_match_left.columns)
    print("✓ Empty result test passed")
    
    print("\n🎉 All semi-join equivalence tests passed!")
    print("The join method with how='semi' works identically to the old exists method!")

def test_inner_join_still_works():
    """Test that inner joins still work as expected."""
    
    print("\nTesting that inner joins still work...")
    
    df1 = DataFrame({
        'assay_id': ['A1', 'A2', 'A3'],
        'name': ['Test1', 'Test2', 'Test3']
    }, name="assays")
    
    df2 = DataFrame({
        'assay_id': ['A1', 'A2', 'A3'],
        'var': [1.5, 2.5, 3.5]
    }, name="params")
    
    # Test inner join (default behavior)
    result1 = df1.join(df2, on='assay_id', source='var')
    assert len(result1) == 3
    assert 'params/var' in result1.columns
    assert result1['params/var'][0] == 1.5
    
    # Test inner join (explicit)
    result2 = df1.join(df2, on='assay_id', source='var', how='inner')
    assert len(result2) == 3
    assert 'params/var' in result2.columns
    assert result2['params/var'][0] == 1.5
    
    # Results should be identical
    assert result1.to_dict() == result2.to_dict()
    
    print("✓ Inner join functionality preserved")

def demonstrate_usage():
    """Demonstrate the new usage patterns."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: New join API with 'how' parameter")
    print("="*60)
    
    # Create sample data
    customers = DataFrame({
        'customer_id': ['C1', 'C2', 'C3', 'C4'],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'region': ['US', 'EU', 'US', 'EU']
    }, name="customers")
    
    orders = DataFrame({
        'customer_id': ['C1', 'C3'],  # Unique customer IDs - C2 and C4 have no orders
        'amount': [100.0, 200.0]
    }, name="orders")
    
    print("Sample data:")
    print("Customers:", customers.to_dict())
    print("Orders:", orders.to_dict())
    
    print("\n1. Inner join (adds columns from right table):")
    inner_result = customers.join(orders, on='customer_id', source='amount', how='inner')
    print("Result:", inner_result.to_dict())
    print("Columns:", inner_result.columns)
    
    print("\n2. Semi-join (filters rows based on existence, no new columns):")
    semi_result = customers.join(orders, on='customer_id', how='semi')
    print("Result:", semi_result.to_dict())
    print("Columns:", semi_result.columns)
    
    print("\nComparison:")
    print(f"Inner join result: {len(inner_result)} rows, {len(inner_result.columns)} columns")
    print(f"Semi-join result: {len(semi_result)} rows, {len(semi_result.columns)} columns")
    print(f"Same customers filtered: {list(inner_result['customer_id']) == list(semi_result['customer_id'])}")

if __name__ == "__main__":
    test_semi_join_equivalence()
    test_inner_join_still_works()
    demonstrate_usage()
    print("\n✅ All tests completed successfully!")
