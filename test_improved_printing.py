#!/usr/bin/env python3
"""
Test script for improved DataFrame printing functionality.
"""

from src.jaxframe import DataFrame
import numpy as np

def test_improved_printing():
    """Test the improved DataFrame printing with various data types."""
    
    print("=== Testing Improved DataFrame Printing ===\n")
    
    # Test 1: Mixed data types similar to the user's example
    print("Test 1: Mixed string and numeric data (similar to user's example)")
    data1 = {
        'volume_fraction': ['0.108', '0.205', '0.142'],
        'pore_diameter_mean_um': ['2.017', '3.456', '1.789'],
        'gel_id': ['0057', '0058', '0059'],
        'numeric_float': [1.234, 5.678, 9.012],
        'numeric_int': [10, 20, 30]
    }
    df1 = DataFrame(data1)
    print(df1)
    print()
    
    # Test 2: Numpy arrays
    print("Test 2: Numpy arrays")
    data2 = {
        'str_vals': ['A', 'B', 'C', 'D'],
        'numpy_int': np.array([100, 200, 300, 400]),
        'numpy_float': np.array([1.11111, 2.22222, 3.33333, 4.44444]),
        'list_int': [1000, 2000, 3000, 4000],
        'list_float': [10.5, 20.7, 30.9, 40.1]
    }
    df2 = DataFrame(data2)
    print(df2)
    print()
    
    # Test 3: JAX arrays (if available)
    print("Test 3: JAX arrays")
    try:
        import jax.numpy as jnp
        data3 = {
            'category': ['cat1', 'cat2', 'cat3'],
            'jax_int': jnp.array([1000, 2000, 3000]),
            'jax_float': jnp.array([123.456789, 234.567890, 345.678901]),
            'mixed_float': [99.99, 88.88, 77.77]
        }
        df3 = DataFrame(data3)
        print(df3)
    except ImportError:
        print("JAX not available - skipping JAX array test")
    print()
    
    # Test 4: Test dtypes method separately
    print("Test 4: Standalone dtypes() method")
    print("df1.dtypes():", df1.dtypes())
    print("df2.dtypes():", df2.dtypes())
    print()
    
    # Test 5: Empty DataFrame
    print("Test 5: Empty DataFrame")
    try:
        empty_data = {'col1': [], 'col2': []}
        df_empty = DataFrame(empty_data)
        print(df_empty)
    except ValueError as e:
        print(f"Empty DataFrame correctly raises error: {e}")
    print()
    
    # Test 6: Single row DataFrame
    print("Test 6: Single row DataFrame")
    single_data = {
        'name': ['Alice'],
        'age': [25],
        'score': [98.5]
    }
    df_single = DataFrame(single_data)
    print(df_single)
    print()
    
    # Test 7: Large DataFrame (to test truncation)
    print("Test 7: Large DataFrame (should show truncation)")
    large_data = {
        'id': list(range(10)),
        'value': [i * 1.5 for i in range(10)],
        'category': [f'cat_{i}' for i in range(10)]
    }
    df_large = DataFrame(large_data)
    print(df_large)
    
    print("\n=== All tests completed! ===")

if __name__ == "__main__":
    test_improved_printing()
