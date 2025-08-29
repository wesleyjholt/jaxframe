"""
Test and demonstration of jit_print functions.

This script tests the JIT-compatible printing functions and compares their output
to the regular printing functionality.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxframe import DataFrame, MaskedArray
from jaxframe.jitprint import (
    jit_print_dataframe,
    jit_print_masked_array,
    jit_print_dataframe_data,
    jit_print_masked_array_data
)


def create_test_objects():
    """Create test DataFrame and MaskedArray objects."""
    # Create a test DataFrame
    df_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.123, 20.456, 30.789, 40.123, 50.456],
        'name': ['apple', 'banana', 'cherry', 'date', 'elderberry']
    }
    df = DataFrame(df_data, name="fruits")
    
    # Create a test MaskedArray
    jax_data = jnp.array([
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [7.7, 8.8, 9.9],
        [10.0, 11.1, 12.2],
        [13.3, 14.4, 15.5]
    ])
    
    mask = np.array([
        [True, True, False],
        [True, False, True],
        [False, True, True],
        [True, True, True],
        [True, False, False]
    ])
    
    # Create index DataFrame for the MaskedArray
    index_data = {
        'row_id': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'C', 'B']
    }
    index_df = DataFrame(index_data)
    
    ma = MaskedArray(jax_data, mask, index_df)
    
    return df, ma


def test_regular_printing():
    """Test regular (non-JIT) printing."""
    print("="*60)
    print("REGULAR PRINTING (for comparison)")
    print("="*60)
    
    df, ma = create_test_objects()
    
    print("DataFrame:")
    print(df)
    print()
    
    print("MaskedArray:")
    print(ma)
    print()


def test_static_argument_printing():
    """Test JIT printing with static arguments."""
    print("="*60)
    print("JIT PRINTING WITH STATIC ARGUMENTS")
    print("="*60)
    
    df, ma = create_test_objects()
    
    # Test DataFrame printing with static argument
    def process_df(x, dataframe):
        jax.debug.print("--- DataFrame inside JIT ---")
        jit_print_dataframe(dataframe)
        return x * 2
    
    process_df_jit = jax.jit(process_df, static_argnames=['dataframe'])
    
    # Test MaskedArray printing with static argument
    def process_ma(x, masked_array):
        jax.debug.print("--- MaskedArray inside JIT ---")
        jit_print_masked_array(masked_array)
        return x + 1
    
    process_ma_jit = jax.jit(process_ma, static_argnames=['masked_array'])
    
    print("Running DataFrame JIT function...")
    result1 = process_df_jit(jnp.array([1.0, 2.0]), df)
    print(f"DataFrame JIT result: {result1}")
    print()
    
    print("Running MaskedArray JIT function...")
    result2 = process_ma_jit(jnp.array([3.0, 4.0]), ma)
    print(f"MaskedArray JIT result: {result2}")
    print()


def test_data_component_printing():
    """Test JIT printing using individual data components."""
    print("="*60)
    print("JIT PRINTING WITH DATA COMPONENTS")
    print("="*60)
    
    df, ma = create_test_objects()
    
    # Test DataFrame data printing
    @jax.jit
    def process_df_data(x):
        jax.debug.print("--- DataFrame data inside JIT ---")
        # Extract data components (these need to be JAX arrays)
        columns = ['id', 'value']  # Subset for demo
        data_dict = {
            'id': jnp.array([1, 2, 3, 4, 5]),
            'value': jnp.array([10.123, 20.456, 30.789, 40.123, 50.456])
        }
        jit_print_dataframe_data(data_dict, columns, 5, "fruits")
        return x * 3
    
    # Test MaskedArray data printing
    @jax.jit
    def process_ma_data(x, data, mask):
        jax.debug.print("--- MaskedArray data inside JIT ---")
        rows, cols = data.shape
        jit_print_masked_array_data(data, mask, rows, cols)
        return jnp.sum(data)
    
    print("Running DataFrame data JIT function...")
    result1 = process_df_data(jnp.array([1.0, 2.0, 3.0]))
    print(f"DataFrame data JIT result: {result1}")
    print()
    
    print("Running MaskedArray data JIT function...")
    # Convert mask to JAX array for JIT compatibility
    jax_mask = jnp.array(ma.mask)
    result2 = process_ma_data(jnp.array([1.0]), ma.data, jax_mask)
    print(f"MaskedArray data JIT result: {result2}")
    print()


def test_edge_cases():
    """Test edge cases like single-row DataFrames."""
    print("="*60)
    print("EDGE CASES")
    print("="*60)
    
    # Single row DataFrame
    single_df = DataFrame({'x': [42], 'y': [3.14]})
    
    def process_single_df(x, dataframe):
        jax.debug.print("--- Single row DataFrame inside JIT ---")
        jit_print_dataframe(dataframe)
        return x
    
    process_single_df_jit = jax.jit(process_single_df, static_argnames=['dataframe'])
    
    print("Testing single row DataFrame...")
    result = process_single_df_jit(jnp.array([2.0]), single_df)
    print(f"Single row DataFrame JIT result: {result}")
    print()
    
    # Large DataFrame (test truncation)
    large_data = {
        'col1': list(range(10)),
        'col2': [i * 1.5 for i in range(10)],
        'col3': [f"item_{i}" for i in range(10)]
    }
    large_df = DataFrame(large_data)
    
    def process_large_df(x, dataframe):
        jax.debug.print("--- Large DataFrame inside JIT (should show truncation) ---")
        jit_print_dataframe(dataframe)
        return x
    
    process_large_df_jit = jax.jit(process_large_df, static_argnames=['dataframe'])
    
    print("Testing large DataFrame (should show '... more rows')...")
    result = process_large_df_jit(jnp.array([3.0]), large_df)
    print(f"Large DataFrame JIT result: {result}")


def main():
    """Run all tests."""
    print("JIT PRINT FUNCTION TESTS")
    print("This script demonstrates JIT-compatible printing for JAXFrame objects.")
    print()
    
    test_regular_printing()
    test_static_argument_printing()
    test_data_component_printing()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
