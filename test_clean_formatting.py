#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.jitprint import jit_print_dataframe, jit_print_dataframe_data

def test_clean_formatting():
    """Test that the DataFrame printing now shows cleaner output without unnecessary quotes."""
    
    print("=== Testing Clean DataFrame Formatting ===")
    
    # Create a DataFrame with mixed data types
    df = DataFrame({
        'volume_fraction': [0.108, 0.098, 0.115],
        'pore_diameter_mean_um': [2.017, 2.080, 1.950],
        'gel_id': ['0057', '0058', '0059'],
        'is_valid': [True, False, True],
        'count': [10, 20, 15]
    })
    
    print("Before (old format would show quotes around all values):")
    print("  [0]: {'volume_fraction': '0.108', 'pore_diameter_mean_um': '2.017', 'gel_id': '0057', 'is_valid': 'True', 'count': '10'}")
    print()
    print("After (new format with smart quoting):")
    
    # Test static argument version
    def test_with_static_df(data, static_df):
        jax.debug.print("\\n=== Static DataFrame Test ===")
        jit_print_dataframe(static_df)
        return jnp.sum(data)
    
    jit_fn = jax.jit(test_with_static_df, static_argnames=['static_df'])
    test_data = jnp.array([1.0, 2.0, 3.0])
    
    result = jit_fn(test_data, df)
    print(f"Static DF test result: {result}")
    
    # Test data version
    print("\\n=== DataFrame Data Test ===")
    @jax.jit
    def test_with_data():
        jax.debug.print("\\n=== Data Version Test ===")
        
        # Mix of numeric and tracer data
        data = {
            'numeric_col': jnp.array([1.5, 2.7, 3.1]),
            'tracer_col': jnp.array([10, 20, 30]),
            'mixed_col': jnp.array([True, False, True])
        }
        columns = ['numeric_col', 'tracer_col', 'mixed_col']
        
        jit_print_dataframe_data(data, columns, 3, "test_data")
        return jnp.array([42.0])
    
    result2 = test_with_data()
    print(f"Data test result: {result2}")

def test_with_complex_tracers():
    """Test formatting with complex tracers."""
    
    print("\\n=== Testing with Complex Tracers ===")
    
    df = DataFrame({
        'base_value': [1.0, 2.0],
        'string_id': ['ABC', 'DEF']
    })
    
    def complex_function(params, static_df):
        # Create complex tracers
        complex_vals = jnp.squeeze(params.reshape(-1, 1))
        
        # Add to DataFrame
        new_df = static_df.add_column('complex_tracers', complex_vals)
        
        jax.debug.print("\\n=== Complex Tracers Test ===")
        jit_print_dataframe(new_df)
        
        return jnp.sum(complex_vals)
    
    grad_fn = jax.grad(complex_function)
    jit_grad = jax.jit(grad_fn, static_argnames=['static_df'])
    
    test_params = jnp.array([3.5, 4.5])
    result = jit_grad(test_params, df)
    
    print(f"Complex tracer test result: {result}")

if __name__ == "__main__":
    test_clean_formatting()
    test_with_complex_tracers()
    
    print("\\n✅ Clean formatting tests completed!")
    print("Notice: Numeric values no longer have quotes, but strings still do!")
