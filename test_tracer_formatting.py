"""
Test script to verify improved JAX tracer handling in jit_print functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxframe import DataFrame, MaskedArray
from jaxframe.jitprint import jit_print_dataframe, jit_print_dataframe_data, _format_value_for_jit_print


def test_tracer_formatting():
    """Test the helper function for formatting tracers."""
    print("=== Testing Tracer Formatting ===")
    
    @jax.jit
    def test_format_inside_jit():
        # Create different types of tracers
        float_tracer = jnp.array([1.0, 2.0, 3.0])
        int_tracer = jnp.array([1, 2, 3])
        scalar_tracer = jnp.array(5.0)
        bool_tracer = jnp.array([True, False, True])
        
        # Test our formatting function
        jax.debug.print("Float array tracer: {}", _format_value_for_jit_print(float_tracer))
        jax.debug.print("Int array tracer: {}", _format_value_for_jit_print(int_tracer))
        jax.debug.print("Scalar tracer: {}", _format_value_for_jit_print(scalar_tracer))
        jax.debug.print("Bool array tracer: {}", _format_value_for_jit_print(bool_tracer))
        
        return float_tracer
    
    print("Running JIT function to test tracer formatting...")
    test_format_inside_jit()
    print()


def test_dataframe_with_tracers():
    """Test DataFrame printing with JAX tracers."""
    print("=== Testing DataFrame with Tracers ===")
    
    # Create a DataFrame with regular data first
    df = DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.7, 30.9]
    })
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Test with tracers using static argument approach
    def process_df_with_tracers(x, dataframe):
        jax.debug.print("=== DataFrame with tracers (static arg) ===")
        jit_print_dataframe(dataframe)
        return x * 2
    
    process_df_jit = jax.jit(process_df_with_tracers, static_argnames=['dataframe'])
    
    print("Running with DataFrame as static argument...")
    result = process_df_jit(jnp.array([1.0, 2.0]), df)
    print(f"Result: {result}")
    print()


def test_dataframe_data_with_tracers():
    """Test DataFrame data printing with JAX tracers."""
    print("=== Testing DataFrame Data with Tracers ===")
    
    @jax.jit
    def process_with_tracer_data():
        # Create JAX arrays (these will be tracers inside JIT)
        ids = jnp.array([1, 2, 3])
        values = jnp.array([10.5, 20.7, 30.9])
        
        data_dict = {'id': ids, 'value': values}
        columns = ['id', 'value']
        
        jax.debug.print("=== DataFrame data with tracers ===")
        jit_print_dataframe_data(data_dict, columns, 3, "test_df")
        
        return jnp.sum(values)
    
    print("Running JIT function with tracer data...")
    result = process_with_tracer_data()
    print(f"Result: {result}")
    print()


def test_mixed_types_with_tracers():
    """Test DataFrame with mixed tracer and non-tracer types."""
    print("=== Testing Mixed Types ===")
    
    @jax.jit
    def process_mixed_data():
        # Mix of JAX arrays and regular values
        jax_values = jnp.array([1.1, 2.2, 3.3])
        regular_strings = ["apple", "banana", "cherry"]  # These will be static
        
        # Note: In practice, string data would need to be passed as static
        # This is more of a conceptual test
        jax.debug.print("JAX tracer formatting: {}", _format_value_for_jit_print(jax_values[0]))
        jax.debug.print("Regular string formatting: {}", _format_value_for_jit_print("apple"))
        jax.debug.print("Regular float formatting: {}", _format_value_for_jit_print(3.14159))
        
        return jax_values
    
    print("Running mixed types test...")
    result = process_mixed_data()
    print(f"Result: {result}")
    print()


def main():
    """Run all tests."""
    print("JAX Tracer Formatting Tests")
    print("=" * 50)
    
    test_tracer_formatting()
    test_dataframe_with_tracers()
    test_dataframe_data_with_tracers()
    test_mixed_types_with_tracers()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
