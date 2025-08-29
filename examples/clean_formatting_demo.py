#!/usr/bin/env python3
"""
Demo showcasing the clean output formatting for JAX JIT-compiled functions.

This example demonstrates how the JAXFrame JIT printing functionality now
provides clean, readable output without unnecessary quotes around numeric values.
"""

import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.jitprint import jit_print_dataframe, jit_print_dataframe_data


def main():
    print("=== Clean Formatting Demo ===\n")
    
    # Create a DataFrame with mixed data types
    df = DataFrame({
        'volume_fraction': [0.108, 0.215, 0.341],
        'pore_diameter_mean_um': [2.017, 1.543, 0.982],
        'gel_id': ['0057', '0123', '0456'],  # String IDs
        'is_valid': [True, False, True],
        'count': [10, 25, 33]
    })
    
    print("DataFrame with mixed types:")
    print(df)
    print()
    
    # Test JIT printing with static DataFrame
    def test_static_dataframe(static_df):
        jax.debug.print("=== Static DataFrame JIT Print ===")
        jit_print_dataframe(static_df)
        return jnp.array([42.0])
    
    jitted_static = jax.jit(test_static_dataframe, static_argnames=['static_df'])
    print("Calling JIT function with static DataFrame:")
    result = jitted_static(static_df=df)
    print(f"Function returned: {result[0]}\n")
    
    # Test JIT printing with tracer data
    def test_tracer_data(params):
        data = {
            'measurements': params,
            'squared': params ** 2,
            'is_positive': params > 0
        }
        columns = ['measurements', 'squared', 'is_positive']
        
        jax.debug.print("=== Tracer Data JIT Print ===")
        jit_print_dataframe_data(data, columns, len(params), "TracerDemo")
        return jnp.sum(params)
    
    jitted_tracer = jax.jit(test_tracer_data)
    print("Calling JIT function with JAX tracers:")
    test_params = jnp.array([1.5, -2.3, 0.8])
    result = jitted_tracer(test_params)
    print(f"Function returned: {result}\n")
    
    # Test with gradient computation (creates more complex tracers)
    def test_gradient_computation(params, static_df):
        # Add tracer column to DataFrame
        enhanced_df = static_df.add_column('computed_values', params)
        
        jax.debug.print("=== Gradient Computation JIT Print ===")
        jit_print_dataframe(enhanced_df)
        
        # Compute some loss
        return jnp.sum(params ** 2)
    
    grad_fn = jax.grad(test_gradient_computation)
    jitted_grad = jax.jit(grad_fn, static_argnames=['static_df'])
    
    print("Calling gradient function (creates complex tracers):")
    test_params = jnp.array([0.5, 1.0, 1.5])
    # Create a smaller DataFrame for the gradient test
    small_df = DataFrame({
        'volume_fraction': [0.108, 0.215, 0.341],
        'gel_id': ['0057', '0123', '0456'],
        'is_valid': [True, False, True]
    })
    gradients = jitted_grad(test_params, static_df=small_df)
    print(f"Gradients: {gradients}\n")
    
    print("=== Output Format Improvements ===")
    print("✅ Numeric values (0.108, 2.017) display without quotes")
    print("✅ Boolean values (True, False) display without quotes") 
    print("✅ JAX tracers (f32, i32[3]) display without quotes")
    print("✅ String values ('0057', 'ABC') retain quotes for clarity")
    print("✅ All formatting is JIT-compatible using jax.debug.print")


if __name__ == "__main__":
    main()
