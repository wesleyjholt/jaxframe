#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.jitprint import jit_print_dataframe, _format_value_for_jit_print

# Test with JVP (Jacobian-Vector Product) tracers
def test_complex_tracers():
    """Test the improved tracer formatting with complex JVP tracers."""
    
    # Create test data
    df = DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [4.0, 5.0, 6.0]
    })
    
    def compute_with_grad(params, static_df):
        """Function that uses gradients, creating JVP tracers."""
        jax.debug.print("=== Testing with JVP tracers ===")
        
        # This will create JVP tracers when called through jax.grad
        result = jnp.sum(params ** 2)
        
        # Try to add a column with the tracer (this would cause the ugly output)
        new_df = static_df.add_column('computed', params)
        jit_print_dataframe(new_df)
        
        return result
    
    # Create a gradient function (this creates JVP tracers)
    grad_fn = jax.grad(compute_with_grad)
    grad_fn = jax.jit(grad_fn, static_argnames=['static_df'])
    
    print("Testing with JVP tracers (from jax.grad)...")
    test_params = jnp.array([1.0, 2.0, 3.0])
    
    try:
        result = grad_fn(test_params, df)
        print(f"Gradient result: {result}")
        print("✓ Complex tracer formatting test passed!")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_complex_tracers()
