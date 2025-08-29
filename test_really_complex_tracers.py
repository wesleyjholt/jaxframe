#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.jitprint import jit_print_dataframe, _format_value_for_jit_print

def test_really_complex_tracers():
    """Test the most complex JAX tracers we can create."""
    
    # Create a DataFrame
    df = DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [4.0, 5.0, 6.0]
    })
    
    def complex_function(params, static_df):
        """Function with multiple transformations creating complex tracers."""
        # This should create very complex nested tracers
        squeezed = jnp.squeeze(params.reshape(-1, 1))  # This creates squeeze operations
        result = jnp.sum(squeezed ** 2)
        
        # Add the complex tracers to the DataFrame
        new_df = static_df.add_column('complex_tracers', squeezed)
        
        print("Testing complex tracer formatting:")
        print(f"Manual format test: {_format_value_for_jit_print(squeezed)}")
        
        jit_print_dataframe(new_df)
        
        return result
    
    # Create multiple levels of transformation to generate complex tracers
    # 1. First apply grad (creates JVP tracers)
    grad_fn = jax.grad(complex_function)
    
    # 2. Then JIT it (creates additional tracer complexity)
    jitted_grad = jax.jit(grad_fn, static_argnames=['static_df'])
    
    print("=== Testing Really Complex Tracers ===")
    test_params = jnp.array([1.0, 2.0, 3.0])
    
    try:
        result = jitted_grad(test_params, df)
        print(f"Result: {result}")
        print("✓ Complex tracer test passed!")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        # Let's also test the specific tracer you mentioned
        print("\n=== Manual Testing ===")
        
        # Try to manually create the problematic scenario
        @jax.jit
        def test_squeeze_in_grad():
            x = jnp.array([5.0]).reshape(-1, 1)
            squeezed = jnp.squeeze(x)
            formatted = _format_value_for_jit_print(squeezed)
            jax.debug.print("Squeezed tracer format: {}", formatted)
            return squeezed
        
        result = test_squeeze_in_grad()
        print(f"Manual test result: {result}")
        return False

if __name__ == "__main__":
    success = test_really_complex_tracers()
    if not success:
        print("\nNeed to investigate further...")
