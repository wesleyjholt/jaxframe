#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.jitprint import jit_print_dataframe, _format_value_for_jit_print

def test_exact_problematic_tracer():
    """Test the exact type of tracer that was causing problems."""
    
    print("=== Testing Exact Problematic Tracer Pattern ===")
    
    # Try to recreate the exact scenario that caused the long tracer string
    df = DataFrame({
        'volume_fraction': [0.108, 0.098],
        'pore_diameter_mean_um': [2.017, 2.080],
        'gel_id': ['0057', '0057']
    })
    
    def problematic_function(params, static_df):
        """Function that creates the exact problematic tracer pattern."""
        # This pattern often creates the complex JVP tracers with squeeze operations
        reshaped = params.reshape(-1, 1)  # Reshape to add dimension
        squeezed = jnp.squeeze(reshaped, axis=-1)  # Squeeze it back (this creates complex tracers)
        
        # Let's test the tracer directly
        print(f"Type of squeezed: {type(squeezed)}")
        print(f"String repr length: {len(str(squeezed))}")
        
        # Test our formatter directly
        formatted = _format_value_for_jit_print(squeezed)
        print(f"Our formatter result: '{formatted}'")
        
        # Add to DataFrame and print
        new_df = static_df.add_column('fab/col_fd', squeezed[:len(static_df._data['volume_fraction'])])
        jit_print_dataframe(new_df)
        
        return jnp.sum(squeezed)
    
    # Create the complex transformation that often causes problematic tracers
    grad_fn = jax.grad(problematic_function)
    jitted_grad = jax.jit(grad_fn, static_argnames=['static_df'])
    
    test_params = jnp.array([0.11994141, 12.791142, 0.4462011])
    
    try:
        result = jitted_grad(test_params, df)
        print(f"Result: {result}")
        print("✓ Exact problematic tracer test passed!")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_tracer_inspection():
    """Manually inspect what attributes different tracers have."""
    
    print("\n=== Manual Tracer Inspection ===")
    
    @jax.jit
    def inspect_tracer():
        x = jnp.array([1.0, 2.0, 3.0])
        
        # Test direct access to tracer attributes
        print(f"Tracer type: {type(x)}")
        print(f"Has aval: {hasattr(x, 'aval')}")
        print(f"Has primal: {hasattr(x, 'primal')}")
        print(f"Has shape: {hasattr(x, 'shape')}")
        print(f"Has dtype: {hasattr(x, 'dtype')}")
        
        if hasattr(x, 'aval'):
            print(f"aval.shape: {x.aval.shape}")
            print(f"aval.dtype: {x.aval.dtype}")
        
        formatted = _format_value_for_jit_print(x)
        jax.debug.print("Formatted: {}", formatted)
        
        return x
    
    result = inspect_tracer()
    print(f"Inspection result shape: {result.shape}")

if __name__ == "__main__":
    success1 = test_exact_problematic_tracer()
    test_manual_tracer_inspection()
    
    if success1:
        print("\n✅ All tracer tests passed!")
    else:
        print("\n❌ Some tests failed - need further investigation")
