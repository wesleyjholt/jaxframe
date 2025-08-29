#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.jitprint import jit_print_dataframe, _format_value_for_jit_print

def debug_tracer_detection():
    """Debug exactly what's happening with tracer detection."""
    
    def detailed_format_test(value):
        """Test our formatting with detailed debug output."""
        print(f"\n=== Debugging value: {type(value)} ===")
        print(f"String representation length: {len(str(value))}")
        print(f"String starts with: {str(value)[:100]}...")
        
        # Test all our detection criteria
        is_traced_type = 'Traced' in str(type(value))
        has_aval = hasattr(value, 'aval')
        has_primal = hasattr(value, 'primal')
        has_tracer_name = 'Tracer' in str(type(value))
        has_jvp = 'JVP' in str(type(value))
        has_jaxpr = 'Jaxpr' in str(type(value))
        
        print(f"Detection criteria:")
        print(f"  'Traced' in type: {is_traced_type}")
        print(f"  has aval: {has_aval}")
        print(f"  has primal: {has_primal}")
        print(f"  'Tracer' in type: {has_tracer_name}")
        print(f"  'JVP' in type: {has_jvp}")
        print(f"  'Jaxpr' in type: {has_jaxpr}")
        
        is_likely_tracer = (is_traced_type or has_aval or has_primal or 
                           has_tracer_name or has_jvp or has_jaxpr)
        print(f"  => is_likely_tracer: {is_likely_tracer}")
        
        if is_likely_tracer:
            print("Attempting aval extraction...")
            
            # Test aval extraction strategies
            if hasattr(value, 'aval'):
                print(f"  Direct aval: {value.aval}")
            elif hasattr(value, 'primal'):
                print(f"  Has primal: {type(value.primal)}")
                if hasattr(value.primal, 'aval'):
                    print(f"  Primal aval: {value.primal.aval}")
            
        formatted_result = _format_value_for_jit_print(value)
        print(f"Final formatted result: '{formatted_result}'")
        
        return formatted_result
    
    # Test with the specific scenario that might be problematic
    df = DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
    
    def problematic_grad_function(params, static_df):
        """Function that creates complex tracers through multiple transformations."""
        # Create the scenario most likely to generate problematic tracers
        reshaped = params.reshape(-1, 1)
        squeezed = jnp.squeeze(reshaped, axis=-1)  
        
        # Debug the tracer before adding to DataFrame
        print("\n=== Debugging tracer before adding to DataFrame ===")
        debug_val = detailed_format_test(squeezed)
        
        # Now test adding to DataFrame
        new_df = static_df.add_column('computed', squeezed[:len(static_df._data['x'])])
        
        print("\n=== Testing DataFrame printing ===")
        jit_print_dataframe(new_df)
        
        return jnp.sum(squeezed)
    
    # Apply grad transformation
    grad_fn = jax.grad(problematic_grad_function)
    jitted_grad = jax.jit(grad_fn, static_argnames=['static_df'])
    
    test_params = jnp.array([1.0, 2.0, 3.0])
    
    print("=== Running debug test ===")
    result = jitted_grad(test_params, df)
    print(f"Final result: {result}")

if __name__ == "__main__":
    debug_tracer_detection()
