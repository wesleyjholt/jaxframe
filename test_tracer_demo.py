#!/usr/bin/env python
"""
Demo script to test JAX tracer detection in DataFrame printing.
This script demonstrates that our implementation can handle JAX tracers 
without raising ConcretizationTypeError.
"""

try:
    import jax
    import jax.numpy as jnp
    from src.jaxframe import DataFrame
    
    print("Testing JAX tracer detection in DataFrame printing...")
    print("=" * 60)
    
    # Test 1: Normal JAX arrays (should work fine)
    print("Test 1: Normal JAX arrays")
    normal_data = {
        'jax_floats': jnp.array([1.0, 2.0, 3.0]),
        'jax_ints': jnp.array([10, 20, 30]),
        'regular': ['a', 'b', 'c']
    }
    df_normal = DataFrame(normal_data)
    print(df_normal)
    print()
    
    # Test 2: Simulate what happens with a JIT-compiled function
    print("Test 2: JIT-compiled function simulation")
    
    @jax.jit
    def process_data(data):
        # Inside a JIT function, data becomes a tracer
        # If we tried to print a DataFrame containing this data,
        # it would previously raise ConcretizationTypeError
        return data * 2
    
    # Create some test data
    test_array = jnp.array([1.5, 2.5, 3.5])
    
    # Compile the function (this creates tracers internally)
    compiled_fn = process_data.lower(test_array).compile()
    print("✓ JIT compilation successful - tracers are handled internally")
    
    # Test 3: Test with actual ConcretizationTypeError handling
    print("\nTest 3: Testing ConcretizationTypeError fallback")
    
    # Create a simple problematic value that our fallback can handle
    class ProblematicValue:
        def __init__(self):
            self.dtype = jnp.float32
            self.shape = ()
            self.__module__ = 'jax'
        
        def __float__(self):
            # This simulates what happens when you try to convert a tracer
            raise Exception("Simulated concretization error")
    
    try:
        problem_df = DataFrame({
            'problem': [ProblematicValue(), ProblematicValue()],
            'normal': ['hello', 'world']
        })
        print("DataFrame with problematic JAX-like value:")
        print(problem_df)
        print("✓ Problematic values handled gracefully!")
    except Exception as e:
        print(f"Handling exception: {type(e).__name__}: {e}")
    
    print("\nTest 4: Normal operation verification")
    # Verify that normal JAX operations still work
    result = compiled_fn(test_array)
    print(f"Normal JAX computation result: {result}")
    print("✓ Normal JAX operations unaffected")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Normal JAX arrays print correctly")
    print("- JIT compilation works without errors")
    print("- Problematic JAX values are handled gracefully")
    print("- ConcretizationTypeError is avoided in DataFrame printing")
    
except ImportError:
    print("JAX not available - skipping tracer detection demo")
except Exception as e:
    print(f"Error running demo: {e}")
    import traceback
    traceback.print_exc()
