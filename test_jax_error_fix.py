#!/usr/bin/env python3
"""
Test script to reproduce and verify the fix for the JAX tracer error.
"""

import numpy as np
from src.jaxframe import DataFrame

def test_jax_tracer_error_fix():
    """Test that the JAX tracer error is fixed."""
    print("=== Testing JAX Tracer Error Fix ===")
    
    try:
        import jax
        import jax.numpy as jnp
        
        print("Testing DataFrame printing inside JIT compilation...")
        
        @jax.jit
        def problematic_function(x):
            # This used to fail with ConcretizationTypeError
            data = {
                'input': x,
                'doubled': x * 2.0,
                'metadata': ['sample1', 'sample2', 'sample3']
            }
            
            df = DataFrame(data, name="tracer_test")
            
            # This print should not cause an error anymore
            print("DataFrame created successfully inside JIT!")
            print("DataFrame repr:")
            print(repr(df))
            
            # Test pprint method too
            print("Using pprint:")
            df.pprint()
            
            return x * 3.0
        
        # Call the function with JAX array (will be traced inside JIT)
        input_data = jnp.array([1.0, 2.0, 3.0])
        result = problematic_function(input_data)
        
        print(f"✅ SUCCESS: Function completed without ConcretizationTypeError!")
        print(f"Result: {result}")
        
    except ImportError:
        print("JAX not available - cannot test the original error scenario")
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_dtypes_with_tracers():
    """Test that dtypes method works with tracers."""
    print("\n=== Testing dtypes() method with tracers ===")
    
    try:
        import jax
        import jax.numpy as jnp
        
        @jax.jit
        def test_dtypes_function(x):
            data = {
                'traced_float': x,
                'traced_int': jnp.array([1, 2, 3], dtype=jnp.int32),
                'normal_str': ['a', 'b', 'c']
            }
            
            df = DataFrame(data)
            
            # Test dtypes method
            dtypes = df.dtypes()
            print("Dtypes inside JIT:")
            for col, dtype in dtypes.items():
                print(f"  {col}: {dtype}")
            
            return x
        
        input_data = jnp.array([1.0, 2.0, 3.0])
        test_dtypes_function(input_data)
        print("✅ dtypes() method works correctly with tracers!")
        
    except ImportError:
        print("JAX not available - cannot test dtypes with tracers")
    except Exception as e:
        print(f"❌ ERROR in dtypes test: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_jax_tracer_error_fix()
    test_dtypes_with_tracers()
    print("\n🎉 All tests completed!")
