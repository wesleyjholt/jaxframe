#!/usr/bin/env python3
"""
Test script for JAX tracer-safe DataFrame printing.
"""

import numpy as np
from src.jaxframe import DataFrame

def test_regular_printing():
    """Test that regular printing still works as before."""
    print("=== Testing Regular DataFrame Printing ===")
    
    data = {
        'str_col': ['A', 'B', 'C'],
        'int_col': [1, 2, 3],
        'float_col': [1.234, 2.345, 3.456],
        'numpy_int': np.array([10, 20, 30]),
        'numpy_float': np.array([1.111, 2.222, 3.333])
    }
    
    df = DataFrame(data, name="test_data")
    print(df)
    print()

def test_jax_tracer_printing():
    """Test printing with JAX tracers."""
    print("=== Testing JAX Tracer-Safe Printing ===")
    
    try:
        import jax
        import jax.numpy as jnp
        
        @jax.jit
        def test_inside_jit(x):
            # Create DataFrame with JAX tracers
            data = {
                'traced_values': x,
                'computed': x * 2.0,
                'strings': ['a', 'b', 'c']
            }
            
            # This should not fail even though x contains tracers
            df = DataFrame(data)
            
            # Test pprint method
            print("Using pprint method:")
            df.pprint()
            
            # Test regular repr
            print("Using repr:")
            print(repr(df))
            
            return x
        
        # Test with JAX arrays (these will become tracers inside JIT)
        test_array = jnp.array([1.0, 2.0, 3.0])
        result = test_inside_jit(test_array)
        print(f"JIT function completed successfully with result: {result}")
        
    except ImportError:
        print("JAX not available - creating mock tracer test")
        
        # Create a mock tracer-like object for testing
        class MockTracer:
            def __init__(self, value, dtype, shape):
                self._value = value
                self.dtype = dtype
                self.shape = shape
                self.aval = self
            
            def __str__(self):
                return f"Traced<ShapedArray(float32[3]):jax.core.Jaxpr object>"
        
        mock_tracer = MockTracer([1.0, 2.0, 3.0], 'float32', (3,))
        
        # This should be detected as a tracer
        df_data = {
            'col1': ['a', 'b', 'c'],
            'col2': [1, 2, 3]
        }
        
        # Add the mock tracer to test detection
        df_data['traced'] = [mock_tracer, mock_tracer, mock_tracer]
        
        df = DataFrame(df_data)
        print("Mock tracer test:")
        print(df)
        
    print()

def test_tracer_detection():
    """Test the tracer detection methods."""
    print("=== Testing Tracer Detection Methods ===")
    
    data = {
        'normal_int': [1, 2, 3],
        'normal_float': [1.1, 2.2, 3.3],
        'normal_str': ['a', 'b', 'c']
    }
    
    df = DataFrame(data)
    
    print(f"Contains JAX tracers: {df._contains_jax_tracers()}")
    print(f"Test _is_jax_tracer on int: {df._is_jax_tracer(42)}")
    print(f"Test _is_jax_tracer on float: {df._is_jax_tracer(3.14)}")
    print(f"Test _is_jax_tracer on string: {df._is_jax_tracer('hello')}")
    
    # Test safe formatting
    print(f"Safe format int: {df._format_value_safe(42)}")
    print(f"Safe format float: {df._format_value_safe(3.14159)}")
    print(f"Safe format string: {df._format_value_safe('hello')}")
    
    print()

if __name__ == "__main__":
    test_regular_printing()
    test_tracer_detection()
    test_jax_tracer_printing()
    
    print("✅ All tracer-safe printing tests completed!")
