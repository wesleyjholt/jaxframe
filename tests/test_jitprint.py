"""
Tests for the jitprint module.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jaxframe import DataFrame, MaskedArray
from jaxframe.jitprint import (
    jit_print_dataframe,
    jit_print_masked_array,
    jit_print_dataframe_data,
    jit_print_masked_array_data,
    _format_value_for_jit_print
)


class TestJitPrint:
    """Test cases for JIT-compatible printing functions."""
    
    def test_format_value_for_jit_print_regular_values(self):
        """Test the value formatting helper with regular (non-tracer) values."""
        # Test float formatting
        assert _format_value_for_jit_print(3.14159) == "3.142"
        assert _format_value_for_jit_print(1.0) == "1.000"
        
        # Test string formatting
        assert _format_value_for_jit_print("hello") == "hello"
        
        # Test integer formatting
        assert _format_value_for_jit_print(42) == "42"
        
        # Test numpy scalar formatting
        np_float = np.float32(2.718)
        assert _format_value_for_jit_print(np_float) == "2.718"
        
        np_int = np.int32(123)
        assert _format_value_for_jit_print(np_int) == "123"
    
    def test_format_value_for_jit_print_with_tracers(self):
        """Test the value formatting helper with JAX tracers."""
        @jax.jit
        def test_inside_jit():
            # Create different types of tracers
            float_array = jnp.array([1.0, 2.0, 3.0])
            int_array = jnp.array([1, 2, 3])
            float_scalar = jnp.array(5.0)
            bool_array = jnp.array([True, False])
            
            # Test that tracers are formatted nicely
            float_format = _format_value_for_jit_print(float_array)
            int_format = _format_value_for_jit_print(int_array)
            scalar_format = _format_value_for_jit_print(float_scalar)
            bool_format = _format_value_for_jit_print(bool_array)
            
            # Use jax.debug.print to verify formats (they should be compact, not ugly tracers)
            jax.debug.print("Float array format: {}", float_format)
            jax.debug.print("Int array format: {}", int_format)
            jax.debug.print("Scalar format: {}", scalar_format)
            jax.debug.print("Bool array format: {}", bool_format)
            
            return float_array
        
        # This test verifies the function runs without error inside JIT
        # The actual format verification happens via visual inspection of jax.debug.print output
        result = test_inside_jit()
        assert result.shape == (3,)
    
    def test_jit_print_dataframe_static_arg(self):
        """Test jit_print_dataframe with static argument."""
        df = DataFrame({
            'x': [1, 2, 3],
            'y': [1.1, 2.2, 3.3]
        })
        
        def process(arr, dataframe):
            jit_print_dataframe(dataframe)
            return jnp.sum(arr)
        
        process_jit = jax.jit(process, static_argnames=['dataframe'])
        
        # This should not raise an error
        result = process_jit(jnp.array([1.0, 2.0, 3.0]), df)
        assert result == 6.0
    
    def test_jit_print_masked_array_static_arg(self):
        """Test jit_print_masked_array with static argument."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        index_df = DataFrame({'id': [1, 2]})
        ma = MaskedArray(data, mask, index_df)
        
        def process(arr, masked_array):
            jit_print_masked_array(masked_array)
            return jnp.sum(arr)
        
        process_jit = jax.jit(process, static_argnames=['masked_array'])
        
        # This should not raise an error
        result = process_jit(jnp.array([1.0, 2.0]), ma)
        assert result == 3.0
    
    def test_jit_print_dataframe_data(self):
        """Test jit_print_dataframe_data function."""
        @jax.jit
        def process():
            data = {
                'a': jnp.array([1, 2, 3]),
                'b': jnp.array([1.1, 2.2, 3.3])
            }
            columns = ['a', 'b']
            jit_print_dataframe_data(data, columns, 3, "test")
            return jnp.array([42.0])
        
        # This should not raise an error
        result = process()
        assert result[0] == 42.0
    
    def test_jit_print_dataframe_data_with_tracers(self):
        """Test jit_print_dataframe_data with JAX tracers (should show clean format)."""
        @jax.jit
        def process():
            # These will be tracers inside JIT
            ids = jnp.array([1, 2, 3])
            values = jnp.array([10.5, 20.7, 30.9])
            
            data = {'id': ids, 'value': values}
            columns = ['id', 'value']
            
            jax.debug.print("=== Testing DataFrame data with tracers ===")
            jit_print_dataframe_data(data, columns, 3, "tracer_test")
            
            return jnp.sum(values)
        
        # This should run without error and show clean tracer formatting
        result = process()
        assert abs(result - 62.1) < 0.001  # Sum of [10.5, 20.7, 30.9]
    
    def test_jit_print_masked_array_data(self):
        """Test jit_print_masked_array_data function."""
        @jax.jit
        def process(data, mask):
            rows, cols = data.shape
            jit_print_masked_array_data(data, mask, rows, cols)
            return jnp.sum(data)
        
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[True, False], [True, True]])
        
        # This should not raise an error
        result = process(data, mask)
        assert result == 10.0
    
    def test_single_row_dataframe_jit_print(self):
        """Test printing a single-row DataFrame in JIT."""
        single_df = DataFrame({'x': [42], 'y': [3.14]})
        
        def process(arr, dataframe):
            jit_print_dataframe(dataframe)
            return jnp.sum(arr)
        
        process_jit = jax.jit(process, static_argnames=['dataframe'])
        
        # This should not raise an error
        result = process_jit(jnp.array([1.0, 2.0]), single_df)
        assert result == 3.0
    
    def test_invalid_objects(self):
        """Test behavior with invalid objects."""
        @jax.jit
        def test_invalid_df():
            jit_print_dataframe("not_a_dataframe")
            return jnp.array([1.0])
        
        @jax.jit
        def test_invalid_ma():
            jit_print_masked_array("not_a_masked_array")
            return jnp.array([1.0])
        
        # These should not raise errors, just print warning messages
        result1 = test_invalid_df()
        result2 = test_invalid_ma()
        
        assert result1[0] == 1.0
        assert result2[0] == 1.0


class TestTracerFormatting:
    """Specific tests for JAX tracer formatting improvements."""
    
    def test_different_dtypes_formatting(self):
        """Test that different JAX dtypes are formatted correctly."""
        @jax.jit
        def test_various_dtypes():
            # Test various dtypes
            f32_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
            f64_arr = jnp.array([1.0, 2.0], dtype=jnp.float64)
            i32_arr = jnp.array([1, 2], dtype=jnp.int32)
            i64_arr = jnp.array([1, 2], dtype=jnp.int64)
            bool_arr = jnp.array([True, False], dtype=jnp.bool_)
            
            jax.debug.print("f32 format: {}", _format_value_for_jit_print(f32_arr))
            jax.debug.print("f64 format: {}", _format_value_for_jit_print(f64_arr))
            jax.debug.print("i32 format: {}", _format_value_for_jit_print(i32_arr))
            jax.debug.print("i64 format: {}", _format_value_for_jit_print(i64_arr))
            jax.debug.print("bool format: {}", _format_value_for_jit_print(bool_arr))
            
            return f32_arr
        
        # This should run without error and show clean formats
        result = test_various_dtypes()
        assert result.shape == (2,)
    
    def test_different_shapes_formatting(self):
        """Test that different shapes are formatted correctly."""
        @jax.jit
        def test_various_shapes():
            scalar = jnp.array(5.0)
            vector = jnp.array([1.0, 2.0, 3.0])
            matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            tensor_3d = jnp.array([[[1.0]]])
            
            jax.debug.print("Scalar format: {}", _format_value_for_jit_print(scalar))
            jax.debug.print("Vector format: {}", _format_value_for_jit_print(vector))
            jax.debug.print("Matrix format: {}", _format_value_for_jit_print(matrix))
            jax.debug.print("3D tensor format: {}", _format_value_for_jit_print(tensor_3d))
            
            return scalar
        
        # This should run without error and show clean formats
        result = test_various_shapes()
        assert result.shape == ()
    
    def test_comparison_with_ugly_tracers(self):
        """Test that our formatting is better than default tracer strings."""
        @jax.jit
        def compare_formats():
            x = jnp.array([1.0, 2.0, 3.0])
            
            # Show the ugly default format
            jax.debug.print("Ugly default: {}", str(x))
            
            # Show our clean format
            jax.debug.print("Clean format: {}", _format_value_for_jit_print(x))
            
            return x
        
        # This demonstrates the improvement - manual verification via output
        result = compare_formats()
        assert result.shape == (3,)
