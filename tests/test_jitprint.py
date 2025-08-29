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
    
    def test_complex_jvp_tracers(self):
        """Test that complex JVP tracers from jax.grad are formatted cleanly."""
        df = DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
        
        def test_function(params, static_df):
            # This will create JVP tracers when called through jax.grad
            result = jnp.sum(params ** 2)
            
            # Add the JVP tracers to the DataFrame
            new_df = static_df.add_column('gradients', params)
            jit_print_dataframe(new_df)
            
            return result
        
        # Create a gradient function (this creates JVP tracers)
        grad_fn = jax.grad(test_function)
        grad_fn = jax.jit(grad_fn, static_argnames=['static_df'])
        
        # This should not crash and should show clean tracer formatting
        test_params = jnp.array([1.0, 2.0])
        result = grad_fn(test_params, df)
        
        # Verify the gradient is correct
        expected = 2 * test_params  # gradient of sum(x^2) is 2*x
        np.testing.assert_allclose(result, expected)
    
    def test_nested_complex_tracers(self):
        """Test even more complex nested tracers."""
        @jax.jit 
        def test_nested():
            # Create some complex nested operations that generate complex tracers
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            
            # Operations that might create complex tracer hierarchies
            z = jnp.sum(x[:, None] * y, axis=0)
            
            jax.debug.print("Complex operation result format: {}", _format_value_for_jit_print(z))
            
            return z
        
        result = test_nested()
        assert result.shape == (2,)
    
    def test_emergency_catch_all_for_complex_tracers(self):
        """Test the emergency catch-all detection for very complex tracers."""
        
        # Create a mock tracer that simulates the problematic case
        class MockProblematicTracer:
            def __init__(self, has_aval=True):
                if has_aval:
                    class MockAval:
                        def __init__(self):
                            self.dtype = 'float32'
                            self.shape = (3,)
                    self.aval = MockAval()
            
            def __str__(self):
                # Very long string that should trigger emergency catch-all
                return ("Traced<ShapedArray(float32[])>with<JVPTrace> with\n  primal = Array(0.11994141, dtype=float32)\n  "
                        "tangent = Traced<ShapedArray(float32[])>with<JaxprTrace> with\n    pval = (ShapedArray(float32[]), None)\n    "
                        "recipe = JaxprEqnRecipe(eqn_id=<object object at 0x1681ea540>, in_tracers=(Traced<ShapedArray(float32[1]):JaxprTrace>,), "
                        "out_tracer_refs=[<weakref at 0x1681f4f40; to 'JaxprTracer' at 0x1681f4550>], out_avals=[ShapedArray(float32[])], "
                        "primitive=squeeze, params={'dimensions': (0,)}, effects=frozenset())")
        
        # Test with aval
        tracer_with_aval = MockProblematicTracer(has_aval=True)
        formatted_with_aval = _format_value_for_jit_print(tracer_with_aval)
        assert formatted_with_aval == "f32[3]", f"Expected 'f32[3]', got '{formatted_with_aval}'"
        
        # Test without aval (should fall back to <tracer>)
        tracer_no_aval = MockProblematicTracer(has_aval=False)
        formatted_no_aval = _format_value_for_jit_print(tracer_no_aval)
        assert formatted_no_aval == "<tracer>", f"Expected '<tracer>', got '{formatted_no_aval}'"
        
        # Test that normal values are not affected
        normal_value = 3.14159
        formatted_normal = _format_value_for_jit_print(normal_value)
        assert formatted_normal == "3.142", f"Expected '3.142', got '{formatted_normal}'"


class TestCleanFormatting:
    """Test the clean formatting without unnecessary quotes."""
    
    def test_should_quote_value_function(self):
        """Test the _should_quote_value helper function."""
        from jaxframe.jitprint import _should_quote_value
        
        # Test numeric values - should not be quoted
        assert not _should_quote_value(42, "42")
        assert not _should_quote_value(3.14, "3.140")
        assert not _should_quote_value(0, "0")
        
        # Test boolean values - should not be quoted
        assert not _should_quote_value(True, "True")
        assert not _should_quote_value(False, "False")
        
        # Test string values - should be quoted
        assert _should_quote_value("hello", "hello")
        assert _should_quote_value("0057", "0057")  # String that looks like number
        
        # Test tracers - should not be quoted
        assert not _should_quote_value("dummy", "f32")
        assert not _should_quote_value("dummy", "i32[3]")
        assert not _should_quote_value("dummy", "<tracer>")
        
        # Test numpy scalars
        import numpy as np
        assert not _should_quote_value(np.float32(1.5), "1.500")
        assert not _should_quote_value(np.int32(10), "10")
    
    def test_dataframe_clean_output_format(self):
        """Test that DataFrame output has clean formatting."""
        df = DataFrame({
            'numeric': [1.5, 2.5],
            'integer': [10, 20],
            'boolean': [True, False],
            'string': ['ABC', 'DEF']
        })
        
        def test_clean_format(static_df):
            # This would previously show quotes around all values
            # Now should only quote strings
            jit_print_dataframe(static_df)
            return jnp.array([1.0])
        
        jitted = jax.jit(test_clean_format, static_argnames=['static_df'])
        result = jitted(static_df=df)
        
        # The test passes if no errors are thrown and formatting works
        assert result[0] == 1.0
    
    def test_dataframe_data_clean_output_format(self):
        """Test that jit_print_dataframe_data has clean formatting."""
        @jax.jit
        def test_data_clean_format():
            data = {
                'floats': jnp.array([1.1, 2.2]),
                'ints': jnp.array([5, 10]),
                'bools': jnp.array([True, False])
            }
            columns = ['floats', 'ints', 'bools']
            
            # Should show clean tracer formatting without unnecessary quotes
            jit_print_dataframe_data(data, columns, 2, "test")
            return jnp.array([42.0])
        
        result = test_data_clean_format()
        assert result[0] == 42.0
    
    def test_mixed_types_clean_formatting(self):
        """Test clean formatting with mixed data types including tracers."""
        df = DataFrame({
            'base_num': [1.0, 2.0],
            'base_str': ['X', 'Y']
        })
        
        def add_tracers(params, static_df):
            # Add JAX tracers to the DataFrame
            new_df = static_df.add_column('tracer_col', params)
            jit_print_dataframe(new_df)
            return jnp.sum(params)
        
        grad_fn = jax.grad(add_tracers)
        jitted = jax.jit(grad_fn, static_argnames=['static_df'])
        
        test_params = jnp.array([3.0, 4.0])
        result = jitted(test_params, df)
        
        # Should work without errors and show clean formatting
        np.testing.assert_allclose(result, [1.0, 1.0])
