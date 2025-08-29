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
    jit_print_masked_array_data
)


class TestJitPrint:
    """Test cases for JIT-compatible printing functions."""
    
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
