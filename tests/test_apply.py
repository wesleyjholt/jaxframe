"""
Tests for DataFrame.apply() method with JAX-compatible functions.

Tests cover:
- Single column transformations
- Multiple column transformations
- JAX compatibility (JIT, vmap, grad)
- Edge cases and error handling
- Different data types
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

from jaxframe import DataFrame


class TestBasicApply:
    """Test basic apply functionality."""
    
    def test_apply_single_column_simple(self):
        """Test applying a simple function to a single column."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3, 4, 5]),
            'y': jnp.array([10, 20, 30, 40, 50])
        })
        
        # Square the values
        result = df.apply(lambda x: x ** 2, 'x')
        
        assert 'x' in result.columns
        assert jnp.array_equal(result['x'], jnp.array([1, 4, 9, 16, 25]))
        assert jnp.array_equal(result['y'], df['y'])  # y unchanged
    
    def test_apply_single_column_with_output_name(self):
        """Test applying function with custom output column name."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3, 4, 5])
        })
        
        result = df.apply(lambda x: x ** 2, 'x', output_column='x_squared')
        
        assert 'x' in result.columns
        assert 'x_squared' in result.columns
        assert jnp.array_equal(result['x'], df['x'])  # Original unchanged
        assert jnp.array_equal(result['x_squared'], jnp.array([1, 4, 9, 16, 25]))
    
    def test_apply_multiple_columns(self):
        """Test applying function to multiple columns."""
        df = DataFrame({
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([10, 20, 30]),
            'c': jnp.array([100, 200, 300])
        })
        
        # Add two columns
        result = df.apply(lambda x, y: x + y, ['a', 'b'], output_column='sum')
        
        assert 'sum' in result.columns
        assert jnp.array_equal(result['sum'], jnp.array([11, 22, 33]))
        assert jnp.array_equal(result['a'], df['a'])  # Originals unchanged
        assert jnp.array_equal(result['b'], df['b'])
    
    def test_apply_multiple_columns_complex(self):
        """Test applying complex function to multiple columns."""
        df = DataFrame({
            'x': jnp.array([1.0, 2.0, 3.0]),
            'y': jnp.array([4.0, 5.0, 6.0]),
            'z': jnp.array([7.0, 8.0, 9.0])
        })
        
        # Compute weighted sum
        result = df.apply(
            lambda x, y, z: 0.5 * x + 0.3 * y + 0.2 * z,
            ['x', 'y', 'z'],
            output_column='weighted_sum'
        )
        
        expected = 0.5 * df['x'] + 0.3 * df['y'] + 0.2 * df['z']
        assert jnp.allclose(result['weighted_sum'], expected)


class TestJAXFunctions:
    """Test with JAX-specific functions."""
    
    def test_apply_jax_log(self):
        """Test applying jnp.log."""
        df = DataFrame({
            'values': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        result = df.apply(jnp.log, 'values', output_column='log_values')
        
        assert jnp.allclose(result['log_values'], jnp.log(df['values']))
    
    def test_apply_jax_exp(self):
        """Test applying jnp.exp."""
        df = DataFrame({
            'x': jnp.array([0.0, 1.0, 2.0])
        })
        
        result = df.apply(jnp.exp, 'x')
        
        assert jnp.allclose(result['x'], jnp.array([1.0, jnp.e, jnp.e**2]))
    
    def test_apply_trigonometric(self):
        """Test applying trigonometric functions."""
        df = DataFrame({
            'angle': jnp.array([0.0, jnp.pi/2, jnp.pi])
        })
        
        result = df.apply(jnp.sin, 'angle', output_column='sin_angle')
        
        assert jnp.allclose(result['sin_angle'], jnp.array([0.0, 1.0, 0.0]), atol=1e-7)
    
    def test_apply_custom_jax_function(self):
        """Test applying custom JAX-compatible function."""
        def normalize(x):
            """Normalize to [0, 1] range."""
            return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
        
        df = DataFrame({
            'values': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        result = df.apply(normalize, 'values', output_column='normalized')
        
        expected = jnp.array([0.0, 1/3, 2/3, 1.0])
        assert jnp.allclose(result['normalized'], expected)


class TestJAXCompatibility:
    """Test JAX compatibility (JIT, vmap, grad)."""
    
    def test_apply_result_jittable(self):
        """Test that result can be used in JIT-compiled functions."""
        df = DataFrame({
            'x': jnp.array([1.0, 2.0, 3.0])
        })
        
        result = df.apply(lambda x: x ** 2, 'x', output_column='x_squared')
        
        @jit
        def compute_sum(arr):
            return jnp.sum(arr)
        
        total = compute_sum(result['x_squared'])
        assert jnp.isclose(total, 14.0)  # 1 + 4 + 9
    
    def test_apply_with_jitted_function(self):
        """Test applying a JIT-compiled function."""
        @jit
        def square_and_add(x):
            return x ** 2 + 1
        
        df = DataFrame({
            'values': jnp.array([1, 2, 3, 4])
        })
        
        result = df.apply(square_and_add, 'values')
        
        assert jnp.array_equal(result['values'], jnp.array([2, 5, 10, 17]))
    
    def test_apply_differentiable(self):
        """Test that applied functions are differentiable."""
        df = DataFrame({
            'x': jnp.array([1.0, 2.0, 3.0])
        })
        
        # Apply a differentiable function
        result = df.apply(lambda x: x ** 2, 'x', output_column='x_squared')
        
        # Define a loss function using the result
        def loss(values):
            squared = values ** 2
            return jnp.sum(squared)
        
        # Compute gradient
        grad_fn = grad(loss)
        gradients = grad_fn(df['x'])
        
        # Gradient of sum(x^2) is 2x
        expected_grad = 2 * df['x']
        assert jnp.allclose(gradients, expected_grad)
    
    def test_apply_with_vmap(self):
        """Test applying function that works with vmap."""
        df = DataFrame({
            'x': jnp.array([1.0, 2.0, 3.0])
        })
        
        def process(x):
            return x ** 2 + 2 * x + 1
        
        result = df.apply(process, 'x', output_column='processed')
        
        # Verify result can be vmapped
        def batch_process(batch):
            return vmap(process)(batch)
        
        batch_input = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        batch_output = batch_process(batch_input)
        
        assert batch_output.shape == (2, 3)
    
    def test_chain_multiple_applies(self):
        """Test chaining multiple apply operations."""
        df = DataFrame({
            'x': jnp.array([1.0, 2.0, 3.0])
        })
        
        result = (df
                  .apply(lambda x: x ** 2, 'x', output_column='x_squared')
                  .apply(lambda x: jnp.sqrt(x), 'x_squared', output_column='x_sqrt_squared')
                  .apply(lambda x: x * 2, 'x', output_column='x_doubled'))
        
        assert jnp.allclose(result['x_squared'], jnp.array([1.0, 4.0, 9.0]))
        assert jnp.allclose(result['x_sqrt_squared'], jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result['x_doubled'], jnp.array([2.0, 4.0, 6.0]))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_apply_nonexistent_column(self):
        """Test error when applying to nonexistent column."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3])
        })
        
        with pytest.raises(KeyError, match="not found"):
            df.apply(lambda x: x ** 2, 'nonexistent')
    
    def test_apply_multiple_columns_no_output_name(self):
        """Test error when applying to multiple columns without output name."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3]),
            'y': jnp.array([4, 5, 6])
        })
        
        with pytest.raises(ValueError, match="output_column must be specified"):
            df.apply(lambda x, y: x + y, ['x', 'y'])
    
    def test_apply_wrong_length_output(self):
        """Test error when function returns wrong length."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3, 4])
        })
        
        # Function that returns wrong length
        def wrong_length(x):
            return jnp.array([1, 2])  # Only 2 elements instead of 4
        
        with pytest.raises(ValueError, match="output length"):
            df.apply(wrong_length, 'x')
    
    def test_apply_to_list_column(self):
        """Test applying to column stored as list."""
        df = DataFrame({
            'x': [1, 2, 3, 4],
            'y': jnp.array([10, 20, 30, 40])
        })
        
        # Should convert list to JAX array internally
        result = df.apply(lambda x: x ** 2, 'x')
        
        assert jnp.array_equal(result['x'], jnp.array([1, 4, 9, 16]))
    
    def test_apply_to_numpy_column(self):
        """Test applying to numpy array column."""
        df = DataFrame({
            'x': np.array([1, 2, 3, 4])
        })
        
        # Should convert numpy to JAX array internally
        result = df.apply(lambda x: x ** 2, 'x')
        
        assert jnp.array_equal(result['x'], jnp.array([1, 4, 9, 16]))
    
    def test_apply_scalar_output(self):
        """Test that scalar output raises appropriate error."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3])
        })
        
        # This should fail because output is scalar, not array
        with pytest.raises(ValueError, match="scalar value"):
            df.apply(lambda x: jnp.sum(x), 'x')


class TestDataTypes:
    """Test with different data types."""
    
    def test_apply_int_arrays(self):
        """Test with integer arrays."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        })
        
        result = df.apply(lambda x: x * 2, 'x')
        
        assert jnp.array_equal(result['x'], jnp.array([2, 4, 6, 8]))
    
    def test_apply_float_arrays(self):
        """Test with float arrays."""
        df = DataFrame({
            'x': jnp.array([1.5, 2.5, 3.5])
        })
        
        result = df.apply(lambda x: x ** 2, 'x')
        
        assert jnp.allclose(result['x'], jnp.array([2.25, 6.25, 12.25]))
    
    def test_apply_mixed_types(self):
        """Test with mixed column types."""
        df = DataFrame({
            'int_col': jnp.array([1, 2, 3]),
            'float_col': jnp.array([1.0, 2.0, 3.0]),
            'str_col': ['a', 'b', 'c']
        })
        
        # Apply to int column
        result1 = df.apply(lambda x: x * 2, 'int_col')
        assert jnp.array_equal(result1['int_col'], jnp.array([2, 4, 6]))
        
        # Apply to float column
        result2 = df.apply(lambda x: x ** 2, 'float_col')
        assert jnp.allclose(result2['float_col'], jnp.array([1.0, 4.0, 9.0]))
    
    def test_apply_type_conversion(self):
        """Test that result type can differ from input."""
        df = DataFrame({
            'x': jnp.array([1, 2, 3])
        })
        
        # Convert int to float
        result = df.apply(lambda x: x / 2.0, 'x')
        
        assert result['x'].dtype == jnp.float32 or result['x'].dtype == jnp.float64


class TestRealWorldExamples:
    """Test real-world use cases."""
    
    def test_normalize_column(self):
        """Test normalizing a column to [0, 1] range."""
        df = DataFrame({
            'prices': jnp.array([100.0, 200.0, 300.0, 400.0])
        })
        
        def normalize(x):
            return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
        
        result = df.apply(normalize, 'prices', output_column='normalized_prices')
        
        assert jnp.allclose(result['normalized_prices'], jnp.array([0.0, 1/3, 2/3, 1.0]))
    
    def test_compute_distance(self):
        """Test computing Euclidean distance."""
        df = DataFrame({
            'x': jnp.array([0.0, 3.0, 0.0]),
            'y': jnp.array([0.0, 0.0, 4.0])
        })
        
        def euclidean_distance(x, y):
            return jnp.sqrt(x**2 + y**2)
        
        result = df.apply(euclidean_distance, ['x', 'y'], output_column='distance')
        
        assert jnp.allclose(result['distance'], jnp.array([0.0, 3.0, 4.0]))
    
    def test_sigmoid_activation(self):
        """Test applying sigmoid activation function."""
        df = DataFrame({
            'logits': jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        })
        
        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))
        
        result = df.apply(sigmoid, 'logits', output_column='probabilities')
        
        # Check sigmoid properties
        assert jnp.all((result['probabilities'] >= 0) & (result['probabilities'] <= 1))
        assert jnp.isclose(result['probabilities'][2], 0.5)  # sigmoid(0) = 0.5
    
    def test_feature_engineering(self):
        """Test creating engineered features."""
        df = DataFrame({
            'age': jnp.array([25, 35, 45, 55]),
            'income': jnp.array([50000, 75000, 100000, 125000])
        })
        
        # Create age squared feature
        result = df.apply(lambda x: x ** 2, 'age', output_column='age_squared')
        
        # Create log income feature
        result = result.apply(jnp.log, 'income', output_column='log_income')
        
        # Create interaction feature
        result = result.apply(
            lambda age, income: age * income / 1000,
            ['age', 'income'],
            output_column='age_income_interaction'
        )
        
        assert 'age_squared' in result.columns
        assert 'log_income' in result.columns
        assert 'age_income_interaction' in result.columns
        assert jnp.array_equal(result['age_squared'], jnp.array([625, 1225, 2025, 3025]))


class TestPerformance:
    """Test performance characteristics."""
    
    def test_apply_preserves_jax_arrays(self):
        """Test that apply preserves JAX array type."""
        df = DataFrame({
            'x': jnp.array([1.0, 2.0, 3.0])
        })
        
        result = df.apply(lambda x: x ** 2, 'x')
        
        # Result should be JAX array
        assert hasattr(result['x'], '__module__') and 'jax' in str(result['x'].__module__)
    
    def test_apply_multiple_times(self):
        """Test applying function multiple times."""
        df = DataFrame({
            'x': jnp.array([1.0, 2.0, 3.0])
        })
        
        # Apply 5 times
        result = df
        for _ in range(5):
            result = result.apply(lambda x: x * 2, 'x')
        
        # 2^5 = 32
        assert jnp.allclose(result['x'], df['x'] * 32)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
