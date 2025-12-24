"""
Tests for custom aggregation functions in GroupBy.agg()

This tests the ability to use custom JAX-compatible functions
alongside built-in aggregations.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from jaxframe import DataFrame


# Helper aggregation functions (no string shortcuts)
import jax.numpy as jnp

def agg_sum(x):
    """Sum aggregation."""
    return jnp.sum(x)

def agg_mean(x):
    """Mean aggregation."""
    return jnp.mean(x)

def agg_std(x):
    """Standard deviation aggregation."""
    return jnp.std(x)

def agg_min(x):
    """Minimum aggregation."""
    return jnp.min(x)

def agg_max(x):
    """Maximum aggregation."""
    return jnp.max(x)

def agg_count(x):
    """Count aggregation."""
    return jnp.array(len(x), dtype=x.dtype)



class TestCustomAggregation:
    """Test custom aggregation functions."""
    
    def test_simple_custom_function(self):
        """Test using a simple custom aggregation function."""
        df = DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Custom function: compute range (max - min)
        def range_func(x):
            return jnp.max(x) - jnp.min(x)
        
        result = df.group_by('category').agg({'value': range_func})
        
        assert 'category' in result.columns
        assert 'value' in result.columns
        assert len(result) == 2
        
        # Category A: max(1, 2) - min(1, 2) = 1
        # Category B: max(3, 4) - min(3, 4) = 1
        assert jnp.allclose(result['value'], jnp.array([1.0, 1.0]))
    
    def test_custom_function_with_name(self):
        """Test that custom function name is used in output column."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        def my_custom_agg(x):
            return jnp.sum(x) / jnp.max(x)
        
        result = df.group_by('group').agg({'value': [('mean', agg_mean), my_custom_agg]})
        
        assert 'value_mean' in result.columns
        assert 'value_my_custom_agg' in result.columns
    
    def test_lambda_function(self):
        """Test using lambda functions."""
        df = DataFrame({
            'category': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 4.0, 9.0, 16.0])
        })
        
        # Geometric mean approximation
        result = df.group_by('category').agg({
            'value': lambda x: jnp.exp(jnp.mean(jnp.log(x)))
        })
        
        assert len(result) == 2
        # Group 1: exp(mean(log(1), log(4))) = exp(mean(0, 1.386)) = exp(0.693) ≈ 2.0
        # Group 2: exp(mean(log(9), log(16))) ≈ 12.0
        assert jnp.allclose(result['value'][0], 2.0, rtol=1e-5)
        assert jnp.allclose(result['value'][1], 12.0, rtol=1e-5)
    
    def test_mix_builtin_and_custom(self):
        """Test mixing built-in and custom aggregations."""
        df = DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A'],
            'value': jnp.array([10.0, 20.0, 30.0, 40.0, 15.0])
        })
        
        def median_func(x):
            return jnp.median(x)
        
        result = df.group_by('category').agg({
            'value': [('mean', agg_mean), ('std', agg_std), median_func]
        })
        
        assert 'value_mean' in result.columns
        assert 'value_std' in result.columns
        assert 'value_median_func' in result.columns
        
        # Category A: [10, 20, 15] -> median = 15
        # Category B: [30, 40] -> median = 35
        a_idx = 0 if result['category'][0] == 'A' else 1
        b_idx = 1 - a_idx
        
        assert jnp.allclose(result['value_median_func'][a_idx], 15.0)
        assert jnp.allclose(result['value_median_func'][b_idx], 35.0)
    
    def test_multiple_columns_custom_agg(self):
        """Test custom aggregations on multiple columns."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'x': jnp.array([1.0, 2.0, 3.0, 4.0]),
            'y': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        def product_sum(x):
            """Sum of products (not a real metric, just for testing)."""
            return jnp.sum(x * x)
        
        result = df.group_by('group').agg({
            'x': product_sum,
            'y': product_sum
        })
        
        assert 'x' in result.columns
        assert 'y' in result.columns
        
        # Group 1 x: 1^2 + 2^2 = 5
        # Group 1 y: 10^2 + 20^2 = 500
        assert jnp.allclose(result['x'][0], 5.0)
        assert jnp.allclose(result['y'][0], 500.0)


class TestJAXFunctions:
    """Test using JAX built-in functions as custom aggregations."""
    
    def test_jax_median(self):
        """Test using jnp.median."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        })
        
        result = df.group_by('category').agg({'value': jnp.median})
        
        assert jnp.allclose(result['value'], jnp.array([2.0, 20.0]))
    
    def test_jax_percentile(self):
        """Test using percentile function."""
        df = DataFrame({
            'group': jnp.array([1, 1, 1, 2, 2, 2]),
            'score': jnp.array([10, 20, 30, 40, 50, 60])
        })
        
        # 75th percentile
        def p75(x):
            return jnp.percentile(x, 75)
        
        result = df.group_by('group').agg({'score': p75})
        
        # Group 1: 75th percentile of [10, 20, 30] = 25
        # Group 2: 75th percentile of [40, 50, 60] = 55
        assert jnp.allclose(result['score'][0], 25.0)
        assert jnp.allclose(result['score'][1], 55.0)
    
    def test_jax_var(self):
        """Test using jnp.var for variance."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
        })
        
        result = df.group_by('category').agg({'value': jnp.var})
        
        # Group 1: var([1, 2, 3]) = 2/3 ≈ 0.667
        # Group 2: var([10, 10, 10]) = 0
        assert jnp.allclose(result['value'][0], 2/3, rtol=1e-5)
        assert jnp.allclose(result['value'][1], 0.0)
    
    def test_multiple_percentiles(self):
        """Test computing multiple percentiles."""
        df = DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': jnp.array([1, 2, 3, 4, 5, 6])
        })
        
        def p25(x):
            return jnp.percentile(x, 25)
        
        def p75(x):
            return jnp.percentile(x, 75)
        
        result = df.group_by('category').agg({'value': [p25, jnp.median, p75]})
        
        assert 'value_p25' in result.columns
        assert 'value_median' in result.columns
        assert 'value_p75' in result.columns


class TestJITCompatibility:
    """Test JIT compilation with custom aggregations."""
    
    def test_jitted_custom_function(self):
        """Test using JIT-compiled custom function."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        @jit
        def harmonic_mean(x):
            """Harmonic mean: n / sum(1/x)."""
            return len(x) / jnp.sum(1.0 / x)
        
        result = df.group_by('group').agg({'value': harmonic_mean})
        
        # Group 1: 2 / (1/1 + 1/2) = 2 / 1.5 = 1.333...
        # Group 2: 2 / (1/3 + 1/4) = 2 / 0.583... = 3.428...
        assert jnp.allclose(result['value'][0], 4/3, rtol=1e-5)
        assert jnp.allclose(result['value'][1], 24/7, rtol=1e-5)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_custom_function_wrong_return_shape(self):
        """Test error when custom function returns non-scalar."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        def bad_func(x):
            return x  # Returns array, not scalar
        
        with pytest.raises(ValueError, match="must return a scalar"):
            df.group_by('group').agg({'value': bad_func})
    
    def test_custom_function_error(self):
        """Test error handling in custom function."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        def error_func(x):
            raise ValueError("Intentional error")
        
        with pytest.raises(ValueError, match="Error applying custom aggregation"):
            df.group_by('group').agg({'value': error_func})
    
    def test_invalid_aggregation_type(self):
        """Test error with invalid aggregation type."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        with pytest.raises(TypeError, match="Aggregation functions must be callables, tuples"):
            df.group_by('group').agg({'value': 123})  # Not a callable or tuple
    
    def test_single_value_groups(self):
        """Test custom aggregation with groups of size 1."""
        df = DataFrame({
            'id': jnp.array([1, 2, 3, 4]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        result = df.group_by('id').agg({'value': jnp.median})
        
        # Each group has one value, so median = that value
        assert jnp.allclose(result['value'], jnp.array([10.0, 20.0, 30.0, 40.0]))


class TestRealWorldExamples:
    """Test real-world use cases."""
    
    def test_coefficient_of_variation(self):
        """Test computing coefficient of variation (std/mean)."""
        df = DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': jnp.array([10.0, 12.0, 14.0, 100.0, 120.0, 140.0])
        })
        
        def cv(x):
            """Coefficient of variation."""
            return jnp.std(x) / jnp.mean(x)
        
        result = df.group_by('category').agg({'value': cv})
        
        # Both groups have similar relative variation
        assert jnp.allclose(result['value'][0], result['value'][1], rtol=1e-2)
    
    def test_interquartile_range(self):
        """Test computing IQR."""
        df = DataFrame({
            'group': jnp.array([1, 1, 1, 1, 2, 2, 2, 2]),
            'value': jnp.array([1, 2, 3, 10, 20, 21, 22, 30])
        })
        
        def iqr(x):
            """Interquartile range: Q3 - Q1."""
            return jnp.percentile(x, 75) - jnp.percentile(x, 25)
        
        result = df.group_by('group').agg({'value': iqr})
        
        assert len(result) == 2
        # Check that IQR is computed
        assert jnp.all(result['value'] > 0)
    
    def test_custom_weighted_mean(self):
        """Test weighted mean aggregation."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 5.0, 15.0, 25.0]),
            'weight': jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 1.0])
        })
        
        # Note: This is a bit tricky since we can only access one column at a time
        # For now, test just the sum (weights would need to be handled differently)
        def weighted_sum(x):
            # In real use, you'd access weights separately
            # This is just a sum for demonstration
            return jnp.sum(x)
        
        result = df.group_by('category').agg({'value': weighted_sum})
        
        assert jnp.allclose(result['value'][0], 60.0)  # 10 + 20 + 30
        assert jnp.allclose(result['value'][1], 45.0)  # 5 + 15 + 25
    
    def test_mode_approximation(self):
        """Test mode-like aggregation (most common value)."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 1, 2, 2, 2, 2]),
            'value': jnp.array([5, 5, 5, 10, 20, 20, 25, 25])
        })
        
        # Simple mode approximation: median of most frequent values
        # For this test, just use median as proxy
        result = df.group_by('category').agg({'value': jnp.median})
        
        # Group 1: median([5, 5, 5, 10]) = 5
        # Group 2: median([20, 20, 25, 25]) = 22.5
        assert jnp.allclose(result['value'][0], 5.0)
        assert jnp.allclose(result['value'][1], 22.5)
    
    def test_range_normalization_factor(self):
        """Test computing range as normalization factor."""
        df = DataFrame({
            'sensor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'reading': jnp.array([10.0, 15.0, 20.0, 100.0, 150.0, 200.0])
        })
        
        def range_span(x):
            return jnp.max(x) - jnp.min(x)
        
        result = df.group_by('sensor').agg({
            'reading': [('min', agg_min), ('max', agg_max), range_span]
        })
        
        assert 'reading_min' in result.columns
        assert 'reading_max' in result.columns
        assert 'reading_range_span' in result.columns
        
        # Sensor A: range = 20 - 10 = 10
        # Sensor B: range = 200 - 100 = 100
        a_idx = 0 if result['sensor'][0] == 'A' else 1
        assert jnp.allclose(result['reading_range_span'][a_idx], 10.0)


class TestComplexScenarios:
    """Test complex aggregation scenarios."""
    
    def test_multi_column_grouping_with_custom(self):
        """Test custom aggregations with multi-column grouping."""
        df = DataFrame({
            'year': jnp.array([2020, 2020, 2021, 2021]),
            'quarter': jnp.array([1, 2, 1, 2]),
            'revenue': jnp.array([100.0, 150.0, 200.0, 250.0])
        })
        
        result = df.group_by(['year', 'quarter']).agg({
            'revenue': lambda x: jnp.sum(x)
        })
        
        assert len(result) == 4
        assert 'year' in result.columns
        assert 'quarter' in result.columns
    
    def test_chain_aggregations(self):
        """Test computing multiple custom aggregations."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        })
        
        def geometric_mean(x):
            return jnp.exp(jnp.mean(jnp.log(x)))
        
        def harmonic_mean(x):
            return len(x) / jnp.sum(1.0 / x)
        
        result = df.group_by('category').agg({
            'value': [('mean', agg_mean), geometric_mean, harmonic_mean]
        })
        
        assert 'value_mean' in result.columns
        assert 'value_geometric_mean' in result.columns
        assert 'value_harmonic_mean' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
