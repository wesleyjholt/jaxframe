"""
Tests for named custom aggregation functions using tuple syntax.

This tests the ability to specify custom names for aggregation functions
using the (name, function) tuple syntax.
"""

import pytest
import numpy as np
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



class TestNamedAggregationTuple:
    """Test tuple (name, function) syntax for custom aggregations."""
    
    def test_single_named_function(self):
        """Test using a single named custom function."""
        df = DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Use tuple to provide custom name
        result = df.group_by('category').agg({
            'value': ('range', lambda x: jnp.max(x) - jnp.min(x))
        })
        
        assert 'category' in result.columns
        assert 'value' in result.columns  # Single agg, so no suffix
        assert len(result) == 2
    
    def test_named_function_with_suffix(self):
        """Test that named functions get proper column suffixes with multiple aggs."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        result = df.group_by('group').agg({
            'value': [
                ('mean', agg_mean),
                ('p90', lambda x: jnp.percentile(x, 90))
            ]
        })
        
        assert 'value_mean' in result.columns
        assert 'value_p90' in result.columns
        assert 'value_p90' in result.columns  # Check custom name is used
    
    def test_multiple_named_functions(self):
        """Test multiple named custom functions."""
        df = DataFrame({
            'category': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 4.0, 9.0, 16.0])
        })
        
        result = df.group_by('category').agg({
            'value': [
                ('p25', lambda x: jnp.percentile(x, 25)),
                ('p75', lambda x: jnp.percentile(x, 75)),
                ('iqr', lambda x: jnp.percentile(x, 75) - jnp.percentile(x, 25))
            ]
        })
        
        assert 'value_p25' in result.columns
        assert 'value_p75' in result.columns
        assert 'value_iqr' in result.columns
    
    def test_mix_named_builtin_and_unnamed(self):
        """Test mixing named custom, built-in, and unnamed custom functions."""
        df = DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        result = df.group_by('category').agg({
            'value': [
                ('mean', agg_mean),                                      # Built-in
                jnp.median,                                  # Unnamed custom
                ('p90', lambda x: jnp.percentile(x, 90))    # Named custom
            ]
        })
        
        assert 'value_mean' in result.columns
        assert 'value_median' in result.columns
        assert 'value_p90' in result.columns
    
    def test_named_function_descriptive_names(self):
        """Test using descriptive names for better readability."""
        df = DataFrame({
            'sensor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'reading': jnp.array([10, 11, 12, 100, 101, 102])
        })
        
        result = df.group_by('sensor').agg({
            'reading': [
                ('min_value', jnp.min),
                ('max_value', jnp.max),
                ('spread', lambda x: jnp.max(x) - jnp.min(x))
            ]
        })
        
        assert 'reading_min_value' in result.columns
        assert 'reading_max_value' in result.columns
        assert 'reading_spread' in result.columns


class TestNamedAggregationWithJAX:
    """Test named aggregations with JAX functions."""
    
    def test_named_jax_functions(self):
        """Test naming JAX built-in functions."""
        df = DataFrame({
            'group': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        })
        
        result = df.group_by('group').agg({
            'value': [
                ('median', jnp.median),
                ('variance', jnp.var),
                ('std_dev', jnp.std)
            ]
        })
        
        assert 'value_median' in result.columns
        assert 'value_variance' in result.columns
        assert 'value_std_dev' in result.columns
    
    def test_named_jit_function(self):
        """Test naming JIT-compiled functions."""
        df = DataFrame({
            'category': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        @jit
        def rms(x):
            return jnp.sqrt(jnp.mean(x ** 2))
        
        result = df.group_by('category').agg({
            'value': ('root_mean_square', rms)
        })
        
        assert 'value' in result.columns  # Single agg, no suffix


class TestMultipleColumns:
    """Test named aggregations on multiple columns."""
    
    def test_multiple_columns_with_named_aggs(self):
        """Test applying named aggregations to multiple columns."""
        df = DataFrame({
            'team': ['A', 'A', 'B', 'B'],
            'points': jnp.array([10, 20, 30, 40]),
            'assists': jnp.array([5, 8, 12, 15])
        })
        
        result = df.group_by('team').agg({
            'points': [
                ('avg', jnp.mean),
                ('top', jnp.max)
            ],
            'assists': [
                ('avg', jnp.mean),
                ('top', jnp.max)
            ]
        })
        
        assert 'points_avg' in result.columns
        assert 'points_top' in result.columns
        assert 'assists_avg' in result.columns
        assert 'assists_top' in result.columns


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_tuple_wrong_length(self):
        """Test error when tuple has wrong length."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        with pytest.raises(ValueError, match="Tuple aggregation must be \\(name, function\\)"):
            df.group_by('group').agg({
                'value': ('name', lambda x: jnp.sum(x), 'extra')
            })
    
    def test_tuple_first_element_not_string(self):
        """Test error when first element of tuple is not a string."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        with pytest.raises(TypeError, match="First element of tuple must be a string name"):
            df.group_by('group').agg({
                'value': (123, lambda x: jnp.sum(x))
            })
    
    def test_tuple_second_element_not_callable(self):
        """Test error when second element of tuple is not callable."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        with pytest.raises(TypeError, match="Second element of tuple must be a callable"):
            df.group_by('group').agg({
                'value': ('custom', 'not_a_function')
            })
    
    def test_empty_name(self):
        """Test that empty string names are allowed (though not recommended)."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Empty name should work but create weird column name
        result = df.group_by('group').agg({
            'value': [('', lambda x: jnp.sum(x)), ('mean', agg_mean)]
        })
        
        assert 'value_' in result.columns  # Empty name results in "value_"
        assert 'value_mean' in result.columns
    
    def test_special_characters_in_name(self):
        """Test that special characters in names are preserved."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        result = df.group_by('group').agg({
            'value': [
                ('p50', jnp.median),
                ('99th_percentile', lambda x: jnp.percentile(x, 99))
            ]
        })
        
        assert 'value_p50' in result.columns
        assert 'value_99th_percentile' in result.columns


class TestRealWorldExamples:
    """Test real-world use cases with named aggregations."""
    
    def test_financial_metrics(self):
        """Test computing multiple financial metrics with clear names."""
        df = DataFrame({
            'stock': ['AAPL', 'AAPL', 'AAPL', 'GOOG', 'GOOG', 'GOOG'],
            'price': jnp.array([150.0, 155.0, 160.0, 2800.0, 2850.0, 2900.0])
        })
        
        result = df.group_by('stock').agg({
            'price': [
                ('low', jnp.min),
                ('high', jnp.max),
                ('avg', jnp.mean),
                ('volatility', jnp.std),
                ('range', lambda x: jnp.max(x) - jnp.min(x))
            ]
        })
        
        assert 'price_low' in result.columns
        assert 'price_high' in result.columns
        assert 'price_avg' in result.columns
        assert 'price_volatility' in result.columns
        assert 'price_range' in result.columns
    
    def test_survey_analysis(self):
        """Test computing survey statistics with descriptive names."""
        df = DataFrame({
            'question': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2'],
            'response': jnp.array([1, 2, 3, 5, 2, 3, 3, 4])
        })
        
        def mode_approximation(x):
            """Simple mode approximation using median."""
            return jnp.median(x)
        
        result = df.group_by('question').agg({
            'response': [
                ('mean_score', jnp.mean),
                ('median_score', jnp.median),
                ('typical_score', mode_approximation),
                ('std_dev', jnp.std)
            ]
        })
        
        assert 'response_mean_score' in result.columns
        assert 'response_median_score' in result.columns
        assert 'response_typical_score' in result.columns
        assert 'response_std_dev' in result.columns
    
    def test_sensor_quality_metrics(self):
        """Test computing sensor quality metrics."""
        df = DataFrame({
            'sensor_id': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2'],
            'reading': jnp.array([10.0, 10.1, 9.9, 50.0, 51.0, 49.0])
        })
        
        def cv(x):
            """Coefficient of variation."""
            return jnp.std(x) / jnp.mean(x)
        
        result = df.group_by('sensor_id').agg({
            'reading': [
                ('average', jnp.mean),
                ('precision', jnp.std),
                ('consistency', cv),
                ('min_reading', jnp.min),
                ('max_reading', jnp.max)
            ]
        })
        
        assert 'reading_average' in result.columns
        assert 'reading_precision' in result.columns
        assert 'reading_consistency' in result.columns
        assert 'reading_min_reading' in result.columns
        assert 'reading_max_reading' in result.columns


class TestComplexScenarios:
    """Test complex aggregation scenarios."""
    
    def test_mix_all_types(self):
        """Test mixing built-in strings, unnamed functions, and named tuples."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([10, 15, 20, 30, 35, 40])
        })
        
        result = df.group_by('category').agg({
            'value': [
                ('sum', agg_sum),                                       # Built-in string
                ('mean', agg_mean),                                      # Built-in string
                jnp.median,                                  # Unnamed function
                jnp.std,                                     # Unnamed function
                ('p90', lambda x: jnp.percentile(x, 90)),   # Named tuple
                ('iqr', lambda x: jnp.percentile(x, 75) - jnp.percentile(x, 25))  # Named tuple
            ]
        })
        
        assert 'value_sum' in result.columns
        assert 'value_mean' in result.columns
        assert 'value_median' in result.columns
        assert 'value_std' in result.columns
        assert 'value_p90' in result.columns
        assert 'value_iqr' in result.columns
    
    def test_single_named_tuple_no_suffix(self):
        """Test that single named tuple doesn't get suffix."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        result = df.group_by('group').agg({
            'value': ('custom_metric', lambda x: jnp.sum(x ** 2))
        })
        
        # Single aggregation should not have suffix
        assert 'value' in result.columns
        assert 'value_custom_metric' not in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
