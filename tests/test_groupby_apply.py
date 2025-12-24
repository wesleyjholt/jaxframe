"""
Tests for GroupBy.apply() method - applying functions within groups.

This tests the pattern: df.group_by('col').apply(func, 'value_col')
which applies a function to each group separately.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad

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



class TestGroupByApply:
    """Test basic group_by().apply() functionality."""
    
    def test_apply_simple_function(self):
        """Test applying simple function within groups."""
        df = DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A'],
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        })
        
        # Double values within each group
        result = df.group_by('category').apply(lambda x: x * 2, 'value')
        
        assert len(result) == len(df)
        assert 'category' in result.columns
        assert 'value' in result.columns
        
        # Values should be doubled
        assert jnp.allclose(result['value'], jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    
    def test_apply_with_output_column(self):
        """Test applying function with new output column name."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'x': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        result = df.group_by('group').apply(lambda x: x ** 2, 'x', output_column='x_squared')
        
        assert 'x' in result.columns
        assert 'x_squared' in result.columns
        assert jnp.array_equal(result['x'], df['x'])  # Original unchanged
        assert jnp.allclose(result['x_squared'], jnp.array([100.0, 400.0, 900.0, 1600.0]))
    
    def test_apply_normalize_within_groups(self):
        """Test normalizing values within each group."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
        })
        
        # Normalize to [0, 1] within each group
        def normalize(x):
            return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
        
        result = df.group_by('category').apply(normalize, 'value', output_column='normalized')
        
        # Group 1: [10, 20, 30] -> [0, 0.5, 1]
        # Group 2: [100, 200, 300] -> [0, 0.5, 1]
        expected = jnp.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        assert jnp.allclose(result['normalized'], expected)
    
    def test_apply_zscore_within_groups(self):
        """Test z-score normalization within groups."""
        df = DataFrame({
            'group': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 5.0, 15.0, 25.0])
        })
        
        def zscore(x):
            return (x - jnp.mean(x)) / jnp.std(x)
        
        result = df.group_by('group').apply(zscore, 'value', output_column='zscore')
        
        # Check that each group has mean ~0 and std ~1
        group1_mask = result['group'] == 1
        group2_mask = result['group'] == 2
        
        assert jnp.allclose(jnp.mean(result['zscore'][group1_mask]), 0.0, atol=1e-6)
        assert jnp.allclose(jnp.mean(result['zscore'][group2_mask]), 0.0, atol=1e-6)
    
    def test_apply_ranking_within_groups(self):
        """Test ranking values within each group."""
        df = DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'score': jnp.array([50, 80, 20, 100, 90, 95])
        })
        
        # Rank within groups (0-indexed)
        def rank(x):
            return jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
        
        result = df.group_by('category').apply(rank, 'score', output_column='rank')
        
        # Category A: [50, 80, 20] -> ranks [1, 2, 0]
        # Category B: [100, 90, 95] -> ranks [2, 0, 1]
        expected = jnp.array([1.0, 2.0, 0.0, 2.0, 0.0, 1.0])
        assert jnp.allclose(result['rank'], expected)


class TestMultiColumnGroupBy:
    """Test apply with multi-column grouping."""
    
    def test_apply_with_two_column_groups(self):
        """Test applying function with two grouping columns."""
        df = DataFrame({
            'year': jnp.array([2020, 2020, 2021, 2021, 2020, 2021]),
            'quarter': jnp.array([1, 1, 1, 1, 2, 2]),
            'revenue': jnp.array([100.0, 150.0, 200.0, 250.0, 120.0, 280.0])
        })
        
        # Normalize within year-quarter groups
        def normalize(x):
            return (x - jnp.mean(x)) / (jnp.std(x) + 1e-8)
        
        result = df.group_by(['year', 'quarter']).apply(
            normalize, 'revenue', output_column='normalized'
        )
        
        assert len(result) == len(df)
        assert 'normalized' in result.columns
        
        # Each year-quarter group should have mean ~0
        for year in [2020, 2021]:
            for quarter in [1, 2]:
                mask = (result['year'] == year) & (result['quarter'] == quarter)
                if jnp.any(mask):
                    group_mean = jnp.mean(result['normalized'][mask])
                    assert jnp.allclose(group_mean, 0.0, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_apply_nonexistent_column(self):
        """Test error when applying to nonexistent column."""
        df = DataFrame({
            'group': jnp.array([1, 2, 1, 2]),
            'value': jnp.array([10, 20, 30, 40])
        })
        
        with pytest.raises(KeyError, match="not found"):
            df.group_by('group').apply(lambda x: x * 2, 'nonexistent')
    
    def test_apply_wrong_length_output(self):
        """Test error when function returns wrong length."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([10, 20, 30, 40])
        })
        
        # Function that returns scalar instead of array
        def bad_func(x):
            return jnp.array([jnp.sum(x)])  # Returns length 1 instead of length of x
        
        with pytest.raises(ValueError, match="same length"):
            df.group_by('group').apply(bad_func, 'value')
    
    def test_apply_single_group(self):
        """Test applying to single group (all rows same group)."""
        df = DataFrame({
            'group': jnp.array([1, 1, 1, 1]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        def normalize(x):
            return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
        
        result = df.group_by('group').apply(normalize, 'value', output_column='normalized')
        
        expected = jnp.array([0.0, 1/3, 2/3, 1.0])
        assert jnp.allclose(result['normalized'], expected)
    
    def test_apply_each_row_unique_group(self):
        """Test when each row is its own group."""
        df = DataFrame({
            'id': jnp.array([1, 2, 3, 4]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        # Each group has one element, so function just processes single values
        result = df.group_by('id').apply(lambda x: x * 2, 'value')
        
        assert jnp.allclose(result['value'], jnp.array([20.0, 40.0, 60.0, 80.0]))


class TestJAXCompatibility:
    """Test JAX compatibility of group_by().apply()."""
    
    def test_apply_with_jax_functions(self):
        """Test using JAX built-in functions."""
        df = DataFrame({
            'category': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1.0, 4.0, 9.0, 16.0])
        })
        
        result = df.group_by('category').apply(jnp.sqrt, 'value', output_column='sqrt_value')
        
        assert jnp.allclose(result['sqrt_value'], jnp.array([1.0, 2.0, 3.0, 4.0]))
    
    def test_apply_with_jitted_function(self):
        """Test applying JIT-compiled function within groups."""
        @jit
        def scale_by_max(x):
            return x / jnp.max(x)
        
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 60.0])
        })
        
        result = df.group_by('group').apply(scale_by_max, 'value', output_column='scaled')
        
        # Group 1: [10, 20] -> [0.5, 1.0]
        # Group 2: [30, 60] -> [0.5, 1.0]
        expected = jnp.array([0.5, 1.0, 0.5, 1.0])
        assert jnp.allclose(result['scaled'], expected)
    
    def test_chain_groupby_apply_and_agg(self):
        """Test chaining apply followed by agg on same groups."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 5.0, 15.0, 25.0])
        })
        
        # First normalize within groups
        def zscore(x):
            return (x - jnp.mean(x)) / jnp.std(x)
        
        normalized = df.group_by('category').apply(zscore, 'value', output_column='zscore')
        
        # Then aggregate the normalized values
        result = normalized.group_by('category').agg({'zscore': [('mean', agg_mean), ('std', agg_std)]})
        
        # Each group should have mean ~0 and std ~1
        assert jnp.allclose(result['zscore_mean'], 0.0, atol=1e-5)
        assert jnp.allclose(result['zscore_std'], 1.0, rtol=1e-5)


class TestRealWorldExamples:
    """Test real-world use cases."""
    
    def test_percentile_rank_within_groups(self):
        """Test computing percentile rank within groups."""
        df = DataFrame({
            'department': ['Sales', 'Sales', 'Sales', 'Engineering', 'Engineering', 'Engineering'],
            'salary': jnp.array([50000, 60000, 70000, 80000, 90000, 100000])
        })
        
        def percentile_rank(x):
            """Compute percentile rank (0-100)."""
            ranks = jnp.argsort(jnp.argsort(x))
            return 100 * ranks / (len(x) - 1)
        
        result = df.group_by('department').apply(
            percentile_rank, 'salary', output_column='percentile'
        )
        
        # Each group should have percentiles [0, 50, 100]
        sales_mask = np.array([d == 'Sales' for d in result['department']])
        eng_mask = np.array([d == 'Engineering' for d in result['department']])
        
        assert jnp.allclose(result['percentile'][sales_mask], jnp.array([0.0, 50.0, 100.0]))
        assert jnp.allclose(result['percentile'][eng_mask], jnp.array([0.0, 50.0, 100.0]))
    
    def test_moving_average_within_groups(self):
        """Test computing group-specific transformations."""
        df = DataFrame({
            'stock': ['AAPL', 'AAPL', 'AAPL', 'GOOG', 'GOOG', 'GOOG'],
            'price': jnp.array([100.0, 110.0, 105.0, 2000.0, 2100.0, 2050.0])
        })
        
        # Compute deviation from group mean
        def deviation_from_mean(x):
            return x - jnp.mean(x)
        
        result = df.group_by('stock').apply(
            deviation_from_mean, 'price', output_column='deviation'
        )
        
        # AAPL mean: 105, deviations: [-5, 5, 0]
        # GOOG mean: 2050, deviations: [-50, 50, 0]
        aapl_mask = np.array([s == 'AAPL' for s in result['stock']])
        goog_mask = np.array([s == 'GOOG' for s in result['stock']])
        
        assert jnp.allclose(result['deviation'][aapl_mask], jnp.array([-5.0, 5.0, 0.0]))
        assert jnp.allclose(result['deviation'][goog_mask], jnp.array([-50.0, 50.0, 0.0]))
    
    def test_feature_scaling_by_category(self):
        """Test scaling features differently per category."""
        df = DataFrame({
            'product_type': ['Electronics', 'Electronics', 'Clothing', 'Clothing'],
            'price': jnp.array([500.0, 1000.0, 20.0, 40.0]),
            'quantity': jnp.array([10, 5, 100, 50])
        })
        
        # Log transform prices within each product type
        def log_transform(x):
            return jnp.log(x + 1)
        
        result = df.group_by('product_type').apply(
            log_transform, 'price', output_column='log_price'
        )
        
        assert 'log_price' in result.columns
        assert jnp.all(result['log_price'] > 0)
        
        # Original prices should be unchanged
        assert jnp.array_equal(result['price'], df['price'])
    
    def test_winsorize_by_group(self):
        """Test winsorizing (capping outliers) within groups."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 1, 2, 2, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 1000.0, 5.0, 15.0, 25.0, 500.0])
        })
        
        def winsorize(x, lower_pct=0.1, upper_pct=0.9):
            """Cap values at percentiles within group."""
            sorted_x = jnp.sort(x)
            n = len(x)
            # Use proper indexing for percentiles
            lower_idx = max(0, int(jnp.floor(n * lower_pct)))
            upper_idx = min(n - 1, int(jnp.ceil(n * upper_pct)) - 1)
            
            lower_bound = sorted_x[lower_idx]
            upper_bound = sorted_x[upper_idx]
            
            return jnp.clip(x, lower_bound, upper_bound)
        
        result = df.group_by('category').apply(
            lambda x: winsorize(x, 0.0, 0.70),  # Use 70th percentile
            'value',
            output_column='winsorized'
        )
        
        # Outliers (1000 and 500) should be capped at 70th percentile
        # Category 1: sorted [10, 20, 30, 1000], 70th percentile ~index 2 -> 30.0
        # Category 2: sorted [5, 15, 25, 500], 70th percentile ~index 2 -> 25.0
        assert result['winsorized'][3] <= 30.0  # Was 1000, now capped at 30
        assert result['winsorized'][7] <= 25.0  # Was 500, now capped at 25


class TestPerformance:
    """Test performance characteristics."""
    
    def test_apply_preserves_order(self):
        """Test that row order is preserved."""
        df = DataFrame({
            'group': jnp.array([2, 1, 2, 1, 2]),
            'value': jnp.array([10, 20, 30, 40, 50]),
            'id': jnp.array([0, 1, 2, 3, 4])
        })
        
        result = df.group_by('group').apply(lambda x: x * 2, 'value')
        
        # Order should be preserved
        assert jnp.array_equal(result['id'], df['id'])
        assert jnp.array_equal(result['group'], df['group'])
    
    def test_apply_multiple_times(self):
        """Test applying multiple transformations sequentially."""
        df = DataFrame({
            'category': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0])
        })
        
        # Apply multiple transformations
        result = (df
                  .group_by('category')
                  .apply(lambda x: x / jnp.max(x), 'value', output_column='normalized')
                  .group_by('category')
                  .apply(lambda x: x * 100, 'normalized', output_column='scaled'))
        
        assert 'normalized' in result.columns
        assert 'scaled' in result.columns
        
        # Scaled should be normalized * 100
        assert jnp.allclose(result['scaled'], result['normalized'] * 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
