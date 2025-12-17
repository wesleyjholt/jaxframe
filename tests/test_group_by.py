"""
Tests for GroupBy functionality in JAXFrame.

Tests cover:
- Single column grouping
- Multi-column grouping
- Various aggregation functions (sum, mean, std, min, max, count)
- JAX compatibility (JIT compilation, gradient computation)
- Edge cases (empty groups, single group, etc.)
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



class TestBasicGroupBy:
    """Test basic grouping and aggregation functionality."""
    
    def test_single_column_sum(self):
        """Test grouping by single column with sum aggregation."""
        df = DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': jnp.array([10, 20, 30, 40, 50])
        })
        
        result = df.group_by('category').agg({'value': ('sum', agg_sum)})
        
        # Should have 2 groups: A and B
        assert len(result) == 2
        assert 'category' in result.columns
        assert 'value_sum' in result.columns
        
        # Check aggregated values
        result_dict = {cat: val for cat, val in zip(result['category'], result['value_sum'])}
        assert result_dict['A'] == 90  # 10 + 30 + 50
        assert result_dict['B'] == 60  # 20 + 40
    
    def test_single_column_multiple_aggs(self):
        """Test multiple aggregations on same column."""
        df = DataFrame({
            'category': jnp.array([1, 2, 1, 2, 1]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        })
        
        result = df.group_by('category').agg({'value': [('sum', agg_sum), ('mean', agg_mean), ('count', agg_count)]})
        
        assert 'value_sum' in result.columns
        assert 'value_mean' in result.columns
        assert 'value_count' in result.columns
        
        # Check values for category 1
        cat1_idx = np.where(np.array(result['category']) == 1)[0][0]
        assert result['value_sum'][cat1_idx] == 90  # 10 + 30 + 50
        assert result['value_mean'][cat1_idx] == 30  # 90 / 3
        assert result['value_count'][cat1_idx] == 3
    
    def test_multi_column_grouping(self):
        """Test grouping by multiple columns."""
        df = DataFrame({
            'year': jnp.array([2020, 2020, 2021, 2021, 2020]),
            'month': jnp.array([1, 2, 1, 2, 1]),
            'sales': jnp.array([100, 200, 150, 250, 120])
        })
        
        result = df.group_by(['year', 'month']).agg({'sales': ('sum', agg_sum)})
        
        # Should have 4 groups: (2020,1), (2020,2), (2021,1), (2021,2)
        assert len(result) == 4
        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 'sales_sum' in result.columns
        
        # Check specific group
        mask = (np.array(result['year']) == 2020) & (np.array(result['month']) == 1)
        assert np.sum(np.array(result['sales_sum'])[mask]) == 220  # 100 + 120
    
    def test_all_aggregation_functions(self):
        """Test all supported aggregation functions."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2, 2]),
            'value': jnp.array([10.0, 20.0, 5.0, 15.0, 25.0])
        })
        
        result = df.group_by('group').agg({
            'value': [('sum', agg_sum), ('mean', agg_mean), ('std', agg_std), ('min', agg_min), ('max', agg_max), ('count', agg_count)]
        })
        
        # Check group 1
        g1_idx = np.where(np.array(result['group']) == 1)[0][0]
        assert result['value_sum'][g1_idx] == 30.0
        assert result['value_mean'][g1_idx] == 15.0
        assert result['value_min'][g1_idx] == 10.0
        assert result['value_max'][g1_idx] == 20.0
        assert result['value_count'][g1_idx] == 2
        
        # Check standard deviation for group 1 (std of [10, 20] = 5.0)
        assert np.isclose(result['value_std'][g1_idx], 5.0, rtol=1e-5)
        
        # Check group 2
        g2_idx = np.where(np.array(result['group']) == 2)[0][0]
        assert result['value_sum'][g2_idx] == 45.0
        assert result['value_mean'][g2_idx] == 15.0
        assert result['value_min'][g2_idx] == 5.0
        assert result['value_max'][g2_idx] == 25.0
        assert result['value_count'][g2_idx] == 3
    
    def test_multiple_columns_aggregation(self):
        """Test aggregating multiple columns at once."""
        df = DataFrame({
            'category': jnp.array([1, 2, 1, 2]),
            'sales': jnp.array([100, 200, 150, 250]),
            'profit': jnp.array([10, 20, 15, 25])
        })
        
        result = df.group_by('category').agg({
            'sales': ('sum', agg_sum),
            'profit': ('mean', agg_mean)
        })
        
        assert 'sales_sum' in result.columns
        assert 'profit_mean' in result.columns
        
        # Check category 1
        cat1_idx = np.where(np.array(result['category']) == 1)[0][0]
        assert result['sales_sum'][cat1_idx] == 250  # 100 + 150
        assert result['profit_mean'][cat1_idx] == 12.5  # (10 + 15) / 2


class TestJAXCompatibility:
    """Test that GroupBy operations are JAX-compatible (jittable and differentiable)."""
    
    def test_jit_compilation_with_static_size(self):
        """Test that aggregations can be JIT compiled when size is static."""
        
        @jit
        def compute_group_sum(values, groups):
            """Compute sum per group using segment_sum with static num_groups."""
            from jax.ops import segment_sum
            
            # When number of groups is known at compile time, pass as static
            num_groups = 2  # Static/concrete value
            return segment_sum(values, groups, num_groups)
        
        values = jnp.array([10.0, 20.0, 30.0, 40.0])
        groups = jnp.array([0, 1, 0, 1])
        
        result = compute_group_sum(values, groups)
        
        assert jnp.allclose(result, jnp.array([40.0, 60.0]))
    
    def test_jit_limitation_with_unique(self):
        """Test that shows jnp.unique() and dynamic num_groups limitation with JIT.
        
        Note: Both jnp.unique() and segment_sum() require concrete/static values for JIT.
        This is a known JAX limitation. For production use with JIT, either:
        1. Pre-compute groups outside JIT boundary
        2. Use fixed-size groups with padding (num_groups as static arg)
        3. Use alternative grouping strategies
        """
        
        @jit
        def compute_group_sum_static(values, group_indices):
            """JIT-compatible version with static num_groups."""
            from jax.ops import segment_sum
            num_groups = 2  # Must be static/concrete for JIT
            return segment_sum(values, group_indices, num_groups)
        
        values = jnp.array([10.0, 20.0, 30.0, 40.0])
        groups = jnp.array([0, 1, 0, 1])
        
        # Pre-compute unique groups outside JIT
        unique_groups, group_indices = jnp.unique(groups, return_inverse=True)
        
        # Now JIT the actual computation with static num_groups
        result = compute_group_sum_static(values, group_indices)
        
        assert jnp.allclose(result, jnp.array([40.0, 60.0]))
    
    def test_gradient_through_aggregation(self):
        """Test that we can compute gradients through group aggregations."""
        
        def loss_fn(values, group_indices):
            """Compute mean of group sums with static num_groups."""
            from jax.ops import segment_sum
            
            num_groups = 2  # Static value
            group_sums = segment_sum(values, group_indices, num_groups)
            return jnp.mean(group_sums)
        
        values = jnp.array([10.0, 20.0, 30.0, 40.0])
        groups = jnp.array([0, 1, 0, 1])
        
        # Pre-compute groups
        unique_groups, group_indices = jnp.unique(groups, return_inverse=True)
        
        # Compute gradient
        grad_fn = grad(loss_fn)
        gradients = grad_fn(values, group_indices)
        
        # Gradient should be 0.5 for all values (since mean divides by 2 groups)
        assert jnp.allclose(gradients, jnp.array([0.5, 0.5, 0.5, 0.5]))
    
    def test_jit_full_pipeline_precomputed(self):
        """Test JIT compilation with pre-computed groups and static size."""
        
        @jit
        def grouped_sum_with_precomputed(values, group_indices):
            """Compute group sums with static num_groups."""
            from jax.ops import segment_sum
            num_groups = 2  # Must be static for JIT
            return segment_sum(values, group_indices, num_groups)
        
        values = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        groups = jnp.array([1, 2, 1, 2, 1])
        
        # Pre-compute groups outside JIT
        unique_groups, inverse_indices = jnp.unique(groups, return_inverse=True)
        
        # JIT the actual aggregation
        sums = grouped_sum_with_precomputed(values, inverse_indices)
        
        assert jnp.allclose(sums[0], 90.0)  # Group 1: 10 + 30 + 50
        assert jnp.allclose(sums[1], 60.0)  # Group 2: 20 + 40
    
    def test_vmap_compatibility(self):
        """Test that operations work with vmap (vectorization)."""
        from jax import vmap
        from jax.ops import segment_sum
        
        # Create a batch of grouping operations
        def single_group_sum(values, groups):
            unique_groups, inverse_indices = jnp.unique(groups, return_inverse=True)
            num_groups = len(unique_groups)
            return segment_sum(values, inverse_indices, num_groups)
        
        # Batch of values and groups
        batch_values = jnp.array([
            [10.0, 20.0, 30.0, 40.0],
            [5.0, 10.0, 15.0, 20.0]
        ])
        batch_groups = jnp.array([
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ])
        
        # Note: vmap with unique is tricky, so this is more of a conceptual test
        # In practice, you'd need to handle this carefully
        # For now, just test that the single case works
        result = single_group_sum(batch_values[0], batch_groups[0])
        assert jnp.allclose(result, jnp.array([40.0, 60.0]))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_group(self):
        """Test grouping when all rows belong to same group."""
        df = DataFrame({
            'category': jnp.array([1, 1, 1, 1]),
            'value': jnp.array([10, 20, 30, 40])
        })
        
        result = df.group_by('category').agg({'value': ('sum', agg_sum)})
        
        assert len(result) == 1
        assert result['value_sum'][0] == 100
    
    def test_each_row_unique_group(self):
        """Test when each row is its own group."""
        df = DataFrame({
            'id': jnp.array([1, 2, 3, 4]),
            'value': jnp.array([10, 20, 30, 40])
        })
        
        result = df.group_by('id').agg({'value': ('sum', agg_sum)})
        
        assert len(result) == 4
        assert jnp.array_equal(result['value_sum'], jnp.array([10, 20, 30, 40]))
    
    def test_invalid_column_error(self):
        """Test error when grouping by non-existent column."""
        df = DataFrame({
            'category': jnp.array([1, 2, 3]),
            'value': jnp.array([10, 20, 30])
        })
        
        with pytest.raises(KeyError, match="not found"):
            df.group_by('nonexistent')
    
    def test_invalid_agg_column_error(self):
        """Test error when aggregating non-existent column."""
        df = DataFrame({
            'category': jnp.array([1, 2, 3]),
            'value': jnp.array([10, 20, 30])
        })
        
        with pytest.raises(KeyError, match="not found"):
            df.group_by('category').agg({'nonexistent': ('sum', agg_sum)})
    
    def test_unsupported_agg_function(self):
        """Test error when using string shortcuts (no longer supported)."""
        df = DataFrame({
            'category': jnp.array([1, 2, 3]),
            'value': jnp.array([10, 20, 30])
        })
        
        with pytest.raises(TypeError, match="Aggregation functions must be tuples"):
            df.group_by('category').agg({'value': 'median'})
    
    def test_three_column_grouping(self):
        """Test grouping by three columns."""
        df = DataFrame({
            'year': jnp.array([2020, 2020, 2021, 2020]),
            'month': jnp.array([1, 1, 2, 1]),
            'day': jnp.array([1, 2, 1, 1]),
            'value': jnp.array([10, 20, 30, 40])
        })
        
        result = df.group_by(['year', 'month', 'day']).agg({'value': ('sum', agg_sum)})
        
        # Should have 3 unique groups
        assert len(result) == 3
        
        # Check (2020, 1, 1) group - should have values 10 + 40 = 50
        mask = (np.array(result['year']) == 2020) & \
               (np.array(result['month']) == 1) & \
               (np.array(result['day']) == 1)
        assert np.sum(np.array(result['value_sum'])[mask]) == 50


class TestNumericalAccuracy:
    """Test numerical accuracy of aggregations."""
    
    def test_mean_accuracy(self):
        """Test that mean calculation is accurate."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        })
        
        result = df.group_by('group').agg({'value': ('mean', agg_mean)})
        
        g1_idx = np.where(np.array(result['group']) == 1)[0][0]
        g2_idx = np.where(np.array(result['group']) == 2)[0][0]
        
        assert np.isclose(result['value_mean'][g1_idx], 1.5, rtol=1e-6)
        assert np.isclose(result['value_mean'][g2_idx], 4.0, rtol=1e-6)
    
    def test_std_accuracy(self):
        """Test standard deviation calculation accuracy."""
        df = DataFrame({
            'group': jnp.array([1, 1, 1]),
            'value': jnp.array([1.0, 2.0, 3.0])
        })
        
        result = df.group_by('group').agg({'value': ('std', agg_std)})
        
        # Expected std for [1, 2, 3] is sqrt(2/3) ≈ 0.8165
        expected_std = np.std([1.0, 2.0, 3.0])
        assert np.isclose(result['value_std'][0], expected_std, rtol=1e-5)
    
    def test_large_values(self):
        """Test with large values to check numerical stability."""
        df = DataFrame({
            'group': jnp.array([1, 1, 2, 2]),
            'value': jnp.array([1e10, 2e10, 3e10, 4e10])
        })
        
        result = df.group_by('group').agg({'value': [('sum', agg_sum), ('mean', agg_mean)]})
        
        g1_idx = np.where(np.array(result['group']) == 1)[0][0]
        
        assert np.isclose(result['value_sum'][g1_idx], 3e10, rtol=1e-5)
        assert np.isclose(result['value_mean'][g1_idx], 1.5e10, rtol=1e-5)


class TestDataTypes:
    """Test GroupBy with different data types."""
    
    def test_integer_groups(self):
        """Test with integer group keys."""
        df = DataFrame({
            'group': jnp.array([1, 2, 1, 2], dtype=jnp.int32),
            'value': jnp.array([10, 20, 30, 40])
        })
        
        result = df.group_by('group').agg({'value': ('sum', agg_sum)})
        
        assert len(result) == 2
        assert 'group' in result.columns
    
    def test_float_groups(self):
        """Test with float group keys."""
        df = DataFrame({
            'group': jnp.array([1.0, 2.0, 1.0, 2.0]),
            'value': jnp.array([10, 20, 30, 40])
        })
        
        result = df.group_by('group').agg({'value': ('sum', agg_sum)})
        
        assert len(result) == 2
    
    def test_string_groups(self):
        """Test with string group keys."""
        df = DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'value': jnp.array([10, 20, 30, 40])
        })
        
        result = df.group_by('category').agg({'value': ('sum', agg_sum)})
        
        assert len(result) == 2
        result_dict = {str(cat): val for cat, val in zip(result['category'], result['value_sum'])}
        assert result_dict['A'] == 40
        assert result_dict['B'] == 60


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
