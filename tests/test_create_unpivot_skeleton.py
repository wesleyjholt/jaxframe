"""Tests for create_unpivot_skeleton function."""

import pytest
import numpy as np

from jaxframe import (
    DataFrame,
    create_unpivot_skeleton,
    pivot_sparse,
    unpivot_sparse,
    to_masked_array,
    from_masked_array,
)

try:
    import jax.numpy as jnp
    from jax import jit
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np


class TestCreateUnpivotSkeletonBasic:
    """Basic tests for create_unpivot_skeleton."""
    
    def test_basic_skeleton_creation(self):
        """Skeleton has correct columns and variable values."""
        df = DataFrame({
            'id_person': ['A', 'A', 'B', 'B', 'B'],
            'id_meas': ['m1', 'm2', 'm3', 'm4', 'm5'],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas'
        )
        
        # Check columns exist
        assert 'id_person' in skeleton.columns
        assert 'variable' in skeleton.columns
        assert 'id_meas' in skeleton.columns
        
        # Check variable values (position within each group)
        # A has 2 measurements: positions 0, 1
        # B has 3 measurements: positions 0, 1, 2
        expected_variables = [0, 1, 0, 1, 2]
        assert list(skeleton['variable']) == expected_variables
        
        # Check id column preserved
        assert list(skeleton['id_meas']) == ['m1', 'm2', 'm3', 'm4', 'm5']
    
    def test_skeleton_without_id_column(self):
        """Skeleton works without id_column."""
        df = DataFrame({
            'id_person': ['A', 'A', 'B'],
            'value': [1.0, 2.0, 3.0]
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value'
        )
        
        assert 'id_person' in skeleton.columns
        assert 'variable' in skeleton.columns
        assert len(skeleton.columns) == 2
    
    def test_skeleton_with_multi_index(self):
        """Skeleton works with multiple index columns."""
        df = DataFrame({
            'id_person': ['A', 'A', 'A', 'B'],
            'id_instrument': ['X', 'X', 'Y', 'X'],
            'id_meas': ['m1', 'm2', 'm3', 'm4'],
            'value': [1.0, 2.0, 3.0, 4.0]
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns=['id_person', 'id_instrument'],
            value_column='value',
            id_column='id_meas'
        )
        
        # (A, X) has 2 measurements: 0, 1
        # (A, Y) has 1 measurement: 0
        # (B, X) has 1 measurement: 0
        expected_variables = [0, 1, 0, 0]
        assert list(skeleton['variable']) == expected_variables


class TestCreateUnpivotSkeletonWithSort:
    """Tests for create_unpivot_skeleton with sort_within_index_group."""
    
    def test_skeleton_with_sort(self):
        """Sorted pivot produces correct variable mapping."""
        # Values are NOT in sorted order within groups
        df = DataFrame({
            'id_person': ['A', 'A', 'A', 'B', 'B'],
            'id_meas': ['m1', 'm2', 'm3', 'm4', 'm5'],
            'value': [3.0, 1.0, 2.0, 5.0, 4.0]  # A: 3,1,2 -> sorted: 1,2,3 -> positions 1,2,0
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas',
            sort_within_index_group=True
        )
        
        # After sorting:
        # A: values [3,1,2] sorted -> [1,2,3] 
        #    m1(3.0) goes to slot 2, m2(1.0) goes to slot 0, m3(2.0) goes to slot 1
        # B: values [5,4] sorted -> [4,5]
        #    m4(5.0) goes to slot 1, m5(4.0) goes to slot 0
        expected_variables = [2, 0, 1, 1, 0]
        assert list(skeleton['variable']) == expected_variables
    
    def test_skeleton_without_sort(self):
        """Unsorted pivot uses position-based variables."""
        df = DataFrame({
            'id_person': ['A', 'A', 'A'],
            'id_meas': ['m1', 'm2', 'm3'],
            'value': [3.0, 1.0, 2.0]  # Order doesn't matter
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas',
            sort_within_index_group=False
        )
        
        # Position-based: 0, 1, 2
        expected_variables = [0, 1, 2]
        assert list(skeleton['variable']) == expected_variables


class TestCreateUnpivotSkeletonWithVarColumn:
    """Tests for create_unpivot_skeleton with explicit var_column."""
    
    def test_skeleton_with_var_column(self):
        """Explicit var_column is used directly."""
        df = DataFrame({
            'id_person': ['A', 'A', 'B'],
            'time_idx': [5, 10, 5],  # Explicit variable indices
            'id_meas': ['m1', 'm2', 'm3'],
            'value': [1.0, 2.0, 3.0]
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas',
            var_column='time_idx'
        )
        
        # var_column values used directly
        assert list(skeleton['variable']) == [5, 10, 5]


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestCreateUnpivotSkeletonWithJAX:
    """Tests for create_unpivot_skeleton with JAX arrays and tracers."""
    
    def test_skeleton_with_jax_array(self):
        """Works with JAX arrays as values."""
        df = DataFrame({
            'id_person': ['A', 'A', 'B'],
            'id_meas': ['m1', 'm2', 'm3'],
            'value': jnp.array([1.0, 2.0, 3.0])
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas'
        )
        
        assert list(skeleton['variable']) == [0, 1, 0]
    
    def test_skeleton_with_jax_tracers(self):
        """Traced values don't cause errors, uses position-based approach."""
        df = DataFrame({
            'id_person': ['A', 'A', 'B'],
            'id_meas': ['m1', 'm2', 'm3'],
            'value': jnp.array([3.0, 1.0, 2.0])  # Would be sorted differently
        })
        
        # Create skeleton inside JIT - values become tracers
        @jit
        def create_skeleton_jit(values):
            df_with_traced = DataFrame({
                'id_person': ['A', 'A', 'B'],
                'id_meas': ['m1', 'm2', 'm3'],
                'value': values
            })
            skeleton = create_unpivot_skeleton(
                df=df_with_traced,
                index_columns='id_person',
                value_column='value',
                id_column='id_meas',
                sort_within_index_group=True  # Would need sorting, but can't with tracers
            )
            # Return just the variable column to verify it works
            return jnp.array(skeleton['variable'])
        
        result = create_skeleton_jit(jnp.array([3.0, 1.0, 2.0]))
        
        # With tracers, falls back to position-based (can't sort tracers)
        expected = jnp.array([0, 1, 0])
        assert jnp.allclose(result, expected)


class TestCreateUnpivotSkeletonRoundtrip:
    """Tests for roundtrip pivot/unpivot with skeleton."""
    
    def test_roundtrip_unsorted(self):
        """pivot → unpivot with skeleton restores original order."""
        # Original long data
        long_df = DataFrame({
            'id_meas': ['m1', 'm2', 'm3', 'm4', 'm5'],
            'id_person': ['A', 'A', 'B', 'B', 'B'],
            'value': np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        })
        
        # Create skeleton
        skeleton = create_unpivot_skeleton(
            df=long_df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas',
            sort_within_index_group=False
        )
        
        # Pivot
        wide_df = pivot_sparse(
            df=long_df,
            index='id_person',
            value='value',
            prefix='value',
            sort_within_index_group=False
        )
        
        # Convert to masked array and back
        masked = to_masked_array(wide_df, index='id_person')
        wide_df_2 = from_masked_array(masked, prefix='value')
        
        # Unpivot with skeleton
        long_restored = unpivot_sparse(
            df=wide_df_2,
            index='id_person',
            value_name='value',
            long_skeleton_df=skeleton,
            long_skeleton_id_column='id_meas'
        )
        
        # Check original order is restored
        assert list(long_restored['id_meas']) == ['m1', 'm2', 'm3', 'm4', 'm5']
        assert np.allclose(long_restored['value'], [1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_roundtrip_sorted(self):
        """Sorted pivot → unpivot with skeleton restores original order."""
        # Original long data - values NOT in sorted order
        long_df = DataFrame({
            'id_meas': ['m1', 'm2', 'm3', 'm4', 'm5'],
            'id_person': ['A', 'A', 'A', 'B', 'B'],
            'value': np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        })
        
        # Create skeleton with sort
        skeleton = create_unpivot_skeleton(
            df=long_df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas',
            sort_within_index_group=True
        )
        
        # Pivot with sort
        wide_df = pivot_sparse(
            df=long_df,
            index='id_person',
            value='value',
            prefix='value',
            sort_within_index_group=True
        )
        
        # Convert to masked array and back
        masked = to_masked_array(wide_df, index='id_person')
        wide_df_2 = from_masked_array(masked, prefix='value')
        
        # Unpivot with skeleton
        long_restored = unpivot_sparse(
            df=wide_df_2,
            index='id_person',
            value_name='value',
            order_name='order',
            long_skeleton_df=skeleton,
            long_skeleton_id_column='id_meas'
        )
        
        # Check original order is restored via id_meas
        assert list(long_restored['id_meas']) == ['m1', 'm2', 'm3', 'm4', 'm5']


class TestCreateUnpivotSkeletonEdgeCases:
    """Edge case tests."""
    
    def test_single_row(self):
        """Works with single row."""
        df = DataFrame({
            'id_person': ['A'],
            'id_meas': ['m1'],
            'value': [1.0]
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas'
        )
        
        assert list(skeleton['variable']) == [0]
    
    def test_single_entity_multiple_obs(self):
        """Works with single entity, multiple observations."""
        df = DataFrame({
            'id_person': ['A', 'A', 'A', 'A'],
            'id_meas': ['m1', 'm2', 'm3', 'm4'],
            'value': [1.0, 2.0, 3.0, 4.0]
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value',
            id_column='id_meas'
        )
        
        assert list(skeleton['variable']) == [0, 1, 2, 3]
    
    def test_missing_id_column_raises(self):
        """Raises error if id_column doesn't exist."""
        df = DataFrame({
            'id_person': ['A', 'A'],
            'value': [1.0, 2.0]
        })
        
        with pytest.raises(ValueError, match="id_column 'nonexistent' not found"):
            create_unpivot_skeleton(
                df=df,
                index_columns='id_person',
                value_column='value',
                id_column='nonexistent'
            )
    
    def test_empty_dataframe(self):
        """Handles empty DataFrame."""
        df = DataFrame({
            'id_person': [],
            'value': []
        })
        
        skeleton = create_unpivot_skeleton(
            df=df,
            index_columns='id_person',
            value_column='value'
        )
        
        assert len(skeleton) == 0
        assert 'variable' in skeleton.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
