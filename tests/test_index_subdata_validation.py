"""
Tests for _validate_index_subdata_columns and JAX tracer handling.

This test suite covers:
1. Basic validation with consistent values
2. Validation with conflicting values  
3. Validation with JAX tracers (should skip validation during tracing)
4. Edge cases (empty dataframes, single rows, etc.)
5. Integration with pivot operations under JIT
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from jaxframe import DataFrame
from jaxframe.transform import (
    _validate_index_subdata_columns,
    _is_jax_tracer,
    pivot_sparse,
    long_to_wide_masked,
    to_masked_array,
)


class TestIsJaxTracer:
    """Tests for _is_jax_tracer helper function."""
    
    def test_regular_python_values(self):
        """Regular Python values should not be detected as tracers."""
        assert not _is_jax_tracer(1)
        assert not _is_jax_tracer(1.0)
        assert not _is_jax_tracer("hello")
        assert not _is_jax_tracer([1, 2, 3])
        assert not _is_jax_tracer(None)
    
    def test_numpy_arrays(self):
        """NumPy arrays should not be detected as tracers."""
        assert not _is_jax_tracer(np.array([1, 2, 3]))
        assert not _is_jax_tracer(np.float32(1.0))
        assert not _is_jax_tracer(np.int32(5))
    
    def test_jax_arrays(self):
        """Concrete JAX arrays should not be detected as tracers."""
        assert not _is_jax_tracer(jnp.array([1, 2, 3]))
        assert not _is_jax_tracer(jnp.float32(1.0))
        assert not _is_jax_tracer(jnp.zeros((2, 3)))
    
    def test_jax_tracers_during_jit(self):
        """JAX tracers during JIT should be detected."""
        detected_as_tracer = []
        
        @jit
        def check_tracer(x):
            detected_as_tracer.append(_is_jax_tracer(x))
            return x
        
        # Call the JIT function - this will trace
        result = check_tracer(jnp.array(1.0))
        
        # During tracing, x should have been detected as a tracer
        assert detected_as_tracer[0] == True
    
    def test_jax_tracers_during_vmap(self):
        """JAX tracers during vmap should be detected."""
        from jax import vmap
        
        detected_as_tracer = []
        
        def check_tracer(x):
            detected_as_tracer.append(_is_jax_tracer(x))
            return x
        
        # vmap also creates tracers
        vmapped_fn = vmap(check_tracer)
        result = vmapped_fn(jnp.array([1.0, 2.0, 3.0]))
        
        # During tracing, elements should have been detected as tracers
        assert any(detected_as_tracer)


class TestValidateIndexSubdataColumns:
    """Tests for _validate_index_subdata_columns function."""
    
    def test_valid_consistent_values(self):
        """Validation should pass when subdata values are consistent within index groups."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'entity_value': [10.0, 10.0, 20.0, 20.0],  # Same value within each entity
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Should not raise
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_value']
        )
    
    def test_valid_with_jax_arrays(self):
        """Validation should pass with JAX array values that are consistent."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'entity_value': jnp.array([10.0, 10.0, 20.0, 20.0]),
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Should not raise
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_value']
        )
    
    def test_invalid_conflicting_values(self):
        """Validation should fail when subdata values conflict within an index group."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'entity_value': [10.0, 15.0, 20.0, 20.0],  # Conflicting values for entity 'a'
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        with pytest.raises(ValueError, match="conflicting values"):
            _validate_index_subdata_columns(
                df, 
                index_columns=['entity'], 
                index_subdata_columns=['entity_value']
            )
    
    def test_invalid_column_not_found(self):
        """Validation should fail when subdata column doesn't exist."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        with pytest.raises(ValueError, match="not in the DataFrame"):
            _validate_index_subdata_columns(
                df, 
                index_columns=['entity'], 
                index_subdata_columns=['nonexistent_column']
            )
    
    def test_multiple_index_columns(self):
        """Validation should work with multiple index columns."""
        df = DataFrame({
            'entity': ['a', 'a', 'a', 'a', 'b', 'b'],
            'operator': ['x', 'x', 'y', 'y', 'x', 'x'],
            'time': [0, 1, 0, 1, 0, 1],
            'operator_offset': jnp.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1]),  # Consistent within (entity, operator)
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        })
        
        # Should not raise - consistent within composite index
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity', 'operator'], 
            index_subdata_columns=['operator_offset']
        )
    
    def test_multiple_index_columns_conflicting(self):
        """Validation should fail with conflicting values in composite index."""
        df = DataFrame({
            'entity': ['a', 'a', 'a', 'a', 'b', 'b'],
            'operator': ['x', 'x', 'y', 'y', 'x', 'x'],
            'time': [0, 1, 0, 1, 0, 1],
            'operator_offset': jnp.array([0.1, 0.15, 0.2, 0.2, 0.1, 0.1]),  # Conflict for (a, x)
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        })
        
        with pytest.raises(ValueError, match="conflicting values"):
            _validate_index_subdata_columns(
                df, 
                index_columns=['entity', 'operator'], 
                index_subdata_columns=['operator_offset']
            )
    
    def test_multiple_subdata_columns(self):
        """Validation should work with multiple subdata columns."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'entity_val1': [10.0, 10.0, 20.0, 20.0],
            'entity_val2': jnp.array([100, 100, 200, 200]),
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Should not raise
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_val1', 'entity_val2']
        )
    
    def test_single_row_per_index(self):
        """Validation should pass when each index has only one row."""
        df = DataFrame({
            'entity': ['a', 'b', 'c'],
            'entity_value': [10.0, 20.0, 30.0],
            'measurement': jnp.array([1.0, 2.0, 3.0])
        })
        
        # Should not raise - no conflicts possible with single rows
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_value']
        )
    
    def test_empty_subdata_columns_list(self):
        """Validation should pass with empty subdata columns list."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Should not raise
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=[]
        )


class TestValidationWithJaxTracers:
    """Tests for validation behavior with JAX tracers."""
    
    def test_validation_skipped_for_tracers_in_jit(self):
        """Validation should skip (not fail) when values are JAX tracers during JIT."""
        validation_called = []
        validation_error = []
        
        def create_df_and_validate(values):
            # Create a DataFrame with traced values
            df = DataFrame({
                'entity': ['a', 'a', 'b', 'b'],
                'time': [0, 1, 0, 1],
                'entity_value': values,  # This will be traced
                'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
            })
            
            try:
                _validate_index_subdata_columns(
                    df, 
                    index_columns=['entity'], 
                    index_subdata_columns=['entity_value']
                )
                validation_called.append(True)
            except ValueError as e:
                validation_error.append(str(e))
            
            return values.sum()  # Return something for JIT
        
        # Wrap in JIT
        jit_fn = jit(create_df_and_validate)
        
        # Call with values that would conflict if compared (but tracers can't be compared)
        result = jit_fn(jnp.array([10.0, 15.0, 20.0, 25.0]))  # These would conflict for entity 'a'
        
        # Validation should have been called without error (tracers skipped)
        assert len(validation_error) == 0
        assert len(validation_called) > 0
    
    def test_mixed_concrete_and_tracer_values(self):
        """Validation should handle DataFrames with mix of concrete and traced values."""
        validation_passed = []
        
        def create_mixed_df_and_validate(traced_values):
            # Create a DataFrame with both concrete and traced columns
            df = DataFrame({
                'entity': ['a', 'a', 'b', 'b'],  # Concrete
                'time': [0, 1, 0, 1],  # Concrete
                'concrete_subdata': [10.0, 10.0, 20.0, 20.0],  # Concrete - consistent
                'traced_subdata': traced_values,  # Traced - would conflict
                'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
            })
            
            try:
                _validate_index_subdata_columns(
                    df, 
                    index_columns=['entity'], 
                    index_subdata_columns=['concrete_subdata', 'traced_subdata']
                )
                validation_passed.append(True)
            except ValueError:
                validation_passed.append(False)
            
            return traced_values.sum()
        
        jit_fn = jit(create_mixed_df_and_validate)
        result = jit_fn(jnp.array([10.0, 15.0, 20.0, 25.0]))  # Would conflict for entity 'a'
        
        # Should pass - concrete values are consistent, traced values are skipped
        assert validation_passed[0] == True


class TestPivotIntegrationWithJIT:
    """Tests for pivot operations with JIT and index_subdata."""
    
    def test_pivot_with_index_subdata_under_jit(self):
        """Pivot operation with index_subdata should work under JIT."""
        # Create base DataFrame structure
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'slot': [0, 1, 0, 1],
            'entity_value': jnp.array([10.0, 10.0, 20.0, 20.0]),
        })
        
        def pivot_fn(values):
            # Add the values column
            df_with_values = df.add_column('measurement', values)
            
            # Pivot with index_subdata
            pivoted = pivot_sparse(
                df=df_with_values,
                index=['entity'],
                value='measurement',
                on=None,
                prefix='measurement',
                fill_type=0.0,
                mask_value=False,
                sort_within_index_group=False,
                index_subdata=['entity_value']
            )
            
            masked = to_masked_array(pivoted, index=['entity'], sort_by_var_index=True)
            return masked.data.sum()
        
        jit_fn = jit(pivot_fn)
        
        # Should work without error
        result = jit_fn(jnp.array([1.0, 2.0, 3.0, 4.0]))
        assert jnp.isfinite(result)
    
    def test_pivot_roundtrip_with_subdata_under_jit(self):
        """Full pivot/unpivot roundtrip should work under JIT with index_subdata."""
        from jaxframe.transform import unpivot_sparse, from_masked_array, wide_to_long_masked
        
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'slot': [0, 1, 0, 1],
            'entity_value': jnp.array([10.0, 10.0, 20.0, 20.0]),
        })
        
        def roundtrip_fn(values):
            df_with_values = df.add_column('measurement', values)
            
            # Pivot
            pivoted = pivot_sparse(
                df=df_with_values,
                index=['entity'],
                value='measurement',
                on=None,
                prefix='measurement',
                fill_type=0.0,
                mask_value=False,
                sort_within_index_group=False,
                index_subdata=['entity_value']
            )
            
            masked = to_masked_array(pivoted, index=['entity'], sort_by_var_index=True)
            
            # Transform the data
            transformed = masked.data * 2.0
            
            # Unpivot
            wide_df = from_masked_array(
                MaskedArray(
                    data=transformed, 
                    mask=masked.mask, 
                    wide_skeleton_df=masked.wide_skeleton_df,
                    index_columns=['entity'],
                    validate=False
                ), 
                prefix='measurement'
            )
            
            long_df = unpivot_sparse(
                df=wide_df,
                index=['entity'],
                value_name='measurement',
                order_name=None,
                long_skeleton_df=None,
                long_skeleton_id_column=None
            )
            
            return jnp.asarray(long_df['measurement'])
        
        jit_fn = jit(roundtrip_fn)
        
        input_values = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = jit_fn(input_values)
        
        # Values should be doubled
        expected = input_values * 2.0
        assert jnp.allclose(result, expected)


class TestEdgeCases:
    """Edge case tests for validation."""
    
    def test_integer_subdata_values(self):
        """Validation should work with integer subdata values."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'entity_id': [100, 100, 200, 200],  # Integer values
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Should not raise
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_id']
        )
    
    def test_string_subdata_values(self):
        """Validation should work with string subdata values."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'entity_name': ['Alice', 'Alice', 'Bob', 'Bob'],
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Should not raise
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_name']
        )
    
    def test_string_subdata_conflicting(self):
        """Validation should fail with conflicting string values."""
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'entity_name': ['Alice', 'Alicia', 'Bob', 'Bob'],  # Conflict for 'a'
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        with pytest.raises(ValueError, match="conflicting values"):
            _validate_index_subdata_columns(
                df, 
                index_columns=['entity'], 
                index_subdata_columns=['entity_name']
            )
    
    def test_same_object_reference_fast_path(self):
        """Validation should use fast path for same object references."""
        shared_value = jnp.array(10.0)
        
        df = DataFrame({
            'entity': ['a', 'a', 'b', 'b'],
            'time': [0, 1, 0, 1],
            'shared': [shared_value, shared_value, shared_value, shared_value],
            'measurement': jnp.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # Should not raise - same object reference
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['shared']
        )
    
    def test_many_rows_same_index(self):
        """Validation should handle many rows with the same index."""
        n = 100
        df = DataFrame({
            'entity': ['a'] * n,
            'time': list(range(n)),
            'entity_value': [42.0] * n,
            'measurement': jnp.arange(n, dtype=jnp.float32)
        })
        
        # Should not raise
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_value']
        )
    
    def test_many_unique_indices(self):
        """Validation should handle many unique indices."""
        n = 100
        df = DataFrame({
            'entity': [f'entity_{i}' for i in range(n)],
            'entity_value': list(range(n)),
            'measurement': jnp.arange(n, dtype=jnp.float32)
        })
        
        # Should not raise - each entity has only one row
        _validate_index_subdata_columns(
            df, 
            index_columns=['entity'], 
            index_subdata_columns=['entity_value']
        )


class TestNotebookScenario:
    """Test the exact scenario from the notebook that was failing."""
    
    def test_weight_measurement_scenario(self):
        """Simulate the weight measurement scenario from the notebook."""
        # Simulate the weight_meas table structure
        weight_meas_table = DataFrame({
            'id_weight_meas': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
            'id_person': ['01', '01', '01', '02', '02', '02', '03', '04', '04'],
            'id_operator': ['01', '01', '02', '01', '01', '01', '02', '02', '03'],
            'time': jnp.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
            'value': jnp.array([170.0, 200.0, 150.0, 190.0, 195.0, 180.0, 160.0, 175.0, 165.0]),
        })
        
        # Simulate adding a column from a join operation (like __input_pk_0)
        # In the real case, this comes from operator table joined to weight_meas
        input_pk_column = ['01', '01', '02', '01', '01', '01', '02', '02', '03']  # Same as id_operator
        table_with_pk = weight_meas_table.add_column('__input_pk_0', input_pk_column)
        
        # Simulate adding input values (like __input_values_0)
        # These are the offset values from operator table, broadcast to weight_meas
        input_values = jnp.array([3.5, 3.5, 2.5, 3.5, 3.5, 3.5, 2.5, 2.5, 1.5])
        table_with_values = table_with_pk.add_column('__input_values_0', input_values)
        
        # Now validate - this should pass since values are consistent within operator groups
        _validate_index_subdata_columns(
            table_with_values,
            index_columns=['__input_pk_0'],
            index_subdata_columns=['__input_values_0']
        )
    
    def test_weight_measurement_under_jit(self):
        """Test weight measurement scenario under JIT."""
        weight_meas_table = DataFrame({
            'id_weight_meas': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
            'id_person': ['01', '01', '01', '02', '02', '02', '03', '04', '04'],
            'id_operator': ['01', '01', '02', '01', '01', '01', '02', '02', '03'],
            'time': jnp.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
            'value': jnp.array([170.0, 200.0, 150.0, 190.0, 195.0, 180.0, 160.0, 175.0, 165.0]),
        })
        
        input_pk_column = ['01', '01', '02', '01', '01', '01', '02', '02', '03']
        table_with_pk = weight_meas_table.add_column('__input_pk_0', input_pk_column)
        
        def process_with_traced_values(offset_values):
            # offset_values will be a tracer during JIT
            table_with_values = table_with_pk.add_column('__input_values_0', offset_values)
            
            # This validation should skip the traced column
            _validate_index_subdata_columns(
                table_with_values,
                index_columns=['__input_pk_0'],
                index_subdata_columns=['__input_values_0']
            )
            
            return offset_values.sum()
        
        jit_fn = jit(process_with_traced_values)
        
        # Use values that would conflict if compared - but tracers should be skipped
        offset_values = jnp.array([3.5, 3.6, 2.5, 3.5, 3.7, 3.5, 2.5, 2.5, 1.5])  # Slight differences
        
        # Should not raise
        result = jit_fn(offset_values)
        assert jnp.isfinite(result)
    
    def test_pivot_with_operator_offset_under_jit(self):
        """Test the full pivot operation with operator offset under JIT."""
        weight_meas_table = DataFrame({
            'id_weight_meas': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
            'id_person': ['01', '01', '01', '02', '02', '02', '03', '04', '04'],
            'id_operator': ['01', '01', '02', '01', '01', '01', '02', '02', '03'],
            '__input_pk_0': ['01', '01', '02', '01', '01', '01', '02', '02', '03'],
            'time': jnp.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
        })
        
        def pivot_with_traced_subdata(offset_values):
            # Add traced column as subdata
            df = weight_meas_table.add_column('__input_values_0', offset_values)
            
            # Pivot operation with index_subdata
            pivoted = pivot_sparse(
                df=df,
                index=['__input_pk_0'],
                value='time',
                on=None,
                prefix='time',
                fill_type=0.0,
                mask_value=False,
                sort_within_index_group=True,
                index_subdata=['__input_values_0']
            )
            
            masked = to_masked_array(pivoted, index=['__input_pk_0'], sort_by_var_index=True)
            return masked.data.sum()
        
        jit_fn = jit(pivot_with_traced_subdata)
        
        offset_values = jnp.array([3.5, 3.5, 2.5, 3.5, 3.5, 3.5, 2.5, 2.5, 1.5])
        
        # Should complete without error
        result = jit_fn(offset_values)
        assert jnp.isfinite(result)


# Import MaskedArray for tests
from jaxframe.masked_array import MaskedArray


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
