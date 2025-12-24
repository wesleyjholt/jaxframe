"""Test JIT compatibility with groupby and aggregate operations."""

import jax.numpy as jnp
from jax import jit
from src.jaxframe import DataFrame


def test_jit_groupby_agg():
    """Test that groupby with aggregation is JIT-compatible."""
    
    # Create a base DataFrame
    weight_meas_table = DataFrame({
        'id_weight_meas': ['01', '02', '03', '04', '05'],
        'id_person': ['01', '02', '03', '04', '04'],
        'id_instrument': ['01', '02', '01', '02', '02'],
        'value': jnp.array([170.0, 200.0, 150.0, 190.0, 195.0]),
    })
    
    # Define the aggregation function
    def aggregate_(x):
        df__ = weight_meas_table.add_column('new_agg', x)
        df__ = df__.group_by('id_person').agg({'new_agg': ('mean', jnp.mean)})
        return df__
    
    # Create a JIT-compiled function
    f = lambda x: aggregate_(x)['new_agg_mean']
    f_jit = jit(f)
    
    # Test input
    test_input = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    # Run non-JIT version first
    print("Running non-JIT version...")
    result_non_jit = f(test_input)
    print(f"Non-JIT result: {result_non_jit}")
    
    # Run JIT version
    print("\nRunning JIT version...")
    result_jit = f_jit(test_input)
    print(f"JIT result: {result_jit}")
    
    # Verify results match
    assert jnp.allclose(result_non_jit, result_jit), "JIT and non-JIT results don't match!"
    print("\n✓ JIT and non-JIT results match!")
    
    # Test with different input
    test_input2 = jnp.array([100.0, 200.0, 300.0, 400.0, 500.0])
    result_jit2 = f_jit(test_input2)
    print(f"\nJIT result with different input: {result_jit2}")
    
    print("\n✓ All tests passed! GroupBy with aggregation is JIT-compatible.")


def test_jit_multiple_aggs():
    """Test JIT with multiple aggregations."""
    
    df = DataFrame({
        'group': ['A', 'B', 'A', 'B', 'A'],
        'value1': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'value2': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    })
    
    def aggregate_multi(x, y):
        df_temp = df.add_column('temp_x', x).add_column('temp_y', y)
        result = df_temp.group_by('group').agg({
            'temp_x': ('mean', jnp.mean),
            'temp_y': ('sum', jnp.sum),
        })
        return result['temp_x_mean'], result['temp_y_sum']
    
    f_jit = jit(aggregate_multi)
    
    x_input = jnp.array([5.0, 6.0, 7.0, 8.0, 9.0])
    y_input = jnp.array([50.0, 60.0, 70.0, 80.0, 90.0])
    
    print("\nTesting multiple aggregations with JIT...")
    result_x, result_y = f_jit(x_input, y_input)
    print(f"Mean result: {result_x}")
    print(f"Sum result: {result_y}")
    print("✓ Multiple aggregations work with JIT!")


if __name__ == '__main__':
    test_jit_groupby_agg()
    test_jit_multiple_aggs()
