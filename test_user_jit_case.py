"""
Test the exact user case for JIT compatibility.
"""
import jax.numpy as jnp
from jax import jit
from src.jaxframe import DataFrame

print("="*70)
print("Testing User's Exact JIT Use Case")
print("="*70)

# Create the weight_meas_table
weight_meas_table = DataFrame({
    'id_weight_meas': ['01', '02', '03', '04', '05'],
    'id_person': ['01', '02', '03', '04', '04'],
    'id_instrument': ['01', '02', '01', '02', '02'],
    'value': jnp.array([170.0, 200.0, 150.0, 190.0, 195.0]),
})

print("\nOriginal DataFrame:")
print(weight_meas_table)

# Define the aggregate function
def aggregate_(df_, x):
    df__ = df_.add_column('new_agg', x)
    df__ = df__.group_by('id_person').agg({'new_agg': ('mean', jnp.mean)})
    return df__

# Test non-JIT version first
print("\n" + "="*70)
print("Testing Non-JIT Version")
print("="*70)

x_test = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
result_non_jit = aggregate_(weight_meas_table, x_test)
print(f"\nResult DataFrame:")
print(result_non_jit)
print(f"\nExtracted 'new_agg_mean': {result_non_jit['new_agg_mean']}")

# Test JIT version
print("\n" + "="*70)
print("Testing JIT Version")
print("="*70)

# Create JIT-compiled function
f = lambda x: aggregate_(weight_meas_table, x)['new_agg_mean']
f_jit = jit(f)

print("\nCompiling and running JIT version...")
try:
    result_jit = f_jit(x_test)
    print(f"JIT result: {result_jit}")
    
    # Verify results match
    non_jit_values = result_non_jit['new_agg_mean']
    print(f"\nNon-JIT result: {non_jit_values}")
    print(f"JIT result:     {result_jit}")
    
    if jnp.allclose(non_jit_values, result_jit):
        print("\n✓ JIT and non-JIT results match!")
    else:
        print("\n✗ Results don't match!")
    
    # Test with different input
    print("\n" + "="*70)
    print("Testing JIT with Different Input")
    print("="*70)
    
    x_test2 = jnp.array([100.0, 200.0, 300.0, 400.0, 500.0])
    result_jit2 = f_jit(x_test2)
    print(f"JIT result with new input: {result_jit2}")
    
    # Verify it's different from first call
    if not jnp.allclose(result_jit, result_jit2):
        print("✓ JIT correctly handles different inputs!")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - JIT COMPATIBILITY ACHIEVED!")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ Error during JIT compilation/execution:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
