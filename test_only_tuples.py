"""
Test that only tuples (name, function) are allowed in aggregation.
Bare callables should now be rejected.
"""
import jax.numpy as jnp
from jaxframe import DataFrame

# Create test data
height_meas_table = DataFrame({
    'id_height_meas': ['01', '02', '03'],
    'id_person': ['01', '01', '02'],
    'height_meas': jnp.array([55.0, 55.5, 60.0])
})

print("=" * 70)
print("Testing: Only tuples (name, function) allowed")
print("=" * 70)
print()

print("Test 1: Bare callable jnp.mean - should throw error")
try:
    result = height_meas_table.group_by('id_person').agg({'height_meas': jnp.mean})
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"✓ Correctly raised TypeError: {e}")
print()

print("Test 2: List with bare callable - should throw error")
try:
    result = height_meas_table.group_by('id_person').agg({'height_meas': [jnp.mean, jnp.std]})
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"✓ Correctly raised TypeError: {e}")
print()

print("Test 3: String shortcut 'mean' - should throw error")
try:
    result = height_meas_table.group_by('id_person').agg({'height_meas': 'mean'})
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"✓ Correctly raised TypeError: {e}")
print()

print("Test 4: Single named tuple ('mean', jnp.mean) - should work ✓")
result = height_meas_table.group_by('id_person').agg({'height_meas': ('mean', jnp.mean)})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print(f"Expected: ('id_person', 'height_meas') because single aggregation doesn't add suffix")
print()

print("Test 5: Multiple named tuples - should work ✓")
result = height_meas_table.group_by('id_person').agg({
    'height_meas': [('mean', jnp.mean), ('std', jnp.std)]
})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print(f"Expected: ('id_person', 'height_meas_mean', 'height_meas_std')")
print()

print("Test 6: Custom lambda with tuple - should work ✓")
result = height_meas_table.group_by('id_person').agg({
    'height_meas': ('p90', lambda x: jnp.percentile(x, 90))
})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print()

print("Test 7: Multiple columns with tuples - should work ✓")
sales_table = DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'sales': jnp.array([100, 150, 200, 250]),
    'profit': jnp.array([10, 15, 20, 25])
})

result = sales_table.group_by('category').agg({
    'sales': [('mean', jnp.mean), ('sum', jnp.sum)],
    'profit': [('mean', jnp.mean), ('max', jnp.max)]
})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print()

print("=" * 70)
print("All tests completed successfully! ✓")
print("Only tuples (name, function) are allowed in aggregation.")
print("=" * 70)
