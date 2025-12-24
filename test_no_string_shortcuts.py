"""
Test that string shortcuts are no longer supported in aggregation.
"""
import jax.numpy as jnp
from jaxframe import DataFrame

# Create test data
height_meas_table = DataFrame({
    'id_height_meas': ['01', '02', '03'],
    'id_person': ['01', '01', '02'],
    'height_meas': jnp.array([55.0, 55.5, 60.0])
})

print("Test 1: Single string shortcut 'mean' - should throw error")
try:
    result = height_meas_table.group_by('id_person').agg({'height_meas': 'mean'})
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"✓ Correctly raised TypeError: {e}")
print()

print("Test 2: List with string shortcuts ['mean', 'max'] - should throw error")
try:
    result = height_meas_table.group_by('id_person').agg({'height_meas': ['mean', 'max']})
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"✓ Correctly raised TypeError: {e}")
print()

print("Test 3: Single named tuple ('mean', jnp.mean) - should create single column")
result = height_meas_table.group_by('id_person').agg({'height_meas': ('mean', jnp.mean)})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print(f"Expected: ('id_person', 'height_meas') because single aggregation doesn't add suffix")
print()

print("Test 4: Multiple named tuples - should create two columns with suffixes")
result = height_meas_table.group_by('id_person').agg({
    'height_meas': [('mean', jnp.mean), ('std', jnp.std)]
})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print(f"Expected: ('id_person', 'height_meas_mean', 'height_meas_std')")
print()

print("Test 5: Mix of named and unnamed functions")
result = height_meas_table.group_by('id_person').agg({
    'height_meas': [('mean', jnp.mean), jnp.median, ('max', jnp.max)]
})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print(f"Expected: ('id_person', 'height_meas_mean', 'height_meas_median', 'height_meas_max')")
print()

print("Test 6: Single unnamed function - should not add suffix")
result = height_meas_table.group_by('id_person').agg({'height_meas': jnp.mean})
print("Result:")
print(result)
print(f"Columns: {result.columns}")
print(f"Expected: ('id_person', 'height_meas') because single aggregation doesn't add suffix")
print()

print("All tests completed successfully! ✓")
