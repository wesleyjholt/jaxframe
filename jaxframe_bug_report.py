"""
Bug Report for jaxframe: Incorrect group_by().agg() results

ISSUE: group_by().agg() produces incorrect mean values for grouped data.

ENVIRONMENT:
- jaxframe version: [please check your version]
- JAX version: [please check your version]
- Python version: 3.13

DESCRIPTION:
When using group_by().agg() with jnp.mean, the aggregation produces incorrect
results that do not match the actual mean of the grouped values.

EXPECTED BEHAVIOR:
The aggregation should compute the correct mean for each group:
  Group A: values [10, 20, 40] → mean = 23.33
  Group B: values [30, 50, 60] → mean = 46.67
  Group C: values [70] → mean = 70.0
  Group D: values [80] → mean = 80.0

ACTUAL BEHAVIOR:
The aggregation produces incorrect means:
  Group A: mean = 15.0  (should be 23.33)
  Group B: mean = 23.75 (should be 46.67)
  Group C: mean = 17.5  (should be 70.0)
  Group D: mean = 18.75 (should be 80.0)

MINIMAL REPRODUCIBLE EXAMPLE:
"""

from jaxframe import DataFrame
import jax.numpy as jnp

print("=" * 70)
print("JAXFRAME BUG: Incorrect group_by().agg() results")
print("=" * 70)
print()

# Create a simple DataFrame with clear grouping structure
df = DataFrame({
    'group_id': ['A', 'A', 'B', 'A', 'B', 'B', 'C', 'D'],
    'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
})

print("Input DataFrame:")
print(df)
print()
print("Row-by-row:")
for i in range(len(df['group_id'])):
    print(f"  Row {i}: group_id={df['group_id'][i]}, value={df['value'][i]}")
print()

# Perform aggregation
result = df.group_by('group_id').agg({'value': ('mean', jnp.mean)})

print("Aggregation result:")
print(result)
print()

# Manual calculation
print("Manual calculation:")
print("  Group A: rows 0,1,3 → values [10.0, 20.0, 40.0] → mean = (10+20+40)/3 = 23.33")
print("  Group B: rows 2,4,5 → values [30.0, 50.0, 60.0] → mean = (30+50+60)/3 = 46.67")
print("  Group C: row 6     → values [70.0]             → mean = 70.0")
print("  Group D: row 7     → values [80.0]             → mean = 80.0")
print()

# Actual results from jaxframe
print("Actual results from jaxframe:")
for group_id, mean_val in zip(result['group_id'], result['value_mean']):
    print(f"  Group {group_id}: mean = {mean_val}")
print()

# Verify the bug
expected_means = {'A': 23.333333, 'B': 46.666667, 'C': 70.0, 'D': 80.0}
print("Bug verification:")
bug_found = False
for group_id, mean_val in zip(result['group_id'], result['value_mean']):
    expected = expected_means[str(group_id)]
    actual = float(mean_val)
    match = "✓ CORRECT" if abs(actual - expected) < 0.01 else "✗ BUG!"
    if "BUG" in match:
        bug_found = True
    print(f"  Group {group_id}: expected={expected:.2f}, actual={actual:.2f} {match}")

print()
if bug_found:
    print("❌ BUG CONFIRMED: group_by().agg() produces incorrect results")
else:
    print("✓ No bug found - results are correct")

"""
ADDITIONAL TEST CASE:

Here's another test showing the issue persists with different data:
"""

print()
print("=" * 70)
print("ADDITIONAL TEST: Different data pattern")
print("=" * 70)
print()

# Test with measurements and person IDs (real-world use case)
meas_df = DataFrame({
    'id_meas': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8'],
    'id_person': ['p1', 'p1', 'p2', 'p1', 'p2', 'p2', 'p3', 'p4'],
    'value': jnp.array([170.0, 171.0, 165.0, 169.0, 166.0, 164.0, 175.0, 160.0])
})

print("Input DataFrame (measurements by person):")
print(meas_df)
print()

agg_result = meas_df.group_by('id_person').agg({'value': ('mean', jnp.mean)})

print("Aggregation result:")
print(agg_result)
print()

print("Manual calculation:")
print("  p1: m1,m2,m4 → [170, 171, 169] → mean = 170.0")
print("  p2: m3,m5,m6 → [165, 166, 164] → mean = 165.0")
print("  p3: m7       → [175]           → mean = 175.0")
print("  p4: m8       → [160]           → mean = 160.0")
print()

expected_person_means = {'p1': 170.0, 'p2': 165.0, 'p3': 175.0, 'p4': 160.0}
print("Verification:")
bug_found2 = False
for person_id, mean_val in zip(agg_result['id_person'], agg_result['value_mean']):
    expected = expected_person_means[str(person_id)]
    actual = float(mean_val)
    match = "✓ CORRECT" if abs(actual - expected) < 0.01 else "✗ BUG!"
    if "BUG" in match:
        bug_found2 = True
    print(f"  Person {person_id}: expected={expected:.2f}, actual={actual:.2f} {match}")

print()
if bug_found2:
    print("❌ BUG CONFIRMED in this test case as well")
else:
    print("✓ This test case passes")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("The bug appears to affect group_by().agg() when:")
print("1. Using jnp.mean as the aggregation function")
print("2. Groups have different numbers of elements")
print("3. Groups are not contiguous in the DataFrame")
print()
print("This suggests a potential issue with how the grouping mask or")
print("indexing is being applied during the aggregation operation.")
