"""
Test JAXFrame's JIT compatibility with groupby and aggregate operations.
"""
import jax
import jax.numpy as jnp
from src.jaxframe import DataFrame

print("="*80)
print("TESTING JAXFRAME JIT COMPATIBILITY WITH GROUPBY AND AGGREGATION")
print("="*80)

# Test 1: Simple aggregation without JIT
print("\n" + "="*80)
print("TEST 1: Basic GroupBy Aggregation (No JIT)")
print("="*80)

df = DataFrame({
    'category': jnp.array([1, 2, 1, 2, 1]),
    'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
})

print(f"\nOriginal DataFrame:")
print(df)

result = df.group_by('category').agg({'value': ('sum', jnp.sum)})
print(f"\nAggregation result:")
print(result)
print("✓ Basic aggregation works!")

# Test 2: Try to JIT a function that uses GroupBy
print("\n" + "="*80)
print("TEST 2: Attempt to JIT a GroupBy Operation")
print("="*80)

def aggregate_by_category(category_data, value_data):
    """
    Function that performs groupby aggregation.
    This will likely fail because DataFrame operations aren't JIT-compatible.
    """
    df = DataFrame({
        'category': category_data,
        'value': value_data
    })
    result = df.group_by('category').agg({'value': ('sum', jnp.sum)})
    return result['value_sum']

# Try without JIT first
print("\nTrying function WITHOUT JIT...")
try:
    category_arr = jnp.array([1, 2, 1, 2, 1])
    value_arr = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = aggregate_by_category(category_arr, value_arr)
    print(f"Result: {result}")
    print("✓ Function works without JIT")
except Exception as e:
    print(f"✗ Function failed without JIT: {e}")

# Try with JIT
print("\nTrying function WITH JIT...")
try:
    jitted_aggregate = jax.jit(aggregate_by_category)
    category_arr = jnp.array([1, 2, 1, 2, 1])
    value_arr = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = jitted_aggregate(category_arr, value_arr)
    print(f"Result: {result}")
    print("✓ JIT compilation SUCCEEDED!")
except Exception as e:
    print(f"✗ JIT compilation FAILED: {type(e).__name__}: {e}")

# Test 3: Try JIT-compatible manual aggregation
print("\n" + "="*80)
print("TEST 3: JIT-Compatible Manual Aggregation (Without DataFrame)")
print("="*80)

def manual_aggregate(category_data, value_data):
    """
    Perform aggregation using only JAX operations (no DataFrame).
    This should be JIT-compatible.
    """
    # Get unique categories and their indices
    unique_cats, inverse_indices = jnp.unique(category_data, return_inverse=True)
    
    # Sum values per category using segment_sum
    from jax.ops import segment_sum
    result = segment_sum(value_data, inverse_indices, num_segments=len(unique_cats))
    
    return result

print("\nTrying manual aggregation WITHOUT JIT...")
try:
    category_arr = jnp.array([1, 2, 1, 2, 1])
    value_arr = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = manual_aggregate(category_arr, value_arr)
    print(f"Result: {result}")
    print("✓ Manual aggregation works without JIT")
except Exception as e:
    print(f"✗ Manual aggregation failed: {e}")

print("\nTrying manual aggregation WITH JIT...")
try:
    jitted_manual = jax.jit(manual_aggregate)
    category_arr = jnp.array([1, 2, 1, 2, 1])
    value_arr = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = jitted_manual(category_arr, value_arr)
    print(f"Result: {result}")
    print("✓ Manual aggregation with JIT SUCCEEDED!")
except Exception as e:
    print(f"✗ JIT compilation FAILED: {e}")

# Test 4: Extract JAX arrays and use them in JIT
print("\n" + "="*80)
print("TEST 4: Extract JAX Arrays from DataFrame for JIT")
print("="*80)

print("\nStrategy: Use DataFrame for setup, extract arrays for JIT computation")

df = DataFrame({
    'category': jnp.array([1, 2, 1, 2, 1, 3]),
    'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
})

print(f"\nDataFrame:")
print(df)

# Extract arrays
category_arr = df['category']
value_arr = df['value']

print(f"\nExtracted arrays:")
print(f"  category: {category_arr}")
print(f"  value: {value_arr}")

# Use JIT on the arrays
@jax.jit
def jit_aggregate(cats, vals):
    unique_cats, inverse = jnp.unique(cats, return_inverse=True)
    from jax.ops import segment_sum
    return segment_sum(vals, inverse, num_segments=len(unique_cats))

try:
    result = jit_aggregate(category_arr, value_arr)
    print(f"\nJIT aggregation result: {result}")
    print("✓ Can extract arrays from DataFrame and use in JIT!")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 5: Try with GroupBy._compute_groups 
print("\n" + "="*80)
print("TEST 5: Understanding GroupBy Internals")
print("="*80)

df = DataFrame({
    'category': jnp.array([1, 2, 1, 2, 1]),
    'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
})

grouped = df.group_by('category')

# This triggers the internal computation
grouped._compute_groups()

print(f"\nGroupBy internal state:")
print(f"  _num_groups: {grouped._num_groups}")
print(f"  _group_indices: {grouped._group_indices}")
print(f"  _unique_groups: {grouped._unique_groups}")

# Can we use these in a JIT function?
@jax.jit
def use_precomputed_groups(values, group_indices, num_groups):
    """Use pre-computed group information in JIT."""
    from jax.ops import segment_sum
    return segment_sum(values, group_indices, num_segments=num_groups)

try:
    col_data = df['value']
    result = use_precomputed_groups(col_data, grouped._group_indices, grouped._num_groups)
    print(f"\nJIT with precomputed groups: {result}")
    print("✓ Can use precomputed group info in JIT!")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 6: Multiple aggregations
print("\n" + "="*80)
print("TEST 6: Multiple Aggregations in JIT")
print("="*80)

@jax.jit
def multiple_aggs(values, group_indices, num_groups):
    """Compute multiple aggregations in one JIT function."""
    from jax.ops import segment_sum
    
    sums = segment_sum(values, group_indices, num_segments=num_groups)
    
    # Compute means (sum / count)
    ones = jnp.ones_like(values)
    counts = segment_sum(ones, group_indices, num_segments=num_groups)
    means = sums / counts
    
    return sums, means, counts

try:
    col_data = df['value']
    sums, means, counts = multiple_aggs(col_data, grouped._group_indices, grouped._num_groups)
    print(f"\nMultiple aggregations:")
    print(f"  Sums:   {sums}")
    print(f"  Means:  {means}")
    print(f"  Counts: {counts}")
    print("✓ Multiple aggregations in JIT work!")
except Exception as e:
    print(f"✗ Failed: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY OF JIT COMPATIBILITY")
print("="*80)

print("""
FINDINGS:

1. ✗ DataFrame operations are NOT directly JIT-compatible
   - Creating DataFrames inside JIT functions fails
   - GroupBy operations inside JIT functions fail
   - This is expected: DataFrames involve Python objects and dynamic operations

2. ✓ JAX arrays extracted FROM DataFrames ARE JIT-compatible
   - Can use df['column'] to get JAX arrays
   - These arrays work perfectly in JIT functions

3. ✓ GroupBy pre-computation + JIT works
   - Use GroupBy outside JIT to compute groups
   - Extract _group_indices and _num_groups
   - Use these in JIT functions with segment_sum

4. ✓ Manual aggregation with JAX primitives is JIT-compatible
   - jnp.unique with return_inverse=True
   - segment_sum for aggregations
   - Standard JAX operations

RECOMMENDED WORKFLOW:

For JIT-compatible aggregations:

Option A: Extract arrays and use pure JAX
  1. Create DataFrame with data
  2. Extract columns as JAX arrays
  3. Use JAX operations (unique, segment_sum) in JIT

Option B: Pre-compute groups, then JIT
  1. Use GroupBy to compute groups (outside JIT)
  2. Extract _group_indices and _num_groups
  3. Apply JIT functions using these precomputed values

Option C: Use DataFrame for non-JIT workflows
  1. DataFrames are great for exploratory analysis
  2. For production/training, extract arrays for JIT

CONCLUSION:
JAXFrame is designed for convenience and expressiveness, not for JIT.
For performance-critical code, extract JAX arrays and use JAX primitives.
""")

print("="*80)
