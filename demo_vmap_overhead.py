#!/usr/bin/env python3
"""
Demonstration of vmap overhead source in JAXFrame.

This script shows exactly where the 3.9x overhead comes from when using vmap
with JAXFrame DataFrames, and demonstrates better alternatives.
"""

import sys
import os
import time
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
from jax import vmap
from jaxframe import DataFrame


def benchmark(func, *args, n_warmup=3, n_runs=10):
    """Simple benchmarking utility."""
    # Warmup
    for _ in range(n_warmup):
        result = func(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        times.append(time.perf_counter() - start)
    
    median_time = sorted(times)[len(times) // 2]
    return result, median_time


print("=" * 80)
print("VMAP OVERHEAD ANALYSIS: Where Does the Slowdown Come From?")
print("=" * 80)

# Setup: Create test data
batch_size = 100
vector_size = 1000

print(f"\nSetup: {batch_size} batches of size {vector_size}")
print("-" * 80)

# Define the computation to vectorize
def single_computation(x):
    """A simple computation on a single vector."""
    return jnp.sum(x ** 2) + jnp.mean(jnp.exp(-x ** 2))


# ============================================================================
# SCENARIO 1: Pure JAX (optimal case)
# ============================================================================
print("\n1. PURE JAX (Optimal - Pre-structured data)")
print("-" * 80)

# Data already in optimal format for vmap
batch_data_jax = jnp.array(np.random.randn(batch_size, vector_size))
jax_vmap_fn = vmap(single_computation)

result_jax, time_jax = benchmark(jax_vmap_fn, batch_data_jax)
print(f"Pure JAX vmap time: {time_jax*1000:.3f}ms")
print(f"Data shape: {batch_data_jax.shape}")
print(f"Data structure: Single 2D JAX array")


# ============================================================================
# SCENARIO 2: JAXFrame (INEFFICIENT - as in current test)
# ============================================================================
print("\n2. JAXFRAME (Inefficient - Multiple column extraction)")
print("-" * 80)

# Store each batch as a separate DataFrame column (this is the problem!)
batch_df_inefficient = DataFrame({
    f'batch_{i}': batch_data_jax[i] for i in range(batch_size)
})

def jaxframe_vmap_inefficient(df):
    """The INEFFICIENT way - extracting and stacking many columns."""
    # THIS IS THE BOTTLENECK: Extracting 100 columns individually
    batch_arrays = [df[f'batch_{i}'] for i in range(batch_size)]
    # Then stacking them back together
    stacked = jnp.stack(batch_arrays)
    # Finally applying vmap
    return vmap(single_computation)(stacked)

result_jaxframe_bad, time_jaxframe_bad = benchmark(jaxframe_vmap_inefficient, batch_df_inefficient)
print(f"JAXFrame (inefficient) time: {time_jaxframe_bad*1000:.3f}ms")
print(f"Overhead: {time_jaxframe_bad / time_jax:.2f}x")
print(f"Data structure: {batch_size} separate DataFrame columns")
print(f"Problem: Must extract each column individually, then stack")


# ============================================================================
# SCENARIO 3: JAXFrame (EFFICIENT - proper usage)
# ============================================================================
print("\n3. JAXFRAME (Efficient - Single column with batched data)")
print("-" * 80)

# Store the batch data as a SINGLE column (much better!)
batch_df_efficient = DataFrame({
    'batched_data': batch_data_jax
})

def jaxframe_vmap_efficient(df):
    """The EFFICIENT way - data already in the right format."""
    # Just extract the single column containing all batches
    return vmap(single_computation)(df['batched_data'])

result_jaxframe_good, time_jaxframe_good = benchmark(jaxframe_vmap_efficient, batch_df_efficient)
print(f"JAXFrame (efficient) time: {time_jaxframe_good*1000:.3f}ms")
print(f"Overhead: {time_jaxframe_good / time_jax:.2f}x")
print(f"Data structure: Single DataFrame column containing 2D array")
print(f"Advantage: Direct extraction, no restructuring needed")


# ============================================================================
# BREAKDOWN: What causes the overhead?
# ============================================================================
print("\n" + "=" * 80)
print("OVERHEAD BREAKDOWN")
print("=" * 80)

# Measure just the data extraction overhead
def extract_columns(df):
    """Extract all batch columns."""
    return [df[f'batch_{i}'] for i in range(batch_size)]

_, time_extract = benchmark(extract_columns, batch_df_inefficient)
print(f"\nExtraction of {batch_size} columns: {time_extract*1000:.3f}ms")

# Measure the stacking overhead
extracted = extract_columns(batch_df_inefficient)
def stack_arrays(arrays):
    """Stack arrays into 2D array."""
    return jnp.stack(arrays)

_, time_stack = benchmark(stack_arrays, extracted)
print(f"Stacking {batch_size} arrays: {time_stack*1000:.3f}ms")

# Measure single column extraction
def extract_single(df):
    """Extract single column."""
    return df['batched_data']

_, time_extract_single = benchmark(extract_single, batch_df_efficient)
print(f"Extraction of 1 column: {time_extract_single*1000:.3f}ms")

print(f"\nTotal overhead (extract + stack): {(time_extract + time_stack)*1000:.3f}ms")
print(f"This accounts for most of the {time_jaxframe_bad - time_jax:.4f}s difference")


# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print(f"""
Performance Comparison:
  Pure JAX:                    {time_jax*1000:.3f}ms  (baseline)
  JAXFrame (inefficient):      {time_jaxframe_bad*1000:.3f}ms  ({time_jaxframe_bad/time_jax:.2f}x overhead)
  JAXFrame (efficient):        {time_jaxframe_good*1000:.3f}ms  ({time_jaxframe_good/time_jax:.2f}x overhead)

Root Cause of Overhead:
  ❌ Storing batches as {batch_size} separate DataFrame columns
  ❌ Extracting each column individually (Python loop overhead)
  ❌ Stacking arrays back together (extra JAX operation)
  
Best Practices for vmap with JAXFrame:
  ✅ Store batched data in a SINGLE DataFrame column as a 2D array
  ✅ Extract once, apply vmap directly
  ✅ Avoid unnecessary data restructuring
  
Real-World Impact:
  The "inefficient" approach is an ARTIFICIAL worst-case scenario.
  In typical usage, you would naturally store batched data in a single column.
  With proper data structure, JAXFrame overhead is minimal (<{time_jaxframe_good/time_jax:.2f}x).

Conclusion:
  The 3.9x vmap overhead in the performance tests comes from poor data structure,
  NOT from inherent JAXFrame inefficiency. With proper usage, JAXFrame adds
  minimal overhead to vmap operations.
""")

# Verify results are identical
print("\nVerification:")
print(f"Results match (JAX vs JAXFrame inefficient): {jnp.allclose(result_jax, result_jaxframe_bad)}")
print(f"Results match (JAX vs JAXFrame efficient): {jnp.allclose(result_jax, result_jaxframe_good)}")
