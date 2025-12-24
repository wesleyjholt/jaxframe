#!/usr/bin/env python3
"""
Debug script to investigate why JAXFrame appears faster than raw JAX.
This is impossible, so there must be a measurement error.
"""

import sys
import os
import time
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
from jaxframe import DataFrame

def timer_decorator(func):
    """Detailed timer decorator."""
    def wrapper(*args, **kwargs):
        # Multiple measurements for accuracy
        times = []
        results = []
        
        for i in range(5):  # 5 measurements
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            results.append(result)
        
        # Check consistency
        for r in results[1:]:
            if isinstance(results[0], (int, float, complex)):
                if not jnp.allclose(results[0], r, rtol=1e-10):
                    print(f"WARNING: Results not consistent in {func.__name__}")
            
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"{func.__name__}:")
        print(f"  Times: {[f'{t:.9f}' for t in times]}")
        print(f"  Average: {avg_time:.9f}s")
        print(f"  Min: {min_time:.9f}s, Max: {max_time:.9f}s")
        print(f"  Result: {results[0]}")
        
        return results[0], avg_time
    return wrapper

def investigate_basic_operations():
    """Investigate the basic operations performance discrepancy."""
    print("=== INVESTIGATING BASIC OPERATIONS ===")
    
    # Test data
    size = 10000
    np.random.seed(42)
    data_np = np.random.randn(size)
    
    print(f"Data size: {size}")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Create JAX and DataFrame data
    jax_data = jnp.array(data_np)
    df_data = DataFrame({'values': jax_data})
    
    print(f"JAX data type: {type(jax_data)}, shape: {jax_data.shape}")
    print(f"DataFrame type: {type(df_data)}")
    print(f"DataFrame column type: {type(df_data['values'])}")
    
    # Warm up both JAX and JAXFrame
    print("\n--- WARMUP ---")
    _ = jnp.sum(jax_data) + jnp.mean(jax_data) * jnp.std(jax_data)
    _ = jnp.sum(df_data['values']) + jnp.mean(df_data['values']) * jnp.std(df_data['values'])
    
    @timer_decorator
    def jax_basic_ops(data):
        return jnp.sum(data) + jnp.mean(data) * jnp.std(data)
    
    @timer_decorator
    def jaxframe_basic_ops(df):
        values = df['values']
        return jnp.sum(values) + jnp.mean(values) * jnp.std(values)
    
    @timer_decorator
    def jaxframe_basic_ops_extracted(values):
        # Direct operations on extracted values (same as JAX)
        return jnp.sum(values) + jnp.mean(values) * jnp.std(values)
    
    print("\n--- PERFORMANCE TEST ---")
    jax_result, jax_time = jax_basic_ops(jax_data)
    jaxframe_result, jaxframe_time = jaxframe_basic_ops(df_data)
    extracted_result, extracted_time = jaxframe_basic_ops_extracted(df_data['values'])
    
    print(f"\n--- RESULTS ---")
    print(f"JAX result: {jax_result}")
    print(f"JAXFrame result: {jaxframe_result}")  
    print(f"Extracted result: {extracted_result}")
    print(f"Results match: JAX vs JAXFrame = {jnp.allclose(jax_result, jaxframe_result)}")
    print(f"Results match: JAX vs Extracted = {jnp.allclose(jax_result, extracted_result)}")
    
    print(f"\n--- TIMING ANALYSIS ---")
    print(f"JAX time: {jax_time:.9f}s")
    print(f"JAXFrame time: {jaxframe_time:.9f}s")
    print(f"Extracted time: {extracted_time:.9f}s")
    print(f"JAXFrame overhead: {jaxframe_time / jax_time:.2f}x")
    print(f"Extracted overhead: {extracted_time / jax_time:.2f}x")
    
    if jaxframe_time < jax_time:
        print(f"\n❌ IMPOSSIBLE: JAXFrame faster than JAX by {jax_time / jaxframe_time:.2f}x")
        print("This suggests a measurement error or caching issue.")
    
    return jax_time, jaxframe_time, extracted_time

if __name__ == "__main__":
    investigate_basic_operations()
    
    print("\n\n=== CONCLUSION ===")
    print("If JAXFrame appears faster than JAX, this indicates:")
    print("1. Measurement error (timing includes compilation/data creation)")
    print("2. Caching effects")  
    print("3. Different operations being performed")
    print("4. JIT compilation artifacts")
    print("\nJAXFrame should NEVER be faster than raw JAX for identical operations!")
