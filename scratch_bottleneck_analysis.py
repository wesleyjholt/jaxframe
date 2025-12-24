"""
Performance Bottleneck Analysis - Finding and Testing Optimizations

This script identifies specific bottlenecks in JAXFrame and tests optimization strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
import jax.numpy as jnp
import numpy as np
from src.jaxframe import DataFrame

# Profiling helper
def profile_function(func, *args, name="Function", n_runs=10):
    """Profile a function with timing and memory info."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"{name}: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    return result

def main():
    print("=== JAXFrame Performance Bottleneck Analysis ===\n")
    
    # Test data
    n_rows = 1000
    test_data = {
        'x': jnp.array(np.random.randn(n_rows).astype(np.float32)),
        'y': jnp.array(np.random.randn(n_rows).astype(np.float32)),
        'id': [f'row_{i}' for i in range(n_rows)]
    }
    
    # 1. Analyze DataFrame creation bottlenecks
    print("1. DataFrame Creation Bottlenecks:")
    print("=" * 40)
    
    # Current implementation
    def create_dataframe_current():
        return DataFrame(test_data)
    
    # Optimized: Skip type detection when we know types
    def create_dataframe_fast():
        # Hypothetical optimized constructor
        df = object.__new__(DataFrame)
        df._data = {k: v for k, v in test_data.items()}
        df._column_types = {
            'x': 'jax_array', 
            'y': 'jax_array', 
            'id': 'list'
        }
        df._length = n_rows
        df._columns = tuple(test_data.keys())
        df._name = None
        return df
    
    profile_function(create_dataframe_current, name="Current DataFrame creation")
    profile_function(create_dataframe_fast, name="Optimized DataFrame creation")
    print()
    
    # 2. Column access bottlenecks
    print("2. Column Access Bottlenecks:")
    print("=" * 40)
    
    df = DataFrame(test_data)
    
    def access_column_current():
        return df['x']
    
    def access_column_direct():
        # Direct access without copy
        return df._data['x']
    
    profile_function(access_column_current, name="Current column access")
    profile_function(access_column_direct, name="Direct column access")
    print()
    
    # 3. JIT compilation overhead analysis
    print("3. JIT Compilation Overhead:")
    print("=" * 40)
    
    # Test different JIT scenarios
    @jax.jit
    def jit_with_dataframe_creation(data_x, data_y):
        # Creates DataFrame inside JIT - expensive
        df = DataFrame({'x': data_x, 'y': data_y})
        return jnp.sum(df['x'] * df['y'])
    
    @jax.jit
    def jit_with_dict_operations(data_dict):
        # Use dict directly - should be faster
        return jnp.sum(data_dict['x'] * data_dict['y'])
    
    @jax.jit
    def jit_pure_arrays(x, y):
        # Pure array operations - fastest
        return jnp.sum(x * y)
    
    # First calls (include compilation time)
    print("Compilation + execution times:")
    start = time.perf_counter()
    result1 = jit_with_dataframe_creation(test_data['x'], test_data['y'])
    time1 = time.perf_counter() - start
    
    start = time.perf_counter()
    result2 = jit_with_dict_operations({'x': test_data['x'], 'y': test_data['y']})
    time2 = time.perf_counter() - start
    
    start = time.perf_counter()
    result3 = jit_pure_arrays(test_data['x'], test_data['y'])
    time3 = time.perf_counter() - start
    
    print(f"JIT + DataFrame creation: {time1*1000:.1f} ms")
    print(f"JIT + dict operations: {time2*1000:.1f} ms") 
    print(f"JIT + pure arrays: {time3*1000:.1f} ms")
    print(f"DataFrame overhead: {(time1/time3 - 1)*100:.1f}%")
    print()
    
    # Cached execution times
    print("Cached execution times:")
    profile_function(lambda: jit_with_dataframe_creation(test_data['x'], test_data['y']), 
                    name="JIT + DataFrame (cached)")
    profile_function(lambda: jit_with_dict_operations({'x': test_data['x'], 'y': test_data['y']}), 
                    name="JIT + dict (cached)")
    profile_function(lambda: jit_pure_arrays(test_data['x'], test_data['y']), 
                    name="JIT + pure arrays (cached)")
    print()

if __name__ == "__main__":
    main()