#!/usr/bin/env python3
"""
Performance comparison tests between JAX numpy operations and JAXFrame.

This module tests compilation and runtime speeds for:
- Basic operations
- JIT compilation
- vmap (vectorization)
- grad (automatic differentiation)
- Operations with missing values (using MaskedArray)
"""

import pytest
import numpy as np
import time
import sys
import os
from typing import Callable, Tuple, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from jaxframe import DataFrame
from jaxframe.masked_array import MaskedArray


def timer(n_warmup: int = 3, n_runs: int = 10):
    """Decorator factory to time function execution with proper warm-up and multiple runs.
    
    Args:
        n_warmup: Number of warm-up runs (not timed)
        n_runs: Number of timed runs for averaging
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Warm-up runs to ensure compilation and caching are done
            for _ in range(n_warmup):
                result = func(*args, **kwargs)
                # Block until ready for JAX operations
                if hasattr(result, 'block_until_ready'):
                    result.block_until_ready()
                elif isinstance(result, (tuple, list)):
                    for r in result:
                        if hasattr(r, 'block_until_ready'):
                            r.block_until_ready()
            
            # Timed runs
            times = []
            results = []
            for _ in range(n_runs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                # Critical: Block until computation is actually complete
                if hasattr(result, 'block_until_ready'):
                    result.block_until_ready()
                elif isinstance(result, (tuple, list)):
                    for r in result:
                        if hasattr(r, 'block_until_ready'):
                            r.block_until_ready()
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                results.append(result)
            
            # Use median time to reduce noise
            median_time = sorted(times)[len(times) // 2]
            return results[0], median_time
        return wrapper
    return decorator


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestPerformanceComparison:
    """Test performance comparison between JAX and JAXFrame operations."""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Set up test data for performance comparisons."""
        np.random.seed(42)
        
        # Small dataset
        self.small_size = 1000
        self.small_jax_data = jnp.array(np.random.randn(self.small_size))
        self.small_df = DataFrame({
            'values': self.small_jax_data,
            'indices': jnp.arange(self.small_size)
        })
        
        # Medium dataset
        self.medium_size = 10000
        self.medium_jax_data = jnp.array(np.random.randn(self.medium_size))
        self.medium_df = DataFrame({
            'values': self.medium_jax_data,
            'indices': jnp.arange(self.medium_size)
        })
        
        # Large dataset
        self.large_size = 100000
        self.large_jax_data = jnp.array(np.random.randn(self.large_size))
        self.large_df = DataFrame({
            'values': self.large_jax_data,
            'indices': jnp.arange(self.large_size)
        })
        
        # Data with missing values
        self.missing_size = 5000
        data_with_missing = np.random.randn(self.missing_size)
        mask = np.random.random(self.missing_size) > 0.2  # 80% valid (True = valid, False = masked)
        
        # Create index DataFrame for MaskedArray
        index_df = DataFrame({'index': list(range(self.missing_size))})
        
        self.masked_array = MaskedArray(
            data=jnp.array(data_with_missing),
            mask=np.array(mask),  # MaskedArray expects numpy mask
            index_df=index_df
        )
        
        self.df_with_missing = DataFrame({
            'values': self.masked_array.data,
            'mask': jnp.array(self.masked_array.mask),  # Convert numpy mask to JAX for DataFrame
            'indices': jnp.arange(self.missing_size)
        })
    
    def test_basic_operations_performance(self):
        """Test performance of basic mathematical operations."""
        
        # JAX operations
        @timer(n_warmup=5, n_runs=20)
        def jax_basic_ops(data):
            return jnp.sum(data) + jnp.mean(data) * jnp.std(data)
        
        # JAXFrame operations
        @timer(n_warmup=5, n_runs=20)
        def jaxframe_basic_ops(df):
            values = df['values']
            return jnp.sum(values) + jnp.mean(values) * jnp.std(values)
        
        # Test on different sizes
        sizes = [
            (self.small_jax_data, self.small_df, "small"),
            (self.medium_jax_data, self.medium_df, "medium"),
            (self.large_jax_data, self.large_df, "large")
        ]
        
        results = {}
        
        for jax_data, df_data, size_name in sizes:
            # JAX performance
            jax_result, jax_time = jax_basic_ops(jax_data)
            
            # JAXFrame performance
            jaxframe_result, jaxframe_time = jaxframe_basic_ops(df_data)
            
            # Results should be identical
            assert jnp.allclose(jax_result, jaxframe_result), f"Results differ for {size_name} data"
            
            results[size_name] = {
                'jax_time': jax_time,
                'jaxframe_time': jaxframe_time,
                'overhead_factor': jaxframe_time / jax_time if jax_time > 0 else float('inf')
            }
            
            print(f"\nBasic Operations ({size_name}):")
            print(f"  JAX time: {jax_time:.6f}s")
            print(f"  JAXFrame time: {jaxframe_time:.6f}s")
            print(f"  Overhead factor: {results[size_name]['overhead_factor']:.2f}x")
        
        # JAXFrame should have reasonable overhead (less than 10x for basic operations)
        for size_name, result in results.items():
            assert result['overhead_factor'] < 10.0, f"JAXFrame overhead too high for {size_name}: {result['overhead_factor']:.2f}x"
    
    def test_jit_compilation_performance(self):
        """Test JIT compilation performance and execution speed."""
        
        # Define functions for JIT compilation
        def jax_compute(data):
            return jnp.sum(data ** 2) + jnp.mean(jnp.sin(data))
        
        def jaxframe_compute_raw(values):
            return jnp.sum(values ** 2) + jnp.mean(jnp.sin(values))
        
        # JIT compile the functions
        jax_jit_compute = jit(jax_compute)
        jaxframe_jit_compute_raw = jit(jaxframe_compute_raw)
        
        # COMPILATION TIME: Measure first call which includes compilation
        def measure_compile_time_jax(data):
            start_time = time.perf_counter()
            result = jax_jit_compute(data)
            result.block_until_ready()  # Ensure compilation and execution are done
            end_time = time.perf_counter()
            return result, end_time - start_time
        
        def measure_compile_time_jaxframe(values):
            start_time = time.perf_counter()
            result = jaxframe_jit_compute_raw(values)
            result.block_until_ready()  # Ensure compilation and execution are done
            end_time = time.perf_counter()
            return result, end_time - start_time
        
        # Measure compilation + first execution
        jax_result_compile, jax_compile_time = measure_compile_time_jax(self.medium_jax_data)
        jaxframe_result_compile, jaxframe_compile_time = measure_compile_time_jaxframe(self.medium_df['values'])
        
        assert jnp.allclose(jax_result_compile, jaxframe_result_compile), "JIT compilation results differ"
        
        # RUNTIME PERFORMANCE: Now functions are compiled, measure pure runtime
        @timer(n_warmup=2, n_runs=20)  # More runs for runtime since it's fast
        def run_compiled_jax(data):
            return jax_jit_compute(data)
        
        @timer(n_warmup=2, n_runs=20)
        def run_compiled_jaxframe(values):
            return jaxframe_jit_compute_raw(values)
        
        # Runtime performance (functions already compiled)
        jax_result_run, jax_run_time = run_compiled_jax(self.medium_jax_data)
        jaxframe_result_run, jaxframe_run_time = run_compiled_jaxframe(self.medium_df['values'])
        
        assert jnp.allclose(jax_result_run, jaxframe_result_run), "JIT runtime results differ"
        
        print(f"\nJIT Compilation Performance:")
        print(f"  JAX compile+first run time: {jax_compile_time:.6f}s")
        print(f"  JAXFrame compile+first run time: {jaxframe_compile_time:.6f}s")
        print(f"  Compile overhead: {jaxframe_compile_time / jax_compile_time:.2f}x")
        
        print(f"\nJIT Runtime Performance (after compilation):")
        print(f"  JAX runtime: {jax_run_time:.9f}s")
        print(f"  JAXFrame runtime: {jaxframe_run_time:.9f}s")
        print(f"  Runtime overhead: {jaxframe_run_time / jax_run_time:.2f}x")
        
        # Compiled JAXFrame should have minimal runtime overhead
        runtime_overhead = jaxframe_run_time / jax_run_time if jax_run_time > 0 else float('inf')
        assert runtime_overhead < 5.0, f"JIT runtime overhead too high: {runtime_overhead:.2f}x"
    
    def test_vmap_performance(self):
        """Test vmap (vectorization) performance."""
        
        # Create batch data
        batch_size = 100
        vector_size = 1000
        
        batch_data_jax = jnp.array(np.random.randn(batch_size, vector_size))
        batch_df = DataFrame({
            f'batch_{i}': batch_data_jax[i] for i in range(batch_size)
        })
        
        # Define function to vectorize
        def single_computation(x):
            return jnp.sum(x ** 2) + jnp.mean(jnp.exp(-x ** 2))
        
        # JAX vmap
        jax_vmap_compute = vmap(single_computation)
        
        # JAXFrame vmap (operating on extracted data)
        def jaxframe_vmap_compute(df):
            # Extract all batch columns as a stacked array
            batch_arrays = [df[f'batch_{i}'] for i in range(batch_size)]
            stacked = jnp.stack(batch_arrays)
            return vmap(single_computation)(stacked)
        
        # Test performance
        @timer(n_warmup=3, n_runs=10)
        def run_jax_vmap(data):
            return jax_vmap_compute(data)
        
        @timer(n_warmup=3, n_runs=10)
        def run_jaxframe_vmap(df):
            return jaxframe_vmap_compute(df)
        
        # Execute and time
        jax_result, jax_time = run_jax_vmap(batch_data_jax)
        jaxframe_result, jaxframe_time = run_jaxframe_vmap(batch_df)
        
        assert jnp.allclose(jax_result, jaxframe_result), "vmap results differ"
        
        print(f"\nvmap Performance:")
        print(f"  JAX vmap time: {jax_time:.6f}s")
        print(f"  JAXFrame vmap time: {jaxframe_time:.6f}s")
        print(f"  vmap overhead: {jaxframe_time / jax_time:.2f}x")
        
        # vmap should have reasonable overhead
        vmap_overhead = jaxframe_time / jax_time if jax_time > 0 else float('inf')
        assert vmap_overhead < 10.0, f"vmap overhead too high: {vmap_overhead:.2f}x"
    
    def test_grad_performance(self):
        """Test automatic differentiation (grad) performance."""
        
        # Define function for differentiation
        def loss_function(params):
            return jnp.sum(params ** 4) - 2 * jnp.sum(params ** 2) + jnp.sum(params)
        
        # JAX grad
        jax_grad_fn = grad(loss_function)
        
        # JAXFrame grad (operating on extracted data)
        def jaxframe_grad_compute(df):
            params = df['values']
            grad_fn = grad(loss_function)
            return grad_fn(params)
        
        # Test data
        params_size = 5000
        params_data = jnp.array(np.random.randn(params_size))
        params_df = DataFrame({'values': params_data})
        
        # Test performance
        @timer(n_warmup=3, n_runs=10)
        def run_jax_grad(params):
            return jax_grad_fn(params)
        
        @timer(n_warmup=3, n_runs=10)
        def run_jaxframe_grad(df):
            return jaxframe_grad_compute(df)
        
        # Execute and time
        jax_result, jax_time = run_jax_grad(params_data)
        jaxframe_result, jaxframe_time = run_jaxframe_grad(params_df)
        
        assert jnp.allclose(jax_result, jaxframe_result), "grad results differ"
        
        print(f"\ngrad Performance:")
        print(f"  JAX grad time: {jax_time:.6f}s")
        print(f"  JAXFrame grad time: {jaxframe_time:.6f}s")
        print(f"  grad overhead: {jaxframe_time / jax_time:.2f}x")
        
        # grad should have reasonable overhead
        grad_overhead = jaxframe_time / jax_time if jax_time > 0 else float('inf')
        assert grad_overhead < 15.0, f"grad overhead too high: {grad_overhead:.2f}x"
    
    def test_missing_values_performance(self):
        """Test performance with missing values using MaskedArray."""
        
        # JAX operations with manual masking
        @timer(n_warmup=3, n_runs=10)
        def jax_masked_operations(data, mask):
            # Manual masking - mask convention: True = valid, False = masked
            # Only use valid data for calculations
            valid_data = data[mask]
            
            if len(valid_data) == 0:
                return 0.0, 0.0, 0.0
            
            # Compute statistics only on valid data
            masked_sum = jnp.sum(valid_data)
            masked_mean = jnp.mean(valid_data)
            masked_var = jnp.var(valid_data)
            
            return masked_sum, masked_mean, masked_var
        
        # JAXFrame operations with MaskedArray
        @timer(n_warmup=3, n_runs=10)
        def jaxframe_masked_operations(masked_array):
            # Use MaskedArray methods
            valid_data = masked_array.get_valid_data()
            
            if len(valid_data) == 0:
                return 0.0, 0.0, 0.0
            
            masked_sum = jnp.sum(valid_data)
            masked_mean = jnp.mean(valid_data)
            masked_var = jnp.var(valid_data)
            
            return masked_sum, masked_mean, masked_var
        
        # Execute and time
        jax_results, jax_time = jax_masked_operations(
            self.masked_array.data, 
            self.masked_array.mask
        )
        
        jaxframe_results, jaxframe_time = jaxframe_masked_operations(self.masked_array)
        
        # Results should be approximately equal (some numerical differences expected)
        for jax_val, jaxframe_val in zip(jax_results, jaxframe_results):
            if not (jnp.isnan(jax_val) and jnp.isnan(jaxframe_val)):
                assert jnp.allclose(jax_val, jaxframe_val, rtol=1e-5), "Masked operations results differ significantly"
        
        print(f"\nMissing Values Performance:")
        print(f"  JAX masked ops time: {jax_time:.6f}s")
        print(f"  JAXFrame masked ops time: {jaxframe_time:.6f}s")
        print(f"  Masked ops overhead: {jaxframe_time / jax_time:.2f}x")
        
        # Missing values operations should have reasonable overhead
        masked_overhead = jaxframe_time / jax_time if jax_time > 0 else float('inf')
        assert masked_overhead < 20.0, f"Masked operations overhead too high: {masked_overhead:.2f}x"
    
    def test_complex_pipeline_performance(self):
        """Test performance of a complex data processing pipeline."""
        
        # Create complex dataset
        pipeline_size = 10000
        pipeline_data = {
            'feature1': jnp.array(np.random.randn(pipeline_size)),
            'feature2': jnp.array(np.random.randn(pipeline_size)),
            'feature3': jnp.array(np.random.randn(pipeline_size)),
            'target': jnp.array(np.random.randn(pipeline_size))
        }
        
        pipeline_df = DataFrame(pipeline_data)
        
        # Complex JAX pipeline
        @timer(n_warmup=3, n_runs=10)
        def jax_complex_pipeline(feature1, feature2, feature3, target):
            # Feature engineering
            interaction = feature1 * feature2
            polynomial = feature1 ** 2 + feature2 ** 2
            normalized_f3 = (feature3 - jnp.mean(feature3)) / jnp.std(feature3)
            
            # Simple linear model computation
            weights = jnp.array([0.5, -0.3, 0.8, 0.2])
            features = jnp.stack([feature1, feature2, interaction, polynomial], axis=1)
            predictions = jnp.dot(features, weights)
            
            # Loss computation
            mse = jnp.mean((predictions - target) ** 2)
            mae = jnp.mean(jnp.abs(predictions - target))
            
            return mse, mae, jnp.mean(predictions)
        
        # JAXFrame pipeline
        @timer(n_warmup=3, n_runs=10)
        def jaxframe_complex_pipeline(df):
            # Feature engineering
            feature1 = df['feature1']
            feature2 = df['feature2']
            feature3 = df['feature3']
            target = df['target']
            
            interaction = feature1 * feature2
            polynomial = feature1 ** 2 + feature2 ** 2
            normalized_f3 = (feature3 - jnp.mean(feature3)) / jnp.std(feature3)
            
            # Simple linear model computation
            weights = jnp.array([0.5, -0.3, 0.8, 0.2])
            features = jnp.stack([feature1, feature2, interaction, polynomial], axis=1)
            predictions = jnp.dot(features, weights)
            
            # Loss computation
            mse = jnp.mean((predictions - target) ** 2)
            mae = jnp.mean(jnp.abs(predictions - target))
            
            return mse, mae, jnp.mean(predictions)
        
        # Execute and time
        jax_results, jax_time = jax_complex_pipeline(
            pipeline_data['feature1'],
            pipeline_data['feature2'],
            pipeline_data['feature3'],
            pipeline_data['target']
        )
        
        jaxframe_results, jaxframe_time = jaxframe_complex_pipeline(pipeline_df)
        
        # Results should be identical
        for jax_val, jaxframe_val in zip(jax_results, jaxframe_results):
            assert jnp.allclose(jax_val, jaxframe_val), "Complex pipeline results differ"
        
        print(f"\nComplex Pipeline Performance:")
        print(f"  JAX pipeline time: {jax_time:.6f}s")
        print(f"  JAXFrame pipeline time: {jaxframe_time:.6f}s")
        print(f"  Pipeline overhead: {jaxframe_time / jax_time:.2f}x")
        
        # Complex pipeline should have reasonable overhead
        pipeline_overhead = jaxframe_time / jax_time if jax_time > 0 else float('inf')
        assert pipeline_overhead < 5.0, f"Complex pipeline overhead too high: {pipeline_overhead:.2f}x"
    
    def test_jit_with_missing_values_performance(self):
        """Test JIT compilation performance with missing values."""
        
        # JIT-compiled function with missing values
        @jit
        def jax_jit_masked_compute(data, mask):
            # mask convention: True = valid, False = masked
            # Use where instead of boolean indexing for JIT compatibility
            masked_data = jnp.where(mask, data, 0.0)
            squared_data = jnp.where(mask, data ** 2, 0.0)
            valid_count = jnp.sum(mask)
            return jnp.sum(squared_data) / jnp.maximum(valid_count, 1)
        
        @jit
        def jaxframe_jit_masked_compute(masked_array_data, masked_array_mask):
            # mask convention: True = valid, False = masked  
            # Use where instead of boolean indexing for JIT compatibility
            squared_data = jnp.where(masked_array_mask, masked_array_data ** 2, 0.0)
            valid_count = jnp.sum(masked_array_mask)
            return jnp.sum(squared_data) / jnp.maximum(valid_count, 1)
        
        # COMPILATION TIME: Measure first call which includes compilation
        def measure_jit_compile_time_jax(data, mask):
            start_time = time.perf_counter()
            result = jax_jit_masked_compute(data, mask)
            result.block_until_ready()
            end_time = time.perf_counter()
            return result, end_time - start_time
        
        def measure_jit_compile_time_jaxframe(df):
            start_time = time.perf_counter()
            result = jaxframe_jit_masked_compute(df['values'], df['mask'])
            result.block_until_ready()
            end_time = time.perf_counter()
            return result, end_time - start_time
        
        # RUNTIME PERFORMANCE: After compilation
        @timer(n_warmup=2, n_runs=20)
        def run_jax_jit_masked_runtime(data, mask):
            return jax_jit_masked_compute(data, mask)
        
        @timer(n_warmup=2, n_runs=20)
        def run_jaxframe_jit_masked_runtime(df):
            return jaxframe_jit_masked_compute(df['values'], df['mask'])
        
        # Measure compilation + first execution
        jax_result_compile, jax_compile_time = measure_jit_compile_time_jax(
            self.masked_array.data, 
            self.masked_array.mask
        )
        
        jaxframe_result_compile, jaxframe_compile_time = measure_jit_compile_time_jaxframe(
            self.df_with_missing
        )
        
        assert jnp.allclose(jax_result_compile, jaxframe_result_compile), "JIT masked compilation results differ"
        
        # Measure pure runtime (functions already compiled)
        jax_result_runtime, jax_runtime_time = run_jax_jit_masked_runtime(
            self.masked_array.data, 
            self.masked_array.mask
        )
        
        jaxframe_result_runtime, jaxframe_runtime_time = run_jaxframe_jit_masked_runtime(
            self.df_with_missing
        )
        
        assert jnp.allclose(jax_result_runtime, jaxframe_result_runtime), "JIT masked runtime results differ"
        
        print(f"\nJIT + Missing Values Performance:")
        print(f"  JAX JIT+compile time: {jax_compile_time:.6f}s")
        print(f"  JAXFrame JIT+compile time: {jaxframe_compile_time:.6f}s")
        print(f"  JIT+compile overhead: {jaxframe_compile_time / jax_compile_time:.2f}x")
        print(f"  JAX runtime time: {jax_runtime_time:.6f}s")
        print(f"  JAXFrame runtime time: {jaxframe_runtime_time:.6f}s")
        print(f"  Runtime overhead: {jaxframe_runtime_time / jax_runtime_time:.2f}x")
        
        # Runtime overhead should be minimal after compilation
        runtime_overhead = jaxframe_runtime_time / jax_runtime_time if jax_runtime_time > 0 else float('inf')
        assert runtime_overhead < 10.0, f"JIT runtime overhead with missing values too high: {runtime_overhead:.2f}x"
    
    def test_memory_efficiency(self):
        """Test memory efficiency of JAXFrame vs raw JAX operations."""
        
        try:
            import psutil
            import os
            
            def get_memory_usage():
                """Get current memory usage in MB."""
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            
            # Create large dataset for memory testing
            memory_test_size = 50000
            
            # Test JAX memory usage
            initial_memory = get_memory_usage()
            
            large_jax_arrays = [
                jnp.array(np.random.randn(memory_test_size)) 
                for _ in range(10)
            ]
            
            jax_memory = get_memory_usage()
            jax_memory_used = jax_memory - initial_memory
            
            # Clean up JAX arrays
            del large_jax_arrays
            
            # Test JAXFrame memory usage
            baseline_memory = get_memory_usage()
            
            large_dataframes = [
                DataFrame({
                    'data': jnp.array(np.random.randn(memory_test_size)),
                    'index': jnp.arange(memory_test_size)
                })
                for _ in range(10)
            ]
            
            jaxframe_memory = get_memory_usage()
            jaxframe_memory_used = jaxframe_memory - baseline_memory
            
            print(f"\nMemory Usage Comparison:")
            print(f"  JAX arrays memory: {jax_memory_used:.2f} MB")
            print(f"  JAXFrame memory: {jaxframe_memory_used:.2f} MB")
            print(f"  Memory overhead: {jaxframe_memory_used / jax_memory_used:.2f}x")
            
            # JAXFrame should not use significantly more memory
            memory_overhead = jaxframe_memory_used / jax_memory_used if jax_memory_used > 0 else float('inf')
            assert memory_overhead < 3.0, f"JAXFrame memory overhead too high: {memory_overhead:.2f}x"
            
            # Clean up
            del large_dataframes
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")


if __name__ == "__main__":
    # Run performance tests with detailed output
    pytest.main([__file__, "-v", "-s"])